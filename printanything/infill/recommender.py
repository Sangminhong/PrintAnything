"""Infill policy recommender (paper, Sec. 3.3 and Table 3).

A bootstrap dataset is built by sampling random ``(pattern, scale)`` policies,
synthesising the corresponding infilled G-plan map and recording the proxy
objectives (see :mod:`printanything.infill.proxies` and
``tools/build_infill_dataset.py``). Two random-forest regressors are then fitted
to predict strength and cost from the candidate policy:

    (phi(M', R', Q'), pattern, scale) -> (S, C)                        Eq. (3)

At inference time every candidate in a discrete grid is scored, the dominated
candidates are discarded, and the policy that maximises a strength-minus-cost
utility on the remaining Pareto frontier is returned.
"""

import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["InfillPolicy", "InfillRecommender", "pareto_front"]

#: The features the recommender is conditioned on.
FEATURE_COLS: Tuple[str, ...] = ("pattern", "scale", "layer_idx")


@dataclass(frozen=True)
class InfillPolicy:
    """One candidate infill policy: a template and the scale it is warped by."""

    pattern: str
    scale: float

    def as_features(self, **extra) -> Dict[str, object]:
        return {"pattern": self.pattern, "scale": float(self.scale), **extra}


def pareto_front(strength: Sequence[float], cost: Sequence[float]) -> List[int]:
    """Indices of the non-dominated candidates (higher S, lower C).

    A candidate is dominated when another one is at least as strong *and* at most
    as expensive, and strictly better in one of the two.
    """
    S = np.asarray(strength, dtype=np.float64)
    C = np.asarray(cost, dtype=np.float64)
    keep = []
    for i in range(len(S)):
        dominated = np.any((S >= S[i]) & (C <= C[i]) & ((S > S[i]) | (C < C[i])))
        if not dominated:
            keep.append(i)
    return keep


def _normalise(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(min(max((value - lo) / (hi - lo), 0.0), 1.0))


class InfillRecommender:
    """Surrogate model over infill policies.

    Args:
        feature_cols: features used by the regressors; ``pattern`` is categorical
            and is one-hot encoded by the vectoriser.
        strength_weight / cost_weight: utility trade-off used to pick one point
            on the Pareto frontier.
    """

    def __init__(
        self,
        feature_cols: Sequence[str] = FEATURE_COLS,
        strength_weight: float = 1.0,
        cost_weight: float = 1.0,
        n_estimators: int = 300,
        max_depth: Optional[int] = 14,
        min_samples_leaf: int = 2,
        seed: int = 42,
    ):
        self.feature_cols = list(feature_cols)
        self.strength_weight = float(strength_weight)
        self.cost_weight = float(cost_weight)
        self._forest_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )
        self.vectorizer = None
        self.model_strength = None
        self.model_cost = None
        # Range of the training targets, used to normalise the utility terms.
        self.strength_range: Tuple[float, float] = (0.0, 1.0)
        self.cost_range: Tuple[float, float] = (0.0, 1.0)

    # -- fitting ----------------------------------------------------------
    def _feature_dicts(self, rows: Sequence[Dict]) -> List[Dict[str, object]]:
        out = []
        for row in rows:
            item: Dict[str, object] = {}
            for col in self.feature_cols:
                value = row.get(col, 0.0)
                item[col] = value if isinstance(value, str) else float(value)
            out.append(item)
        return out

    def fit(self, rows: Sequence[Dict], strength_col: str = "strength_proxy", cost_col: str = "cost_time"):
        """Fit both regressors on bootstrap trials.

        Each row holds the features plus the measured proxy objectives.
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_extraction import DictVectorizer

        if not rows:
            raise ValueError("No training rows given.")

        self.vectorizer = DictVectorizer(sparse=False)
        X = self.vectorizer.fit_transform(self._feature_dicts(rows))
        y_s = np.asarray([float(r[strength_col]) for r in rows], dtype=np.float64)
        y_c = np.asarray([float(r[cost_col]) for r in rows], dtype=np.float64)

        self.model_strength = RandomForestRegressor(**self._forest_kwargs).fit(X, y_s)
        self.model_cost = RandomForestRegressor(**self._forest_kwargs).fit(X, y_c)
        self.strength_range = (float(y_s.min()), float(y_s.max()))
        self.cost_range = (float(y_c.min()), float(y_c.max()))
        return self

    # -- inference --------------------------------------------------------
    def predict(self, rows: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Predicted ``(strength, cost)`` for a batch of candidate rows."""
        if self.model_strength is None or self.vectorizer is None:
            raise RuntimeError("The recommender has not been fitted or loaded.")
        X = self.vectorizer.transform(self._feature_dicts(rows))
        return self.model_strength.predict(X), self.model_cost.predict(X)

    def utility(self, strength: np.ndarray, cost: np.ndarray) -> np.ndarray:
        s = np.array([_normalise(v, *self.strength_range) for v in np.atleast_1d(strength)])
        c = np.array([_normalise(v, *self.cost_range) for v in np.atleast_1d(cost)])
        return self.strength_weight * s - self.cost_weight * c

    def recommend(
        self, candidates: Sequence[InfillPolicy], **context
    ) -> Tuple[InfillPolicy, Dict[str, float]]:
        """Pick the best policy out of ``candidates``.

        ``context`` supplies any remaining feature (e.g. ``layer_idx``).

        Returns:
            ``(policy, info)`` where ``info`` carries the predicted strength and
            cost of the pick and the size of the Pareto frontier it came from.
        """
        if not candidates:
            raise ValueError("No candidate policies given.")

        rows = [c.as_features(**context) for c in candidates]
        pred_s, pred_c = self.predict(rows)

        front = pareto_front(pred_s, pred_c)
        utility = self.utility(pred_s, pred_c)
        best = max(front, key=lambda i: utility[i])

        return candidates[best], {
            "pred_strength": float(pred_s[best]),
            "pred_cost": float(pred_c[best]),
            "utility": float(utility[best]),
            "pareto_size": len(front),
        }

    # -- persistence ------------------------------------------------------
    def save(self, path) -> str:
        with open(str(path), "wb") as fh:
            pickle.dump(
                {
                    "version": 2,
                    "feature_cols": self.feature_cols,
                    "vectorizer": self.vectorizer,
                    "model_strength": self.model_strength,
                    "model_cost": self.model_cost,
                    "strength_range": self.strength_range,
                    "cost_range": self.cost_range,
                    "strength_weight": self.strength_weight,
                    "cost_weight": self.cost_weight,
                },
                fh,
            )
        return str(path)

    @classmethod
    def load(cls, path) -> "InfillRecommender":
        with open(str(path), "rb") as fh:
            blob = pickle.load(fh)
        obj = cls(
            feature_cols=blob.get("feature_cols", FEATURE_COLS),
            strength_weight=blob.get("strength_weight", 1.0),
            cost_weight=blob.get("cost_weight", 1.0),
        )
        obj.vectorizer = blob["vectorizer"]
        obj.model_strength = blob["model_strength"]
        obj.model_cost = blob["model_cost"]
        obj.strength_range = tuple(blob.get("strength_range", (0.0, 1.0)))
        obj.cost_range = tuple(blob.get("cost_range", (0.0, 1.0)))
        return obj
