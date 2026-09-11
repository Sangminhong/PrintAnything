"""PrintAnything: Learning Geometric Plan Map for 3D Printing G-code Generation
from Unoriented Point Clouds (ECCV 2026).

The package is organised after the three stages of the paper:

    printanything.gplan       Geometric plan (G-plan) map: definition, GT
                              construction from G-code, slice-wise projection of
                              the input point cloud, and the G-code compiler.
    printanything.models      GPNet: point-cloud encoder + FiLM-conditioned U-Net
                              that predicts (M, R, Q) for every slice.
    printanything.infill      Infill pattern dictionary, proxy objectives and the
                              strength/cost recommender.
    printanything.engine      Losses and train/val loops.
    printanything.evaluation  CD / F1_3D / F1_2D and extrusion-flow metrics.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
