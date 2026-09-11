# Third-party code

## PointTransformerV3

Vendored from https://github.com/Pointcept/PointTransformerV3 (commit `ac4ef87`),
used as the global point-cloud encoder `E_global` described in Sec. 3.2 of the
paper. Only `model.py` and `serialization/` are kept — the parts the encoder
imports. The original licence is preserved in
`PointTransformerV3/LICENSE`.

It needs `spconv` (matching your CUDA build), `torch-scatter`, `timm` and
`addict`; see `requirements.txt`. `--point_encoder pointnet` avoids all of them.
