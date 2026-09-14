from __future__ import annotations

"""Optional micro-model for conditional point residual prediction.

The deterministic StyleProfile is the default MVP. This module becomes useful after the
user has accumulated hundreds/thousands of fitted stroke pairs.
"""

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]  # optional dependency
    nn = None  # type: ignore[assignment]  # optional dependency


if nn is not None:

    class TinyStyleNet(nn.Module):
        """~100k parameter MLP predicting dx/dy for one trajectory point.

        Input features (example, 12 dims):
          x,y,t, dx,dy, curvature,
          stroke_index_norm, stroke_count_norm,
          bbox_x,bbox_y,bbox_w,bbox_h
        A learned writer embedding can be concatenated later for multi-writer training.
        """

        def __init__(self, in_dim: int = 12, hidden: int = 192, out_dim: int = 2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, out_dim),
            )

        def forward(self, x):
            return self.net(x)
else:

    class TinyStyleNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Install ML extras: pip install -e '.[ml]'")
