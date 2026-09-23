"""TimesFM-3 wrapper for multivariate pollen-tip forecasting.

Targets (jointly forecast, cross-series attention couples them):
  tip_x, tip_y, tube_length
Past-only covariate (history only; H144 ablation):
  accepted flag (measurement trust). Tip-speed / length-gain activity
  covariates were ablated — no signal on points or burst-F1 — and removed.
Past-future (dynamic) covariate (known into the horizon):
  time_minutes — the scope's fixed sampling schedule, the exact analogue
  of the "planned promotions" covariate in the TimesFM-3 announcement:
  known future sample times shape expected growth.
"""

from __future__ import annotations

import numpy as np
from timesfm3 import ModelConfig, TimesFM3Evaluator

CHECKPOINT = "google/timesfm-3.0-pytorch"
Q10, Q50, Q90 = 0, 4, 8  # 9 quantiles, 0.1 .. 0.9


class TipForecaster:
    def __init__(self, device: str = "cpu", per_core_batch_size: int = 8):
        self.evaluator = TimesFM3Evaluator(
            ModelConfig(
                checkpoint_path=CHECKPOINT,
                per_core_batch_size=per_core_batch_size,
                device=device,
            )
        )

    def predict_contexts(
        self,
        targets: list[np.ndarray],
        past_covs: list[np.ndarray],
        future_times: list[np.ndarray],
        horizon: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Batch forecast.

        targets:      [(3, T)] tip_x, tip_y, length over context
        past_covs:    [(1, T)] accepted trust over context (H144)
        future_times: [(T + H,)] time_minutes over context + horizon
        returns [(forecast (3, H), quantiles (3, H, 9))]
        """
        past_future = [ft[None, :].astype(np.float32) for ft in future_times]
        outputs = self.evaluator.predict_batch(
            contexts=[t.astype(np.float32) for t in targets],
            horizon=horizon,
            past_only_covariates=[c.astype(np.float32) for c in past_covs],
            past_future_covariates=past_future,
            return_quantiles=True,
            use_symmetric_averaging=False,
        )
        return [
            (np.asarray(o.forecast, dtype=float), np.asarray(o.quantiles, dtype=float))
            for o in outputs
        ]

    def predict_one(
        self,
        targets: np.ndarray,
        past_cov: np.ndarray,
        future_time: np.ndarray,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.predict_contexts([targets], [past_cov], [future_time], horizon)[0]
