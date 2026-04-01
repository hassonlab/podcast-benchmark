from __future__ import annotations

from typing import Optional

import mne
import numpy as np
import torch

from utils import data_utils


class LazyEpochCollator:
    """Materialize raw windows and apply configured preprocessing per batch."""

    def __init__(
        self,
        raws: list[mne.io.Raw],
        lag: int,
        window_width: float,
        preprocessing_fns=None,
        preprocessor_params: Optional[dict | list[dict]] = None,
    ) -> None:
        self.raws = raws
        self.lag = lag
        self.window_width = window_width
        self.preprocessing_fns = preprocessing_fns
        self.preprocessor_params = preprocessor_params
        self.tmin = lag / 1000 - window_width / 2
        self.tmax = lag / 1000 + window_width / 2 - 2e-3

    def __call__(self, batch):
        starts, input_dicts, targets = zip(*batch)
        starts = np.asarray(starts, dtype=np.float32)

        datas = []
        for raw in self.raws:
            events = np.zeros((len(starts), 3), dtype=int)
            events[:, 0] = (starts * raw.info["sfreq"]).astype(int)
            epochs = mne.Epochs(
                raw,
                events,
                tmin=self.tmin,
                tmax=self.tmax,
                baseline=None,
                proj=False,
                event_id=None,
                preload=True,
                on_missing="ignore",
                event_repeated="merge",
                verbose="ERROR",
            )
            if len(epochs.selection) != len(starts):
                raise ValueError(
                    "LazyEpochCollator received invalid starts for at least one raw."
                )
            datas.append(epochs.get_data(copy=False))

        batch_data = np.concatenate(datas, axis=1)
        batch_data = data_utils._apply_preprocessing(
            batch_data, self.preprocessing_fns, self.preprocessor_params
        )
        xb = torch.from_numpy(np.asarray(batch_data, dtype=np.float32))

        merged_inputs = {}
        for key in input_dicts[0].keys():
            vals = [inp[key] for inp in input_dicts]
            merged_inputs[key] = torch.stack(vals)

        first_target = targets[0]
        if torch.is_tensor(first_target):
            yb = torch.stack(list(targets))
        else:
            yb = torch.as_tensor(targets)

        return xb, merged_inputs, yb
