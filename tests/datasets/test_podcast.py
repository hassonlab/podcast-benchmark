import mne
import numpy as np

from datasets.podcast import STORED_IN_MICROVOLTS, _to_volts


def _raw(value):
    return mne.io.RawArray(
        np.full((2, 10), value), mne.create_info(["A", "B"], 512.0, ch_types="ecog"), verbose=False
    )


def test_sub01_microvolts_are_rescaled_to_volts():
    assert 1 in STORED_IN_MICROVOLTS
    raw = _to_volts(_raw(23.0), subject_id=1)
    np.testing.assert_allclose(raw.get_data(), 23.0e-6)


def test_other_subjects_are_unchanged():
    raw = _to_volts(_raw(3.0e-5), subject_id=9)
    np.testing.assert_allclose(raw.get_data(), 3.0e-5)
