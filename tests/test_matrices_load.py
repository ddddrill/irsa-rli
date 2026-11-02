"""Smoke-тест моделей рассеивателей (*.pkl) — задача F7 (План фиксов §9).

Все 18 файлов matrices/matrices_*.pkl: проверка ключей intens/x0/y0,
конечность, неотрицательность интенсивностей.

`matrices_cloudSAT_high.pkl` усечён с initial release (git c53dc39;
UnpicklingError: pickle data was truncated) — строгий xfail: после
перезаливки файла владельцем тест станет зелёным, а xfail не даст забыть.
"""
import glob

import numpy as np
import pytest

from models.target import Target

PKL_FILES = sorted(glob.glob("matrices/matrices_*.pkl"))


def test_all_matrices_present():
    """18 .pkl в репозитории (приёмка F7:  rendered-set integrity)."""
    assert len(PKL_FILES) == 18, len(PKL_FILES)


@pytest.mark.parametrize("path", PKL_FILES)
def test_matrix_smoke(path):
    if "cloudSAT_high" in path:
        pytest.skip("усечён с c53dc39 — отдельный строгий xfail ниже")
    t = Target(path)
    sc = t._load_scatterers()
    assert {"intens", "x0", "y0"} <= set(sc)
    assert np.isfinite(sc["intens"]).all()
    assert np.isfinite(sc["x0"]).all() and np.isfinite(sc["y0"]).all()
    assert (sc["intens"] > 0).all()
    assert np.abs(sc["x0"]).max() > 0 or np.abs(sc["y0"]).max() > 0


@pytest.mark.xfail(strict=True,
                   reason="matrices_cloudSAT_high.pkl усечён с c53dc39;"
                          "ожидает перезаливки владельцем")
def test_cloudSAT_high_loads():
    t = Target("matrices/matrices_cloudSAT_high.pkl")
    sc = t._load_scatterers()
    assert np.isfinite(sc["intens"]).all()


def test_cloudSAT_high_friendly_error():
    """§9.3: повреждённый pkl → понятная ошибка ValueError (не сырой
    traceback UnpicklingError в глубине конвейера)."""
    import pickle

    from models.radar import Radar
    from processing.data_processor import DataProcessor

    pytest.importorskip("PyQt5")
    proc = DataProcessor(
        filename="matrices/matrices_cloudSAT_high.pkl", method="стандартный",
        f_c=8.0, Xmax=40.0, Ymax=40.0, spectr_w=0.5, ang=9.0,
        range_m=500_000.0, nifft_size=256,
        V=0.0, alpha=0.0, omega=0.0, pri=0.05, num_pulses=256,
    )
    with pytest.raises(ValueError) as e:
        proc.generate_raw_data()
    msg = str(e.value)
    assert "matrices_cloudSAT_high.pkl" in msg
    assert "повреждён" in msg or "усечён" in msg
    # исходная причина сохранена
    assert isinstance(e.value.__cause__, pickle.UnpicklingError)
