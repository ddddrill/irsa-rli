"""Общие фикстуры тестов: фиксированный радар и цель."""
import math

import numpy as np
import pytest

from models.radar import Radar
from models.target import Target

SPEED_OF_LIGHT = 3e8

MATRIX = "matrices/matrices_solarB_high.pkl"


@pytest.fixture(scope="session")
def radar():
    return Radar(
        c=SPEED_OF_LIGHT, f_c=8e9, Xmax=20.0, Ymax=20.0,
        spectr_w=0.5e9, ph_c=0.0, ang=9.0, nifft_size=256,
    )


@pytest.fixture(scope="session")
def target():
    omega = math.radians(9.0) / 16.0
    return Target(
        MATRIX, pri=1.0, num_pulses=16, R0=500000.0,
        V=0.0, alpha=0.0, omega=omega,
    )


@pytest.fixture()
def reference_matrix(radar, target):
    """Эталон по классической формуле из README/Оздемира (без эффектов)."""
    sc = target._load_scatterers()
    R = target.get_ranges_all()                      # (N, I)
    _, _, f_r, *_ = radar.base_img_params()
    f_col = f_r[:, None, None]
    return (sc["intens"][None, None, :] *
            np.exp(1j * (-4.0 * np.pi * f_col / SPEED_OF_LIGHT * R[None, :, :]))
            ).sum(axis=2)


@pytest.fixture(scope="session")
def radar_tf():
    """Радар для TF-анализа: X-диапазон, 10 ГГц (план §7.5)."""
    return Radar(
        c=SPEED_OF_LIGHT, f_c=10e9, Xmax=20.0, Ymax=20.0,
        spectr_w=0.5e9, ph_c=0.0, ang=14.0, nifft_size=256,
    )


@pytest.fixture(scope="session")
def target_tf():
    """Цель для TF-анализа: медлённая вращающаяся солнечная панель (§7.5)."""
    omega = math.radians(9.0) / 256.0
    return Target(
        MATRIX, pri=0.05, num_pulses=256, R0=500000.0,
        V=0.0, alpha=0.0, omega=omega,
    )
