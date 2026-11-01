"""Общие фикстуры тестов: фиксированный радар и цель.

Фикстуры валидны по guard'ам C1a–C4 (задача D1, План фиксов §8.2):
    - survey 40 м -> окно дальности ±20.10 м > max|δR| (solarB: 12.05), C2 1.67;
    - Δθ = 0.8° за апертуру (omega = radians(0.8)/(N*PRI)) — валиден и
      по Найквисту (C1a 14.5×), и по порогу малого угла (C4 1.10×);
    - Nf = 134 (чётный) — осознанно: тренировка исправленной конвенции
      fftshift (задача F9) в реальном использовании.
Прежние фикстуры (N=16, PRI=1, Xmax=20) были в алиасинг-режиме ×12.4 —
см. CHANGELOG «физическая корректность демо».
"""
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
        c=SPEED_OF_LIGHT, f_c=8e9, Xmax=40.0, Ymax=40.0,
        spectr_w=0.5e9, ph_c=0.0, ang=9.0, nifft_size=256,
    )


@pytest.fixture(scope="session")
def target():
    omega = math.radians(0.8) / (256 * 0.05)      # Δθ = 0.8° за апертуру
    return Target(
        MATRIX, pri=0.05, num_pulses=256, R0=500000.0,
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
