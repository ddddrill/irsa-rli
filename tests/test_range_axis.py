"""Тесты полу-бинового сдвига оси дальности (План фиксов, задача F9).

Конвенция fftshift: DC-бин после fftshift оказывается в индексе M//2.
Прежняя ось (arange(M) - (M-1)/2)*dr ставила ноль оси в (M-1)/2 = M//2 - 1/2,
из-за чего при чётном M весь пик уходил на +dr/2 (измерено: Nf=200, dR=0 ->
+0.150 м). При нечётном M сдвига не было (дефект молчал на дефолте Nf=67).

Эталон конвенции в проекте: azimuth_compress.py (doppler_axis, N//2).
"""
import math

import numpy as np
import pytest

from models.radar import Radar
from processing.range_compress import range_compress

SPEED_OF_LIGHT = 3e8
R0 = 500_000.0  # км -> м; дешифровка в range_compress вычитает R0


def _radar(Xmax):
    """Радар с производственной сеткой частот (Nf = ceil(2*B*Xmax/c)).

    dr = c/(2*Nf*df) = c/(2*spectr_w) = 0.3 м — для ВСЕХ Nf (B=0.5 ГГц),
    потому dR = 2.1 м = 7*dr ложится точно в центр бина для любой чётности.
    """
    return Radar(
        c=SPEED_OF_LIGHT, f_c=8e9, Xmax=Xmax, Ymax=20.0,
        spectr_w=0.5e9, ph_c=0.0, ang=9.0, nifft_size=256,
    )


def _single_scatterer_E(f_r, dR, N=4):
    """E-матрица одиночного рассеивателя на дальности R0 + dR (V=0, omega=0).

    Прямая модель [CM14] (2.1): E[m,n] = exp(-j*4*pi*f_m*(R0+dR)/c).
    Стационарна по медленному времени (вращение и трансляция нулевые).
    """
    M = f_r.shape[0]
    phase = -4.0 * math.pi * f_r * (R0 + dR) / SPEED_OF_LIGHT
    return np.tile(np.exp(1j * phase)[:, None], (1, N))


@pytest.mark.parametrize("Xmax,Nf", [(20.0, 67), (30.0, 100), (40.0, 134), (60.0, 200)])
@pytest.mark.parametrize("dR", [0.0, 2.1])
def test_range_axis_exact(Xmax, Nf, dR):
    """Пик одиночного рассеивателя обязан стоять на своём dR для любой чётности Nf."""
    radar = _radar(Xmax)
    assert radar.Nf == Nf  # чёт 100/134/200 и нечёт 67 — обе ветки бага
    _, _, f_r, *_ = radar.base_img_params()

    P, P_abs, range_axis, dr = range_compress(
        _single_scatterer_E(f_r, dR), f_r, R0=R0, window="none")

    k = int(np.argmax(P_abs[:, 0]))
    assert abs(range_axis[k] - dR) <= 0.02 * dr


def test_range_axis_odd_nf_bitwise_stable():
    """Нечётный Nf не меняется вовсе (баг молчал на дефолте Nf=67)."""
    radar = _radar(20.0)
    _, _, f_r, *_ = radar.base_img_params()
    assert radar.Nf % 2 == 1

    E = _single_scatterer_E(f_r, 2.1)
    P, P_abs, range_axis, dr = range_compress(E, f_r, R0=R0, window="none")
    k = int(np.argmax(P_abs[:, 0]))
    assert range_axis[k] == pytest.approx(2.1, abs=1e-9)


def test_range_axis_antisymmetric():
    """Ось симметрична: range_axis[M//2] == 0 и шаг между соседями == dr."""
    radar = _radar(40.0)  # Nf=134 (чёт) — случай, где был дефект
    _, _, f_r, *_ = radar.base_img_params()
    _, _, range_axis, dr = range_compress(
        _single_scatterer_E(f_r, 0.0), f_r, R0=R0, window="none")

    assert range_axis[len(range_axis) // 2] == 0.0
    assert np.allclose(np.diff(range_axis), dr)
