"""PSF-тесты разрешения и взвешивания по дальности/азимуту (задача R7).

Проверяют:
    - битовую идентичность `window="none"` старому boxcar-коду;
    - уровень боковых лепестков окон по дальности и азимуту;
    - уширение главного лепестка hann относительно boxcar (<= 1.7x);
    - разрешение по Рэлею: два рассеивателя через ячейку dr.

Источники: взвешивание C&M, §6.3.1; окно перед IFFT/FFT C&M, §3.3.
Метрики PSF описаны в processing/psf_metrics.py.
"""
import numpy as np
import pytest
from scipy.fft import ifft, fft, fftshift

from processing.azimuth_compress import azimuth_compress
from processing.psf_metrics import psf_metrics
from processing.range_compress import range_compress

SPEED_OF_LIGHT = 3e8

# Синтетика: одиночный рассеиватель на нулевой дальности -> E = единичный
# вектор-столбец -> PSF = |окно| в дальностной области.
# Оверсемплинг PSF через zero-padding спектра до 8M.
M = 256
f_r = np.linspace(7.75e9, 8.25e9, M)          # B = 0.5 ГГц, dr ~= 0.3 м
E = np.ones((M, 8), dtype=complex)            # 8 импульсов

OVERSAMPLE = 32

# Полуширина исключаемой области главного лепестка (в бинах) для поиска
# боковых лепестков. Ориентир: положение первого нуля PSF окна (C&M, §6.3.1)
# + запас на дискретность.
MAIN_LOBE_HALFWIDTH_BINS = {
    "none": 1.5, "hann": 2.5, "hamming": 2.5,
    "kaiser": 3.0, "chebwin": 3.0,
}

# Ожидаемый уровень первого бокового лепестка (дБ).
WINDOW_SLL = [
    ("none", -13.2), ("hann", -31.5), ("hamming", -42.7),
    ("kaiser", -55.0), ("chebwin", -57.0),
]


def _db(profile, eps=1e-30):
    p = 20.0 * np.log10(np.abs(profile) + eps)
    return p - p.max()


def _oversample(profile, factor=OVERSAMPLE):
    """Оверсемплинг комплексного профиля дополнением спектра нулями.

    Спектр профиля восстанавливается точным FFT/IFT и дополняется нулями
    в (factor - 1) раз -> IFFT даёт интерполированный профиль. Пик
    центрируется в середине массива (для psf_metrics).

    Args:
        profile: 1D комплексный профиль PSF.
        factor: кратность оверсемплинга.

    Returns:
        np.abs интерполяции профиля, length = factor * len(profile).
    """
    spec = fft(fftshift(profile))
    padded = np.concatenate([spec, np.zeros((factor - 1) * len(spec), dtype=complex)])
    return np.abs(fftshift(ifft(padded)))


def test_range_compress_none_bitwise_identity():
    """range_compress(window="none") битово идентичен старому boxcar-коду."""
    for R0 in (0.0, 5.0):
        E_ref = E * np.exp(1j * 4.0 * np.pi * f_r / SPEED_OF_LIGHT * R0)[:, np.newaxis]
        reference = fftshift(ifft(E_ref, axis=0), axes=0)
        P, _, _, _ = range_compress(E, f_r, R0=R0, window="none")
        assert np.array_equal(P, reference)


def test_azimuth_compress_none_bitwise_identity():
    """azimuth_compress(window="none") не меняет матрицу побитово."""
    N = 64
    n = np.arange(N)
    P_comp = np.tile(np.exp(2j * np.pi * 16 * n / N), (M, 1))
    reference = fftshift(fft(P_comp, axis=1), axes=1)
    I, _, _ = azimuth_compress(P_comp, omega=0.1, f_c=8e9, pri=1.0, window="none")
    assert np.array_equal(I, reference)


@pytest.mark.parametrize("window,expected_sll", WINDOW_SLL)
def test_range_psf_sidelobes(window, expected_sll):
    """SLL окон по дальности не хуже ожидаемого + допуск 3 дБ."""
    P, _, _, _ = range_compress(E, f_r, window=window)
    profile = _oversample(P[:, 0])
    _, sll = psf_metrics(_db(profile),
                         MAIN_LOBE_HALFWIDTH_BINS[window] * OVERSAMPLE)
    assert sll <= expected_sll + 3.0


def test_range_psf_main_lobe_broadening():
    """hann расширяет главный лепесток не более чем в 1.7 раза от boxcar."""
    widths = {}
    for window in ("none", "hann"):
        P, _, _, _ = range_compress(E, f_r, window=window)
        width, _ = psf_metrics(_db(_oversample(P[:, 0])), 3 * OVERSAMPLE)
        widths[window] = width
    assert widths["hann"] / widths["none"] <= 1.7


def test_two_scatterers_rayleigh():
    """Два рассеивателя через dr: boxcar различимы по Рэлею, kaiser сливаются."""
    B = f_r[-1] - f_r[0]
    df = B / (M - 1)
    dr = SPEED_OF_LIGHT / (2.0 * M * df)
    echo = np.ones(M, dtype=complex) + np.exp(
        -1j * 4.0 * np.pi * f_r * dr / SPEED_OF_LIGHT)
    E_two = np.tile(echo[:, np.newaxis], (1, 8))

    dips = {}
    for window in ("none", "kaiser"):
        P, _, _, _ = range_compress(E_two, f_r, window=window)
        d = _db(_oversample(P[:, 0]))
        dips[window] = d.max() - d[len(d) // 2]

    assert dips["none"] >= 1.5   # провал между пиками: по Рэлею различимы
    assert dips["kaiser"] < 1.2  # ГЛ шире dr: пики сливаются в плато


@pytest.mark.parametrize("window,expected_sll", WINDOW_SLL)
def test_azimuth_psf_sidelobes(window, expected_sll):
    """Симметричный тест азимутального PSF: строка exp(j*2*pi*k0*n/N)."""
    N = 256
    n = np.arange(N)
    k0 = N // 4
    row = np.exp(2j * np.pi * k0 * n / N)
    P_comp = np.tile(row, (M, 1))
    I, _, _ = azimuth_compress(P_comp, omega=0.1, f_c=8e9, pri=1.0, window=window)
    profile = _oversample(I[0, :])
    _, sll = psf_metrics(_db(profile),
                         MAIN_LOBE_HALFWIDTH_BINS[window] * OVERSAMPLE)
    assert sll <= expected_sll + 3.0


def test_azimuth_psf_symmetric():
    """Азимутальный PSF симметричен вокруг пика (окно симметрично)."""
    N = 256
    n = np.arange(N)
    row = np.exp(2j * np.pi * (N // 4) * n / N)
    P_comp = np.tile(row, (M, 1))
    I, _, _ = azimuth_compress(P_comp, omega=0.1, f_c=8e9, pri=1.0, window="hann")
    profile = np.abs(I[0, :])
    peak = int(np.argmax(profile))
    extent = min(peak, len(profile) - peak - 1)
    left = profile[peak - extent:peak + 1][::-1]
    right = profile[peak:peak + extent + 1]
    assert np.allclose(left, right, rtol=1e-6, atol=1e-9)
