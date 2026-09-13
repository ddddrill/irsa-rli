"""Метрики функции рассеяния точки (PSF): ширина ГЛ и уровень боковых лепестков.

Используется для автотестов разрешения (tests/test_psf.py) и для
дальнейших задач плана R2/R3/R6 (инструментарий переиспользуем).
"""
import numpy as np


def psf_metrics(profile_db, main_lobe_halfwidth_bins):
    """Метрики 1D PSF: ширина главного лепестка и максимальный SLL.

    Ширина ГЛ измеряется по уровню −3 дБ от пика (интерполяция между
    отсчётами в точках пересечения). Боковые лепестки ищутся вне области
    главного лепестка: пик ± main_lobe_halfwidth_bins.

    Args:
        profile_db: 1D профиль PSF в дБ (пик должен быть ≈ 0 дБ), shape (N,).
        main_lobe_halfwidth_bins: полуширина области главного лепестка
            (в отсчётах), исключаемой из поиска боковых лепестков.

    Returns:
        width_minus3db_bins: ширина главного лепестка по −3 дБ (в отсчётах).
        max_sll_db: максимум профиля вне области главного лепестка (в дБ).
    """
    profile_db = np.asarray(profile_db, dtype=float)
    peak_idx = int(np.argmax(profile_db))
    hl = int(main_lobe_halfwidth_bins)

    lo = max(0, peak_idx - hl)
    hi = min(len(profile_db), peak_idx + hl + 1)
    outside = np.concatenate([profile_db[:lo], profile_db[hi:]])
    max_sll_db = float(outside.max()) if outside.size else -np.inf

    threshold = profile_db[peak_idx] - 3.0
    crossing_l = _left_crossing(profile_db, peak_idx, threshold)
    crossing_r = _right_crossing(profile_db, peak_idx, threshold)
    width_minus3db_bins = crossing_r - crossing_l
    return width_minus3db_bins, max_sll_db


def _left_crossing(profile_db, peak_idx, threshold):
    idx = peak_idx
    while idx > 0 and profile_db[idx - 1] >= threshold:
        idx -= 1
    for i in range(max(idx - 1, 0), peak_idx):
        if profile_db[i] < threshold <= profile_db[i + 1]:
            return i + (threshold - profile_db[i]) / (
                profile_db[i + 1] - profile_db[i])
    return 0.0


def _right_crossing(profile_db, peak_idx, threshold):
    idx = peak_idx
    while idx < len(profile_db) - 1 and profile_db[idx + 1] >= threshold:
        idx += 1
    for i in range(idx, len(profile_db) - 1):
        if profile_db[i] >= threshold > profile_db[i + 1]:
            return i + (threshold - profile_db[i]) / (
                profile_db[i + 1] - profile_db[i])
    return float(len(profile_db) - 1)
