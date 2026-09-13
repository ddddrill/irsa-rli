"""Кросс-масштабирование по данным: оценка эффективной угловой скорости.

Задача R3: по сжатому по дальности и азимуту комплексному изображению
оценить линейный тренд скорости изменения (чирпа) вдоль координаты,
восстановить эффективную угловую скорость Omega_eff и сформировать
азимутальную ось в метрах.

Конвенция знаков. В симуляторе ФАР (models.radar / simulation.raw_generator)
фаза сигнала от точечной цели имеет вид
    s(t) = exp( j * (2*pi*f_D*t + pi*mu*t^2) ),   mu = (2*f_c/c) * x * Omega^2,
где x - кросс-дальность, f_c - несущая, c - скорость света, Omega - угловая
скорость вращения. Положительная координата x даёт положительный чирп
(в отличие от конвенции -a2/pi в плане §8.2.1, соответствующих сопряжённой
записи эха; здесь используется фактически измеряемый знак чирпа).

Метод "phasefit": линейно-фазовая регрессия по сдвинутому произведению
(лаг-произведение). Для сигнала s[n] = A*exp(j*(theta[n] + pi*mu*(n*dt)^2))
значение
    z[n] = s[n+1] * conj(s[n]) = A^2 * exp(j*(2*pi*mu*dt^2*n + phi_dc))
есть монохроматический комплексный тон с частотой f_z = mu*dt^2 (в единицах
нормированной частоты DFT). Наклон фазы arg(z) после разворачивания фазы
даёт mu_hat = a1 / (2*pi*dt^2). Метод устойчив к постоянному доплеровскому
сдвигу f_D (он входит только в phi_dc) и не требует разрешения спектра.
Область однозначности |mu| * dt^2 < 0.5 (на границе - алиасинг).

Метод "lpft": локальное преобразование Фурье (дечирп + максимум контраста).
Подбирает mu на сетке [mu_init +/- mu_grid_rel] так, чтобы максимизировать
отношение std/mean модуля спектра дечирпированного сигнала. Рекомендован для
хорошо сфокусированных сцен; стартует с оценки phasefit.

Примечание о полосе. Восстановление строки из ограниченной (±3 доплеровских
бина) области спектра систематически занижает измеряемое mu на 12–20 % из-за
усечения спектральной протяжённости чирпа (T^2*mu > 8 бинов). Поэтому
azimuth_decompress восстанавливает строку по всей доплеровской полосе
(полный обратный оператор к azimuth_compress). Строка изображения
I = fftshift(fft(P_comp, axis=1), axes=1), обратный оператор:
azimuth_decompress(row) = ifft(ifftshift(row)).

Ссылка на опорный бин. После автофокусировки (mocomp) фаза компенсирована
относительно опорного рассеивателя r_ref, поэтому измеряемое mu на строке y
равно K*(y - r_ref). Линейная регрессия mu = K*y + b сохраняет наклон K
(и, следовательно, оценку Omega_eff), а точка пересечения b поглощает
смещение r_ref (см. план §8.2.3); b далее не используется.
"""

import numpy as np

SPEED_OF_LIGHT = 3e8

_WEIGHT_TRIM_SIGMA = 2.0
_MAX_TRIM_ITER = 2


def _unwrap_phase(z):
    """Развернуть фазу z (устойчиво к скачкам 2*pi по отсчётам)."""
    return np.unwrap(np.angle(z))


def _phasefit_slope(s, dt):
    """Оценка mu методом линейной фазовой регрессии лаг-произведения.

    Возвращает (mu_hat, r2), где r2 - коэффициент детерминации R^2 линейной
    подгонки фазы; близость к 1 означает чистый чирп.
    """
    if len(s) < 3:
        return 0.0, 0.0
    z = s[1:] * np.conj(s[:-1])
    phases = _unwrap_phase(z)
    n = np.arange(len(phases), dtype=float)
    a1, a0 = np.polyfit(n, phases, 1)
    resid = phases - (a1 * n + a0)
    denom = np.sum((phases - phases.mean()) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / denom if denom > 0.0 else 1.0
    mu_hat = a1 / (2.0 * np.pi * dt ** 2)
    return float(mu_hat), float(r2)


def _lpft_contrast(s, t, mu_grid):
    """Векторизованный локальный FFT: контраст = std/mean |FFT| на сетке mu."""
    dechi = s[np.newaxis, :] * np.exp(-1j * np.pi * mu_grid[:, np.newaxis] * t[np.newaxis, :] ** 2)
    spec = np.abs(np.fft.fft(dechi, axis=1))
    mean = spec.mean(axis=1)
    std = spec.std(axis=1)
    return std / np.maximum(mean, 1e-300)


def segment_dominant_scatterers(
    I_complex,
    range_axis=None,
    doppler_axis=None,
    max_scatterers=12,
    min_sep_range_bins=4,
    min_sep_doppler_bins=8,
    threshold_sigma=3.0,
):
    """Выделить доминирующие рассеиватели через заливку (по манхэттенской связности).

    Порог амплитуды: mean + threshold_sigma * std по всей области изображения.
    Найденные области сортируются по пиковой амплитуде; по умолчанию не более
    max_scatterers.

    Returns:
        Список словарей с ключами:
            range_bin, doppler_bin - индексы пика в изображении;
            y_abs - абсолютная координата (если задан range_axis);
            f_d - доплер (если задан doppler_axis);
            peak_amp - пиковая амплитуда.
    """
    I = np.asarray(I_complex, dtype=complex)
    if I.ndim != 2 or I.size == 0:
        return []
    amp = np.abs(I)
    amp_max = amp.max()
    if amp_max <= 0:
        return []
    thr = float(amp.mean() + threshold_sigma * amp.std())

    mask = np.zeros(amp.shape, dtype=bool)
    segments = []
    for _ in range(max_scatterers):
        if not mask.all():
            idx = np.argmax(amp * (1 - mask))
        else:
            break
        seed = np.unravel_index(idx, amp.shape)
        if amp[seed] <= thr:
            break
        stack = [seed]
        seen = {seed}
        while stack:
            cy, cd = stack.pop()
            for ny, nd in ((cy + 1, cd), (cy - 1, cd), (cy, cd + 1), (cy, cd - 1)):
                if 0 <= ny < amp.shape[0] and 0 <= nd < amp.shape[1]:
                    key = (ny, nd)
                    if key not in seen and amp[key] > thr:
                        seen.add(key)
                        stack.append(key)
        reg_rows = np.array(list(seen))
        ys = reg_rows[:, 0]
        ds = reg_rows[:, 1]
        pix = amp[ys, ds]
        pk = int(np.argmax(pix))
        y_pk, d_pk = int(ys[pk]), int(ds[pk])
        seg = {
            "range_bin": y_pk,
            "doppler_bin": d_pk,
            "peak_amp": float(amp[y_pk, d_pk]),
        }
        if range_axis is not None and len(range_axis) == amp.shape[0]:
            seg["y_abs"] = float(np.interp(y_pk, np.arange(len(range_axis)), np.asarray(range_axis, float)))
        if doppler_axis is not None and len(doppler_axis) == amp.shape[1]:
            seg["f_d"] = float(np.interp(d_pk, np.arange(len(doppler_axis)), np.asarray(doppler_axis, float)))
        segments.append(seg)
        ys = np.clip(ys, 0, amp.shape[0] - 1)
        ds = np.clip(ds, 0, amp.shape[1] - 1)
        mask[ys, ds] = True
    segments.sort(key=lambda s: s["peak_amp"], reverse=True)
    return segments[:max_scatterers]


def azimuth_decompress(row_or_image, axis_doppler=1):
    """Обратный оператор к azimuth_compress: ifft(ifftshift(...)).

    Для I = fftshift(fft(P_comp, axis=1), axes=1) возвращает P_comp
    (точность до машинной погрешности). Для одномерной входной строки
    выполняется процедура вдоль единственной оси.

    Args:
        row_or_image: комплексная строка (N,) или подизображение (…, N).
        axis_doppler: ось доплера для многомерного входа (default 1).
    """
    x = np.asarray(row_or_image, dtype=complex)
    if x.ndim == 1:
        return np.fft.ifft(np.fft.ifftshift(x))
    return np.fft.ifft(np.fft.ifftshift(x, axes=axis_doppler), axis=axis_doppler)


def estimate_chirp_rate(s, fs, method="lpft", mu_init=None, mu_grid_rel=0.25, n_grid=101):
    """Оценка скорости изменения чирпа mu сигнала s(t) на интервале N*dt.

    Args:
        s: комплексный сигнал (N,).
        fs: частота дискретизации (доплер-спектр уже в Гц; dt = 1/fs).
        method: "lpft" (default) или "phasefit". Качество (quality) -
            контраст LD-спектра для lpft, R^2 линейной подгонки для phasefit.
        mu_init: стартовая точка сетки lpft (None -> оценка phasefit).
        mu_grid_rel: полуширина сетки относительно |mu_init| (0.25 = ±25 %).
        n_grid: число точек сетки (нечётное >= 3).

    Returns:
        (mu_hat, quality).
    """
    s = np.asarray(s, dtype=complex)
    n = len(s)
    dt = 1.0 / fs
    mu_pf, q_pf = _phasefit_slope(s, dt)
    if method == "phasefit":
        return mu_pf, q_pf
    if method != "lpft":
        raise ValueError(f"unknown method '{method}'; expected 'lpft' or 'phasefit'")
    if n < 3:
        return 0.0, 0.0
    init = float(mu_init) if mu_init is not None else mu_pf
    if abs(init) < 1e-12:
        return mu_pf, q_pf
    width = mu_grid_rel * abs(init)
    low, high = init - width, init + width
    if high - low < 1e-300:
        return mu_pf, q_pf
    n_grid = max(int(n_grid), 3)
    if n_grid % 2 == 0:
        n_grid += 1
    grid = np.linspace(low, high, n_grid)
    t = np.arange(n) * dt
    contrast = _lpft_contrast(s, t, grid)
    best = int(np.argmax(contrast))
    return float(grid[best]), float(contrast[best])


def _weighted_fit(x, y, w):
    """Взвешенный метод наименьших квадратов: y = K*x + b.

    Returns:
        (K, b, residuals).
    """
    X = np.stack([x, np.ones_like(x)], axis=1)
    W = np.sqrt(np.clip(w, 0.0, None))
    coef, *_ = np.linalg.lstsq(X * W[:, np.newaxis], y * W, rcond=None)
    pred = X @ coef
    return float(coef[0]), float(coef[1]), y - pred


def _r2_weighted(y, resid, w):
    denom = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    if denom <= 0.0:
        return 1.0
    return 1.0 - np.sum(w * resid ** 2) / denom


def cross_range_scale(
    I_complex,
    range_axis,
    doppler_axis,
    f_c,
    fs,
    method="lpft",
    min_r2=0.85,
    decompress_window=None,
    **seg_kwargs,
):
    """Оценить эффективную угловую скорость по кросс-дальностной регрессии чирпа.

    Восстанавливает сигналы строк изображения (полная полоса, если
    decompress_window=None; иначе окно ±decompress_window доплеровских бинов
    вокруг пика — НЕ рекомендуется из-за систематического занижения mu,
    см. docstring модуля), оценивает mu(i) каждой строки и подгоняет
    mu = K*y + b взвешенным МНК с двукратным исключением выбросов по 2-sigma.

    Args:
        I_complex: комплексное изображение (M, N).
        range_axis: ось дальности (M,) в метрах.
        doppler_axis: ось доплера (N,) в Гц.
        f_c: несущая частота, Гц.
        fs: частота дискретизации медленного времени, Гц.
        method: "lpft" (default) или "phasefit" (см. estimate_chirp_rate).
        min_r2: минимальный R^2 подгонки для признания оценки.
        decompress_window: полная полоса (None, default) или ±бина окна.
        seg_kwargs: параметры segment_dominant_scatterers.

    Returns:
        Словарь:
            omega_eff - оцененная угловая скорость, рад/с (0 при fallback);
            r2, n_used - R^2 и число использованных точек;
            points - список (y_m, mu_hat, weight);
            azimuth_axis_m - азимутальная ось в метрах (по doppler_axis);
            azimuth_axis_hz - исходная доплеровская ось;
            fallback - True, если оценка недостоверна.
    """
    I = np.asarray(I_complex, dtype=complex)
    if I.ndim != 2:
        raise ValueError("I_complex must be a 2-D complex image")
    M, N = I.shape
    range_axis = np.asarray(range_axis, float)
    doppler_axis = np.asarray(doppler_axis, float)
    if len(range_axis) != M:
        raise ValueError("range_axis length must match image rows")
    if len(doppler_axis) != N:
        raise ValueError("doppler_axis length must match image columns")

    segments = segment_dominant_scatterers(I, range_axis, doppler_axis, **seg_kwargs)
    empty = dict(omega_eff=0.0, K=None, intercept=None, r2=0.0, n_used=0,
                 points=[], azimuth_axis_m=None,
                 azimuth_axis_hz=np.asarray(doppler_axis, float).copy(),
                 fallback=True)
    if len(segments) < 3:
        return empty

    used_bins = set()
    points = []
    for seg in segments:
        yb = seg["range_bin"]
        if yb in used_bins:
            continue
        used_bins.add(yb)
        row = I[yb, :].copy()
        if decompress_window is not None:
            cb = seg["doppler_bin"]
            lo = max(0, cb - int(decompress_window))
            hi = min(N, cb + int(decompress_window) + 1)
            band = row[lo:hi]
            sig = azimuth_decompress(band)
        else:
            sig = azimuth_decompress(row)
        mu_hat, quality = estimate_chirp_rate(sig, fs, method=method)
        points.append((seg.get("y_abs", float(yb)), mu_hat, seg["peak_amp"]))

    if len(points) < 3:
        return empty

    amps = np.array([p[2] for p in points])
    p75 = float(np.percentile(amps, 75))
    weights = np.minimum(amps, p75)

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    ws = np.array(weights)

    K, b, resid = _weighted_fit(xs, ys, ws)
    r2 = _r2_weighted(ys, resid, ws)

    for _ in range(_MAX_TRIM_ITER):
        sigma = float(np.sqrt(np.average(
            resid ** 2, weights=ws)) if ws.sum() > 0 and np.average(resid ** 2, weights=ws) > 0 else 0.0)
        keep = np.abs(resid) <= _WEIGHT_TRIM_SIGMA * sigma if sigma > 0 else np.ones_like(resid, dtype=bool)
        if keep.sum() >= 3 and keep.sum() < len(xs):
            xs, ys, ws = xs[keep], ys[keep], ws[keep]
            K, b, resid = _weighted_fit(xs, ys, ws)
            r2 = _r2_weighted(ys, resid, ws)
        else:
            break

    n_used = int(len(xs))
    if K <= 0.0 or r2 < min_r2 or n_used < 3:
        return empty

    omega_eff = float(np.sqrt(SPEED_OF_LIGHT * K / (2.0 * f_c)))
    azimuth_axis_m = doppler_axis * SPEED_OF_LIGHT / (2.0 * f_c * omega_eff)
    pts = [(float(xs[i]), float(ys[i]), float(ws[i])) for i in range(len(xs))]
    return {
        "omega_eff": omega_eff,
        "K": float(K),
        "intercept": float(b),
        "r2": float(r2),
        "n_used": n_used,
        "points": pts,
        "azimuth_axis_m": azimuth_axis_m,
        "azimuth_axis_hz": np.asarray(doppler_axis, float).copy(),
        "fallback": False,
    }
