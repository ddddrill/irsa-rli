"""Частотно-временной анализ для ISAR/микро-Доплер.

Источники: [TF] гл. 2 (преобразования) и гл. 5 (RID); [MD] §1.8 (сигнатуры).
Все функции — чистые (детерминированные).

Ось частот STFT — в Гц, та же конвенция, что и `doppler_axis` в
`processing/azimuth_compress` (fftshift, возрастающий порядок от -fs/2).
STFT считаем через `scipy.signal.stft` (scaling="spectrum"), окна берём из
`processing/windows.get_weighting` (нормировка w/mean(w) — единый источник).

Калибровка SCALE (см. ``SCALE`` и формулу (7.2.4) плана): для четного
nperseg и hop=1 полная сумма кадров спектрограммы |Z|^2 на on-бинном тоне
воспроизводит |FFT|^2 маргинально с точностью SCALE-калибровки:

    SCALE = N^2 / S_cal,  S_cal = (N - L + 1) + 2 * sum_{m=1}^{L/2}
                             ( sum_{n=m}^{L-1} w[n] / sum(w) )^2

(краевые кадры STFT усекают окно на границах сигнала; S_cal учитывает их
вклад в полную сумму). ``hop > 1`` нарушает точное маргинальное условие —
такие кубы помечаются approximate=True.
"""
import warnings

import numpy as np
from scipy import signal

from processing.windows import get_weighting

__all__ = [
    "SCALE",
    "stft_spectrogram",
    "strongest_range_bins",
    "micro_doppler_signature",
    "total_echo_signature",
    "rid_cube",
    "rid_frame",
    "rid_marginal",
    "wigner_ville",
]


def _marginal_scale(N, nperseg, window):
    """SCALE калибровочная константа маргинального условия (7.2.4).

    Точное значение для конкретных (N, nperseg, окно). Тон x = exp(j2πf0t)
    с f0·N ∈ Z при hop=1 даёт  max_f Σ_t |Z(t,f)|^2 · SCALE == пик |FFT|^2
    с точностью до машинной ошибки (проверено для nperseg ∈ {128, 256}).

    Args:
        N: длина сигнала (число медленных отсчётов).
        nperseg: длина окна STFT (должна быть чётной).
        window: имя окна из processing.windows.WINDOW_NAMES.

    Returns:
        float:
    """
    w = get_weighting(window, nperseg)
    wsum = float(w.sum())
    partials = np.array([w[m:].sum() for m in range(1, nperseg // 2 + 1)])
    s_cal = float(N - nperseg + 1) + 2.0 * float(((partials / wsum) ** 2.0).sum())
    return N * N / s_cal


# SCALE для эталонной конфигурации модуля: N=256, nperseg=128, окно hann.
SCALE = _marginal_scale(256, 128, "hann")


def _even_nperseg(nperseg):
    return nperseg if nperseg % 2 == 0 else nperseg + 1


def stft_spectrogram(x, fs=1.0, nperseg=None, noverlap=None,
                     window="hann", scaling_calibrated=True):
    """Краткосрочное преобразование Фурье одного сигнала.

    Args:
        x: комплексный/вещественный сигнал (строка = дальностный бин),
            длина N.
        fs: частота дискретизации, Гц (для медленного времени fs = 1/pri).
        nperseg: длина окна; по умолчанию max(4, N//8); чётное (SCALE).
        noverlap: перекрытие кадров; по умолчанию round(0.75*nperseg) [TF, гл.5].
        window: имя окна (processing.windows.WINDOW_NAMES), по умолчанию "hann".
        scaling_calibrated: если True, |Z| умножается на sqrt(SCALE) —
            маргинальная сумма кадров воспроизводит |FFT|^2 (формула (7.2.4)).
            Для визуализации и относительных величин калибровка не влияет
            (S_db нормируется на максимум).

    Returns:
        dict(f, t, Z, S_db):
            f: ось частот, Гц (fftshift, [-fs/2, fs/2)), длина F=nperseg;
            t: моменты времени кадров, с;
            Z: комплексная матрица (F, T);
            S_db: 20*log10|Z|, нормированная к максимуму (0 дБ).
    """
    x = np.asarray(x)
    N = x.shape[-1]
    if nperseg is None:
        nperseg = max(4, N // 8)
    nperseg = _even_nperseg(int(nperseg))
    if noverlap is None:
        noverlap = int(round(0.75 * nperseg))
    if nperseg > N:
        nperseg = _even_nperseg(N)
    w = get_weighting(window, nperseg)

    f, t, Z = signal.stft(x, fs=fs, window=w, nperseg=nperseg,
                          noverlap=noverlap, boundary="zeros", padded=True,
                          return_onesided=False, scaling="spectrum")
    f = np.fft.fftshift(f)
    Z = np.fft.fftshift(Z, axes=0)
    if scaling_calibrated:
        scale = _marginal_scale(N, nperseg, window)
        Z = Z * np.sqrt(scale)

    amp = np.abs(Z)
    S_db = 20.0 * np.log10(amp + 1e-12)
    S_db -= float(S_db.max())
    return {"f": f, "t": t, "Z": Z, "S_db": S_db}


def strongest_range_bins(P_comp, k=5):
    """Индексы топ-K дальностных бинов по суммарной энергии |P|^2.

    Args:
        P_comp: матрица (M, N): строки — дальностные бины, столбцы — медленное время.
        k: сколько бинов вернуть.

    Returns:
        list[int]: индексы бинов по убыванию энергии (длина min(k, M)).
    """
    energy = (np.abs(P_comp) ** 2.0).sum(axis=1)
    order = np.argsort(energy)[::-1]
    return [int(i) for i in order[:k]]


def micro_doppler_signature(P_comp, bin_index, fs, **stft_kwargs):
    """Сигнатура выбранного дальностного бина (обёртка stft_spectrogram).

    Args:
        P_comp: матрица (M, N) после дальностного сжатия и MOCO.
        bin_index: индекс строки (дальностного бина).
        fs: частота дискретизации медленного времени, fs = 1/pri, Гц.
        **stft_kwargs: параметры stft_spectrogram (nperseg, noverlap, window).

    Returns:
        tuple(f, t, S_db) — те же поля, что у stft_spectrogram.
    """
    st = stft_spectrogram(P_comp[bin_index], fs=fs, **stft_kwargs)
    _warn_aliasing(st["Z"], st["f"])
    return st["f"], st["t"], st["S_db"]


def _warn_aliasing(Z, f, threshold=1e-3):
    """Предупреждение о нарушении критерия Котельникова (PRF ≤ 2·f_max) [MD §3.2.6]."""
    amp2 = np.abs(Z) ** 2.0
    tot = amp2.sum()
    if tot <= 0:
        return
    F = Z.shape[0]
    outer = amp2[:2].sum() + amp2[-2:].sum()
    if outer / tot > threshold:
        warnings.warn(
            "Микро-Доплеровская полоса достигает ±fs/2: возможно нарушение "
            "критерия Котельникова (PRF < 2*f_max) — ссылка [MD, §3.2.6].",
            RuntimeWarning,
            stacklevel=3,
        )


def total_echo_signature(P_comp, fs, **stft_kwargs):
    """Спектрограмма суммарного эха для распределённых целей [MD, §1.8].

    s(t) = Σ_m P[m, t] — когерентное объединение дальностных бинов (вариант
    [MD, §1.8]); дополнительно возвращается энергетическая сумма спектрограмм
    бинов S_db_sum = 20·log10( sqrt( Σ_m |STFT(P[m,:])|^2 ) ).

    Args:
        P_comp: матрица (M, N).
        fs: частота дискретизации, Гц.
        **stft_kwargs: параметры stft_spectrogram.

    Returns:
        dict(f, t, S_db, S_db_sum):
            S_db: спектрограмма суммарного эха (дБ, нормировано к макс);
            S_db_sum: энергетическая сумма по бинам (дБ, норм. к макс).
    """
    P_comp = np.asarray(P_comp)
    st = stft_spectrogram(np.sum(P_comp, axis=0), fs=fs, **stft_kwargs)
    Z = np.asarray(st["Z"])
    f = np.asarray(st["f"])
    t = np.asarray(st["t"])
    _warn_aliasing(Z, f)

    kw = dict(stft_kwargs)
    kw.setdefault("nperseg", Z.shape[0])
    kw.setdefault("window", "hann")
    kw.setdefault("noverlap", None)
    pow_sum = np.zeros(Z.shape, dtype=np.float64)
    for m in range(P_comp.shape[0]):
        zs = stft_spectrogram(P_comp[m], fs=fs, **kw)["Z"]
        pow_sum += np.abs(zs) ** 2.0
    S_db_sum = 20.0 * np.log10(np.sqrt(pow_sum) + 1e-12)
    S_db_sum -= float(S_db_sum.max())
    return {"f": f, "t": t, "S_db": np.asarray(st["S_db"]), "S_db_sum": S_db_sum}


RID_MEMORY_BUDGET_BYTES = 100 * 1024 * 1024
RID_MAX_BINS = 256


def rid_cube(P_comp, fs=1.0, n_frames=16, nperseg=None, window="hann",
             hop=1, dtype_mag=np.float32):
    """Куб RID Q(r, f, t) по [TF, §5.3]: STFT каждой строки по медленному времени.

    Маргинал куба I(r, f) = Σ_t |Q(r, f, t)|^2 воспроизводит обычное
    RD-изображение (|FFT|^2 по медленному времени) с точностью SCALE-калибровки
    при hop=1 (формула (7.2.4)). Кадры Q хранятся как |Q| float32 (память!);
    комплексные кадры лениво пересчитывает rid_frame_complex() (R5b).

    Бюджет памяти ≤100 МБ (float32): при превышении автоматически прорежаются
    дальностные бины до топ-256 (strongest_range_bins) и/или увеличивается hop
    до T ≤ 512; оба случая помечаются approximate=True.

    Args:
        P_comp: матрица (M, N) после дальностного сжатия и MOCO.
        fs: частота дискретизации, Гц (fs = 1/pri).
        n_frames: число равноотстоящих кадров для отображения (16-32 [TF, §5.3]).
        nperseg: длина окна STFT; по умолчанию max(4, N//8).
        window: имя окна (processing.windows.WINDOW_NAMES).
        hop: шаг между STFT-отсчётами; hop=1 — точное маргинальное условие
            (7.2.4). Для больших N допускается hop = N//512 (T ≤ 512) с пометкой
            approximate=True.
        dtype_mag: dtype хранимой амплитуды (float32 по умолчанию).

    Returns:
        dict(Q_mag=(M,F,T), f, t, frame_idx, approximate, meta):
            Q_mag: амплитуды |SCALE-калиброванных STFT|, shape (M, F, T);
            f, t: оси частот (Гц) и времени кадров (с);
            frame_idx: индексы n_frames равноотстоящих кадров;
            approximate: True, если маргинал не точен (hop>1 или прореживание);
            meta: {"nperseg", "hop", "window", "scale", "bins_used", "n_frames"}.
    """
    P_comp = np.asarray(P_comp)
    M_full, N = P_comp.shape
    if nperseg is None:
        nperseg = max(4, N // 8)
    nperseg = _even_nperseg(int(nperseg))
    if nperseg > N:
        nperseg = _even_nperseg(N)
    hop = max(1, int(hop))

    # --- бюджет памяти ---
    approximate = hop != 1
    bins_used = list(range(M_full))
    P_work = P_comp
    M = M_full
    t_est = N // hop + 1
    while M * nperseg * t_est * np.dtype(dtype_mag).itemsize > RID_MEMORY_BUDGET_BYTES \
            and M > RID_MAX_BINS:
        P_work = P_work[strongest_range_bins(P_work, k=RID_MAX_BINS)]
        M = P_work.shape[0]
        bins_used = strongest_range_bins(P_comp, k=RID_MAX_BINS)
        approximate = True
    hop_eff = hop
    while M * nperseg * (N // hop_eff + 1) * np.dtype(dtype_mag).itemsize \
            > RID_MEMORY_BUDGET_BYTES:
        hop_eff *= 2
        approximate = True

    scale = _marginal_scale(N, nperseg, window)
    w = get_weighting(window, nperseg)
    noverlap = nperseg - hop_eff

    Q = np.empty((M, nperseg, N // hop_eff + 1), dtype=dtype_mag)
    f_ax = None
    for m in range(M):
        _, f, Z = signal.stft(P_work[m], fs=fs, window=w, nperseg=nperseg,
                              noverlap=noverlap, boundary="zeros", padded=True,
                              return_onesided=False, scaling="spectrum")
        f = np.fft.fftshift(f)
        Z = np.fft.fftshift(Z, axes=0) * np.sqrt(scale)
        Q[m] = np.abs(Z).astype(dtype_mag)
        if f_ax is None:
            f_ax = f
            t_ax = np.arange(Z.shape[1], dtype=np.float64) * hop_eff / fs

    T = Q.shape[2]
    frame_idx = np.linspace(0, T - 1, int(n_frames)).astype(int)
    meta = {
        "nperseg": nperseg,
        "hop": hop,
        "hop_eff": hop_eff,
        "window": window,
        "scale": float(scale),
        "bins_used": bins_used,
        "n_frames": int(n_frames),
    }
    return {
        "Q_mag": Q,
        "f": f_ax,
        "t": t_ax,
        "frame_idx": frame_idx,
        "approximate": bool(approximate),
        "meta": meta,
    }


def rid_frame(rid, i):
    """Мгновенный RID-кадр I_i(r, f) = |Q[:, :, frame_idx[i]]| в дБ.

    Args:
        rid: словарь из rid_cube().
        i: номер кадра (индекс в frame_idx).

    Returns:
        ndarray (M, F): 20*log10(амплитуда), нормировано к максимуму (0 дБ),
            готово для imshow.
    """
    idx = int(rid["frame_idx"][i])
    Q = np.asarray(rid["Q_mag"])
    frame = 20.0 * np.log10(Q[:, :, idx] + 1e-12)
    return frame - float(frame.max())


def rid_marginal(rid):
    """Маргинал куба RID I(r, f) = Σ_t |Q(r, f, t)|^2 — [TF, (5.18)].

    При hop=1 и SCALE-калибровке воспроизводит |FFT|^2 по медленному времени
    (сравнение с азимутальным Фурье-изображением в тестах/демо).

    Args:
        rid: словарь из rid_cube().

    Returns:
        ndarray (M, F): маргинальное изображение, энергия.
    """
    Q = np.asarray(rid["Q_mag"])
    return (np.abs(Q) ** 2.0).sum(axis=2)


def wigner_ville(x, fs=1.0):
    """Распределение Вигнера-Вилле комплексного сигнала (WVD).

    WVD(t, f) = FFT_τ[ s(t+τ/2)·s*(t−τ/2) ] — реализация через lag-матрицу
    (N×N) и FFT по оси лагов, O(N^2) память [TF, гл.2]. Дискретизация требует
    лагов τ, кратных 2Δt («even-lag» схема, см. §7.4.5): частота отображается
    как f_k = k·fs/(2N) (односторонняя ось [0, fs/2)). Перед расчётом берётся
    аналитический сигнал (анти-алиасинг по частоте). Кросс-термы — свойство
    билинейных распределений, а не ошибка (тест документирует).

    Args:
        x: сигнал, длина N (комплексный или вещественный).
        fs: частота дискретизации, Гц (по умолчанию 1.0 — ось нормирована).

    Returns:
        tuple(t_axis, f_axis, W):
            t_axis: моменты времени (с);
            f_axis: частоты (Гц), длина N, 0..fs/2;
            W: комплексная матрица (N, N), нормирована на max|W|.
    """
    x = np.asarray(x)
    N = x.shape[0]
    if np.isrealobj(x):
        xa = signal.hilbert(x)
    else:
        xa = x
    pad = N
    xp = np.zeros(2 * pad + N, dtype=np.complex128)
    xp[pad:pad + N] = xa
    t0 = np.arange(N) + pad
    p = np.arange(N)
    lag = xp[t0[:, None] + p[None, :]] * np.conj(xp[t0[:, None] - p[None, :]])
    W = np.fft.fft(lag, axis=1)
    denom = np.abs(W).max()
    if denom > 0:
        W = W / denom
    t_axis = np.arange(N) / fs
    f_axis = np.arange(N) * (fs / (2.0 * N))
    return t_axis, f_axis, W