"""Тесты TF-ядра: STFT/маргиналы/RID/WVD и сигнатуры микро-Доплера.

Соответствие плану §7.7 (задача R5a): 7 тестов, все детерминированные
(np.random.default_rng seed). F5 добавил биполярный WVD (§7.4–7.5).
"""
import pytest
import numpy as np

from processing.tf_analysis import (
    micro_doppler_signature,
    rid_cube,
    rid_frame,
    rid_marginal,
    strongest_range_bins,
    stft_spectrogram,
    total_echo_signature,
    wigner_ville,
)


def _fixture_p_comp(radar_tf, target_tf):
    """Обычный конвейер фикстур -> сжатая по дальности/выровненная матрица."""
    from processing.mocomp import mocomp
    from processing.range_compress import range_compress
    from simulation.raw_generator import generate_raw_matrix

    E, f_r = generate_raw_matrix(radar_tf, target_tf)
    P, _, range_axis, _ = range_compress(E, f_r, R0=target_tf.R0)
    P_comp, _, _, _ = mocomp(P, range_axis)
    return P_comp


# ---------------------------------------------------------------- 1. STFT-тон
def test_stft_single_tone():
    """§7.7/1: одиночный тон -> гребень спектрограммы на f0 ± 1 бин;
    энергия вдоль времени постоянна (дисперсия по t < 5%)."""
    rng = np.random.default_rng(1)
    N, fs, f0 = 512, 1.0, 0.1
    t = np.arange(N) / fs
    x = np.exp(1j * 2.0 * np.pi * f0 * t) * rng.uniform(0.9, 1.0)
    st = stft_spectrogram(x, fs=fs)
    f = np.asarray(st["f"])
    Z = np.asarray(st["Z"])
    df = fs / Z.shape[0]
    ridge = int(np.argmax((np.abs(Z) ** 2.0).sum(axis=1)))
    assert abs(f[ridge] - f0) <= df, "гребень должен быть на f0 ± 1 бин"

    E_t = np.abs(Z[ridge, :]) ** 2.0
    t_frame = np.asarray(st["t"])
    L = Z.shape[0]
    interior = (t_frame >= L / (2.0 * fs)) & (t_frame <= (N - L / 2.0) / fs)
    E_int = E_t[interior]
    assert E_int.size >= 4
    cv2 = float(E_int.var()) / float(E_int.mean() ** 2.0)
    assert cv2 < 0.05, f"разброс энергии по t = {cv2:.4f} >= 5%"


# --------------------------------------------------- 2. Полоса вибрации
def test_vibration_signature_band():
    """§7.7/2 [MD §1.8]: вибрационное эхо s(t)=exp(j(4π/λ)A·sin(2πf_v t)).
    95% энергии внутри полосы Карсона (B+1)f_v; >=90% внутри ±15% мгновенной
    полосы; сигнатура периодична с периодом 1/f_v."""
    lam, A_v, f_v, fs, N = 0.03, 0.005, 3.0, 40.0, 256
    B = 4.0 * np.pi * A_v / lam
    t = np.arange(N) / fs
    sv = np.exp(1j * B * np.sin(2.0 * np.pi * f_v * t))

    f, t_s, S_db = micro_doppler_signature(
        sv.reshape(1, -1), 0, fs=fs, nperseg=64, noverlap=63)
    S_db = np.asarray(S_db)
    mag2 = 10.0 ** (S_db.astype(np.float64) / 10.0)
    tot = mag2.sum()
    band = (4.0 * np.pi / lam) * A_v * f_v
    carson = (B + 1.0) * f_v
    frac_carson = mag2[np.abs(f) <= carson].sum() / tot
    frac_band15 = mag2[np.abs(f) <= 1.15 * band].sum() / tot
    assert frac_carson >= 0.95, f"полоса Карсона даёт {frac_carson:.4f}"
    assert frac_band15 >= 0.90, f"полоса ±15% даёт {frac_band15:.4f}"
    te = total_echo_signature(sv.reshape(1, -1), fs=fs, nperseg=64, noverlap=63)
    assert np.isfinite(np.asarray(te["S_db_sum"])).all()

    ridge = f[np.argmax(S_db, axis=0)]
    ar = np.correlate(ridge - ridge.mean(), ridge - ridge.mean(), mode="full")
    n2 = ar.size // 2
    center = np.arange(ar.size) - n2
    iac = center[(np.abs(center) > int(round(fs / (2.0 * f_v)))) &
                 (np.abs(center) < int(round(2.0 * fs / f_v)))]
    peak_lag = iac[np.argmax(ar[iac + n2])]
    period = abs(peak_lag) / fs
    assert abs(period - 1.0 / f_v) / (1.0 / f_v) <= 0.15, \
        f"период сигнатуры {period:.3f} с != 1/f_v={1 / f_v:.3f} с"


# ------------------------------------------- 3. Маргинал == |FFT|^2 (SCALE)
def test_rid_marginal_matches_fft():
    """§7.7/3, §7.4.1: на on-бинных тонах при hop=1 маргинал RID
    воспроизводит |FFT|^2 с точностью SCALE-калибровки (< 1e-3).
    hop>1 помечается approximate=True."""
    N, fs = 256, 1.0
    bins = [8, 64, 100, 212]
    t = np.arange(N)
    P = np.zeros((len(bins), N), dtype=complex)
    for m, b in enumerate(bins):
        P[m] = np.exp(1j * 2.0 * np.pi * b / N * t)

    L = N // 2
    rid = rid_cube(P, fs=fs, nperseg=L, hop=1)
    marg = rid_marginal(rid)                      # (M, L) = SCALE*sum_t|Z|^2
    F2 = np.abs(np.fft.fftshift(np.fft.fft(P, axis=1))) ** 2.0
    F2r = F2.reshape(len(bins), L, 2).sum(axis=2)  # 256-сетка -> L-сетка
    assert marg.shape == F2r.shape
    for m in range(len(bins)):
        rel = abs(marg[m].max() - F2r[m].max()) / F2r[m].max()
        assert rel < 1e-3, f"SCALE-ошибка маргинала строки {m}: {rel:.2e}"

    assert rid["approximate"] is False
    rid2 = rid_cube(P, fs=fs, nperseg=L, hop=2)
    assert rid2["approximate"] is True


# --------------------------------------------- 4. Формы куба и кадры
def test_rid_shapes_and_frames(radar_tf, target_tf):
    """§7.7/4: Q_mag.shape == (M, F, T); len(frame_idx) == 16; кадры
    равноотстоящие; float32; meta описывает использованные бины."""
    P_comp = _fixture_p_comp(radar_tf, target_tf)
    top = strongest_range_bins(P_comp, k=8)
    assert len(top) == 8
    sub = P_comp[top]

    fs = 1.0 / target_tf.pri
    rid = rid_cube(sub, fs=fs, nperseg=64, hop=1)
    Q = rid["Q_mag"]
    M, F, T = sub.shape[0], 64, Q.shape[2]
    assert Q.shape == (M, F, T)
    assert Q.dtype == np.float32
    fr = np.asarray(rid["frame_idx"])
    assert len(fr) == 16
    assert np.ptp(np.diff(fr)) <= 1, "кадры равноотстоящие (с точностью до бина)"
    assert 0 <= fr.min() and fr.max() < T
    assert rid["approximate"] is False
    assert "bins_used" in rid["meta"]
    assert len(rid["meta"]["bins_used"]) == M   # без прореживания = все строки

    frame = rid_frame(rid, 4)
    assert frame.shape == (M, F)
    assert np.isfinite(frame).all()

    rid_hop = rid_cube(sub, fs=fs, nperseg=64, hop=8)
    assert rid_hop["Q_mag"].shape[2] < T
    assert rid_hop["approximate"] is True


# ------------------------------------------- 5. Концентрация WVD лчм-сигнала
def test_wvd_lfm_concentration():
    """§7.7/5: для ЛЧМ энергия WVD в полосе вокруг мгновенной частоты заметно
    выше, чем у STFT той же полосы."""
    N, fs = 256, 1.0
    f0, f1 = 0.08, 0.28
    rng = np.random.default_rng(3)
    t = np.arange(N)
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / N * t ** 2.0)
    lfm = np.exp(1j * phase) * rng.uniform(0.9, 1.0, N)
    inst = f0 + (f1 - f0) * t / N
    half = 0.01

    _, f_ax, W = wigner_ville(lfm, fs=fs)
    A = np.abs(W) ** 2.0
    frac_w = 0.0
    for n in range(N):
        frac_w += A[n, np.abs(f_ax - inst[n]) <= half].sum()
    frac_w /= A.sum()

    st = stft_spectrogram(lfm, fs=fs, nperseg=32)
    fS = np.asarray(st["f"])
    ZS = np.asarray(st["Z"])
    GS = np.abs(ZS) ** 2.0
    tS = np.asarray(st["t"])
    frac_s = 0.0
    for j in range(len(tS)):
        fi = inst[min(N - 1, max(0, int(round(tS[j] * fs))))]
        frac_s += GS[:, j][np.abs(fS - fi) <= half].sum()
    frac_s /= GS.sum()

    assert frac_w >= 0.8, f"WVD-полоса держит {frac_w:.3f}"
    assert frac_w >= 1.5 * frac_s, \
        f"концентрация WVD ({frac_w:.3f}) не выше STFT ({frac_s:.3f})"


# ---------------------------------------------- 6. Кросс-термы WVD
def test_wvd_cross_terms_documented():
    """§7.7/6: два тона -> энергия WVD на (f1+f2)/2 > -30 дБ от max
    (кросс-терм — ожидаемое свойство билинейного распределения).
    Биполярная ось (F5): f_ax[j] = (j − N)·fs/(2N), j=0..2N−1."""
    N, fs = 256, 1.0
    b1, b2 = 40, 120
    t = np.arange(N)
    x = np.exp(1j * 2.0 * np.pi * b1 / N * t) + np.exp(1j * 2.0 * np.pi * b2 / N * t)
    _, f_ax, W = wigner_ville(x, fs=fs)
    A = np.abs(W) ** 2.0
    # f0 = b·fs/N соответствует j = N + 2b на биполярной оси (fs/(2N) на бин)
    k_mid = N + b1 + b2                # (f1+f2)/2 = (b1+b2)/2 · fs/N
    assert abs(f_ax[k_mid] - 0.5 * (b1 + b2) * fs / N) < 1e-9
    level = 10.0 * np.log10(A[:, k_mid].max() / A.max())
    assert level > -30.0, f"кросс-терм на {level:.1f} дБ ниже max"


# ----------------------------------------------- 7. Детерминированность
def test_determinism():
    """§7.7/7: rid_cube детерминирован: два вызова -> np.array_equal."""
    rng = np.random.default_rng(7)
    P = (rng.normal(size=(8, 256)) +
         1j * rng.normal(size=(8, 256))).astype(np.complex128)
    r1 = rid_cube(P, fs=1.0, nperseg=64)
    r2 = rid_cube(P, fs=1.0, nperseg=64)
    assert np.array_equal(r1["Q_mag"], r2["Q_mag"])
    assert np.array_equal(r1["frame_idx"], r2["frame_idx"])


# ------------------------------- 9. WVD биполярный (F5, план §7.4–7.5)
def test_wvd_bipolar_tones():
    """F5 §7.5: тоны ±5 Гц (fs=40, N=256) → пики на ±5 ± 2·fs/(2N);
    форма (N, 2N); f_axis[0] = −fs/2.

    Прежняя реализация с целочисленными лагами была π-периодичной (fs/2):
    тон −5 Гц сворачивался на +15 Гц — для биполярного доплера ISAR
    любая вращающаяся цель показывала неверную картину."""
    N, fs = 256, 40.0
    t = np.arange(N) / fs
    df2 = fs / (2 * N)                      # разрешение новой оси
    for f0 in (+5.0, -5.0):
        x = np.exp(1j * 2.0 * np.pi * f0 * t)
        _, f_ax, W = wigner_ville(x, fs=fs)
        assert W.shape == (N, 2 * N)
        assert f_ax[0] == pytest.approx(-fs / 2)
        assert len(f_ax) == 2 * N
        ridge = f_ax[np.argmax(np.abs(W).sum(axis=0))]
        assert abs(ridge - f0) <= 2 * df2, (f0, ridge)


def test_wvd_bipolar_pair_cross_term():
    """F5 §7.4: смесь +5/−3 (ампл. 0.6/0.4) → пики на −2.97/+5.0 и
    кросс-терм на средней частоте +1.0 (документирующий ассерт: свойство
    билинейного распределения, не ошибка)."""
    N, fs = 256, 40.0
    t = np.arange(N) / fs
    x = 0.6 * np.exp(1j * 2.0 * np.pi * 5.0 * t) + \
        0.4 * np.exp(1j * 2.0 * np.pi * (-3.0) * t)
    _, f_ax, W = wigner_ville(x, fs=fs)
    spec = np.abs(W).sum(axis=0)
    # авто-терм +5 на месте (допуск 2 бина новой оси)
    ridge5 = f_ax[np.argmax(spec)]
    assert abs(ridge5 - 5.0) <= 2 * fs / (2 * N), ridge5
    # авто-терм −3 присутствует в своём окне (раньше сворачивался на +17)
    near_m = spec[np.abs(f_ax + 3.0) <= 0.3].max()
    assert near_m > 0.1, near_m
    # кросс-терм на средней частоте пары +1 Гц — выше −30 дБ от max
    mid = (5.0 + (-3.0)) / 2.0                        # +1 Гц
    k_mid = int(np.argmin(np.abs(f_ax - mid)))
    level = 10.0 * np.log10(spec[k_mid] / spec.max())
    assert level > -30.0, f"кросс-терм на {level:.1f} дБ ниже max"


def test_wvd_real_input_regression():
    """F5 §7.5: вещественный cos(2π·5t) → пик на +5 Гц (аналитический сигнал
    по-прежнему предотвращает зеркальную компоненту на −5)."""
    N, fs = 256, 40.0
    t = np.arange(N) / fs
    x = np.cos(2.0 * np.pi * 5.0 * t)
    _, f_ax, W = wigner_ville(x, fs=fs)
    spec = np.abs(W).sum(axis=0)
    ridge = f_ax[np.argmax(spec)]
    assert abs(ridge - 5.0) <= 2 * (fs / (2 * N)), ridge


# --------------------------------------- 8. Guard nperseg при нечётном N
def test_odd_n_nperseg_guard():
    """nperseg > N при нечётном N не падает (чётный nperseg <= N)."""
    rng = np.random.default_rng(11)
    x = (rng.normal(size=129) + 1j * rng.normal(size=129)).astype(np.complex128)
    rid = rid_cube(x[None, :], fs=1.0, nperseg=256)
    assert rid["Q_mag"].shape[1] == 128         # nperseg = 128 (чётное <= 129)
    st = stft_spectrogram(x, fs=1.0, nperseg=256)
    assert st["Z"].shape[0] == 128              # nperseg = 128, fftshift