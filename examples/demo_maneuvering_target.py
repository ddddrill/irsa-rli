"""Демо: маневрирующая цель — RD vs STFT vs RID-кадр (γ ≠ 0).

Сценарий (§6.5 плана; [TF, §5.4]; [C&M, §3.2]): равномерное вращение ω и
угловое ускорение γ, угол поворота θ(t) = ω·t + 0.5·γ·t² — C&M, (1.13).
Мгновенная угловая скорость θ̇(t) = ω + γ·t растёт, карта
«кросс-рейндж → Доплер» перестаёт быть постоянной (§6.2.1): Доплер каждого
рассеивателя — линейный чирп со скоростью

    df_D/dt = −(2/λ)·(γ·u + ω²·r),                 (6.2.3)

(u — кросс-рейндж, r — дальность точки). RD-изображение (азимутальное БПФ за
всю апертуру T) честно размазывается по критерию QPE: (6.2.5) нарушается в
~30 раз. STFT-спектрограмма показывает чирпирующий (линейный/квадратичный)
Доплер; отдельный RID-кадр — «мгновенный снимок» с локально-постоянным
Доплером (§6.2.3, §7.2) и существенно меньшей энтропией (критерий §6.5:
разрыв ≥ 15 %).

Параметры (план §6.5, подобраны при отладке под критерии):
    f_c = 10 ГГц (λ = 3 см), B = 300 МГц (dr = 0.5 м),
    PRI = 0.05 с, N = 256 (T = 12.8 с),
    ω = 0.01 рад/с, γ = 2e-4 рад/с²   →  Δθ_total ≈ 0.144 рад ≈ 8.3°.
Цель — matrices_solarB_mid (кросс-рейндж до ≈ 7 м): миграция дальности
y·ω·T ≈ 0.88 м ≈ 1.8 dr; веер Доплера цели ≈ 9.2 Гц < fs/2 = 10 Гц, без
алиасинга. RID: nperseg = 32 (T_w = 1.6 с) — внутрикадровый чирп
|df/dt|·T_w ≈ 0.27 Гц меньше разрешения кадра 1/T_w = 0.625 Гц.

Энтропия — compute_image_entropy (processing/polar_reformat, §7.4; одинаковая
для RD и RID: по |I|^2 / |Q|^2). Маргинал куба Σ_t |Q|^2 [TF, (5.18)]
сопоставляется с RD на STFT-сетке (группировка |FFT|^2 в F бинов — конвенция
demo_rid_imaging / R5a).

Запуск:
    python -m examples.demo_maneuvering_target
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.radar import Radar
from models.target import Target
from simulation.raw_generator import SimEffects, generate_raw_matrix
from processing.range_compress import range_compress
from processing.azimuth_compress import azimuth_compress, compute_amplitude
from processing.tf_analysis import rid_cube, rid_frame, rid_marginal, stft_spectrogram
from processing.polar_reformat import compute_image_entropy

SPEED_OF_LIGHT = 3e8

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "demo_maneuvering_target.png",
)


def compute_maneuvering_metrics(f_c=10e9, spectr_w=300e6, nifft_size=512,
                                pri=0.05, num_pulses=256, omega=0.01,
                                gamma=2e-4, nperseg=32, r0=500000.0,
                                matrix="matrices/matrices_solarB_mid.pkl",
                                seed=42):
    """Детерминированный расчёт демо (§6.5): энтропии RD/RID и метрики.

    Args:
        f_c: несущая частота (Гц).
        spectr_w: полоса (Гц) — B = 300 МГц → dr = 0.5 м.
        nifft_size: параметр сетки частот радара.
        pri: период повторения импульсов (с).
        num_pulses: число импульсов N (T = N·pri = 12.8 с).
        omega: угловая скорость вращения (рад/с).
        gamma: угловое ускорение (рад/с²), C&M (1.13).
        nperseg: длина окна STFT/RID (T_w = nperseg·pri = 1.6 с).
        r0: начальная дальность до ЦМ (м), дешифровка.
        matrix: путь к матрице рассеивателей.
        seed: seed генераторов эффектов (без шума — всё детерминировано).

    Returns:
        dict: e_rd, e_rd_stft, e_marginal, e_frame, gap_pct,
        gap_stftgrid_pct, peak_frame, peak_rd, top_energy_frac, dr, fs, F,
        T, omega, gamma, lam, а также I, rid, stft, mid_frame_idx и оси.
    """
    lam = SPEED_OF_LIGHT / f_c
    radar = Radar(c=SPEED_OF_LIGHT, f_c=f_c, Xmax=25.0, Ymax=25.0,
                  spectr_w=spectr_w, ph_c=0.0, ang=9.0, nifft_size=nifft_size)
    target = Target(matrix, pri=pri, num_pulses=num_pulses, R0=r0,
                    V=0.0, alpha=0.0, omega=omega, gamma=gamma)

    E, f_r = generate_raw_matrix(radar, target, effects=SimEffects(seed=seed))
    P, P_abs, range_axis, dr = range_compress(E, f_r, R0=r0)
    I, azimuth_axis, doppler_axis = azimuth_compress(P, omega, f_c, pri=pri)

    fs = 1.0 / pri
    rid = rid_cube(P, fs=fs, n_frames=16, nperseg=nperseg, hop=1)
    marg_e = rid_marginal(rid)
    F = marg_e.shape[1]
    mid_idx = len(rid["frame_idx"]) // 2
    mid_frame_idx = int(rid["frame_idx"][mid_idx])
    frame_mag = rid["Q_mag"][:, :, mid_frame_idx]

    # --- энтропии (compute_image_entropy, одинаково для RD и RID) ---
    e_rd = compute_image_entropy(I)
    # RD на STFT-сетке: группировка |FFT|^2 в F бинов (конвенция R5a).
    rr = I.shape[1] // F
    rd_pow = (np.abs(I) ** 2)[:, :rr * F].reshape(I.shape[0], rr, F).sum(axis=1)
    e_rd_stft = compute_image_entropy(np.sqrt(rd_pow))
    e_marginal = compute_image_entropy(np.sqrt(marg_e))
    e_frame = compute_image_entropy(frame_mag)

    # --- «острота»: норм. пик и концентрация топ-20 ячеек (§6.2.5) ---
    def _top_frac(X, k=20):
        s = np.sort(X.ravel())[::-1]
        return float(s[:k].sum() / s.sum())

    peak_frame = float(frame_mag.max() / frame_mag.sum())
    peak_rd = float(rd_pow.max() / rd_pow.sum())
    top_energy_frac = (_top_frac(frame_mag ** 2.0), _top_frac(rd_pow))

    # --- STFT сильнейшего дальностного бина (панель 2) ---
    strongest_bin = int((P_abs ** 2.0).sum(axis=1).argmax())
    st = stft_spectrogram(P[strongest_bin], fs=fs, nperseg=nperseg,
                          noverlap=nperseg - 1)

    return dict(
        f_c=f_c, lam=lam, spectr_w=spectr_w, pri=pri, num_pulses=num_pulses,
        omega=omega, gamma=gamma, nperseg=nperseg, r0=r0, matrix=matrix,
        seed=seed, T=num_pulses * pri, fs=fs, dr=dr, F=F,
        e_rd=e_rd, e_rd_stft=e_rd_stft, e_marginal=e_marginal, e_frame=e_frame,
        gap_pct=(1.0 - e_frame / e_rd) * 100.0,
        gap_stftgrid_pct=(1.0 - e_frame / e_rd_stft) * 100.0,
        peak_frame=peak_frame, peak_rd=peak_rd,
        top_energy_frac=top_energy_frac,
        strongest_bin=strongest_bin,
        stft=st, rid=rid, mid_frame_idx=mid_frame_idx,
        I=I, P_abs=P_abs, range_axis=range_axis,
        doppler_axis=doppler_axis, azimuth_axis=azimuth_axis,
    )


def run(verbose=True, out_path=DEFAULT_OUT):
    """Полный прогон: расчёт + 2×2 панели + сохранение PNG + печать метрик.

    Args:
        verbose: печатать параметры/метрики.
        out_path: путь PNG.

    Returns:
        dict метрик из compute_maneuvering_metrics.
    """
    m = compute_maneuvering_metrics()

    T = m["T"]
    lam = m["lam"]
    rad = m["omega"] * T + 0.5 * m["gamma"] * T ** 2.0
    if verbose:
        print("=" * 62)
        print("  МАНЕВРИРУЮЩАЯ ЦЕЛЬ: RD vs STFT vs RID-кадр (gamma != 0)")
        print("=" * 62)
        print(f"  f_c = {m['f_c'] / 1e9:.0f} ГГц (lam = {lam * 1e2:.1f} см), "
              f"B = {m['spectr_w'] / 1e6:.0f} МГц (dr = {m['dr']:.2f} м)")
        print(f"  PRI = {m['pri'] * 1e3:.0f} мс, N = {m['num_pulses']}, "
              f"T = {T:.1f} с")
        print(f"  omega = {m['omega']} рад/с, gamma = {m['gamma']:.0e} рад/с^2 "
              f"-> d_theta ~ {rad:.3f} рад ~ {rad * 57.2958:.1f} град")
        print(f"  Цель: {m['matrix']}  "
              f"(веер Доплера цели < fs/2 = {m['fs'] / 2:.0f} Гц)")
        print("-" * 62)
        print("  ЭНТРОПИЯ (compute_image_entropy, бит):")
        print(f"    RD-изображение (M x N):          {m['e_rd']:.3f}")
        print(f"    RD на STFT-сетке (M x F):        {m['e_rd_stft']:.3f}")
        print(f"    маргинал RID (сумма по времени): {m['e_marginal']:.3f}")
        print(f"    RID-кадр t={m['rid']['t'][m['mid_frame_idx']]:.2f} c: "
              f"{m['e_frame']:.3f}")
        print(f"    Разрыв кадр vs RD:               "
              f"{m['gap_pct']:.1f}% (критерий §6.5: >= 15%)")
        print(f"    Разрыв на STFT-сетке:            "
              f"{m['gap_stftgrid_pct']:.1f}%")
        print(f"    Концентрация топ-20 ячеек:       кадр "
              f"{m['top_energy_frac'][0]:.3f} vs RD "
              f"{m['top_energy_frac'][1]:.3f}")

    rid, st = m["rid"], m["stft"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Маневрирующая цель: ω=const, γ≠0 — RD размазано, STFT видит чирп, "
        f"RID-кадр «замораживает» вращение (γ = {m['gamma']:.0e} рад/с²)",
        fontsize=12)

    # Панель 1: Range-Doppler
    ax = axes[0, 0]
    amp_mm = compute_amplitude(m["I"], mode="db")
    extent_rd = [m["doppler_axis"][0], m["doppler_axis"][-1],
                 m["range_axis"][-1], m["range_axis"][0]]
    im = ax.imshow(amp_mm, aspect="auto", cmap="jet", extent=extent_rd,
                   vmin=-40, vmax=0)
    ax.set_title(f"RD-изображение (апертура {T:.1f} с)\nразмазано: чирп "
                 f"(6.2.3) + миграция дальности")
    ax.set_xlabel("Доплер, Гц")
    ax.set_ylabel("Дальность, м")
    plt.colorbar(im, ax=ax, label="дБ")

    # Панель 2: STFT-спектрограмма сильного дальностного бина
    ax = axes[0, 1]
    f, t, Z, S_db = st["f"], st["t"], st["Z"], st["S_db"]
    im = ax.imshow(S_db, aspect="auto", cmap="jet", origin="lower",
                   extent=[t[0], t[-1], f[0], f[-1]], vmin=-40, vmax=0)
    ridge = f[np.argmax(np.abs(Z) ** 2.0, axis=0)]
    ax.plot(t, ridge, "w--", lw=0.8, alpha=0.7)
    ax.set_title(f"STFT дальн. бина №{m['strongest_bin']}\n"
                 f"чирпирующий Доплер f_D(t) = −(2/λ)·u·(ω+γt)")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Доплер, Гц")
    plt.colorbar(im, ax=ax, label="дБ")

    # Панель 3: RID-кадр в середине апертуры
    ax = axes[1, 0]
    rid_mid = len(rid["frame_idx"]) // 2
    frame = rid_frame(rid, rid_mid)
    t_frame = rid["t"][rid["frame_idx"][rid_mid]]
    im = ax.imshow(frame, aspect="auto", cmap="jet", origin="lower",
                   extent=[rid["f"][0], rid["f"][-1],
                           m["range_axis"][-1], m["range_axis"][0]],
                   vmin=-40, vmax=0)
    ax.set_title(f"RID-кадр t={t_frame:.2f} с (T_w={m['nperseg'] * m['pri']:.2f} с)\n"
                 f"«мгновенный снимок» — энтропия {m['e_frame']:.2f} бит")
    ax.set_xlabel("Доплер, Гц")
    ax.set_ylabel("Дальность, м")
    plt.colorbar(im, ax=ax, label="дБ")

    # Панель 4: сравнение энтропий
    ax = axes[1, 1]
    labels = ["RD (M×N)", "RD на STFT-\nсетке (M×F)",
              "маргинал\nRID", "RID-кадр"]
    vals = [m["e_rd"], m["e_rd_stft"], m["e_marginal"], m["e_frame"]]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.05, f"{v:.2f}",
                ha="center", fontsize=9)
    ax.axhline(m["e_rd"] * 0.85, color="k", ls=":", lw=1.0)
    ax.text(len(labels) - 0.5, m["e_rd"] * 0.85, "15% от RD",
            ha="right", va="top", fontsize=8)
    ax.set_title(f"Энтропия (бит): разрыв кадр vs RD = {m['gap_pct']:.1f}%")
    ax.set_ylabel("Шеннон-энтропия, бит")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.abspath(out_path)
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"\nСохранено: {fig_path}")
    return m


def main():
    run(verbose=True)


if __name__ == "__main__":
    main()
