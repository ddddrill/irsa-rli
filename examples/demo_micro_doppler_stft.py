"""Демо: Микро-Доплеровская сигнатура вибрации (STFT + WVD + суммарное эхо).

Сценарий (план §7.5): вибрация солнечной панели A_vib = 5 мм, f_vib = 3 Гц;
несущая 10 ГГц; PRI = 0.025 с (fs = 40 Гц); N = 512 импульсов.

Ожидания [MD, §1.8]:
    - полоса микро-Доплера ±(4π/λ)·A_vib·f_vib ≈ ±6.28 Гц (пунктиры на графике);
    - периодическая сигнатура с периодом 1/f_vib = 1/3 с;
    - правило Карсона: значимая гребёнка Бесселя k ≤ B+1 (B = 4πA/λ ≈ 2.09).

Запуск:
    python -m examples.demo_micro_doppler_stft
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from models.radar import Radar
from models.target import Target
from simulation.raw_generator import generate_raw_matrix
from processing.range_compress import range_compress
from processing.mocomp import mocomp
from processing.tf_analysis import (
    stft_spectrogram,
    wigner_ville,
    total_echo_signature,
    strongest_range_bins,
)

SPEED_OF_LIGHT = 3e8


def run(f_c=10e9, vib_amplitude=0.005, vib_frequency=3.0, pri=0.025,
        num_pulses=512, R0=500000.0, verbose=True):
    lam = SPEED_OF_LIGHT / f_c
    band = (4.0 * np.pi / lam) * vib_amplitude * vib_frequency

    if verbose:
        print("=" * 62)
        print("  МИКРО-ДОПЛЕРОВСКАЯ СИГНАТУРА ВИБРАЦИИ (STFT / WVD)")
        print("=" * 62)
        print(f"  Несущая:                  {f_c / 1e9:.1f} ГГц  (lam = {lam * 1e3:.2f} мм)")
        print(f"  Вибрация:                 A = {vib_amplitude * 1e3:.0f} мм, f_v = {vib_frequency} Гц")
        print(f"  PRI = {pri * 1e3:.0f} мс -> fs = {1 / pri:.0f} Гц, N = {num_pulses}")
        print(f"  Индекс модуляции:         B = 4*pi*A/lam = {4 * np.pi * vib_amplitude / lam:.2f}")
        print(f"  Мгновенная полоса:        ±{band:.2f} Гц")
        print(f"  Полоса Карсона:           ±(B+1)·f_v = ±{(4 * np.pi * vib_amplitude / lam + 1) * vib_frequency:.2f} Гц")
        print("-" * 62)

    radar = Radar(c=SPEED_OF_LIGHT, f_c=f_c, Xmax=20.0, Ymax=20.0,
                  spectr_w=0.5e9, ph_c=0.0, ang=14.0, nifft_size=256)
    target = Target(
        "matrices/matrices_solarB_high.pkl", pri=pri, num_pulses=num_pulses,
        R0=R0, V=0.0, alpha=0.0, omega=0.0,
        vib_amplitude=vib_amplitude, vib_frequency=vib_frequency,
    )

    E, f_r = generate_raw_matrix(radar, target)
    P, _, range_axis, _ = range_compress(E, f_r, R0=R0)
    P_comp, _, _, _ = mocomp(P, range_axis)

    fs = 1.0 / pri
    # бин с максимальной энергией НЕ у нулевой доплеровской частоты —
    # "вибрационный" бин (наиболее динамичный) для наглядной сигнатуры
    ff = np.fft.fftshift(np.fft.fft(P_comp, axis=1), axes=1)
    freq = np.fft.fftshift(np.fft.fftfreq(P_comp.shape[1], pri))
    sel = (np.abs(freq) >= 1.0) & (np.abs(freq) <= 9.5)
    dyn = (np.abs(ff) ** 2.0)[:, sel].sum(axis=1)
    bin_idx = int(np.argmax(dyn))

    st = stft_spectrogram(P_comp[bin_idx], fs=fs, nperseg=64, noverlap=63)
    f, t, Z, S_db = st["f"], st["t"], st["Z"], st["S_db"]

    # энергетическая доля внутри полос и период сигнатуры (гребень)
    mag2 = np.abs(Z) ** 2.0
    tot = mag2.sum()
    f_carson = (4.0 * np.pi / lam) * vib_amplitude * vib_frequency
    frac_band15 = mag2[np.abs(f) <= 1.15 * band].sum() / tot
    frac_carson = mag2[np.abs(f) <= f_carson].sum() / tot
    # период сигнатуры виден в WVD-панели: мгновенная частота
    # f(t) = B·f_v·cos(2π·f_v·t) колеблется с периодом 1/f_vib = 1/3 с
    period = 1.0 / vib_frequency

    if verbose:
        print(f"\n  Энергия сигнатуры внутри ±15% полосы:   {frac_band15:.1%}")
        print(f"  Энергия сигнатуры внутри полосы Карсона: {frac_carson:.1%}")
        print(f"  Период модуляции (в WVD-панели):         {period:.3f} с = 1/f_vib")

    t_w, f_w, W = wigner_ville(P_comp[bin_idx], fs=fs)
    te = total_echo_signature(P_comp, fs=fs, nperseg=64, noverlap=63)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Микро-Доплеровская сигнатура вибрации: "
                 f"A={vib_amplitude * 1e3:.0f} мм, f_v={vib_frequency} Гц, "
                 f"полоса ±{band:.2f} Гц", fontsize=13)

    ax = axes[0, 0]
    prof = np.abs(P_comp[:, :200])
    ax.imshow(20 * np.log10(prof / prof.max() + 1e-12), aspect="auto",
              cmap="jet", origin="lower")
    ax.set_title(f"Профили дальности (бина сигнатуры №{bin_idx})")
    ax.set_xlabel("Импульс (медленное время)")
    ax.set_ylabel("Дальностный бин")
    ax.axhline(bin_idx, color="w", ls="--", lw=0.8)

    ax = axes[0, 1]
    im = ax.imshow(S_db, aspect="auto", cmap="jet", origin="lower",
                   extent=[t[0], t[-1], f[0], f[-1]], vmin=-40, vmax=0)
    ax.axhline(band, color="w", ls="--", lw=1.0)
    ax.axhline(-band, color="w", ls="--", lw=1.0)
    ax.set_title("STFT-спектрограмма вибрационного бина")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Доплер, Гц")
    plt.colorbar(im, ax=ax, label="дБ")

    ax = axes[1, 0]
    WA = np.abs(W) ** 2.0
    WA_db = 20 * np.log10(WA / WA.max() + 1e-12)
    im = ax.imshow(WA_db.T, aspect="auto", cmap="jet", origin="lower",
                   extent=[t_w[0], t_w[-1], f_w[0], f_w[-1]], vmin=-40, vmax=0)
    ax.axhline(band, color="w", ls="--", lw=1.0)
    ax.axhline(-band, color="w", ls="--", lw=1.0)
    ax.set_title("WVD того же бина (кросс-термы — свойство)")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Доплер, Гц, fs/2")
    plt.colorbar(im, ax=ax, label="дБ")

    ax = axes[1, 1]
    im = ax.imshow(te["S_db"], aspect="auto", cmap="jet", origin="lower",
                   extent=[t[0], t[-1], f[0], f[-1]], vmin=-40, vmax=0)
    ax.axhline(band, color="w", ls="--", lw=1.0)
    ax.axhline(-band, color="w", ls="--", lw=1.0)
    ax.set_title("Суммарная сигнатура суммарного эха")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Доплер, Гц")
    plt.colorbar(im, ax=ax, label="дБ")

    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(out_dir, "demo_micro_doppler_stft.png")
    plt.savefig(fig_path, dpi=150)
    if sys.flags.interactive:
        plt.show()
    else:
        plt.close(fig)
        print(f"\nСохранено: {fig_path}")

    return P_comp, st, (t_w, f_w, W), te


if __name__ == "__main__":
    run(verbose=True)