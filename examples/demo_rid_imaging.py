"""Демо: RID-кадры и маргинал куба vs азимутальное БПФ (равномерное вращение).

Сценарий (план §7.5): та же цель, что и в фикстурах test_tf_analysis
(ω = radians(9°)/256 на импульс, PRI = 0.05 с, N = 256). Демонстрирует:
    1. 16 равноотстоящих RID-кадров [TF, §5.3];
    2. маргинальная сумма I(r, f) = Σ_t |Q|^2 [TF, (5.18)] сравнение с
       |FFT|^2 по медленному времени — изображения совпадают на глаз,
       печатается относительная ошибка (дБ).

Запуск:
    python -m examples.demo_rid_imaging
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import matplotlib.pyplot as plt

from models.radar import Radar
from models.target import Target
from simulation.raw_generator import generate_raw_matrix
from processing.range_compress import range_compress
from processing.mocomp import mocomp
from processing.tf_analysis import rid_cube, rid_frame, rid_marginal, SCALE

SPEED_OF_LIGHT = 3e8


def run(f_c=10e9, pri=0.05, num_pulses=256, R0=500000.0, verbose=True):
    omega = math.radians(9.0) / 256.0
    if verbose:
        print("=" * 62)
        print("  RID-КАДРЫ И МАРГИНАЛ vs АЗИМУТАЛЬНОЕ БПФ")
        print("=" * 62)
        print(f"  Несущая: {f_c / 1e9:.1f} ГГц,  omega = {omega:.5f} рад/имп (9°/256)")
        print(f"  PRI = {pri * 1e3:.0f} мс,  N = {num_pulses} импульсов")
        print(f"  SCALE (N=256, nperseg=128, hann) = {SCALE:.4f}")
        print("-" * 62)

    radar = Radar(c=SPEED_OF_LIGHT, f_c=f_c, Xmax=20.0, Ymax=20.0,
                  spectr_w=0.5e9, ph_c=0.0, ang=14.0, nifft_size=256)
    target = Target(
        "matrices/matrices_solarB_high.pkl", pri=pri, num_pulses=num_pulses,
        R0=R0, V=0.0, alpha=0.0, omega=omega,
    )

    E, f_r = generate_raw_matrix(radar, target)
    P, _, range_axis, _ = range_compress(E, f_r, R0=R0)
    P_comp, _, _, _ = mocomp(P, range_axis)

    fs = 1.0 / pri
    rid = rid_cube(P_comp, fs=fs, n_frames=16, nperseg=128, hop=1)
    Q, f_ax, t_ax = rid["Q_mag"], rid["f"], rid["t"]

    # маргинал vs азимутальное |FFT|^2 (|FFT| прорежается на STFT-сетку)
    marg = rid_marginal(rid)
    F2 = np.abs(np.fft.fftshift(np.fft.fft(P_comp, axis=1))) ** 2.0
    grp = P_comp.shape[1] // marg.shape[1]
    F2 = F2.reshape(P_comp.shape[0], marg.shape[1], grp).sum(axis=2)

    def _db_norm_rows(X):
        X = np.abs(X)
        if X.ndim == 2:
            return 20.0 * np.log10(X / X.max(axis=1, keepdims=True) + 1e-12)
        return 20.0 * np.log10(X / X.max() + 1e-12)

    per_row = [np.linalg.norm(_db_norm_rows(F2[m]) - _db_norm_rows(marg[m])) /
               np.linalg.norm(_db_norm_rows(F2[m]))
               for m in range(P_comp.shape[0])]
    err_db = float(np.mean(per_row))
    corr = np.corrcoef(marg.ravel(), F2.ravel())[0, 1]

    if verbose:
        print(f"\n  Маргинал:          {marg.shape}")
        print(f"  Кадров в кубе:     {Q.shape[2]}, показано {len(rid['frame_idx'])}")
        print(f"  Ср. отн. ошибка (дБ): {err_db:.3f}")
        print(f"  Корреляция изображений: {corr:.4f} (на глаз совпадают, если близка к 1)")

    fig = plt.figure(figsize=(13, 10))
    fig.suptitle("RID-кадры (16 шт.) и маргинал vs азимутальное БПФ",
                 fontsize=13)

    grid = fig.add_gridspec(5, 4)
    for i in range(16):
        ax = fig.add_subplot(grid[i // 4, i % 4])
        fr = rid_frame(rid, i)
        ax.imshow(fr, aspect="auto", cmap="jet", origin="lower",
                  vmin=-40, vmax=0)
        ax.set_title(f"t={t_ax[rid['frame_idx'][i]]:.2f} c")
        ax.set_xticks([])
        ax.set_yticks([])

    ax1 = fig.add_subplot(grid[4, 0:2])
    im1 = ax1.imshow(_db_norm_rows(marg), aspect="auto", cmap="jet",
                     origin="lower", vmin=-40, vmax=0)
    ax1.set_title(f"RID-маргинал Σ_t |Q|^2 (SCALE={SCALE:.1f})")
    ax1.set_xlabel("Доплер, Гц")
    ax1.set_ylabel("Дальность")
    plt.colorbar(im1, ax=ax1, label="дБ")

    ax2 = fig.add_subplot(grid[4, 2:4])
    im2 = ax2.imshow(_db_norm_rows(F2), aspect="auto", cmap="jet",
                     origin="lower", vmin=-40, vmax=0)
    ax2.set_title(f"|FFT|^2 по медленному времени (err_dB={err_db:.3f}, "
                  f"r={corr:.3f})")
    ax2.set_xlabel("Доплер, Гц")
    ax2.set_ylabel("Дальность")
    plt.colorbar(im2, ax=ax2, label="дБ")

    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(out_dir, "demo_rid_imaging.png")
    plt.savefig(fig_path, dpi=150)
    if sys.flags.interactive:
        plt.show()
    else:
        plt.close(fig)
        print(f"\nСохранено: {fig_path}")

    return rid, marg, F2


if __name__ == "__main__":
    run(verbose=True)