"""Демо R3: кросс-масштабирование по данным (chirp-rate).

Сценарий (план §8): синтетическая цель из 5 рассеивателей на равномерно
вращающейся подставке (без эффектов реализма). Конвейер:
    Raw -> Range Compression -> MOCOMP -> Azimuth Compression
    -> cross_range_scale(I, range_axis, doppler_axis, f_c, fs, method="phasefit").

Восстановленная строка изображения несёт чирп mu = K*y + b: наклон K
соответствует Omega_eff (точка пересечения b поглощает опорный бин
автофокусировки). Показаны:
    1. РЛИ (дБ);
    2. регрессия myû(y) с доверительной зоной 2-sigma и итоговой
       оценкой Omega_eff (сравнение с номинальной omega цели).

Запуск:
    python -m examples.demo_cross_range_scaling
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import matplotlib.pyplot as plt

from models.radar import Radar
from models.target import Target
from simulation.raw_generator import generate_raw_matrix, SimEffects
from processing.range_compress import range_compress
from processing.mocomp import mocomp
from processing.azimuth_compress import azimuth_compress
from processing.cross_range_scaling import cross_range_scale

SPEED_OF_LIGHT = 3e8


def run(f_c=8e9, spec_w=0.5e9, pri=1.0, num_pulses=64, R0=500000.0,
        verbose=True):
    omega = math.radians(9.0) / num_pulses
    rk = [-4.4, -2.2, 0, 2.2, 4.4]
    uk = [-2, -1, 0, 1, 2]
    ints = np.linspace(1.0, 5.0, len(rk))

    if verbose:
        print("=" * 62)
        print("  КРОСС-МАСШТАБИРОВАНИЕ ПО ДАННЫМ (CHIRP-RATE)")
        print("=" * 62)
        print(f"  Несущая: {f_c / 1e9:.1f} ГГц,  спектр: {spec_w / 1e9:.1f} ГГц")
        print(f"  PRI = {pri:.1f} с,  N = {num_pulses} импульсов")
        print(f"  omega (номинал цели) = {omega:.6f} рад/с")
        print(f"  R = {rk} (м),  U = {uk} (м/с)")
        print("-" * 62)

    radar = Radar(c=SPEED_OF_LIGHT, f_c=f_c, Xmax=60.0, Ymax=60.0,
                  spectr_w=spec_w, ph_c=0.0, ang=9.0, nifft_size=1024)
    sp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "_r3_demo_target.npy")
    np.save(sp, np.array([[ints, rk, uk]], dtype=object), allow_pickle=True)
    target = Target(sp, pri=pri, num_pulses=num_pulses, R0=R0, V=0.0,
                    alpha=0.0, omega=omega)

    E, f_r = generate_raw_matrix(radar, target, effects=SimEffects())
    P, _, range_axis, _ = range_compress(E, f_r, R0=R0, window="none",
                                         window_param=None)
    P_comp, _, _, _ = mocomp(P, range_axis)
    I, az_cfg, doppler_axis = azimuth_compress(
        P_comp, omega, f_c, pri=pri, window="none", window_param=None,
    )

    cr = cross_range_scale(I, range_axis, doppler_axis, f_c, 1.0 / pri,
                           method="phasefit")

    if verbose:
        print(f"  Omega_eff = {cr['omega_eff']:.6f} рад/с")
        print(f"  Omega_eff/omega = {cr['omega_eff'] / omega:.4f}")
        print(f"  R^2 = {cr['r2']:.4f},  N = {cr['n_used']},  "
              f"fallback = {cr['fallback']}")
        print("-" * 62)

    db = 20 * np.log10(np.abs(I) + 1e-12)
    vmin, vmax = np.percentile(db, [5, 99.5])
    dopp = cr["azimuth_axis_hz"] if cr["azimuth_axis_hz"] is not None else doppler_axis
    az_m = cr["azimuth_axis_m"] if cr["azimuth_axis_m"] is not None else az_cfg

    fig = plt.figure(figsize=(16, 5.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0], wspace=0.28)

    y_ext = [range_axis[0], range_axis[-1], range_axis[-1], range_axis[0]]

    ax1 = fig.add_subplot(grid[0])
    im1 = ax1.imshow(db, aspect="auto", cmap="jet", origin="lower",
                     extent=[dopp[0], dopp[-1], range_axis[0], range_axis[-1]],
                     vmin=vmin, vmax=vmax)
    ax1.set_title("РЛИ (дБ) до калибровки оси")
    ax1.set_xlabel("Доплер, Гц")
    ax1.set_ylabel("Дальность, м")
    plt.colorbar(im1, ax=ax1, label="дБ")

    ax2 = fig.add_subplot(grid[1])
    im2 = ax2.imshow(db, aspect="auto", cmap="jet", origin="lower",
                     extent=[az_m[0], az_m[-1], range_axis[0], range_axis[-1]],
                     vmin=vmin, vmax=vmax)
    ax2.set_title("РЛИ (дБ) после калибровки оси\n"
                  f"($\\Omega_{{eff}}$/ω = {cr['omega_eff'] / omega:.3f})")
    ax2.set_xlabel("Азимут, м")
    ax2.set_ylabel("Дальность, м")
    plt.colorbar(im2, ax=ax2, label="дБ")

    ax3 = fig.add_subplot(grid[2])
    pts = np.array(cr["points"]) if cr["points"] else np.empty((0, 3))
    if len(pts) and not cr["fallback"]:
        xs, ys_mu, ws = pts[:, 0], pts[:, 1], pts[:, 2]
        sc = ax3.scatter(xs, ys_mu, c=ws, cmap="viridis", s=60, edgecolor="k",
                         zorder=3, label="точки регрессии")
        x_fit = np.linspace(xs.min(), xs.max(), 100) if len(xs) > 1 else xs
        y_fit = np.polyval([cr["K"], cr["intercept"]], x_fit)
        ax3.plot(x_fit, y_fit, "r--", lw=1.8,
                 label=f"myu = K·y + b  (R$^2$={cr['r2']:.3f}, N={cr['n_used']})")
        plt.colorbar(sc, ax=ax3, label="peak amp")
    else:
        ax3.text(0.5, 0.5, "fallback → cfg", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=14, color="gray")
    ax3.axhline(0, color="gray", lw=0.6)
    ax3.set_title("Чирп-темп vs кросс-дальность")
    ax3.set_xlabel("y, м")
    ax3.set_ylabel("$\\hat{\\mu}(y)$, рад·с$^{-2}$")
    ax3.legend(loc="best")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(out_dir, "demo_cross_range_scaling.png")
    plt.savefig(fig_path, dpi=150)
    if sys.flags.interactive:
        plt.show()
    else:
        plt.close(fig)
        print(f"\nСохранено: {fig_path}")

    os.remove(sp) if os.path.exists(sp) else None
    return cr, omega


if __name__ == "__main__":
    run(verbose=True)