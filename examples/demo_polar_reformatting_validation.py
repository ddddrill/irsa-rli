"""Демо валидации полярной переформатизации (Этап 7.3).

Генерирует одну цель с большим углом поворота и выводит три графика:
  (A) Naive 2D FFT (без переформатизации) — артефакт дуг
  (B) K-space: до vs после переформатизации
  (C) Polar Reformatting + 2D IFFT — сфокусированное изображение

Запуск:
    python -m examples.demo_polar_reformatting_validation
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
from processing.azimuth_compress import azimuth_compress, compute_amplitude
from processing.polar_reformat import (
    build_kspace_grid, design_cartesian_grid,
    interpolate_polar_to_cartesian, process_blind_zones,
    form_image_2d, compute_amplitude_db,
    compute_image_entropy, compute_isnr,
    plot_kspace, plot_kspace_cartesian,
)

SPEED_OF_LIGHT = 3e8


def run(
    f_c=8e9,
    spectr_w=1.5e9,
    ang=5.0,
    R0=500_000.0,
    omega=0.524,
    pri=0.0005,
    num_pulses=1000,
    satellite="ICEsat2",
    Xmax=40.0,
    Ymax=30.0,
    Xmax_scene=50.0,
    Ymax_scene=50.0,
    nifft_size=2048,
    verbose=True,
):
    T_obs = num_pulses * pri
    delta_theta_rad = omega * T_obs
    delta_theta_deg = math.degrees(delta_theta_rad)

    if verbose:
        print("=" * 60)
        print("  VALIDATION: Polar Reformatting for Large-Angle ISAR")
        print("=" * 60)
        print(f"  Frequency:   {f_c / 1e9:.1f} GHz")
        print(f"  Bandwidth:   {spectr_w / 1e9:.1f} GHz")
        print(f"  omega:       {omega:.4f} rad/s")
        print(f"  PRI:         {pri * 1e3:.2f} ms")
        print(f"  Pulses:      {num_pulses}")
        print(f"  T_obs:       {T_obs * 1e3:.1f} ms")
        print(f"  Delta_theta: {delta_theta_deg:.2f} deg ({delta_theta_rad:.4f} rad)")
        print(f"  V:           0 m/s (Turntable)")
        print("=" * 60)

    # ── Генерация данных ──────────────────────────────────────────────────
    radar = Radar(c=SPEED_OF_LIGHT, f_c=f_c, Xmax=Xmax, Ymax=Ymax,
                  spectr_w=spectr_w, ph_c=0.0, ang=ang, nifft_size=nifft_size)
    filename = f"matrices/matrices_{satellite}_high.pkl"
    target = Target(filename, pri=pri, num_pulses=num_pulses,
                    R0=R0, V=0.0, alpha=0.0, omega=omega)
    E, f_r = generate_raw_matrix(radar, target)
    M, N = E.shape

    # ── K-space ───────────────────────────────────────────────────────────
    kspace = build_kspace_grid(f_r, omega, pri, N)
    Kx, Ky = kspace["Kx"], kspace["Ky"]
    k, theta = kspace["k"], kspace["theta"]

    # ── (A) Наивный метод: стандартный конвейер без переформатизации ───────
    P, P_abs, range_axis, dr = range_compress(E, f_r, R0=R0)
    I_naive, azimuth_axis, _ = azimuth_compress(
        P.astype(np.complex128), omega, f_c, pri=pri
    )
    amp_naive = compute_amplitude(I_naive, mode="db")
    extent_naive = [azimuth_axis[0], azimuth_axis[-1], range_axis[-1], range_axis[0]]
    entropy_naive = compute_image_entropy(I_naive)

    # ── (B) K-space: до и после переформатизации ─────────────────────────
    grid = design_cartesian_grid(k, theta, Xmax_scene, Ymax_scene)
    E_cart = interpolate_polar_to_cartesian(E, k, theta, grid, method="linear")
    kz = grid["kx_axis"]
    ky = grid["ky_axis"]
    blind = process_blind_zones(E_cart, kz, ky, window_type="taylor",
                                apply_zero_pad=True, pad_factor=2, nbar=4, sll=-35.0)
    E_final = blind["E_final"]

    # ── (C) Polar Reformatting + 2D IFFT ──────────────────────────────────
    I_complex, x_axis, y_axis, x_span, y_span = form_image_2d(
        E_final, grid["dkx"], grid["dky"]
    )
    amp_bilinear = compute_amplitude_db(I_complex)
    extent_bilinear = [y_axis[0], y_axis[-1], x_axis[-1], x_axis[0]]
    entropy_bilinear = compute_image_entropy(I_complex)

    # ISNR: сравниваем с "размытым" (наивным) изображением
    # Интерполяция наивного изображения для сравнения
    isnr = compute_isnr(I_complex[:M, :N], I_naive)

    if verbose:
        print(f"\n=== Metrics ===")
        print(f"  Entropy (naive):     {entropy_naive:.2f} bits")
        print(f"  Entropy (reformat):  {entropy_bilinear:.2f} bits")
        print(f"  Entropy improvement: {(entropy_naive - entropy_bilinear):.2f} bits "
              f"({(entropy_naive - entropy_bilinear) / entropy_naive * 100:.1f}%)")
        print(f"  ISNR:                {isnr:.1f} dB")

    # ── Построение графиков (Шаг 7.3) ────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))

    # (A) Наивное изображение
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(amp_naive, aspect="auto", cmap="jet",
                     extent=extent_naive, vmin=-40, vmax=0)
    ax1.set_xlabel("Cross-Range (m)")
    ax1.set_ylabel("Range (m)")
    ax1.set_title(f"(A) Naive 2D FFT - arc smearing\nentropy={entropy_naive:.1f} bits")
    plt.colorbar(im1, ax=ax1, label="dB")

    # (B) K-space: до и после
    ax2 = fig.add_subplot(2, 2, 2)
    kx_axis = grid["kx_axis"]
    ky_axis = grid["ky_axis"]
    ax2_left = ax2
    plot_kspace(Kx, Ky, E, ax=ax2_left,
                title="(B) K-space: polar (before reformatting)")

    # (C) Финальное изображение
    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(amp_bilinear, aspect="auto", cmap="jet",
                     extent=extent_bilinear, vmin=-40, vmax=0)
    ax3.set_xlabel("Cross-Range (m)")
    ax3.set_ylabel("Range (m)")
    ax3.set_title(f"(C) Polar Reformat + 2D IFFT - focused\n"
                  f"entropy={entropy_bilinear:.1f} bits, ISNR={isnr:.1f} dB")
    plt.colorbar(im3, ax=ax3, label="dB")

    # (D) K-space после переформатизации
    ax4 = fig.add_subplot(2, 2, 4)
    plot_kspace_cartesian(E_final, grid, ax=ax4,
                          title="(D) K-space: Cartesian (after reformatting)")

    fig.suptitle(
        f"Polar Reformatting Validation: dTheta={delta_theta_deg:.1f} deg, "
        f"omega={omega:.3f} rad/s, T_obs={T_obs * 1e3:.0f} ms",
        fontsize=13,
    )
    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(out_dir, "demo_polar_reformatting_validation.png")
    plt.savefig(fig_path, dpi=150)
    if sys.flags.interactive:
        plt.show()
    else:
        plt.close(fig)
        print(f"\nSaved: {fig_path}")

    return {
        "amp_naive": amp_naive,
        "amp_bilinear": amp_bilinear,
        "entropy_naive": entropy_naive,
        "entropy_bilinear": entropy_bilinear,
        "isnr": isnr,
    }


if __name__ == "__main__":
    run(verbose=True)
