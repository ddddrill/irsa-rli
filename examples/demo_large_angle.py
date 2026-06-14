"""Демо: Большой угол поворота цели (15°) — индуцирование артефактов.

Сценарий: V=0, omega=0.524 рад/с, N=1000, PRI=0.5 мс.
Суммарный угол поворота: Delta_theta = omega * N * PRI = 15°.

Демонстрирует:
  1. K-space: кольцевой сектор вместо прямоугольника
  2. Проектирование декартовой сетки (Этап 3)
  3. РЛИ: размытие в дуги из-за кривизны данных в k-space

Запуск:
    python -m examples.demo_large_angle
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
    build_kspace_grid, plot_kspace,
    design_cartesian_grid, plot_grid_comparison,
    interpolate_polar_to_cartesian, plot_kspace_cartesian,
    process_blind_zones, form_image_2d, compute_amplitude_db,
    compute_image_entropy,
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
    nifft_size=2048,
    verbose=True,
):
    T_obs = num_pulses * pri
    delta_theta_rad = omega * T_obs
    delta_theta_deg = math.degrees(delta_theta_rad)

    if verbose:
        print("=" * 60)
        print("  ДЕМОНСТРАЦИЯ БОЛЬШОГО УГЛА ПОВОРОТА ЦЕЛИ")
        print("=" * 60)
        print(f"  Частота несущей:        {f_c / 1e9:.1f} ГГц")
        print(f"  Ширина спектра:         {spectr_w / 1e9:.1f} ГГц")
        print(f"  Угол места (beam_width): {ang:.1f}°")
        print(f"  omega:                  {omega:.4f} рад/с")
        print(f"  PRI:                    {pri * 1e3:.2f} мс")
        print(f"  Число импульсов:        {num_pulses}")
        print(f"  Время наблюдения:       {T_obs * 1e3:.1f} мс")
        print(f"  Суммарный угол:         {delta_theta_deg:.2f}° ({delta_theta_rad:.4f} рад)")
        print(f"  V (поступательное):     0.0 м/с (Turntable)")
        print(f"  MOCOMP:                 отключен")
        print("=" * 60)

    radar = Radar(
        c=SPEED_OF_LIGHT,
        f_c=f_c,
        Xmax=Xmax,
        Ymax=Ymax,
        spectr_w=spectr_w,
        ph_c=0.0,
        ang=ang,
        nifft_size=nifft_size,
    )

    filename = f"matrices/matrices_{satellite}_high.pkl"
    target = Target(
        filename,
        pri=pri,
        num_pulses=num_pulses,
        R0=R0,
        V=0.0,
        alpha=0.0,
        omega=omega,
    )

    E, f_r = generate_raw_matrix(radar, target)
    if verbose:
        print(f"\nRaw data matrix E: {E.shape} (M freq x N pulses)")
        print(f"Frequency vector f_r: [{f_r[0] / 1e9:.3f}, {f_r[-1] / 1e9:.3f}] GHz")

    # ── K-space assembly (Etapa 2: polar_reformat module) ─────────────────
    kspace = build_kspace_grid(f_r, omega, pri, E.shape[1])
    Kx, Ky = kspace["Kx"], kspace["Ky"]
    k_min, k_max = kspace["k_min"], kspace["k_max"]
    theta_max = kspace["bounds"]["theta_max"]

    if verbose:
        print(f"\nK-space (Etapa 2):")
        print(f"  k_min = {k_min:.1f} rad/m,  k_max = {k_max:.1f} rad/m")
        print(f"  theta_max = {theta_max:.4f} rad = {delta_theta_deg:.2f} deg")
        print(f"  Kx: [{Kx.min():.1f}, {Kx.max():.1f}] rad/m")
        print(f"  Ky: [{Ky.min():.1f}, {Ky.max():.1f}] rad/m")

    # ── Etapa 3: Cartesian grid design ────────────────────────────────────
    # Xmax_scene/Ymax_scene: размер сцены для антиалиасинга
    Xmax_scene = 50.0  # м (по дальности)
    Ymax_scene = 50.0  # м (по азимуту)
    grid = design_cartesian_grid(kspace["k"], kspace["theta"], Xmax_scene, Ymax_scene)

    if verbose:
        print(f"\nCartesian grid (Etapa 3):")
        print(f"  Scene size: {Xmax_scene} x {Ymax_scene} m")
        print(f"  dkx = {grid['dkx']:.4f} rad/m,  dky = {grid['dky']:.4f} rad/m")
        print(f"  Grid size: {grid['M_new']} x {grid['N_new']} points")
        print(f"  Kx range: [{grid['bounds']['kx_min']:.1f}, {grid['bounds']['kx_max']:.1f}] rad/m")
        print(f"  Ky range: [{grid['bounds']['ky_min']:.1f}, {grid['bounds']['ky_max']:.1f}] rad/m")

    # ── Etapa 4: Interpolation (polar -> Cartesian) ───────────────────────
    E_cart_bilinear = interpolate_polar_to_cartesian(
        E, kspace["k"], kspace["theta"], grid, method="linear"
    )
    E_cart_nearest = interpolate_polar_to_cartesian(
        E, kspace["k"], kspace["theta"], grid, method="nearest"
    )

    if verbose:
        print(f"\nInterpolation (Etapa 4):")
        print(f"  E_cart (bilinear): {E_cart_bilinear.shape}, "
              f"nonzeros: {np.count_nonzero(np.abs(E_cart_bilinear))}")
        print(f"  E_cart (nearest):  {E_cart_nearest.shape}, "
              f"nonzeros: {np.count_nonzero(np.abs(E_cart_nearest))}")
        print(f"  Zero-filled corners: "
              f"{np.count_nonzero(np.abs(E_cart_bilinear) == 0)} / {E_cart_bilinear.size}")

    # ── Etapa 5: Blind zones + Windowing + Zero Padding ────────────────────
    result = process_blind_zones(
        E_cart_bilinear, grid["kx_axis"], grid["ky_axis"],
        window_type="taylor", apply_zero_pad=True, pad_factor=2,
        nbar=4, sll=-35.0,
    )
    E_final = result["E_final"]
    mask = result["mask"]
    W_2D = result["W_2D"]
    pad_info = result["pad_info"]

    if verbose:
        print(f"\nBlind zones + Windowing (Etapa 5):")
        print(f"  Mask: {mask.shape}, data points: {int(mask.sum())}")
        print(f"  Window: {W_2D.shape}, range: [{W_2D.min():.4f}, {W_2D.max():.4f}]")
        print(f"  E_final (after window+mask): {result['E_windowed'].shape}")
        if pad_info:
            print(f"  After zero padding: {E_final.shape} "
                  f"(from {pad_info['M']}x{pad_info['N']})")

    # ── Standard pipeline (no MOCOMP) — for comparison ────────────────────
    M, N = E.shape
    P, P_abs, range_axis, dr = range_compress(E, f_r, R0=R0)
    I, azimuth_axis, doppler_axis = azimuth_compress(
        P.astype(np.complex128), omega, f_c, pri=pri
    )
    amplitude = compute_amplitude(I, mode="linear")
    amp_db = compute_amplitude(I, mode="db")

    # ── Etapa 6: 2D IFFT + axis scaling ───────────────────────────────────
    I_reformat, x_axis, y_axis, x_span, y_span = form_image_2d(
        E_final, grid["dkx"], grid["dky"]
    )
    amp_reformat = compute_amplitude_db(I_reformat)
    entropy_naive = compute_image_entropy(I)
    entropy_reformat = compute_image_entropy(I_reformat)

    # Nearest neighbor: same pipeline but nearest interpolation
    result_nearest = process_blind_zones(
        E_cart_nearest, grid["kx_axis"], grid["ky_axis"],
        window_type="taylor", apply_zero_pad=True, pad_factor=2, nbar=4, sll=-35.0,
    )
    I_nearest, _, _, _, _ = form_image_2d(
        result_nearest["E_final"], grid["dkx"], grid["dky"]
    )
    amp_nearest = compute_amplitude_db(I_nearest)

    if verbose:
        print(f"\nISAR (standard): {I.shape}, peak={amplitude.max():.1f}")
        print(f"ISAR (reformat): {I_reformat.shape}, span: {x_span:.1f} x {y_span:.1f} m")
        print(f"Entropy (naive):    {entropy_naive:.2f} bits")
        print(f"Entropy (reformat): {entropy_reformat:.2f} bits")

    # ── Plotting ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    # Row 1: K-space analysis
    # 1. K-space original (polar)
    ax1 = axes[0, 0]
    plot_kspace(Kx, Ky, E, ax=ax1,
                title=f"K-space: polar sector (dTheta={delta_theta_deg:.1f} deg)")

    # 2. Grid comparison (Etapa 3)
    ax2 = axes[0, 1]
    plot_grid_comparison(Kx, Ky, grid, ax=ax2)

    # 3. Data mask (Etapa 5.2)
    ax3 = axes[0, 2]
    kx_axis = grid["kx_axis"]
    ky_axis = grid["ky_axis"]
    extent_mask = [ky_axis[0], ky_axis[-1], kx_axis[-1], kx_axis[0]]
    ax3.imshow(mask, aspect="auto", cmap="gray", extent=extent_mask)
    ax3.set_xlabel("Ky (rad/m)")
    ax3.set_ylabel("Kx (rad/m)")
    ax3.set_title(f"Data mask (Etapa 5.2): {int(mask.sum())} points")

    # Row 2: Processing pipeline
    # 4. Taylor window 2D (Etapa 5.3)
    ax4 = axes[1, 0]
    im4 = ax4.imshow(W_2D, aspect="auto", cmap="viridis", extent=extent_mask)
    ax4.set_xlabel("Ky (rad/m)")
    ax4.set_ylabel("Kx (rad/m)")
    ax4.set_title("2D Taylor window (Etapa 5.3)")
    plt.colorbar(im4, ax=ax4, label="Window weight")

    # 5. E after windowing + mask
    ax5 = axes[1, 1]
    E_before_pad = result["E_windowed"] * mask
    amp5 = np.abs(E_before_pad)
    im5 = ax5.imshow(amp5, aspect="auto", cmap="plasma", extent=extent_mask)
    ax5.set_xlabel("Ky (rad/m)")
    ax5.set_ylabel("Kx (rad/m)")
    ax5.set_title("E after window + mask (Etapa 5.3)")
    plt.colorbar(im5, ax=ax5, label="|E|")

    # 6. Final k-space after zero padding (Etapa 5.4)
    ax6 = axes[1, 2]
    if pad_info:
        kx_padded = np.linspace(grid["bounds"]["kx_min"], grid["bounds"]["kx_max"], E_final.shape[0])
        ky_padded = np.linspace(grid["bounds"]["ky_min"], grid["bounds"]["ky_max"], E_final.shape[1])
        extent_padded = [ky_padded[0], ky_padded[-1], kx_padded[-1], kx_padded[0]]
    else:
        extent_padded = extent_mask
    amp6 = np.abs(E_final)
    im6 = ax6.imshow(amp6, aspect="auto", cmap="plasma", extent=extent_padded)
    ax6.set_xlabel("Ky (rad/m)")
    ax6.set_ylabel("Kx (rad/m)")
    ax6.set_title(f"Final k-space (Etapa 5.4): {E_final.shape}")
    plt.colorbar(im6, ax=ax6, label="|E_final|")

    # Row 3: Three-way comparison (Plan.md Step 7.2)
    # 7. (A) No reformat - arc smearing
    ax7 = axes[2, 0]
    extent_isar = [azimuth_axis[0], azimuth_axis[-1], range_axis[-1], range_axis[0]]
    im7 = ax7.imshow(amp_db, aspect="auto", cmap="jet", extent=extent_isar, vmin=-40, vmax=0)
    ax7.set_xlabel("Cross-Range (m)")
    ax7.set_ylabel("Range (m)")
    ax7.set_title(f"(A) No reformat - 2D FFT\nentropy={entropy_naive:.1f} bits")
    plt.colorbar(im7, ax=ax7, label="dB")

    # 8. (B) Nearest Neighbor - aliasing artifacts
    ax8 = axes[2, 1]
    extent_reformat = [y_axis[0], y_axis[-1], x_axis[-1], x_axis[0]]
    im8 = ax8.imshow(amp_nearest, aspect="auto", cmap="jet", extent=extent_reformat, vmin=-40, vmax=0)
    ax8.set_xlabel("Cross-Range (m)")
    ax8.set_ylabel("Range (m)")
    ax8.set_title("(B) Nearest Neighbor\naliasing artifacts")
    plt.colorbar(im8, ax=ax8, label="dB")

    # 9. (C) Bilinear + Taylor - focused
    ax9 = axes[2, 2]
    im9 = ax9.imshow(amp_reformat, aspect="auto", cmap="jet", extent=extent_reformat, vmin=-40, vmax=0)
    ax9.set_xlabel("Cross-Range (m)")
    ax9.set_ylabel("Range (m)")
    ax9.set_title(f"(C) Bilinear + Taylor window\nentropy={entropy_reformat:.1f} bits")
    plt.colorbar(im9, ax=ax9, label="dB")

    fig.suptitle(
        f"Polar Reformatting: dTheta={delta_theta_deg:.1f} deg, omega={omega:.3f} rad/s, "
        f"T_obs={T_obs * 1e3:.0f} ms | "
        f"Entropy: {entropy_naive:.1f} -> {entropy_reformat:.1f} bits",
        fontsize=13,
    )
    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(out_dir, "demo_large_angle.png")
    plt.savefig(fig_path, dpi=150)
    if sys.flags.interactive:
        plt.show()
    else:
        plt.close(fig)
        print(f"\nSaved: {fig_path}")

    return E, f_r, I, amplitude, range_axis, azimuth_axis


if __name__ == "__main__":
    run(verbose=True)
