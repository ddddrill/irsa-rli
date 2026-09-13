"""Демо: keystone-перешкалирование (задача R6, §9) — RD «до/после».

Мотив рис. 5.25 книги C&M (Mig-25 before/after): вращающийся точечный
рассеиватель со стенда §9.5 размазан по дальности без keystone и фокусируется
с ним.

Стенд: f_c = 3 ГГц, B = 0.5 ГГц (dr ≈ 0.3 м), M = N = 256, PRI = 0.05 с
(T = 12.8 с), ω = 0.02 рад/с, точечный рассеиватель (x_p = 5 м LOS,
y_p = 10 м кросс-рейндж). Миграция дальности y_p·ω·T = 2.56 м ≈ 8.6·dr.

Запуск:
    python -m examples.demo_keystone
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from processing.keystone import keystone_transform
from processing.range_compress import range_compress
from processing.azimuth_compress import azimuth_compress, compute_amplitude
from processing.polar_reformat import compute_image_entropy

SPEED_OF_LIGHT = 3e8

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "demo_keystone.png",
)

# Стенд §9.5
F_C = 3e9
B = 0.5e9
PRI = 0.05
OMEGA = 0.02
X_P = 5.0
Y_P = 10.0
M_PULSES = 256
N_PULSES = 256


def _stand_rows():
    """Сырые строки E точечного рассеивателя стенда §9.5."""
    f_r = np.linspace(F_C - B / 2.0, F_C + B / 2.0, M_PULSES)
    t = np.arange(N_PULSES) * PRI
    R = X_P * np.cos(OMEGA * t) - Y_P * np.sin(OMEGA * t)
    E = np.exp(-1j * 4.0 * np.pi * f_r[:, None] / SPEED_OF_LIGHT * R[None, :])
    return E, f_r


def _rd(E, f_r, keystone=False):
    """Пайплайн §9.4 п.2: E → (keystone) → range_compress → azimuth_compress."""
    if keystone:
        E = keystone_transform(E, f_r, F_C)
    P, _, range_axis, dr = range_compress(E, f_r, R0=0.0)
    I, azimuth_axis, doppler_axis = azimuth_compress(P, OMEGA, F_C, pri=PRI)
    return I, range_axis, dr, doppler_axis


def _peak(I, range_axis, doppler_axis):
    row, col = np.unravel_index(np.argmax(np.abs(I)), I.shape)
    return float(range_axis[row]), float(doppler_axis[col])


def run(verbose=True, out_path=DEFAULT_OUT):
    """Расчёт «до/после» + 2 панели + сохранение PNG + печать метрик."""
    E, f_r = _stand_rows()
    T = N_PULSES * PRI
    lam = SPEED_OF_LIGHT / F_C

    I_no, range_axis, dr, doppler_axis = _rd(E, f_r, keystone=False)
    I_ks, _, _, _ = _rd(E, f_r, keystone=True)

    H_no = compute_image_entropy(I_no)
    H_ks = compute_image_entropy(I_ks)
    gain = float(np.abs(I_ks).max() / np.abs(I_no).max())
    r_no, d_no = _peak(I_no, range_axis, doppler_axis)
    r_ks, d_ks = _peak(I_ks, range_axis, doppler_axis)
    f_D_true = (2.0 / lam) * Y_P * OMEGA

    if verbose:
        print("=" * 62)
        print("  KEYSTONE-ПЕРЕШКАЛИРОВАНИЕ: RD до/после (стенд §9.5)")
        print("=" * 62)
        print(f"  f_c = {F_C / 1e9:.0f} ГГц (lam = {lam * 1e2:.1f} см), "
              f"B = {B / 1e9:.1f} ГГц")
        print(f"  PRI = {PRI * 1e3:.0f} мс, N = {N_PULSES}, T = {T:.1f} с, "
              f"omega = {OMEGA} рад/с")
        print(f"  Цель: x_p = {X_P:.0f} м (LOS), y_p = {Y_P:.0f} м (кросс-рейндж)")
        print(f"  Миграция дальности: y_p*omega*T = {Y_P * OMEGA * T:.2f} м "
              f"= {Y_P * OMEGA * T / dr:.1f} dr (dr = {dr:.3f} м)")
        print("-" * 62)
        print(f"  Энтропия, бит:           без keystone {H_no:.3f} "
              f"vs с keystone {H_ks:.3f}")
        print(f"  Пик |I|:                 {np.abs(I_no).max():.1f} -> "
              f"{np.abs(I_ks).max():.1f}  (x{gain:.2f})")
        print(f"  Координаты пика:         без ({r_no:+.2f} м, {d_no:+.1f} Гц) "
              f"vs с ({r_ks:+.2f} м, {d_ks:+.1f} Гц)")
        print(f"  Ожидаемый доплер (по модулю): "
              f"f_D = 2/lam*y_p*omega = {f_D_true:+.1f} Гц")

    amp_no = compute_amplitude(I_no, mode="db")
    amp_ks = compute_amplitude(I_ks, mode="db")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    fig.suptitle(
        "Keystone-перешкалирование (§9): вращающийся рассеиватель "
        f"(y_p = {Y_P:g} м, migration = {Y_P * OMEGA * T:.2f} м "
        f"~ {Y_P * OMEGA * T / dr:.0f}·dr)",
        fontsize=12,
    )

    extent = [doppler_axis[0], doppler_axis[-1],
              range_axis[-1], range_axis[0]]

    ax = axes[0]
    im = ax.imshow(amp_no, aspect="auto", cmap="jet", extent=extent,
                   vmin=-40, vmax=0)
    ax.set_title(f"RD БЕЗ keystone — миграция (~{Y_P * OMEGA * T / dr:.0f} dr)\n"
                 f"энергия расплыта, H = {H_no:.2f} бит")
    ax.set_xlabel(f"Доплер, Гц\nпик: ({r_no:+.2f} м, {d_no:+.1f} Гц)")
    ax.set_ylabel("Дальность, м")
    plt.colorbar(im, ax=ax, label="дБ")

    ax = axes[1]
    im = ax.imshow(amp_ks, aspect="auto", cmap="jet", extent=extent,
                   vmin=-40, vmax=0)
    ax.set_title(f"RD С keystone — фокус (x{gain:.2f})\n"
                 f"H = {H_ks:.2f} бит")
    ax.set_xlabel(f"Доплер, Гц\nпик: ({r_ks:+.2f} м, {d_ks:+.1f} Гц)")
    plt.colorbar(im, ax=ax, label="дБ")

    plt.tight_layout()
    fig_path = os.path.abspath(out_path)
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"\nСохранено: {fig_path}")
    return I_no, I_ks


def main():
    run()


if __name__ == "__main__":
    main()