"""Демо: Влияние эффектов реализма на качество РЛИ.

Сравнение 6 сценариев на 6 панелях:
  1. Идеальный (все эффекты выключены)
  2. Комплексные амплитуды (случайные фазы рассеивателей)
  3. Анизотропия + затенение
  4. Антенна + фазовый шум
  5. Нелинейное движение (ускорение + микровибрация)
  6. Всё вместе

Запуск:
    python -m examples.demo_realism_effects
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import matplotlib.pyplot as plt

from models.radar import Radar
from models.target import Target
from simulation.raw_generator import SimEffects, generate_raw_matrix
from processing.range_compress import range_compress
from processing.azimuth_compress import azimuth_compress, compute_amplitude

SPEED_OF_LIGHT = 3e8
SEED = 42


def make_effects(complex_sigma=False, anisotropy=False, shadowing=False,
                 antenna=False, phase_noise=False, accel=0.0,
                 vib_amplitude=0.0):
    return SimEffects(
        seed=SEED,
        use_complex_sigma=complex_sigma,
        use_anisotropy=anisotropy,
        anisotropy_width_deg=10.0,
        use_shadowing=shadowing,
        shadow_radius_m=0.4,
        use_antenna_pattern=antenna,
        pointing_error_deg=2.0,
        antenna_drift_deg=1.0,
        use_phase_noise=phase_noise,
        phase_noise_std_deg=10.0,
    )


def run_pipeline(radar, target, effects=None):
    E, f_r = generate_raw_matrix(radar, target, effects=effects)
    P, P_abs, range_axis, dr = range_compress(E, f_r, R0=target.R0)
    I, azimuth_axis, doppler_axis = azimuth_compress(
        P.astype(np.complex128), target.omega, radar.f_c, pri=target.pri,
    )
    return compute_amplitude(I, mode="db"), azimuth_axis, range_axis


def run(f_c=8e9, spectr_w=0.5e9, ang=9.0, R0=500000.0,
        satellite="ICEsat2", num_pulses=256, verbose=True):
    omega = math.radians(ang) / 16.0

    radar = Radar(c=SPEED_OF_LIGHT, f_c=f_c, Xmax=20.0, Ymax=20.0,
                  spectr_w=spectr_w, ph_c=0.0, ang=ang, nifft_size=1024)

    filename = "matrices/matrices_%s_high.pkl" % satellite

    scenarios = [
        ("1. Идеальный", make_effects(), None),
        ("2. Комплексные фазы", make_effects(complex_sigma=True), None),
        ("3. Анизотропия + тень",
         make_effects(anisotropy=True, shadowing=True), None),
        ("4. Антенна + фаз. шум",
         make_effects(antenna=True, phase_noise=True), None),
        ("5. Нелинейное движение",
         make_effects(), {"accel": 0.5, "vib_amplitude": 0.002}),
        ("6. Всё вместе",
         make_effects(complex_sigma=True, anisotropy=True, shadowing=True,
                      antenna=True, phase_noise=True),
         {"accel": 0.5, "vib_amplitude": 0.002}),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (title, effects, kin_kwargs) in enumerate(scenarios):
        target_kw = dict(
            file=filename, pri=1.0, num_pulses=num_pulses, R0=R0,
            V=0.0, alpha=0.0, omega=omega,
        )
        if kin_kwargs:
            target_kw.update(kin_kwargs)

        target = Target(**target_kw)

        amp_db, azimuth_axis, range_axis = run_pipeline(radar, target, effects)

        extent = [azimuth_axis[0], azimuth_axis[-1],
                  range_axis[-1], range_axis[0]]
        im = axes[idx].imshow(amp_db, aspect="auto", cmap="jet",
                              extent=extent, vmin=-30, vmax=0)
        axes[idx].set_title(title, fontsize=12)
        axes[idx].set_xlabel("Cross-Range (m)")
        axes[idx].set_ylabel("Range (m)")
        plt.colorbar(im, ax=axes[idx], label="dB", shrink=0.8)

        if verbose:
            print("  %s — peak: %.1f dB" % (title, amp_db.max()))

    fig.suptitle("Эффекты реализма: сравнение РЛИ (v2.4.0)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("examples/demo_realism_effects.png", dpi=150)
    if sys.flags.interactive:
        plt.show()
    else:
        plt.close(fig)
        print("Saved: examples/demo_realism_effects.png")


if __name__ == "__main__":
    run(verbose=True)
