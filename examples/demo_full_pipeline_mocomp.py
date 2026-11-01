"""Демо 3: Полный конвейер с MOCOMP — физически валидный режим.

Сценарий идентичен Demo 2 (N=256, PRI=5e-4, Δθ=3°, V=10 м/с, survey 40),
MOCOMP включен: честное сравнение «до/после»:
  - косые линии -> выравнивание огибающей + PPP-фазовая коррекция -> фокус;
  - пик ×2.45, ход 1.28 м (4.3 бина) полностью компенсируется.

Примечание (План фиксов §8.3): на вращающейся цели выравнивание огибающей
включает вращательную составляющую ярких элементов — применённые сдвиги
могут превышать аналитический ход трансляции; суб-бин точность ENVELOPE —
на трансляционно-доминированном эхе.
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
from simulation.applicability import check_applicability, format_report
from processing.range_compress import range_compress
from processing.mocomp import mocomp
from processing.azimuth_compress import azimuth_compress, compute_amplitude

SPEED_OF_LIGHT = 3e8

N_PULSES = 256
PRI = 5e-4
DELTA_THETA_DEG = 3.0
V_SC = 10.0
ALPHA_DEG = 0.3
SURVEY = 40.0
R0 = 500000.0


def run(f_c=8e9, spectr_w=0.5e9, ang=DELTA_THETA_DEG, R0=R0,
        V=V_SC, alpha=ALPHA_DEG, omega=None,
        satellite="ICEsat2", verbose=True):
    omega = (omega if omega is not None
             else math.radians(DELTA_THETA_DEG) / (N_PULSES * PRI))
    pri = PRI

    radar = Radar(c=SPEED_OF_LIGHT, f_c=f_c, Xmax=SURVEY, Ymax=SURVEY,
                  spectr_w=spectr_w, ph_c=0.0, ang=9.0, nifft_size=256)

    filename = "matrices/matrices_%s_high.pkl" % satellite
    target = Target(filename, pri=pri, num_pulses=N_PULSES, R0=R0,
                    V=V, alpha=alpha, omega=omega)

    if verbose:
        print("=== Full Pipeline with MOCOMP (V=%.1f, alpha=%.2f°, Δθ=%.1f°) ==="
              % (V, alpha, math.degrees(omega * N_PULSES * pri)))
        print(format_report(check_applicability(radar, target)))

    E, f_r = generate_raw_matrix(radar, target)
    P, P_abs, range_axis, dr = range_compress(E, f_r, R0=R0)

    I_raw, az_axis, _ = azimuth_compress(P.astype(np.complex128), omega, f_c, pri=pri)
    amp_raw = compute_amplitude(I_raw)

    P_comp, shifts, ref_bin, phase_errors = mocomp(P, range_axis)
    I_moc, _, _ = azimuth_compress(P_comp, omega, f_c, pri=pri)
    amp_moc = compute_amplitude(I_moc)

    def entropy(x):
        p = x / x.sum()
        return -np.sum(p * np.log(p + 1e-30))

    e_raw = entropy(amp_raw)
    e_moc = entropy(amp_moc)

    # Честные метрики (План фиксов §8.3): аналитический ход + применённые
    # сдвиги MOCOMP в метрах вместо «drift = argmax-разность» (старая метрика
    # измеряла заворот, а не дрейф)
    walk_analytic = V * math.cos(math.radians(alpha)) * N_PULSES * pri   # м
    shifts_m = np.asarray(shifts) * dr                                   # метры

    if verbose:
        print("  Range walk (аналитика трансляции): %.2f м = %.1f бина"
              % (walk_analytic, walk_analytic / dr))
        print("  Применённые сдвиги MOCOMP: %+.2f..%+.2f м (включают "
              "вращательную составляющую ЯЛЭ)" % (shifts_m.min(), shifts_m.max()))
        print("  Peak: raw=%.0f, mocomp=%.0f (improvement %.2fx)"
              % (amp_raw.max(), amp_moc.max(), amp_moc.max() / amp_raw.max()))
        print("  Entropy: raw=%.2f, mocomp=%.2f" % (e_raw, e_moc))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    extent_range = [0, P_abs.shape[1] - 1, range_axis[-1], range_axis[0]]
    axes[0, 0].imshow(P_abs, aspect="auto", cmap="plasma", extent=extent_range)
    axes[0, 0].set_title("Before MOCOMP (walk=%.1f bins)" % (walk_analytic / dr))
    axes[0, 0].set_xlabel("Pulse #")
    axes[0, 0].set_ylabel("Range (m)")

    P_comp_abs = np.abs(P_comp)
    axes[0, 1].imshow(P_comp_abs, aspect="auto", cmap="plasma", extent=extent_range)
    axes[0, 1].set_title("After MOCOMP (walk compensated)")
    axes[0, 1].set_xlabel("Pulse #")

    axes[0, 2].plot(shifts_m, "o-", markersize=4)
    axes[0, 2].set_title("Applied Shifts")
    axes[0, 2].set_xlabel("Pulse #")
    axes[0, 2].set_ylabel("Shift (m)")
    axes[0, 2].grid(True, alpha=0.3)

    extent_isar = [az_axis[0], az_axis[-1], range_axis[-1], range_axis[0]]
    im0 = axes[1, 0].imshow(amp_raw, aspect="auto", cmap="jet", extent=extent_isar)
    axes[1, 0].set_title("Without MOCOMP\n(peak=%.0f, ent=%.2f)" % (amp_raw.max(), e_raw))
    axes[1, 0].set_xlabel("Cross-Range (m)")
    axes[1, 0].set_ylabel("Range (m)")
    plt.colorbar(im0, ax=axes[1, 0], label="Amplitude")

    im1 = axes[1, 1].imshow(amp_moc, aspect="auto", cmap="jet", extent=extent_isar)
    axes[1, 1].set_title("With MOCOMP\n(peak=%.0f, ent=%.2f)" % (amp_moc.max(), e_moc))
    axes[1, 1].set_xlabel("Cross-Range (m)")
    plt.colorbar(im1, ax=axes[1, 1], label="Amplitude")

    amp_moc_db = compute_amplitude(I_moc, mode="db")
    im2 = axes[1, 2].imshow(amp_moc_db, aspect="auto", cmap="jet", extent=extent_isar, vmin=-30, vmax=0)
    axes[1, 2].set_title("With MOCOMP (dB)")
    axes[1, 2].set_xlabel("Cross-Range (m)")
    plt.colorbar(im2, ax=axes[1, 2], label="Amplitude (dB)")

    fig.suptitle("Demo 3: Full Pipeline with MOCOMP (V=%.1f m/s, Δθ=%.1f°)"
                 % (V, math.degrees(omega * N_PULSES * pri)), fontsize=13)
    plt.tight_layout()
    plt.savefig("examples/demo_full_pipeline_mocomp.png", dpi=150)
    if sys.flags.interactive:
        plt.show()
    else:
        plt.close(fig)
        print("Saved: examples/demo_full_pipeline_mocomp.png")


if __name__ == "__main__":
    run(verbose=True)
