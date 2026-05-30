"""Демо 3: Полный конвейер с MOCOMP (Full Pipeline).

Сценарий: V!=0, omega!=0. MOCOMP включен.
Показывает всю магию ISAR:
  - Косые линии -> Выравнивание -> Фокусировка
  - От размытого пятна к чётким точкам цели

Запуск:
    python -m examples.demo_full_pipeline_mocomp
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
from processing.azimuth_compress import azimuth_compress, compute_amplitude

SPEED_OF_LIGHT = 3e8


def run(f_c=8e9, spectr_w=0.5e9, ang=9.0, R0=500000.0,
        V=50.0, alpha=0.3, omega=None,
        satellite="ICEsat2", verbose=True):
    if omega is None:
        omega = math.radians(ang) / 16.0

    radar = Radar(c=SPEED_OF_LIGHT, f_c=f_c, Xmax=20.0, Ymax=20.0,
                  spectr_w=spectr_w, ph_c=0.0, ang=ang, nifft_size=1024)

    filename = "matrices/matrices_%s_high.pkl" % satellite
    target = Target(filename, pri=1.0, num_pulses=16, R0=R0, V=V, alpha=alpha, omega=omega)

    E, f_r = generate_raw_matrix(radar, target)
    P, P_abs, range_axis, dr = range_compress(E, f_r, R0=R0)

    I_raw, az_axis, _ = azimuth_compress(P.astype(np.complex128), omega, f_c, pri=1.0)
    amp_raw = compute_amplitude(I_raw)

    P_comp, shifts, ref_bin, phase_errors = mocomp(P, range_axis)
    I_moc, _, _ = azimuth_compress(P_comp, omega, f_c, pri=1.0)
    amp_moc = compute_amplitude(I_moc)

    def entropy(x):
        p = x / x.sum()
        return -np.sum(p * np.log(p + 1e-30))

    e_raw = entropy(amp_raw)
    e_moc = entropy(amp_moc)

    peak_bins_before = np.argmax(P_abs, axis=0)
    peak_bins_after = np.argmax(np.abs(P_comp), axis=0)
    drift_before = int(peak_bins_before[-1]) - int(peak_bins_before[0])
    drift_after = int(peak_bins_after[-1]) - int(peak_bins_after[0])

    if verbose:
        print("=== Full Pipeline with MOCOMP (V=%.1f, alpha=%.2f, w!=0) ===" % (V, alpha))
        print("  Before MOCOMP: drift=%d bins, entropy=%.2f, peak=%.0f" % (drift_before, e_raw, amp_raw.max()))
        print("  After MOCOMP:  drift=%d bins, entropy=%.2f, peak=%.0f" % (drift_after, e_moc, amp_moc.max()))
        print("  Peak improvement: %.1fx" % (amp_moc.max() / amp_raw.max()))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    extent_range = [0, P_abs.shape[1] - 1, range_axis[-1], range_axis[0]]
    axes[0, 0].imshow(P_abs, aspect="auto", cmap="plasma", extent=extent_range)
    axes[0, 0].set_title("Before MOCOMP\n(drift=%d bins)" % drift_before)
    axes[0, 0].set_xlabel("Pulse #")
    axes[0, 0].set_ylabel("Range (m)")

    P_comp_abs = np.abs(P_comp)
    axes[0, 1].imshow(P_comp_abs, aspect="auto", cmap="plasma", extent=extent_range)
    axes[0, 1].set_title("After MOCOMP\n(drift=%d bins)" % drift_after)
    axes[0, 1].set_xlabel("Pulse #")

    axes[0, 2].plot(shifts, "o-", markersize=4)
    axes[0, 2].set_title("Applied Shifts")
    axes[0, 2].set_xlabel("Pulse #")
    axes[0, 2].set_ylabel("Shift (bins)")
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

    fig.suptitle("Demo 3: Full Pipeline with MOCOMP (V=%.1f, alpha=%.2f, w!=0)" % (V, alpha), fontsize=13)
    plt.tight_layout()
    plt.savefig("examples/demo_full_pipeline_mocomp.png", dpi=150)
    if sys.flags.interactive:
        plt.show()
    else:
        plt.close(fig)
        print("Saved: examples/demo_full_pipeline_mocomp.png")


if __name__ == "__main__":
    run(verbose=True)
