"""Визуальная валидация Этапа 5: компенсация поступательного движения (MOCOMP).

Сравнивает три картинки:
  1. Без MOCOMP  — размытое пятно (Motion Blur).
  2. После MOCOMP — чёткие точки цели.
  3. БПФ по азимуту после MOCOMP — финальное РЛИ.

Запуск:
    python -m simulation.validate_mocomp
"""
import math
import numpy as np
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

from models.radar import Radar
from models.target import Target
from simulation.raw_generator import generate_raw_matrix
from processing.range_compress import range_compress
from processing.mocomp import mocomp

SPEED_OF_LIGHT = 3e8


def azimuth_compress(P_comp):
    """БПФ по оси медленного времени (по столбцам, axis=1) -> РЛИ."""
    isar = np.abs(fftshift(fft(P_comp, axis=1), axes=1))
    return isar


def run_validation():
    f_c = 8e9
    spectr_w = 0.5e9
    ang = 9.0
    R0 = 500000.0
    omega = math.radians(ang) / 16.0

    radar = Radar(
        c=SPEED_OF_LIGHT, f_c=f_c, Xmax=20.0, Ymax=20.0,
        spectr_w=spectr_w, ph_c=0.0, ang=ang, nifft_size=1024,
    )

    scenarios = [
        {"V": 100.0, "alpha": 0.0, "omega": 0.0,
         "label": "V=100, w=0 (pure translation)"},
        {"V": 50.0, "alpha": 0.3, "omega": omega,
         "label": "V=50, alpha=0.3, w!=0 (combined)"},
        {"V": 0.0, "alpha": 0.0, "omega": omega,
         "label": "V=0, w!=0 (pure rotation)"},
    ]

    for sc in scenarios:
        target = Target(
            "matrices/matrices_ICEsat2_high.pkl",
            pri=1.0, num_pulses=16,
            R0=R0, V=sc["V"], alpha=sc["alpha"], omega=sc["omega"],
        )
        E, f_r = generate_raw_matrix(radar, target)
        P, P_abs, range_axis, dr = range_compress(E, f_r, R0=R0)
        P_comp, shifts, ref_bin, phase_errors = mocomp(P, range_axis)

        isar_before = azimuth_compress(P.astype(np.complex128))
        isar_after = azimuth_compress(P_comp)

        e_before = -np.sum((isar_before/isar_before.sum()) * np.log(isar_before/isar_before.sum() + 1e-30))
        e_after = -np.sum((isar_after/isar_after.sum()) * np.log(isar_after/isar_after.sum() + 1e-30))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].imshow(P_abs, aspect="auto", cmap="plasma",
                       extent=[0, P_abs.shape[1]-1, range_axis[-1], range_axis[0]])
        axes[0].set_title("Before MOCOMP\n(drift=%d bins)" % (int(np.argmax(P_abs, axis=0)[-1]) - int(np.argmax(P_abs, axis=0)[0])))
        axes[0].set_xlabel("Pulse #")
        axes[0].set_ylabel("Range (m)")

        P_comp_abs = np.abs(P_comp)
        axes[1].imshow(P_comp_abs, aspect="auto", cmap="plasma",
                       extent=[0, P_comp_abs.shape[1]-1, range_axis[-1], range_axis[0]])
        axes[1].set_title("After MOCOMP\n(ref_bin=%d, phase_std=%.3f)" % (ref_bin, phase_errors.std()))
        axes[1].set_xlabel("Pulse #")

        axes[2].imshow(isar_after, aspect="auto", cmap="plasma")
        axes[2].set_title("ISAR after MOCOMP\n(peak=%.0f, entropy=%.2f)" % (isar_after.max(), e_after))
        axes[2].set_xlabel("Azimuth bin")
        axes[2].set_ylabel("Range bin")

        fig.suptitle(sc["label"] + "\nentropy: %.2f -> %.2f" % (e_before, e_after), fontsize=13)
        plt.tight_layout()
        plt.savefig("simulation/mocomp_%s.png" % sc["label"].replace(" ", "_").replace(",", ""), dpi=150)
        plt.close(fig)

        print(sc["label"])
        print("  Before: entropy=%.2f, peak=%.0f" % (e_before, isar_before.max()))
        print("  After:  entropy=%.2f, peak=%.0f" % (e_after, isar_after.max()))
        print("  Improvement: entropy %.2f%%, peak %.1fx" % (
            (e_before - e_after) / e_before * 100, isar_after.max() / isar_before.max()))
        print()

    print("All MOCOMP validation tests completed")


if __name__ == "__main__":
    run_validation()
