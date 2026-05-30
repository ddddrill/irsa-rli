"""Финальная валидация Этапа 6: формирование РЛИ (Range-Doppler ISAR).

Сравнивает три картинки:
  1. Без MOCOMP  — размытое пятно (Motion Blur).
  2. С MOCOMP    — чёткие точки цели (Focused).
  3. Идеал       — чистое вращение без трансляции (Reference).

Запуск:
    python -m simulation.validate_isar
"""
import math
import numpy as np
import matplotlib.pyplot as plt

from processing import DataProcessor
from filenames import get_matrix_filename

SPEED_OF_LIGHT = 3e8


def run_validation():
    filename = get_matrix_filename("ICEsat2")
    f_c = 8.0
    spectr_w = 0.5
    ang = 9.0
    range_m = 500000.0
    omega = math.radians(ang) / 16.0

    scenarios = [
        {"V": 100.0, "alpha": 0.0, "omega": omega,
         "label": "V=100, w=0 (pure translation)"},
        {"V": 50.0, "alpha": 0.3, "omega": omega,
         "label": "V=50, alpha=0.3, w!=0 (combined)"},
    ]

    for sc in scenarios:
        print(sc["label"])
        print("=" * 50)

        proc = DataProcessor(
            filename=filename, method="стандартный",
            f_c=f_c, Xmax=20.0, Ymax=20.0, spectr_w=spectr_w,
            ang=ang, range_m=range_m,
            V=sc["V"], alpha=sc["alpha"], omega=sc["omega"],
        )

        I_raw, amp_raw, range_axis, az_axis, target = proc.compute_isar(use_mocomp=False)
        I_moc, amp_moc, _, _, _ = proc.compute_isar(use_mocomp=True)

        proc_ideal = DataProcessor(
            filename=filename, method="стандартный",
            f_c=f_c, Xmax=20.0, Ymax=20.0, spectr_w=spectr_w,
            ang=ang, range_m=range_m,
            V=0.0, alpha=0.0, omega=sc["omega"],
        )
        I_ideal, amp_ideal, _, _, _ = proc_ideal.compute_isar(use_mocomp=False)

        def entropy(x):
            p = x / x.sum()
            return -np.sum(p * np.log(p + 1e-30))

        e_raw = entropy(amp_raw)
        e_moc = entropy(amp_moc)
        e_ideal = entropy(amp_ideal)

        print("  Without MOCOMP: peak=%7.1f  entropy=%.2f" % (amp_raw.max(), e_raw))
        print("  With MOCOMP:    peak=%7.1f  entropy=%.2f" % (amp_moc.max(), e_moc))
        print("  Ideal (no V):   peak=%7.1f  entropy=%.2f" % (amp_ideal.max(), e_ideal))
        print("  Improvement:    peak %.1fx, entropy %.1f%%" % (
            amp_moc.max() / amp_raw.max(), (e_raw - e_moc) / e_raw * 100))
        print()

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].imshow(amp_raw, aspect="auto", cmap="plasma",
                       extent=[az_axis[0], az_axis[-1], range_axis[-1], range_axis[0]])
        axes[0].set_title("Without MOCOMP\n(peak=%.0f, ent=%.2f)" % (amp_raw.max(), e_raw))
        axes[0].set_xlabel("Cross-Range (m)")
        axes[0].set_ylabel("Range (m)")

        axes[1].imshow(amp_moc, aspect="auto", cmap="plasma",
                       extent=[az_axis[0], az_axis[-1], range_axis[-1], range_axis[0]])
        axes[1].set_title("With MOCOMP\n(peak=%.0f, ent=%.2f)" % (amp_moc.max(), e_moc))
        axes[1].set_xlabel("Cross-Range (m)")

        axes[2].imshow(amp_ideal, aspect="auto", cmap="plasma",
                       extent=[az_axis[0], az_axis[-1], range_axis[-1], range_axis[0]])
        axes[2].set_title("Ideal (no translation)\n(peak=%.0f, ent=%.2f)" % (amp_ideal.max(), e_ideal))
        axes[2].set_xlabel("Cross-Range (m)")

        fig.suptitle(sc["label"], fontsize=13)
        plt.tight_layout()
        plt.savefig("simulation/isar_%s.png" % sc["label"].replace(" ", "_").replace(",", ""), dpi=150)
        plt.close(fig)

    print("All ISAR validation tests completed")


if __name__ == "__main__":
    run_validation()
