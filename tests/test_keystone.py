"""Тесты keystone-перешкалирования (`processing.keystone`, задача R6, §9).

Проверяется контракт плана §9.6 (без 9.6.4 — интеграционная) и ключевая
страховка направления индексации (§9.2.4): выходной отсчёт j берётся из
исходной строки по индексу j·(f_c/f_r[m]), а НЕ j·(f_r[m]/f_c).

Стенд §9.5 (тесты 2–3): f_c = 3e9 Гц, B = 0.5e9 Гц, M = N = 256,
PRI = 0.05 с (T = 12.8 с), ω = 0.02 рад/с, точечный рассеиватель
(x_p = 5 м, y_p = 10 м). Миграция дальности y_p·ω·T = 2.56 м ≈ 8.6·dr
(dr ≈ 0.3 м) — энергия без keystone расплывается в ~9 ячеек.

Тест 1 использует точечный тон с 12 ПОЛНЫМИ периодами на окне 64 отсчёта
(f0 = 12/64 = 0.1875·fs) — «неутекающий» тон: дискретное преобразование
не имеет спектральной утечки, и ошибка ресемплинга изолирует качество
самого интерполятора (с нулевой приставкой она была бы замаскирована
модельным полом ~4e-2 — см. report R6-core).
"""
import os

import numpy as np
import pytest
from scipy.fft import fft, ifftshift

from processing.keystone import keystone_transform
from processing.range_compress import range_compress
from processing.azimuth_compress import azimuth_compress
from processing.polar_reformat import compute_image_entropy
from processing.mocomp import mocomp

SPEED_OF_LIGHT = 3e8

F_C = 3e9
B = 0.5e9
PRI = 0.05
OMEGA = 0.02
X_P = 5.0
Y_P = 10.0


def _migrating_rows(N=256, M=256, V=0.0, omega=OMEGA, x_p=X_P, y_p=Y_P):
    """Сырые строки E (M×N) точечного рассеивателя (§9.5), R0 = 0.

    R(t) = x_p + V·t − y_p·sin(ω·t): чистое вращение (keystone-часть) +
    необязательный поступательный дрейф (тест 4, V ≠ 0). LOS-координата
    цели x_p = 5 м, кросс-рейндж y_p = 10 м.
    """
    f_r = np.linspace(F_C - B / 2.0, F_C + B / 2.0, M)
    t = np.arange(N) * PRI
    R = x_p + V * t - y_p * np.sin(omega * t)
    return np.exp(-1j * 4.0 * np.pi * f_r[:, None] / SPEED_OF_LIGHT * R[None, :]), f_r


def _rd_image(E, f_r, keystone=True):
    """Пайплайн §9.4 п.2: E → (keystone) → range_compress → azimuth_compress."""
    if keystone:
        E = keystone_transform(E, f_r, F_C)
    P, _, range_axis, dr = range_compress(E, f_r, R0=0.0)
    I, _, _ = azimuth_compress(P, OMEGA, F_C, pri=PRI)
    return I, range_axis, dr


def test_row_resampler_accuracy():
    """§9.6.1: точность ресемплинга строки + страховка направления (§9.2.4).

    x(t) = exp(j2π·f0·t), f0 = 0.1875·fs (12 полных периодов на N=64), —
    ресемплинг на дробных индексах j·(f_c/f_r) должен давать x без искажений
    максимум |y − x(idx)| < 1e-3 (sinc) / < 1e-2 (cubic). Индексы про
    НЕПРАВИЛЬНОЕ направление (j·(f_r/f_c)) не сходятся (ошибка ~1) — этим
    зафиксирован знак перешкалирования.
    """
    N = 64
    f0 = 12.0 / N
    pulses = np.arange(N)
    x = np.exp(1j * 2.0 * np.pi * f0 * pulses)
    E = x[np.newaxis, :]
    f_r = np.array([1.0])

    for ratio in (1.03, 1.0 / 1.03):
        q = pulses * ratio
        interior = (np.floor(q) >= 2) & (np.floor(q) <= N - 4)
        ans = np.exp(1j * 2.0 * np.pi * f0 * q)

        for method, tol in (("sinc", 1e-3), ("cubic", 1e-2)):
            y = keystone_transform(E, f_r, f_c=ratio, method=method,
                                   pad=0)[0]
            err_correct = np.abs(y[interior] - ans[interior]).max()
            assert err_correct < tol, \
                f"{method}: correct-direction error {err_correct:.2e} >= {tol:.0e}"

            # инверсия направления: индексы j·(f_r/f_c) — не сходится, ошибка ~1
            y_inv = keystone_transform(E, f_r, f_c=1.0 / ratio,
                                       method=method, pad=0)[0]
            err_inv = np.abs(y_inv[interior] - ans[interior]).max()
            assert err_inv > 0.5 and err_inv > 10.0 * err_correct, \
                f"{method}: sign-guard broken, inverted error {err_inv:.3f}"


def test_keystone_focuses_migrating_scatterer():
    """§9.6.2: keystone фокусирует мигрирующий рассеиватель стенда §9.5.

    Без keystone энергия одной точки расплывается на ~9 ячеек; с keystone
    пик |I| >= 3× пика без и энтропия(I_ks) < 0.9·энтропии(I_no).
    """
    E, f_r = _migrating_rows()
    I_no, _, _ = _rd_image(E, f_r, keystone=False)
    I_ks, _, _ = _rd_image(E, f_r, keystone=True)

    peak_no = float(np.abs(I_no).max())
    peak_ratio = float(np.abs(I_ks).max() / peak_no)
    assert peak_ratio >= 3.0, f"peak ratio {peak_ratio:.2f} < 3"

    H_no = compute_image_entropy(I_no)
    H_ks = compute_image_entropy(I_ks)
    assert H_ks < 0.9 * H_no, \
        f"entropy not improved: H_ks={H_ks:.3f} vs H_no={H_no:.3f}"

    # fallback-метод тоже фокусирует пик (страховка pipeline-использования)
    I_cubic, _, _ = _rd_image(keystone_transform(E, f_r, F_C, method="cubic"),
                              f_r, keystone=False)
    assert float(np.abs(I_cubic).max() / peak_no) >= 3.0


def test_keystone_range_position():
    """§9.6.3: позиция пика по дальности ±1 dr от истинной x_p (LOS).

    Координаты дешифрованной сетки range_compress (R0=0) отсчитываются
    от ЦМ: истинная дальность точки x_p = 5 м.
    """
    E, f_r = _migrating_rows()
    I_ks, range_axis, dr = _rd_image(E, f_r, keystone=True)

    row_peak = int(np.unravel_index(np.argmax(np.abs(I_ks)), I_ks.shape)[0])
    r_peak = float(range_axis[row_peak])
    assert abs(r_peak - X_P) <= dr, \
        f"range peak {r_peak:.3f} м, truth {X_P} м, dr {dr:.3f} м"


def test_translational_case_documented():
    """§9.6.5: поступательное движение (V≠0) без MOCOMP keystone НЕ фокусирует.

    Keystone компенсирует только вращательную (линейную по t) миграцию.
    Сценарий: V_x = 1 м/с + вращение. Сравниваются два пайплайна:
      (a) E → keystone → range → azimuth      (keystone в одиночку);
      (b) E → range → mocomp → fft(ifftshift(P_comp)) → keystone →
          range → azimuth                     (гибрид MOCOMP → keystone).
    Гибрид существенно лучше: пик (b) > 3× пика (a) — документировано,
    что чисто keystone для V≠0 недостаточен.
    """
    E, f_r = _migrating_rows(V=1.0)

    I_alone, _, _ = _rd_image(E, f_r, keystone=True)

    P, _, range_axis, _ = range_compress(E, f_r, R0=0.0)
    P_comp, _, _, _ = mocomp(P, range_axis)
    E_comp = fft(ifftshift(P_comp, axes=0), axis=0)
    E_hybrid = keystone_transform(E_comp, f_r, F_C)
    P2, _, _, _ = range_compress(E_hybrid, f_r, R0=0.0)
    I_hybrid, _, _ = azimuth_compress(P2, OMEGA, F_C, pri=PRI)

    peak_alone = float(np.abs(I_alone).max())
    peak_hybrid = float(np.abs(I_hybrid).max())
    assert peak_hybrid > 3.0 * peak_alone, \
        f"hybrid {peak_hybrid:.2f} not > 3x alone {peak_alone:.2f}"
    assert compute_image_entropy(I_hybrid) < compute_image_entropy(I_alone)


def test_disabled_is_bitwise_identical():
    """§9.6.4: use_keystone=False — битовая идентичность прежнему конвейеру.

    Интеграционный инвариант: DataProcessor с флагом use_keystone=False
    (как и конструкция без этого параметра — дефолт False) выдаёт битово
    то же комплексное изображение I и амплитуду, что и «старый» конвейер.
    Сравнение — np.array_equal на комплексном I (не только амплитуда), по
    мотиву bitwise-тестов test_cross_range_scaling/test_tf_gui.
    """
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from processing.data_processor import DataProcessor

    common = dict(
        filename="matrices/matrices_solarB_high.pkl",
        method="стандартный",
        f_c=8.0, Xmax=20.0, Ymax=20.0, spectr_w=0.5,
        ang=9.0, range_m=500000.0, nifft_size=256,
        V=0.0, alpha=0.0, omega=0.004, pri=0.01,
        num_pulses=128, use_mocomp=True, window="none",
    )
    proc_def = DataProcessor(**common)
    proc_off = DataProcessor(**common, use_keystone=False)

    E, f_r, target = proc_def.generate_raw_data()
    I1, amp1, ra1, az1, _ = proc_def._compute_isar_standard(E, f_r, target, "dB")
    I2, amp2, ra2, az2, _ = proc_off._compute_isar_standard(E, f_r, target, "dB")

    assert np.array_equal(I1, I2), "комплексное I различается при use_keystone=False"
    assert np.array_equal(amp1, amp2), "амплитуда различается при use_keystone=False"
