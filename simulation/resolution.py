"""Инверсия параметризации: разрешения Δr/Δx вместо «обзора» (задача RP1).

Пользователь задаёт ЦЕЛЕВОЕ разрешение системы; всё остальное выводится:

    B   = c/(2·Δr)              — полоса SFCW (радиосетка частот)
    Δθ  = λ/(2·Δx), λ = c/f_c   — полный угол поворота за апертуру
    ω   = Δθ/(N·PRI)            — угловая скорость (N, PRI — пользовательские)
    Xmax = max(40, 2.4·max|δR|) — авто-обзор по точному dR (окно ≥ 1.2×)
    Ymax = max(40, 2.4·(max|y′| + |V·sinα|·T_obs)) — авто-обзор сцены
    Nf  = ⌈2·B·Xmax/c⌉          — число частотных шагов

Эмпирика — по ТОЧНОЙ кинематике `Target.get_ranges_all()` (принцип guard'ов
G1, План фиксов §0.5): без приближённых формул «2·ω·y/λ». Ограничения
знают три неочевидности:

1. C1a не зависит от PRI: условие сворачивается к N·λ ≳ 4·span_y·Δθ — при
   нарушении помогает ТОЛЬКО увеличение N (в warning'е называется N_min,
   вычисленное эмпирическим масштабированием fd_diff ∝ 1/N).
2. Δx → Δθ → fd: мелкое Δx для крупной цели при малом N неосуществимо
   (пример: Δx=0.20 м на сетке demo-1 требует N ≳ 1300).
3. V·T_obs растёт с N·PRI → авто-Xmax и Nf растут с N (оценивается сразу).

Два снэпа ради бит-совместимости с legacy-конвейером:

* B снэпится к сетке 1 МГц (Δr=0.30 м → B ≡ 5e8 Гц, как задание 0.5 ГГц);
* Δθ снэпится к сетке 0.05° (k = round(deg/0.05); шаг = k/20.0 — точное
  деление, поэтому Δθ ≡ radians(0.80)/radians(3.00) бит-равны legacy).
  Без снэпа обратное деление λ/(2·Δx)→Δθ→ω→θ(t) давало бы отличия фазы в
  последних битах и НЕ достигало бы побитового равенства E с прямым
  заданием (тест RP4). Фактическое Δx после снэпа — в da_effective.
"""
import math
from dataclasses import dataclass, field

import numpy as np

SPEED_OF_LIGHT = 3e8

DELTATHETA_SNAP_DEG = 0.05        # каноническая сетка полного угла
NF_CAP = 4096                     # предупреждение о размере матрицы
SURVEY_FLOOR_M = 40.0             # дефолт-пол GUI: авто-обзор не ниже 40 м
SURVEY_MARGIN = 2.4               # окно ≥ 1.2·max|δR|  ⇔  Xmax ≥ 2.4·max|δR|
ROUND_STEP_M = 5.0                # авто-обзор округляется вверх кратно 5 м
B_SNAP_HZ = 1e6                   # сетка снэпа полосы (1 МГц)


@dataclass(frozen=True)
class ReconPlan:
    """Результат деривации «разрешения → параметры радара/кинематики».

    Поля:
        f_c_hz, spectr_w_hz, Xmax_m, Ymax_m, nifft... — аргументы Radar;
        omega_rad_s, num_pulses, pri_s — кинематика Target;
        dr_effective/da_effective — фактические разрешения ПОСЛЕ снэпов;
        dR_max_m, y_extent_m — эмпирика точной кинематике;
        fd_diff_hz, N_min_c1a — диф. доплер и минимальное N (C1a);
        warnings — строки «WARN (RP): …» для progress_message/logging.
    """
    f_c_hz: float
    dr_requested_m: float
    da_requested_m: float
    spectr_w_hz: float
    dr_effective_m: float
    delta_theta_rad: float
    da_effective_m: float
    omega_rad_s: float
    num_pulses: int
    pri_s: float
    Xmax_m: float
    Ymax_m: float
    Nf: int
    dR_max_m: float
    y_extent_m: float
    fd_diff_hz: float
    N_min_c1a: int
    warnings: list = field(default_factory=list)


def target_geometry_from_pkl(filename):
    """Лёгкое извлечение координат рассеивателей без создания Radar.

    Обрывает цикл «survey → Nf → Radar → target»: геометрия нужна ДО
    создания радара. Координаты центрированы вычитанием среднего — та же
    конвенция, что в Target._load_scatterers.
    """
    from models.target import Target

    tgt = Target(filename)
    sc = tgt._load_scatterers()
    return {"x0": np.asarray(sc["x0"], dtype=float),
            "y0": np.asarray(sc["y0"], dtype=float)}


def _ceil_to_step(value, step):
    return math.ceil(value / step) * step


def _survey_floor(dmax, ymax):
    """Авто-обзор: «пол» дефолта GUI + кратность 5 м (байт-совместимость)."""
    xmax = max(SURVEY_FLOOR_M, _ceil_to_step(SURVEY_MARGIN * dmax, ROUND_STEP_M))
    ymax = max(SURVEY_FLOOR_M, _ceil_to_step(SURVEY_MARGIN * ymax, ROUND_STEP_M))
    return xmax, ymax


def derive_recon(f_c_hz, dr_m, da_m, pri, num_pulses, R0_m, V=0.0, alpha=0.0,
                 filename=None, gamma=0.0) -> ReconPlan:
    """Деривация: целевые разрешения → (B, ω, Xmax, Ymax, Nf) + предупреждения.

    Args:
        f_c_hz: несущая, Гц.
        dr_m: целевое разрешение по дальности, м (→ B = c/(2·dr)).
        da_m: целевое разрешение по азимуту, м (→ Δθ = λ/(2·da)).
        pri, num_pulses: период повторения, с; число импульсов.
        R0_m, V, alpha: дальность (м), скорость/направление трансляции.
        filename: путь к .pkl модели (геометрия/кинематика под неё).
        gamma: угловое ускорение, рад/с² (учитывается в эмпирике).

    Returns:
        ReconPlan. Расчёт не блокируется; проблемы — в `warnings`.

    Raises:
        ValueError: некорректные входные величины (dr/da <= 0 и т.п.).
    """
    if f_c_hz <= 0:
        raise ValueError(f"f_c_hz должен быть > 0, получено {f_c_hz}")
    if dr_m <= 0 or da_m <= 0:
        raise ValueError(f"разрешения dr_m/da_m должны быть > 0: {dr_m}/{da_m}")
    if pri <= 0:
        raise ValueError(f"pri должен быть > 0, получено {pri}")
    if num_pulses < 2:
        raise ValueError(f"num_pulses должно быть >= 2, получено {num_pulses}")

    warnings = []
    S = SPEED_OF_LIGHT
    lam = S / f_c_hz

    # ---- B: полоса из Δr, снэп к сетке 1 МГц (бит-совместимость) ----------
    b_exact = S / (2.0 * dr_m)
    spectr_w = round(b_exact / B_SNAP_HZ) * B_SNAP_HZ
    dr_eff = S / (2.0 * spectr_w)
    if abs(spectr_w - b_exact) > 1e-9:
        warnings.append(
            f"WARN (RP): полоса снэпнута к сетке {B_SNAP_HZ / 1e6:.0f} МГц: "
            f"B = {b_exact / 1e9:.6f} → {spectr_w / 1e9:.3f} ГГц "
            f"(фактическое Δr = {dr_eff:.4f} м)")
    if spectr_w <= 0:
        raise ValueError(f"полоса после снэпа не положительна: {spectr_w}")

    # ---- Δθ: угол из Δx, снэп к сетке 0.05° (бит-совместимость) -----------
    dt_exact = lam / (2.0 * da_m)
    deg_exact = math.degrees(dt_exact)
    k_snap = round(deg_exact / DELTATHETA_SNAP_DEG)
    deg_snap = k_snap / 20.0                       # точное деление: k/20
    if abs(deg_snap - deg_exact) > 1e-12:
        warnings.append(
            f"WARN (RP): Δθ снэпнут к сетке {DELTATHETA_SNAP_DEG}°: "
            f"{deg_exact:.4f}° → {deg_snap:.2f}° (фактическое Δx = "
            f"{lam / (2.0 * math.radians(deg_snap)):.4f} м)")
    delta_theta = math.radians(deg_snap)
    da_eff = lam / (2.0 * delta_theta)
    if delta_theta <= 0:
        raise ValueError(f"Δθ после снэпа не положителен: {delta_theta}")

    # ---- ω: производная — ПРИНЦИП инверсии (N, PRI остаются вводом) -------
    omega = delta_theta / (num_pulses * pri)

    # ---- Эмпирика exact кинематике (principle G1) ------------------------
    from models.target import Target

    t_full = Target(filename, pri=pri, num_pulses=num_pulses, R0=R0_m,
                    V=V, alpha=alpha, omega=omega, gamma=gamma)
    R_all = t_full.get_ranges_all()                     # (N, I) — абсолютные
    dR_max = float(np.abs(R_all - R0_m).max())

    if num_pulses >= 2:
        v_los = np.diff(R_all, axis=0) / pri
        spread = float(v_los.max() - v_los.min())
        fd_diff = 2.0 * f_c_hz / S * spread
        fd_tot = 2.0 * f_c_hz / S * float(np.abs(v_los).max())
        walk_pp = float(np.abs(v_los).max()) * pri
    else:
        spread = fd_diff = fd_tot = walk_pp = 0.0

    # y′ — только вращение (трансляция по кросс-рейнджу дает общий сдвиг,
    # который уберёт MOCOMP; для сцены полярной сетки добавляем отдельно)
    t_rot = Target(filename, pri=pri, num_pulses=num_pulses, R0=R0_m,
                   V=0.0, alpha=0.0, omega=omega, gamma=gamma)
    y_all = t_rot.get_cross_range_all()
    y_extent = float(np.abs(y_all).max()) + abs(V * np.sin(np.radians(alpha))) * num_pulses * pri

    # ---- Предупреждения (рецепты, не блокировка) -------------------------
    nyquist = 1.0 / (2.0 * pri)
    n_min = int(math.ceil(num_pulses * fd_diff / nyquist)) if fd_diff > 0 else num_pulses
    if fd_diff >= nyquist:
        warnings.append(
            f"WARN (RP): C1a (диф. доплер {fd_diff:.1f} Гц ≥ Найквист "
            f"{nyquist:.1f} Гц): Δx = {da_m:.3f} м неосуществимо при "
            f"N = {num_pulses}. C1a от PRI не зависит; минимум импульсов "
            f"N_min = {n_min} (увеличьте N до N_min, ослабьте Δx до "
            f"≥ {da_eff * n_min / max(num_pulses, 1):.3f} м "
            f"или уменьшите f_c). Справочно (чистое вращение): "
            f"N_min ≈ 4·span_y·Δθ/λ.")
    elif fd_tot >= nyquist:
        warnings.append(
            f"WARN (RP): C1b (полный доплер с трансляцией {fd_tot:.1f} Гц ≥ "
            f"Найквист {nyquist:.1f} Гц): изображение сдвинуто по кросс-"
            f"рейнджу — включите MOCOMP (PPP) или увеличьте PRF.")
    if walk_pp > 0.5 * dr_eff:
        warnings.append(
            f"WARN (RP): WALK {walk_pp * 1000:.0f} мм/импульс > 0.5·Δr "
            f"({0.5 * dr_eff * 1000:.1f} мм): уменьшите PRI или V.")

    # ---- Обзор из геометрии → Nf -----------------------------------------
    Xmax, Ymax = _survey_floor(dR_max, y_extent)
    Nf = int(math.ceil((Xmax * spectr_w * 2) / S))      # выражение Radar.Nf
    if Nf > NF_CAP:
        warnings.append(
            f"WARN (RP): Nf = {Nf} > {NF_CAP}: матрица E {Nf}×{num_pulses} "
            f"комплексных (≈ {Nf * num_pulses * 16 / 1e6:.0f} МБ) — "
            f"генерация/обработка будут медленными. Ослабьте Δr "
            f"(сейчас {dr_m:.4f} м) или уменьшите обзор.")

    return ReconPlan(
        f_c_hz=f_c_hz,
        dr_requested_m=dr_m,
        da_requested_m=da_m,
        spectr_w_hz=spectr_w,
        dr_effective_m=dr_eff,
        delta_theta_rad=delta_theta,
        da_effective_m=da_eff,
        omega_rad_s=omega,
        num_pulses=num_pulses,
        pri_s=pri,
        Xmax_m=Xmax,
        Ymax_m=Ymax,
        Nf=Nf,
        dR_max_m=dR_max,
        y_extent_m=y_extent,
        fd_diff_hz=fd_diff,
        N_min_c1a=n_min,
        warnings=warnings,
    )
