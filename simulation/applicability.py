"""Guard'ы применимости физической модели ISAR (условия C1–C5).

Задача G1 (План фиксов §5): в стандартном тракте не было ни одной проверки
границ — невалидные конфигурации (алиасинг доплера, заворот дальности)
молча давали геометрически недостоверные изображения.

Ключевое решение: все условия считаются ЭМПИРИЧЕСКИ по точной кинематике
Target.get_ranges_all() (конечные разности дальностей по медленному времени),
без приближённых формул вида «2*omega*y/lambda». Причина: полный доплер
включает трансляцию ЦМ (измерено: V=1 м/с при PRI=0.05 c даёт fd=65 Гц при
Найквисте 10 Гц, а контроль «2*omega*y/lambda» показывал «OK»).

Условия (План фиксов §0.5):
    C1a  дифференциальный доплер < PRF/2     — HARD (заворот спектра,
         рассеиватели отображаются на aliased-позиции, восстановление
         невозможно)
    C1b  полный доплер < PRF/2               — WARN (изображение циклически
         сдвинуто по кросс-рейнджу; PPP MOCOMP убирает общий сдвиг, без
         MOCOMP позиции по кросс-рейнджу недостоверны)
    C2   max|dR| < окно c/(4*Δf)             — HARD (заворот в противоположный
         край профиля происходит молча)
    C3   R0 > 2*D^2/lambda (Фраунгофер)      — WARN (статическое искажение
         r^2/(2*R0))
    C4   Δθ < sqrt(lambda/L_LOS)/2           — WARN (arc-smearing; C&M §3.1)
    VIB  f_vib < PRF/2 и скорость вибрации < PRF/2 в доплере — WARN
    WALK ход за импульс < 0.5*dr             — WARN (subbin-здоровье)

Guard по умолчанию НЕ блокирует расчёт (совместимость с GUI): испускаются
сообщения WARN / ОШИБКА КОНФИГУРАЦИИ. Блокировка — опция enforce=True у
DataProcessor (программный API).
"""
import math
from dataclasses import dataclass

import numpy as np

from processing.polar_reformat import polar_threshold_deg, LEGACY_THRESHOLD_DEG

SPEED_OF_LIGHT = 3e8

SEVERITY_HARD = "hard"
SEVERITY_WARN = "warn"


@dataclass(frozen=True)
class Check:
    """Результат одного условия применимости.

    margin: запас конфигурации: >= 1.0 — условие выполнено; < 1.0 — нарушено;
            inf — неприменимо (например, вибрация выключена).
    severity: "hard" — изображение геометрически недостоверно, не
              восстанавливается; "warn" — допустимо с оговоркой.
    message: готовая строка для progress_message / печати в демо.
    """
    code: str
    ok: bool
    margin: float
    severity: str
    message: str


def check_applicability(radar, target) -> list[Check]:
    """Проверить условия применимости C1a/C1b/C2/C3/C4/VIB/WALK.

    Все величины — эмпирические, по точной геометрии:

        v_los[i,n] = (R[i, n+1] - R[i, n]) / PRI        # конечные разности
        fd_tot  = 2*f_c/c * max |v_los|                 -> C1b (полный доплер)
        fd_diff = 2*f_c/c * max_n( max_i v_los - min_i v_los ) -> C1a
        dR_max  = max |R[i,n] - R0|                     -> C2 (окно c/(4*Δf))
        walk_pp = max|v_los| * PRI                      -> WALK (< 0.5*dr)

    Args:
        radar: models.radar.Radar (f_c, spectr_w, Xmax, ...).
        target: models.target.Target (кинематика + матрица рассеивателей).

    Returns:
        Список Check; расчёт не прерывается ни при каком результате.
    """
    S = SPEED_OF_LIGHT
    _, _, f_r, *_ = radar.base_img_params()
    fc = radar.f_c
    lam = S / fc
    pri = target.pri
    N = target.num_pulses
    prf = 1.0 / pri
    nyquist = prf / 2.0

    # Разрешение по дальности (согласовано с range_compress: dr = c/(2*M*Δf),
    # M*Δf = spectr_w) и окно однозначности по дальности ±c/(4*Δf)
    dr = S / (2.0 * radar.spectr_w)
    unambiguous_half = S / (4.0 * radar.df)   # radar.df = spectr_w/Nf

    R_all = target.get_ranges_all()           # (N, I) — точные дальности

    checks: list[Check] = []

    # ---------- C1a: дифференциальный доплер (HARD) ----------
    if N >= 2:
        v_los = np.diff(R_all, axis=0) / pri          # (N-1, I)
        spread = float(v_los.max() - v_los.min())     # макс-разброс LOS-скоростей
        fd_diff = 2.0 * fc / S * spread
        margin_c1a = nyquist / fd_diff if fd_diff > 0 else math.inf
        checks.append(_format_c1a(spread, fd_diff, nyquist, margin_c1a, pri))
    else:
        checks.append(Check("C1a", True, math.inf, SEVERITY_WARN,
                            "C1a: не применимо (N < 2 импульсов) — доплер не определён"))

    # ---------- C1b: полный доплер, включая трансляцию (WARN) ----------
    if N >= 2:
        fd_tot = 2.0 * fc / S * float(np.abs(v_los).max())
        margin_c1b = nyquist / fd_tot if fd_tot > 0 else math.inf
        checks.append(_format_c1b(fd_tot, nyquist, margin_c1b))
    else:
        checks.append(Check("C1b", True, math.inf, SEVERITY_WARN,
                            "C1b: не применимо (N < 2 импульсов)"))

    # ---------- C2: окно однозначности по дальности (HARD) ----------
    dR_max = float(np.abs(R_all - target.R0).max())
    margin_c2 = unambiguous_half / dR_max if dR_max > 0 else math.inf
    checks.append(_format_c2(dR_max, unambiguous_half, margin_c2))

    # ---------- C3: Фраунгофер (WARN) ----------
    sc = target._load_scatterers()
    r_max = float(np.sqrt(sc["x0"] ** 2 + sc["y0"] ** 2).max()) if len(sc["intens"]) else 0.0
    if r_max > 0:
        D = 2.0 * r_max                                # габарит цели
        fraunhofer = 2.0 * D * D / lam
        margin_c3 = target.R0 / fraunhofer
        qpe = r_max ** 2 / (2.0 * target.R0)           # фактическое искажение, м
        checks.append(_format_c3(target.R0, fraunhofer, margin_c3, qpe, dr, D, lam))
    else:
        checks.append(Check("C3", True, math.inf, SEVERITY_WARN,
                            "C3: не применимо (нет рассеивателей)"))

    # ---------- C4: малый угол RD (WARN, C&M §3.1) ----------
    d_theta_deg = math.degrees(target.omega * N * pri)
    threshold_deg = polar_threshold_deg(lam, radar.Xmax)
    if threshold_deg is not None and threshold_deg > 0:
        margin_c4 = threshold_deg / d_theta_deg if d_theta_deg > 0 else math.inf
        checks.append(_format_c4(d_theta_deg, threshold_deg, margin_c4))
    else:
        checks.append(Check("C4", True, math.inf, SEVERITY_WARN,
                            "C4: не применимо (порог не вычисляется)"))

    # ---------- VIB: микровибрация (WARN) ----------
    checks.append(_format_vib(target, nyquist, fc, S))

    # ---------- WALK: ход за импульс < 0.5*dr (WARN) ----------
    if N >= 2:
        walk_pp = float(np.abs(v_los).max()) * pri
        margin_walk = (0.5 * dr) / walk_pp if walk_pp > 0 else math.inf
        checks.append(_format_walk(walk_pp, dr, margin_walk, pri))
    else:
        checks.append(Check("WALK", True, math.inf, SEVERITY_WARN,
                            "WALK: не применимо (N < 2 импульсов)"))

    return checks


def _format_c1a(spread, fd_diff, nyquist, margin, _pri) -> Check:
    if margin >= 1.0:
        msg = (f"OK (C1a): дифференциальный доплер {fd_diff:.1f} Гц < PRF/2 "
               f"(запас {margin:.2f}); макс разброс LOS-скоростей {spread:.2f} м/с")
        return Check("C1a", True, margin, SEVERITY_HARD, msg)
    msg = (f"ОШИБКА КОНФИГУРАЦИИ (C1a): дифференциальный доплер {fd_diff:.1f} Гц "
           f"≥ PRF/2 = {nyquist:.1f} Гц (запас {margin:.2f}). Заворот спектра: "
           f"часть рассеивателей отображается на aliased-позиции по "
           f"кросс-рейнджу — изображение геометрически недостоверно "
           f"(позиции ложные, восстановление невозможно). Уменьшите omega "
           f"(или gamma), увеличьте PRF (меньше PRI={_pri:.4g} c) или "
           f"уменьшите f_c")
    return Check("C1a", False, margin, SEVERITY_HARD, msg)


def _format_c1b(fd_tot, nyquist, margin) -> Check:
    if margin >= 1.0:
        msg = (f"OK (C1b): полный доплер (с трансляцией) {fd_tot:.1f} Гц < PRF/2 "
               f"(запас {margin:.2f})")
        return Check("C1b", True, margin, SEVERITY_WARN, msg)
    msg = (f"WARN (C1b): полный доплер {fd_tot:.1f} Гц ≥ PRF/2 = {nyquist:.1f} Гц "
           f"(запас {margin:.2f}). Общий доплер заворачивается: изображение "
           f"циклически сдвинуто по кросс-рейнджу. При use_mocomp=True PPP-"
           f"автофокус убирает общий сдвиг (опорный штрих); без MOCOMP позиции "
           f"по кросс-рейнджу недостоверны. Уменьшите V/PRI или увеличьте PRF")
    return Check("C1b", False, margin, SEVERITY_WARN, msg)


def _format_c2(dR_max, half_window, margin) -> Check:
    if margin >= 1.0:
        msg = (f"OK (C2): окно дальности ±{half_window:.2f} м > макс |δR| "
               f"{dR_max:.2f} м (запас {margin:.2f})")
        return Check("C2", True, margin, SEVERITY_HARD, msg)
    needed = 1.2 * dR_max
    msg = (f"ОШИБКА КОНФИГУРАЦИИ (C2): макс |δR| {dR_max:.2f} м ≥ окна "
           f"±{half_window:.2f} м (survey_length задаёт окно ±c/(4Δf)). "
           f"Заворот дальности: края цели молча переносятся на противоположный "
           f"край профиля. Увеличьте survey_length до ≥ {needed:.1f} м "
           f"(запас 1.2)")
    return Check("C2", False, margin, SEVERITY_HARD, msg)


def _format_c3(r0, fraunhofer, margin, qpe, dr, D, lam) -> Check:
    if margin >= 1.0:
        msg = (f"OK (C3): R0 = {r0 / 1e3:.1f} км > 2·D²/λ = {fraunhofer / 1e3:.1f} км "
               f"(запас {margin:.2f}; D = габарит цели); квадратичная фазовая "
               f"ошибка r_max²/(2·R0) = {qpe * 1e3:.2f} мм ≪ dr = {dr:.2f} м")
        return Check("C3", True, margin, SEVERITY_WARN, msg)
    needed = fraunhofer * 1.2
    msg = (f"WARN (C3): R0 = {r0 / 1e3:.1f} км < 2·D²/λ = {fraunhofer / 1e3:.1f} км "
           f"(запас {margin:.2f}; D = {D:.1f} м). Зона Фраунгофера нарушена — "
           f"статическое искажение {qpe * 1e3:.1f} мм против dr = {dr * 1e2:.1f} см. "
           f"Увеличьте дальность R0 до ≥ {needed / 1e3:.1f} км")
    return Check("C3", False, margin, SEVERITY_WARN, msg)


def _format_c4(d_theta_deg, threshold_deg, margin) -> Check:
    if margin >= 1.0:
        msg = (f"OK (C4): суммарный угол Δθ = {d_theta_deg:.2f}° < физического "
               f"порога {threshold_deg:.2f}° (C&M §3.1, запас {margin:.2f})")
        return Check("C4", True, margin, SEVERITY_WARN, msg)
    msg = (f"WARN (C4): Δθ = {d_theta_deg:.2f}° > физический порог "
           f"{threshold_deg:.2f}° (C&M §3.1) — аппроксимация малых углов (RD) "
           f"даёт arc-smearing и искажение дальности краевых рассеивателей; "
           f"рекомендуется метод с полярным переформатированием "
           f"(legacy-порог {LEGACY_THRESHOLD_DEG:.2f}°)")
    return Check("C4", False, margin, SEVERITY_WARN, msg)


def _format_vib(target, nyquist, fc, S) -> Check:
    fv = target.vib_frequency
    amp = target.vib_amplitude
    if amp == 0.0 or fv == 0.0:
        return Check("VIB", True, math.inf, SEVERITY_WARN,
                     "VIB: не применимо (вибрация выключена)")
    v_los_vib = 2.0 * math.pi * fv * amp          # пиковая LOS-скорость вибрации, м/с
    fd_vib = 2.0 * fc / S * v_los_vib
    margin_freq = nyquist / fv if fv > 0 else math.inf
    margin_dop = nyquist / fd_vib if fd_vib > 0 else math.inf
    margin = min(margin_freq, margin_dop)
    ok = margin >= 1.0
    if ok:
        msg = (f"OK (VIB): f_vib = {fv:.1f} Гц < PRF/2 = {nyquist:.1f} Гц, "
               f"доплер вибрации {fd_vib:.1f} Гц < PRF/2 (запас {margin:.2f})")
        return Check("VIB", True, margin, SEVERITY_WARN, msg)
    parts = []
    if margin_freq < 1.0:
        parts.append(f"f_vib = {fv:.1f} Гц ≥ PRF/2 = {nyquist:.1f} Гц — "
                     f"подпись вибрации алиасится (период сгибается)")
    if margin_dop < 1.0:
        parts.append(f"доплер вибрации {fd_vib:.1f} Гц ≥ PRF/2 — боковые "
                     f"частоты заворачиваются")
    msg = (f"WARN (VIB): { '; '.join(parts) }. Уменьшите vib_frequency или "
           f"амплитуду вибрации, либо увеличьте PRF")
    return Check("VIB", False, margin, SEVERITY_WARN, msg)


def _format_walk(walk_pp, dr, margin, pri) -> Check:
    if margin >= 1.0:
        msg = (f"OK (WALK): ход за импульс {walk_pp * 1e3:.1f} мм < 0.5·dr "
               f"({0.5 * dr * 1e3:.1f} мм, запас {margin:.2f}) — суб-бин "
               f"здоровье профиля")
        return Check("WALK", True, margin, SEVERITY_WARN, msg)
    msg = (f"WARN (WALK): ход за импульс {walk_pp * 1e3:.1f} мм ≥ 0.5·dr = "
           f"{0.5 * dr * 1e3:.1f} мм (запас {margin:.2f}) — профиль мигрирует "
           f"быстрее полбина за PRI; по-пиковая корреляция огибающей (MOCOMP) "
           f"может терять захват. Уменьшите PRI = {pri:.4g} c или скорость "
           f"движения")
    return Check("WALK", False, margin, SEVERITY_WARN, msg)


def hard_failures(checks) -> list[Check]:
    """Не-OK проверки класса hard — для enforce-блокировки (программный API)."""
    return [c for c in checks if (not c.ok) and c.severity == SEVERITY_HARD]


def format_report(checks) -> str:
    """Таблица-отчёт для печати в демо/stdout (коды C1–C5, запасы)."""
    lines = ["Проверка применимости (guard C1–C5):"]
    for c in checks:
        if math.isinf(c.margin):
            status = "-- "
            margin_txt = "   n/a"
        else:
            status = "OK " if c.ok else ("!!! " if c.severity == SEVERITY_HARD else "WRN")
            margin_txt = f"{c.margin:8.2f}×"
        first, rest = c.message.split(":", 1)
        lines.append(f"  {c.code:4s} [{status}] запас {margin_txt}{rest}")
    failed_hard = hard_failures(checks)
    if failed_hard:
        codes = ", ".join(c.code for c in failed_hard)
        lines.append(f"  ИТОГ: ЗАПРЕЩЕНО ({codes}) — изображение будет "
                     f"геометрически недостоверным")
    else:
        warns = [c.code for c in checks if not c.ok]
        if warns:
            lines.append(f"  ИТОГ: допустимо с оговорками ({', '.join(warns)})")
        else:
            lines.append("  ИТОГ: все условия выполнены")
    return "\n".join(lines)
