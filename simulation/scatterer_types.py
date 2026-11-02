"""Библиотека RCS элементарных рассеивателей (канонических форм).

Оптическое приближение (когда размеры >> длины волны) для моностатической
геометрии. Максимумы RCS — канонические формулы:

    Источники (сверены между собой):
    - Skolnik, "Radar Handbook", 3rd ed., гл. "Radar Cross Section"
      (таблица канонических форм);
    - Knott, Shaeffer, Tuley, "Radar Cross Section", 2nd ed., Artech House, 1993;
    - Özdemir, "Inverse Synthetic Aperture Radar Imaging with MATLAB
      Algorithms", 2012, гл. 2 (базовый референс проекта);
    - Pouliguen et al., JPIER 2008 — аналитический RCS пластины от ракурса
      (https://www.jpier.org/ac_api/download.php?id=08081902);
    - Balanis, "Antenna Theory" — диаграмма полуволнового диполя.

Канонические максимумы (λ — длина волны, k = 2π/λ):
    сфера         σ = π·r²                                  (изотропна)
    пластина      σ = 4π·a²b²/λ²                            (нормальное падение)
    цилиндр       σ = 2π·r·L²/λ                             (боком, LOS ⊥ оси)
    диэдр         σ = 8π·a²b²/λ²                            (вдоль биссектрисы)
    триэдр        σ = 4π·a⁴/(3λ²)                           (вдоль диагонали)
    диполь        σ = 0.86·λ²                               (боком, резонанс λ/2)

Конвенции:
    - Ось X — линия визия (радар -> цель), Y — поперечник, Z — третья ось.
    - Локальная СК элемента: пластина лежит в локальной плоскости XY,
      нормаль — локальная Z; ось цилиндра и диполя — локальная Z;
      биссектриса диэдра и диагональ триэдра — локальная Z.
    - Эйлеровы углы (phi, theta, psi): конвенция ZYZ, АКТИВНАЯ матрица
      поворота R = Rz(phi)·Ry(theta)·Rz(psi), переводящая вектор из
      локальной СК элемента в СК цели (градусы на входе).
    - los_unit — ЕДИНИЧНЫЙ вектор направления на радар В СК ЦЕЛИ
      (или на цель из радара — формулы симметричны к знаку).
"""
import math
from dataclasses import dataclass, field

import numpy as np

SPEED_OF_LIGHT = 3e8

# Типы элементов, поддерживаемые библиотекой
ELEMENT_TYPES = ("sphere", "plate", "cylinder", "dihedral", "trihedral", "dipole")


@dataclass
class Element:
    """Элементарный рассеиватель в системе координат цели (ЦМ в нуле).

    Атрибуты:
        type: один из ELEMENT_TYPES.
        params: размеры в метрах, ключи зависят от типа:
            sphere   -> {"r": радиус}
            plate    -> {"a": сторона по X, "b": сторона по Y}
            cylinder -> {"r": радиус, "L": длина (ось = локальная Z)}
            dihedral -> {"a": сторона, "b": сторона}
            trihedral-> {"a": сторона}
            dipole   -> {"L": длина (ось = локальная Z)}
        position: (x, y, z) центра элемента, м.
        orientation_deg: эйлеровы углы ZYZ (phi, theta, psi), град —
            ориентация локальной СК элемента в СК цели.
        phase: константная фаза центра рассеяния (рад).
            None -> случайная фаза из RNG генератора (spawn_rng).
    """
    type: str
    params: dict
    position: tuple
    orientation_deg: tuple
    phase: float = None


def _sinc(x):
    """Нормированный sinc: sin(x)/x (в numpy sinc(x)=sin(pi x)/(pi x))."""
    return np.sinc(np.asarray(x, dtype=float) / np.pi)


def rotation_matrix_from_euler(phi_deg, theta_deg, psi_deg):
    """Активная матрица поворота, конвенция ZYZ.

    R = Rz(phi) · Ry(theta) · Rz(psi)  (углы в ГРАДУСАХ на входе).

    Переводит вектор из локальной СК элемента в СК цели:
        v_world = R @ v_local.

    Args:
        phi_deg, theta_deg, psi_deg: эйлеровы углы, град.

    Returns:
        R: матрица 3×3 (np.ndarray), ортогональная, det=+1.
    """
    p = math.radians(phi_deg)
    t = math.radians(theta_deg)
    s = math.radians(psi_deg)

    cp, sp = math.cos(p), math.sin(p)
    ct, st = math.cos(t), math.sin(t)
    cs, ss = math.cos(s), math.sin(s)

    Rz_psi = np.array([[cs, -ss, 0.0], [ss, cs, 0.0], [0.0, 0.0, 1.0]])
    Ry_t = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
    Rz_phi = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])

    return Rz_phi @ Ry_t @ Rz_psi


def element_rotation(element):
    """Матрица поворота элемента (кэш не нужен: элементов <= 20)."""
    return rotation_matrix_from_euler(*element.orientation_deg)


def aspect_in_element_frame(element, los_unit):
    """Единичный вектор LOS в локальной СК элемента.

        l_local = R^T · l_world
    """
    R = element_rotation(element)
    return R.T @ np.asarray(los_unit, dtype=float)


def _angle_between(a, b):
    """Угол между векторами (рад), вход единичные."""
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.acos(c)


def _rcs_sphere(params, lam, l_local):
    """σ = π·r² — изотропна (оптическая область, |r| >> λ).

    Источник: Skolnik, Radar Handbook (каноническая форма); справедлива
    при 2πr/λ >> 1.
    """
    r = params["r"]
    return math.pi * r * r


def _rcs_plate(params, lam, l_local):
    """σ пластины a×b от ракурса (физическая оптика, моностатика).

        σ(θ) = (4π·a²b²/λ²) · l_z² · sinc²(k·a·l_x/2) · sinc²(k·b·l_y/2)

    где l = (l_x, l_y, l_z) — единичный LOS в СК элемента (нормаль = ẑ):
    l_x, l_y задают отклонение от нормали в плоскостях сторон (Pouliguen
    2008, ф. 3-5; Knott et al., гл. 6 — прямоугольная пластина).
    Множитель l_z² — обличность (cos²θ), зануляет на скользящих углах.
    """
    a, b = params["a"], params["b"]
    k = 2.0 * math.pi / lam
    lx, ly, lz = float(l_local[0]), float(l_local[1]), float(l_local[2])
    sigma_max = 4.0 * math.pi * a * a * b * b / (lam * lam)
    if lz < 0.0:
        # Пластина тонкая: тыльная сторона зеркальна к лицевой
        lz = -lz
    return sigma_max * lz * lz * float(_sinc(k * a * lx / 2.0)) ** 2 \
        * float(_sinc(k * b * ly / 2.0)) ** 2


def _rcs_cylinder(params, lam, l_local):
    """σ цилиндра (ось = локальная Z) от ракурса (физическая оптика).

        σ(ψ) = (2π·r·L²/λ) · sinc²((k·L/2)·cos ψ)

    ψ — угол между LOS и осью; максимум при ψ=90° (боком):
    σ_max = 2πrL²/λ (Skolnik). Инвариант к вращению вокруг оси
    (поверхность вращения).
    """
    r, L = params["r"], params["L"]
    k = 2.0 * math.pi / lam
    cos_psi = float(np.clip(l_local[2], -1.0, 1.0))
    sigma_max = 2.0 * math.pi * r * L * L / lam
    return sigma_max * float(_sinc(k * L * cos_psi / 2.0)) ** 2


def _rcs_dihedral(params, lam, l_local):
    """σ диэдрального уголка (биссектриса = локальная Z).

        σ_max = 8π·a²b²/λ²  (Skolnik, Knott)

    Угловая зависимость — ИНЖЕНЕРНАЯ АППРОКСИМАЦИЯ двойного отражения:
    плато cos⁸(2·θ_b) внутри полураствора 45° от биссектрисы, ноль вне.
    Точная PO-модель даёт серию лепестков с максимумом на биссектрисе
    (зеркальная точка, ψ=0°) и раздвоением лепестков при удалении
    (Knott, гл. 8); для моделирования ISAR гладкое плато с максимумом
    на биссектрисе достаточно.
    """
    a, b = params["a"], params["b"]
    sigma_max = 8.0 * math.pi * a * a * b * b / (lam * lam)
    theta_b = _angle_between(np.array([0.0, 0.0, 1.0]), l_local)
    if theta_b > math.pi / 4.0:
        return 0.0
    c = math.cos(2.0 * theta_b)
    return sigma_max * c ** 8


def _rcs_trihedral(params, lam, l_local):
    """σ триэдрального уголка (диагональ = локальная Z).

        σ_max = 4π·a⁴/(3λ²)  (Skolnik, Knott)

    Угловая зависимость — ИНЖЕНЕРНАЯ АППРОКСИМАЦИЯ: трёхкратное отражение
    держит плато в широком конусе ~±50° от диагонали (cos⁸(1.5·θ_d),
    зануление вне 60°). Точное поведение: монотонный спад с осцилляциями.
    """
    a = params["a"]
    sigma_max = 4.0 * math.pi * a ** 4 / (3.0 * lam * lam)
    theta_d = _angle_between(np.array([0.0, 0.0, 1.0]), l_local)
    if theta_d > math.pi / 3.0:
        return 0.0
    c = math.cos(1.5 * theta_d)
    return sigma_max * c ** 8


def _dipole_pattern(psi):
    """Диаграмма полуволнового диполя |F(ψ)|, нормированная к 1 в broadside.

        F(ψ) = cos((π/2)·cos ψ) / sin ψ,   ψ — угол от оси диполя.

    Источник: Balanis, "Antenna Theory", гл. 4 (полуволновая антенна).
    При ψ→0 предел F=0 (излучение вдоль оси отсутствует).
    """
    psi = float(np.clip(psi, 0.0, math.pi))
    if psi < 1e-9 or abs(psi - math.pi) < 1e-9:
        return 0.0
    return abs(math.cos(math.pi / 2.0 * math.cos(psi)) / math.sin(psi))


def _rcs_dipole(params, lam, l_local):
    """σ резонансного полуволнового диполя от ракурса.

        σ(ψ) = 0.86·λ² · F⁴(ψ)

    Максимум 0.86λ² при ψ=90° (боком) — Skolnik, Radar Handbook;
    F(ψ) — диаграмма диполя (Balanis), ψ — угол между LOS и осью.
    Зависимость ~F⁴: антенная переизлучающая модель (эхо дважды
    фильтруется диаграммой — приём и переизлучение).
    """
    axis = np.array([0.0, 0.0, 1.0])
    psi = _angle_between(axis, l_local)
    return 0.86 * lam * lam * _dipole_pattern(psi) ** 4


_RCS_FUNCS = {
    "sphere": _rcs_sphere,
    "plate": _rcs_plate,
    "cylinder": _rcs_cylinder,
    "dihedral": _rcs_dihedral,
    "trihedral": _rcs_trihedral,
    "dipole": _rcs_dipole,
}

# Обязательные ключи params для каждого типа
REQUIRED_PARAMS = {
    "sphere": ("r",),
    "plate": ("a", "b"),
    "cylinder": ("r", "L"),
    "dihedral": ("a", "b"),
    "trihedral": ("a",),
    "dipole": ("L",),
}


def validate_element(el_dict):
    """Проверить словарь элемента (из JSON) на корректность.

    Raises:
        ValueError: неизвестный тип или отсутствуют обязательные размеры.
    """
    etype = el_dict.get("type")
    if etype not in ELEMENT_TYPES:
        raise ValueError(
            f"Неизвестный тип элемента: {etype!r}. "
            f"Доступны: {', '.join(ELEMENT_TYPES)}"
        )
    params = el_dict.get("params", {})
    for key in REQUIRED_PARAMS[etype]:
        if key not in params:
            raise ValueError(
                f"Тип '{etype}' требует params с ключом '{key}', "
                f"получено: {sorted(params.keys())}"
            )
    return True


def element_from_dict(el_dict):
    """Создать Element из словаря (строка JSON)."""
    validate_element(el_dict)
    return Element(
        type=el_dict["type"],
        params=dict(el_dict["params"]),
        position=tuple(el_dict.get("position", (0.0, 0.0, 0.0))),
        orientation_deg=tuple(el_dict.get("orientation_deg", (0.0, 0.0, 0.0))),
        phase=el_dict.get("phase", None),
    )


def rcs_m2(element, wavelength, los_unit):
    """ЭПР элемента в м² для заданного ракурса.

    Args:
        element: Element (модели или вручную собранный).
        wavelength: длина волны, м.
        los_unit: единичный вектор направления на радар, СК ЦЕЛИ.

    Returns:
        sigma: ЭПР, м² (вещественная, >= 0).
    """
    if element.type not in _RCS_FUNCS:
        raise ValueError(f"Неизвестный тип элемента: {element.type!r}")
    l_local = aspect_in_element_frame(element, los_unit)
    return _RCS_FUNCS[element.type](element.params, wavelength, l_local)


def rcs_amplitude(element, wavelength, los_unit, rng=None):
    """Комплексная амплитуда вклада элемента: sqrt(σ)·exp(j·φ).

    φ — константная фаза центра рассеяния: element.phase, если задана,
    иначе случайная из переданного RNG (генератор передаётся вызывающим
    кодом — см. raw_generator, spawn_rng(seed, "element_phase")).

    Геометрическая фаза распространения exp(-j·4πfR/c) здесь НЕ входит —
    она вычисляется генератором по дальности.
    """
    sigma = rcs_m2(element, wavelength, los_unit)
    if element.phase is not None:
        phase = element.phase
    elif rng is not None:
        phase = rng.uniform(0.0, 2.0 * math.pi)
    else:
        phase = 0.0
    return math.sqrt(sigma) * np.exp(1j * phase)
