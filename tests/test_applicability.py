"""Тесты guard'ов применимости (задача G1, План фиксов §5.5–5.6).

По каждому условию C1a/C1b/C2/C3/C4/VIB/WALK — синтетические конфигурации
по обе стороны границы: невалидная (ok=False, верное сообщение с числами и
рецептом) и валидная (ok=True). GUI-уровень — progress_message.
"""
import math

import numpy as np
import pytest

from models.radar import Radar
from models.target import Target
from simulation.applicability import (
    check_applicability,
    format_report,
    hard_failures,
)

SPEED_OF_LIGHT = 3e8
MATRIX = "matrices/matrices_solarB_high.pkl"


def _radar(Xmax=40.0, f_c=8e9):
    return Radar(c=SPEED_OF_LIGHT, f_c=f_c, Xmax=Xmax, Ymax=Xmax,
                 spectr_w=0.5e9, ph_c=0.0, ang=9.0, nifft_size=256)


def _target(pri=0.05, num_pulses=256, omega_deg=0.8, V=0.0, R0=500_000.0,
            vib_amplitude=0.0, vib_frequency=0.0):
    omega = math.radians(omega_deg) / (num_pulses * pri)
    return Target(MATRIX, pri=pri, num_pulses=num_pulses, R0=R0,
                  V=V, alpha=0.0, omega=omega,
                  vib_amplitude=vib_amplitude, vib_frequency=vib_frequency)


def _by_code(checks, code):
    return [c for c in checks if c.code == code][0]


# ---------------- Валидная конфигурация (маржины плана, §Приложение Г) ------

def test_valid_config_all_ok():
    """Демо-1-конфигурация плана: все ok, у каждой жёсткой — severity hard."""
    checks = check_applicability(_radar(), _target())
    codes = {c.code: c for c in checks}
    assert set(codes) == {"C1a", "C1b", "C2", "C3", "C4", "VIB", "WALK"}
    for code, c in codes.items():
        assert c.ok, f"{code}: {c.message}"
        assert c.margin >= 1.0 or math.isinf(c.margin)
    # числа из плана (§Приложение Г): C1a 14.5, C1b 14.3, C2 1.67; C1a/C1b
    # зависят от распределения y0 модели — допуск широкий, C2 детерминирован
    assert codes["C1a"].margin == pytest.approx(14.47, rel=0.5)
    assert codes["C1b"].margin == pytest.approx(14.32, rel=0.5)
    assert codes["C2"].margin == pytest.approx(1.67, rel=0.05)


def test_format_report_mentions_all_codes_and_total():
    checks = check_applicability(_radar(), _target())
    report = format_report(checks)
    for code in ("C1a", "C1b", "C2", "C3", "C4", "VIB", "WALK"):
        assert code in report
    assert "ИТОГ" in report


# ---------------- C1a: дифференциальный доплер (HARD) -----------------------

def test_c1a_hard_fail_low_prf():
    """PRF занижен: дифф. доплер за Найквистом -> ok=False, hard, рецепт."""
    pri = 1.0  # PRF/2 = 0.5 Гц при fc=8 ГГц: вращение 9°/16 имп. заведомо за
    t = _target(pri=pri, num_pulses=16, omega_deg=9.0)
    c = _by_code(check_applicability(_radar(20.0), t), "C1a")
    assert not c.ok
    assert c.severity == "hard"
    assert c.margin < 1.0
    assert c.message.startswith("ОШИБКА КОНФИГУРАЦИИ (C1a)")
    assert "PRF/2" in c.message and f"{pri}" in c.message or "PRI" in c.message


def test_c1a_ok_valid():
    t = _target()
    c = _by_code(check_applicability(_radar(), t), "C1a")
    assert c.ok and c.message.startswith("OK (C1a)")


# ---------------- C1b: полный доплер с трансляцией (WARN) ------------------

def test_c1b_translation_dominates_fail():
    """V=1.2 м/с, PRI=0.05: fd≈64 Гц > Найквист 10 Гц, margin≈0.15 (стенд T).

    Контроль «2*omega*y/lambda» при V!=0 показывал бы «OK» — guard считает
    эмпирически и ловит трансляцию (главный аргумент эмпирического C1b).
    """
    t = _target(pri=0.05, num_pulses=64, omega_deg=0.0, V=1.2)
    checks = check_applicability(_radar(), t)
    c1b = _by_code(checks, "C1b")
    assert c1b.ok is False
    assert c1b.severity == "warn"
    assert c1b.margin == pytest.approx(0.15, rel=0.25)
    assert c1b.message.startswith("WARN (C1b)")
    assert "циклически сдвинуто" in c1b.message
    # Трансляция не видна дифференциальным доплером: C1a — «OK» (это и есть
    # главный аргумент эмпирического C1b из плана §0.5, стенд T)
    assert _by_code(checks, "C1a").ok
    assert all(c.code != "C1b" for c in hard_failures(checks))


def test_c1b_translation_ok():
    t = _target(pri=0.05, num_pulses=64, omega_deg=0.0, V=0.15)
    c = _by_code(check_applicability(_radar(), t), "C1b")
    assert c.ok and c.margin > 1.0


# ---------------- C2: окно однозначности по дальности (HARD) ---------------

def test_c2_fail_survey20_solarB():
    """survey=20 (окно ±10.05) против solarB_high max|δR|≈12.05: mute wrap."""
    t = _target(pri=0.05, num_pulses=64, omega_deg=0.0)
    c = _by_code(check_applicability(_radar(Xmax=20.0), t), "C2")
    assert not c.ok and c.severity == "hard"
    # окно 10.05 м (Xmax=20 → Nf=67), max|δR| ≈ max x0 = 12.05 м
    assert c.margin == pytest.approx(10.05 / 12.05, rel=0.1)
    assert c.message.startswith("ОШИБКА КОНФИГУРАЦИИ (C2)")
    assert "survey_length" in c.message and "±10.05" in c.message
    assert "Увеличьте survey_length" in c.message


def test_c2_ok_survey40_solarB():
    """survey=40 → окно ±20.10 м, max|δR|≈12.05: запас 1.67 (план)."""
    t = _target(pri=0.05, num_pulses=64, omega_deg=0.0)
    c = _by_code(check_applicability(_radar(Xmax=40.0), t), "C2")
    assert c.ok and c.margin == pytest.approx(1.67, rel=0.05)


# ---------------- C3: Фраунгофер (WARN) ------------------------------------

def test_c3_fail_short_range():
    """R0=1 км с D≈33 м на X-band: 2·D²/λ≈60 км ≫ R0."""
    t = _target(R0=1_000.0)
    c = _by_code(check_applicability(_radar(), t), "C3")
    assert not c.ok and c.severity == "warn"
    assert c.message.startswith("WARN (C3)")
    assert "Увеличьте дальность" in c.message


def test_c3_ok_500km():
    t = _target()
    c = _by_code(check_applicability(_radar(), t), "C3")
    assert c.ok and c.message.startswith("OK (C3)")


# ---------------- C4: малый угол RD (WARN) ---------------------------------

def test_c4_fail_large_angle():
    """Δθ=9° при пороге ≈0.88° (X-band, L=40 м): arc-smearing."""
    t = _target(pri=0.05, num_pulses=256, omega_deg=9.0)
    c = _by_code(check_applicability(_radar(), t), "C4")
    assert not c.ok and c.severity == "warn"
    assert c.message.startswith("WARN (C4)")
    assert "0.88" in c.message
    assert "полярным" in c.message


def test_c4_ok_small_angle():
    t = _target(pri=0.05, num_pulses=256, omega_deg=0.8)
    c = _by_code(check_applicability(_radar(), t), "C4")
    assert c.ok


# ---------------- VIB: микровибрация (WARN) --------------------------------

def test_vib_fail_above_nyquist():
    """f_vib=30 Гц при PRF=20 Гц: алиасинг подписи вибрации."""
    t = _target(pri=0.05, num_pulses=256, omega_deg=0.0,
                vib_amplitude=0.001, vib_frequency=30.0)
    c = _by_code(check_applicability(_radar(), t), "VIB")
    assert not c.ok and c.severity == "warn"
    assert c.message.startswith("WARN (VIB)")
    assert "30.0" in c.message and "10.0" in c.message


def test_vib_ok_below_nyquist():
    t = _target(pri=0.05, num_pulses=256, omega_deg=0.0,
                vib_amplitude=0.001, vib_frequency=4.0)
    c = _by_code(check_applicability(_radar(), t), "VIB")
    assert c.ok


def test_vib_off_not_applicable():
    t = _target()
    c = _by_code(check_applicability(_radar(), t), "VIB")
    assert c.ok and math.isinf(c.margin)


# ---------------- WALK: ход за импульс < 0.5·dr (WARN) ---------------------

def test_walk_fail_too_fast():
    """V=30 м/с при PRI=0.05: ход 1.5 м = 5·dr за импульс."""
    t = _target(pri=0.05, num_pulses=64, omega_deg=0.0, V=30.0)
    c = _by_code(check_applicability(_radar(), t), "WALK")
    assert not c.ok and c.severity == "warn"
    assert c.message.startswith("WARN (WALK)")


def test_walk_ok_slow():
    """Демо 2/3 плана: ход/импульс 0.022 бина."""
    t = _target(pri=5e-4, num_pulses=256, omega_deg=3.0, V=10.0)
    c = _by_code(check_applicability(_radar(), t), "WALK")
    assert c.ok


# ---------------- enforce: программный API ---------------------------------

def test_enforce_blocks_hard_failures():
    pytest.importorskip("PyQt5")
    from processing.data_processor import DataProcessor

    proc = DataProcessor(
        filename=MATRIX, method="стандартный", f_c=8.0, Xmax=20.0, Ymax=20.0,
        spectr_w=0.5, ang=9.0, range_m=500_000.0, nifft_size=256,
        V=0.0, alpha=0.0, omega=math.radians(9.0) / 16.0,
        pri=1.0, num_pulses=16, enforce=True,
    )
    with pytest.raises(ValueError):
        proc.compute_isar_notified(display_mode="dB")


def test_guard_emits_but_does_not_block_gui_path():
    """Дефолт (enforce=False): расчёт завершается, сообщения в progress_message.
    Старый демо-1 режим (N=16, PRI=1) — по C1a/C2 hard-нарушение, но расчёт
    НЕ прерывается (инвариант совместимости с GUI)."""
    pytest.importorskip("PyQt5")
    from processing.data_processor import DataProcessor

    proc = DataProcessor(
        filename=MATRIX, method="стандартный", f_c=8.0, Xmax=20.0, Ymax=20.0,
        spectr_w=0.5, ang=9.0, range_m=500_000.0, nifft_size=256,
        V=0.0, alpha=0.0, omega=math.radians(9.0) / 16.0,
        pri=1.0, num_pulses=16,
    )
    messages = []
    proc.progress_message.connect(messages.append)
    proc.compute_isar_notified(display_mode="dB")

    assert any(m.startswith("ОШИБКА КОНФИГУРАЦИИ (C1a)") for m in messages)
    assert any(m.startswith("ОШИБКА КОНФИГУРАЦИИ (C2)") for m in messages)
