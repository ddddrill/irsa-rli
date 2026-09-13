"""Тесты физического порога полярного конвейера (задача R4).

Критерий применимости малой угловой аппроксимации,
C&M, §3.1: Δθ_max ≈ √(λ/L_LOS)/2.
"""
import math

import pytest

from processing.polar_reformat import polar_threshold_deg, should_use_polar_reformat


def test_physical_threshold_values():
    """Опорные значения: X-band/L=20 м → 1.24°, L-band/L=20 м → 3.06°."""
    assert abs(polar_threshold_deg(0.0375, 20.0) - 1.24) < 0.13
    assert abs(polar_threshold_deg(0.23, 20.0) - 3.06) < 0.3


def test_grey_zone_flips_decision():
    """Δθ = 2°: legacy-порог сказал бы "standard", физический — "polar"."""
    use_polar, *_ = should_use_polar_reformat(
        omega=math.radians(2.0) / 16.0, pri=1.0, num_pulses=16,
        wavelength=0.0375, target_size_los=20.0)
    assert use_polar is True


def test_legacy_fallback():
    """Без wavelength/size — фолбэк 3°, source="legacy", поведение как раньше."""
    use_polar, _, delta_theta_deg, threshold_used_deg, threshold_source = (
        should_use_polar_reformat(
            omega=math.radians(2.0) / 16.0, pri=1.0, num_pulses=16)
    )
    assert threshold_used_deg == 3.0
    assert threshold_source == "legacy"
    assert delta_theta_deg == 2.0
    assert use_polar is False  # Δθ = 2° < legacy 3°


def test_explicit_threshold_deg_positional_still_supported():
    """Старый позиционный вызов threshold_deg не сломан (обратная совместимость)."""
    use_polar, _, _, threshold_used_deg, threshold_source = (
        should_use_polar_reformat(0.1, 1.0, 16, 3.0)
    )
    assert threshold_source == "legacy"
    assert threshold_used_deg == 3.0
    assert use_polar is True  # Δθ = 91.7° ≥ 3°


def test_physical_threshold_invalid_inputs():
    """Некорректные входы (None, ≤ 0) → None (сигнал для фолбэка)."""
    assert polar_threshold_deg(None, 20.0) is None
    assert polar_threshold_deg(0.0375, None) is None
    assert polar_threshold_deg(0.0, 20.0) is None
    assert polar_threshold_deg(0.0375, -5.0) is None


def test_warning_emitted_grey_zone_standard():
    """Прессет ω = 2°/16 импульсов, метод «стандартный»: GUI-предупреждение."""
    pytest.importorskip("PyQt5")
    from processing.data_processor import DataProcessor

    proc = DataProcessor(
        filename="matrices/matrices_solarB_high.pkl",
        method="стандартный", f_c=8.0, Xmax=20.0, Ymax=20.0, spectr_w=0.5,
        ang=2.0, range_m=500000.0, nifft_size=256,
        V=0.0, alpha=0.0, omega=math.radians(2.0) / 16.0,
        pri=1.0, num_pulses=16,
    )
    messages = []
    proc.progress_message.connect(messages.append)
    proc.compute_isar_notified(display_mode="dB")

    warns = [m for m in messages if m.startswith("WARN: Δθ")]
    assert len(warns) == 1, messages
    assert "физический порог 1.24°" in warns[0]  # X-band, L_LOS=20 м
    assert "legacy-порог 3.00°" in warns[0]      # legacy 3° тоже выводится


def test_no_warning_for_polar_method():
    """Метод «с полярным переформатированием» предупреждение не эмитит."""
    pytest.importorskip("PyQt5")
    from processing.data_processor import DataProcessor

    proc = DataProcessor(
        filename="matrices/matrices_solarB_high.pkl",
        method="с полярным переформатированием",
        f_c=8.0, Xmax=20.0, Ymax=20.0, spectr_w=0.5,
        ang=2.0, range_m=500000.0, nifft_size=256,
        V=0.0, alpha=0.0, omega=math.radians(2.0) / 16.0,
        pri=1.0, num_pulses=16,
    )
    messages = []
    proc.progress_message.connect(messages.append)
    proc.compute_isar_notified(display_mode="dB")

    assert not any(m.startswith("WARN: Δθ") for m in messages)


def test_no_warning_below_physical_threshold():
    """Δθ = 1° < физический порог 1.24°: предупреждения нет."""
    pytest.importorskip("PyQt5")
    from processing.data_processor import DataProcessor

    proc = DataProcessor(
        filename="matrices/matrices_solarB_high.pkl",
        method="стандартный", f_c=8.0, Xmax=20.0, Ymax=20.0, spectr_w=0.5,
        ang=1.0, range_m=500000.0, nifft_size=256,
        V=0.0, alpha=0.0, omega=math.radians(1.0) / 16.0,
        pri=1.0, num_pulses=16,
    )
    messages = []
    proc.progress_message.connect(messages.append)
    proc.compute_isar_notified(display_mode="dB")

    WARN = [m for m in messages if m.startswith("WARN")]
    assert WARN == [], WARN