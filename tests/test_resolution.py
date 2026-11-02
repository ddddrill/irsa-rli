"""Тесты инверсии параметризации «разрешения → параметры» (задача RP1).

Эталоны (план RP §4):
    demo-1: Δr=0.30 м → B ≡ 0.5 ГГц (бит-равенство legacy 0.5e9);
            Δx=1.34 м → Δθ снэп к 0.80° → ω бит-равно
            math.radians(0.8)/(256·0.05) фикстуры conftest;
            Xmax/Ymax = 40 м (пол дефолта GUI) → Nf = 134 (как demo-1).
    demo-3: Δx = λ/(2·radians(3.0)) = 0.3581 м, V=10, PRI=5e-4 → те же
            значения очзора, бит-равенство legacy-конфигурации.
Канонические Δθ демо лежат на сетке 0.05°, поэтому снэп НЕ меняет их.
"""
import math
import time

import numpy as np
import pytest

from simulation.resolution import (
    ReconPlan,
    derive_recon,
    target_geometry_from_pkl,
)

SPEED_OF_LIGHT = 3e8
SOLARB = "matrices/matrices_solarB_high.pkl"
ICEST2 = "matrices/matrices_ICEsat2_high.pkl"

FC = 8e9
LAM = SPEED_OF_LIGHT / FC

# demo-1: Δθ = 0.8° == radians(0.8); фактическое Δx = λ/(2·Δθ)
DA_DEMO1 = 1.34                                    # → 0.801° → снэп 0.80°
DA_DEMO1_EXACT = LAM / (2.0 * math.radians(0.8))   # 1.3429 м (без снэпа)
DR_DEMO1 = 0.30
N_DEMO1, PRI_DEMO1 = 256, 0.05

# demo-3: Δθ = 3°, V=10, PRI=5e-4 (ICEsat2_high)
DA_DEMO3_EXACT = LAM / (2.0 * math.radians(3.0))   # 0.3581 м
PRI_DEMO3, V_DEMO3 = 5e-4, 10.0


def test_canonical_demo1_derivation():
    """Этан demo-1: B/Δθ/ω/Xmax/Nf бит-равны legacy-конфигурации."""
    plan = derive_recon(
        f_c_hz=FC, dr_m=DR_DEMO1, da_m=DA_DEMO1, pri=PRI_DEMO1,
        num_pulses=N_DEMO1, R0_m=500_000.0, V=0.0, alpha=0.0,
        filename=SOLARB)

    assert plan.spectr_w_hz == 0.5 * 1e9              # бит-равенство legacy
    # Δθ снэпан к 0.80° (запрошено 0.801° из Δx=1.34)
    assert plan.delta_theta_rad == math.radians(0.8)  # бит-равенство conftest
    assert plan.omega_rad_s == math.radians(0.8) / (256 * 0.05)
    assert plan.dr_effective_m == pytest.approx(DR_DEMO1, rel=1e-4)
    assert plan.da_effective_m == pytest.approx(DA_DEMO1_EXACT, rel=1e-3)
    assert plan.Xmax_m == 40.0 and plan.Ymax_m == 40.0
    assert plan.Nf == 134                             # как в demo-1/фикстурах
    assert plan.fd_diff_hz > 0
    assert plan.fd_diff_hz < 0.5 / PRI_DEMO1          # C1a выполнен
    assert not any("N_min" in w for w in plan.warnings)
    assert plan.dR_max_m == pytest.approx(12.05, abs=0.2)


def test_roundtrip_dx_to_dtheta():
    """Δx → Δθ → ω → Δx: на канонической сетке возврат бит-точный."""
    da = LAM / (2.0 * math.radians(1.0))              # Δθ = 1.00°, на сетке
    plan = derive_recon(
        f_c_hz=FC, dr_m=DR_DEMO1, da_m=da, pri=PRI_DEMO1,
        num_pulses=N_DEMO1, R0_m=500_000.0, filename=SOLARB)
    assert plan.delta_theta_rad == math.radians(1.0)  # снэп не сдвинул
    assert plan.da_effective_m == da                  # бит-равенство
    assert plan.dr_effective_m == pytest.approx(DR_DEMO1, rel=1e-4)


def test_c1a_warning_with_n_min():
    """«Мелкое» Δx при малом N → warning с N_min (план §4, сценарий 2)."""
    plan = derive_recon(
        f_c_hz=FC, dr_m=DR_DEMO1, da_m=0.20,          # Δθ≈5.35° после снэпа
        pri=PRI_DEMO1, num_pulses=64, R0_m=500_000.0,
        filename=SOLARB)
    nyq = 0.5 / PRI_DEMO1
    assert plan.fd_diff_hz >= nyq                     # C1a нарушен
    n_min_msg = [w for w in plan.warnings if "N_min" in w]
    assert n_min_msg, plan.warnings
    assert str(plan.N_min_c1a) in n_min_msg[0]
    assert plan.N_min_c1a > 64                        # рецепт: больше N
    # масштабная проверка fd ∝ 1/N: при N = N_min диф. доплер ≈ Найквист
    # (замах сверху только из-за ceil внутри N_min)
    fd_at_nmin = plan.fd_diff_hz * 64 / plan.N_min_c1a
    assert fd_at_nmin <= nyq, (fd_at_nmin, nyq)
    assert fd_at_nmin > nyq * (1.0 - 1.0 / plan.N_min_c1a)


def test_translation_demo3_derivation():
    """demo-3: Δx=λ/(2·3°), V=10, PRI=5e-4 → Xmax=40, Nf=134, без C1a."""
    plan = derive_recon(
        f_c_hz=FC, dr_m=DR_DEMO1, da_m=DA_DEMO3_EXACT, pri=PRI_DEMO3,
        num_pulses=N_DEMO1, R0_m=500_000.0, V=V_DEMO3, alpha=0.3,
        filename=ICEST2)
    assert plan.delta_theta_rad == math.radians(3.0)  # бит-равенство demo-3
    assert plan.spectr_w_hz == 0.5 * 1e9
    assert plan.Xmax_m == 40.0 and plan.Ymax_m == 40.0
    assert plan.Nf == 134
    # C1b (полный доплер с трансляцией): demo-3 запас 1.40 → нет warning
    assert not any("C1b" in w for w in plan.warnings), plan.warnings
    assert not any("N_min" in w for w in plan.warnings)
    assert plan.fd_diff_hz > 0
    # margin demo-3 (отчёт guard'а DOC): C1a 2.70× — пер-cross-check тут
    nyq = 0.5 / PRI_DEMO3                             # 1000 Гц
    assert nyq / plan.fd_diff_hz == pytest.approx(2.70, rel=0.1)


def test_nf_cap_warning():
    """Δr=8 мм → B=18.75 ГГц → Nf=5000 > 4096: cap-warning с размером."""

    plan = derive_recon(
        f_c_hz=FC, dr_m=0.008, da_m=1.0, pri=PRI_DEMO1,
        num_pulses=N_DEMO1, R0_m=500_000.0, filename=SOLARB)
    assert plan.Nf == 5000
    cap = [w for w in plan.warnings if "Nf" in w and "4096" in w]
    assert cap, plan.warnings
    assert "МБ" in cap[0]                             # бюджет памяти в сообщении


def test_translation_grows_survey_and_nf():
    """V ≠ 0: авто-Xmax растёт (дрейф V·T_obs) → Nf растёт (план §1.2.3)."""
    base = derive_recon(
        f_c_hz=FC, dr_m=0.10, da_m=DA_DEMO1_EXACT, pri=0.05, num_pulses=256,
        R0_m=500_000.0, V=0.0, alpha=0.0, filename=SOLARB)
    moving = derive_recon(
        f_c_hz=FC, dr_m=0.10, da_m=DA_DEMO1_EXACT, pri=0.05, num_pulses=256,
        R0_m=500_000.0, V=200.0, alpha=0.0, filename=SOLARB)
    assert moving.Xmax_m > base.Xmax_m                # дрейф 200·12.8=2560 м
    assert moving.Nf > base.Nf
    assert any("Nf" in w for w in moving.warnings)    # cap сработает
    # B одинаков (Δr неизменен) — только обзор вырос
    assert moving.spectr_w_hz == base.spectr_w_hz


def test_geometry_from_pkl_centrifugal_convention():
    """target_geometry_from_pkl: координаты центрированы, конечны."""
    geom = target_geometry_from_pkl(SOLARB)
    assert np.isfinite(geom["x0"]).all() and np.isfinite(geom["y0"]).all()
    assert abs(geom["x0"].mean()) < 1e-9              # центр масс в 0


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        derive_recon(f_c_hz=8e9, dr_m=0.0, da_m=1.0, pri=0.05,
                     num_pulses=256, R0_m=500_000.0, filename=SOLARB)
    with pytest.raises(ValueError):
        derive_recon(f_c_hz=8e9, dr_m=0.3, da_m=1.0, pri=0.05,
                     num_pulses=1, R0_m=500_000.0, filename=SOLARB)
    with pytest.raises(ValueError):
        derive_recon(f_c_hz=-8e9, dr_m=0.3, da_m=1.0, pri=0.05,
                     num_pulses=256, R0_m=500_000.0, filename=SOLARB)


def test_derivation_performance_budget():
    """Бюджет производительности: LRO_high (265 точек), N=1024.

    План RP1: ≤ 50 мс. Жёсткий assert — 0.2 с (защита деградации);
    измеренное число фиксируется в CHANGELOG.
    """
    t0 = time.perf_counter()
    plan = derive_recon(
        f_c_hz=FC, dr_m=DR_DEMO1, da_m=DA_DEMO1, pri=0.05, num_pulses=1024,
        R0_m=500_000.0, filename="matrices/matrices_LRO_high.pkl")
    elapsed = time.perf_counter() - t0
    assert plan.Nf == 134
    assert elapsed < 0.2, elapsed
    # измерение печатается для CHANGELOG
    print(f"\nderive_recon(LRO, N=1024): {elapsed * 1000:.1f} мс")
