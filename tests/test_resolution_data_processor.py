"""Тесты DataProcessor-шины инверсии параметризации (задача RP4).

Приёмка (план RP §3 RP4): инстанс с res_*-kwargs даёт БИТ-идентичную
E-матрицу прямому заданию производных spectr_w/Xmax/Ymax/omega.

Эталоны demo-1 (снэпы дают бит-равенство):
    Δr=0.30 → B ≡ 5e8 Гц; Δx=1.34 → Δθ ≡ radians(0.80) → ω бит-равно
    radians(0.8)/(256·0.05); Xmax=Ymax=40 → Nf=134.

Важно: предупреждения деривации НЕ эмитятся в __init__ (сигналы коннектятся
После) — эмит отложен в compute_isar_notified/_compute_and_save; тест
инспектирует proc._recon_warnings.
"""
import numpy as np
import pytest

SOLARB = "matrices/matrices_solarB_high.pkl"


def _kwargs(res_mode, dr=0.30, da=1.34):
    """Базовый конфиг demo-1 в двух эквивалентных формах."""
    base = dict(
        filename=SOLARB, method="стандартный",
        f_c=8.0, spectr_w=0.5, ang=9.0,
        Xmax=40.0, Ymax=40.0,
        range_m=500_000.0, nifft_size=256,
        V=0.0, alpha=0.0, pri=0.05, num_pulses=256,
        use_mocomp=False, window="none",
    )
    if res_mode == "res":
        base["res_range_m"] = dr
        base["res_azimuth_m"] = da
        base["omega"] = 0.0
    elif res_mode == "legacy":
        import math

        from simulation.resolution import derive_recon
        plan = derive_recon(
            f_c_hz=8e9, dr_m=dr, da_m=da, pri=0.05, num_pulses=256,
            R0_m=500_000.0, V=0.0, alpha=0.0, filename=SOLARB)
        base["spectr_w"] = plan.spectr_w_hz / 1e9     # ГГц → *1e9 внутри
        base["omega"] = plan.omega_rad_s
    else:
        base["spectr_w"] = 0.5
        base["omega"] = 0.0
    return base


def test_res_kwargs_bitwise_equal_explicit_derivation():
    """RP4.4: res_*-kwargs ≡ прямое задание производных (бит-равенство E)."""
    from processing.data_processor import DataProcessor

    proc_res = DataProcessor(**_kwargs("res"))
    proc_exp = DataProcessor(**_kwargs("legacy"))

    # производные совпали побитово:
    assert proc_res.spectr_w == proc_exp.spectr_w
    assert proc_res.omega == proc_exp.omega
    assert proc_res.Xmax == proc_exp.Xmax == 40.0
    assert proc_res.Ymax == proc_exp.Ymax == 40.0
    assert proc_res.radar.Nf == proc_exp.radar.Nf == 134
    assert proc_res.radar.spectr_w == proc_exp.radar.spectr_w

    E_res, f_r1, _ = proc_res.generate_raw_data()
    E_exp, f_r2, _ = proc_exp.generate_raw_data()
    assert np.array_equal(E_res, E_exp), "E должна быть бит-равной"
    assert np.array_equal(f_r1, f_r2)


def test_res_kwargs_override_config_values():
    """res_*-kwargs подавляют вступившие spectr_w/Xmax/omega (человеку-вводу)."""
    from processing.data_processor import DataProcessor

    kw = _kwargs("res")
    kw.update(spectr_w=9.99, Xmax=20.0, omega=777.0)   # ввод-мусор
    proc = DataProcessor(**kw)
    assert proc.spectr_w == 5e8                        # не 9.99e9
    assert proc.Xmax == 40.0                           # не 20
    assert proc.omega == pytest.approx(
        __import__("math").radians(0.8) / (256 * 0.05), rel=1e-12)
    # подменный omega=777 не применился
    assert proc.omega != 777.0


def test_none_kwargs_keeps_legacy_branch_verbatim():
    """Без res_*-kwargs — прежняя ветка: spectr_w*1e9, auto-ω из ang."""
    from processing.data_processor import DataProcessor

    proc = DataProcessor(**_kwargs("plain"))
    assert proc.spectr_w == 0.5 * 1e9
    assert proc.Xmax == 40.0
    # auto-ω: angle за апертуру = beam_width (ветка data_processor:140)
    import math
    assert proc.omega == math.radians(9.0) / (256 * 0.05)
    assert proc._recon_plan is None
    assert proc._recon_warnings == []


def test_recon_warnings_deferred_not_emitted_in_init():
    """Предупреждения деривации не эмитятся в __init__ (сигналы later)."""
    from processing.data_processor import DataProcessor

    messages = []
    kw = _kwargs("res", dr=0.008)                     # Δr=8 мм → Nf=5000 cap
    proc = DataProcessor(**kw)
    assert proc._recon_warnings, "ожидался cap-warning для Δr=8 мм"
    assert any("Nf" in w for w in proc._recon_warnings)

    proc.progress_message.connect(messages.append)
    proc.compute_isar_notified()                      # эмит ИЗ точки расчёта
    assert any("WARN (RP)" in m for m in messages), messages[:5]
    # расчёт не блокируется: isar_ready должен прийти
    assert len(messages) > 0


# --------------------------------------------- RP6: фиксы прогона -refmsg
def test_memory_pre_guard_raises_config_error():
    """RP6: E > 1.5 ГБ → ValueError с рецептом ДО аллокации (нет MemoryError)."""
    from processing.data_processor import DataProcessor

    proc = DataProcessor(
        filename="matrices/matrices_solarB_high.pkl", method="стандартный",
        f_c=8.0, Xmax=40.0, Ymax=40.0, spectr_w=0.5, ang=9.0,
        range_m=500_000.0, nifft_size=256, V=0.0, alpha=0.0, omega=0.0,
        pri=0.05, num_pulses=400, use_mocomp=False, window="none",
        res_range_m=0.008, res_azimuth_m=1.34)        # Nf=5000 × N=40000
    proc.num_pulses = 40000                           # после деривации
    emitted = []
    proc.progress_message.connect(emitted.append)
    with pytest.raises(ValueError, match="ОШИБКА КОНФИГУРАЦИИ"):
        proc.generate_raw_data()
    assert any("превышает лимит" in m for m in emitted)
