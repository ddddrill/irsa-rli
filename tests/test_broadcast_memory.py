"""Тесты памяти broadcast-матрицы в генераторе сырых данных (P2-фикс).

Суть проблемы: генератор строит фазу как broadcast (Nf, N, I), где
I — число рассеивателей модели. При Δr=8 мм (Nf=5000), N=256, I=182
временные массивы phi/exp занимают ≈3.7 ГБ — в ~187 раз больше
финальной E (20 МБ) → OOM на машинах с 4 ГБ RAM, хотя memory-cap
проверяет только E.

Проверяем:
  1. чанкование по частотам НЕ меняет результат (бит-в-бит);
  2. пик памяти ограничен бюджетом (чанков > 1 при малом бюджете);
  3. слишком большой broadcast даёт честную ошибку конфигурации
     (не MemoryError) с рецептом — и она доходит до progress_message.
"""
import numpy as np
import pytest

SOLARB = "matrices/matrices_solarB_high.pkl"


def _processor():
    from processing.data_processor import DataProcessor

    return DataProcessor(
        filename=SOLARB, method="стандартный",
        f_c=8.0, spectr_w=0.5, ang=9.0,
        Xmax=40.0, Ymax=40.0, range_m=500_000.0, nifft_size=256,
        V=0.0, alpha=0.0, omega=0.0, pri=0.05, num_pulses=256,
        use_mocomp=False, window="none",
    )


def test_chunked_generation_is_bitwise_identical(monkeypatch):
    """Чанкование по осям частот не меняет ни одного бита E."""
    from simulation import raw_generator

    proc = _processor()
    E_full, f_full = proc.generate_raw_data()[:2]

    # бюджет меньше одной строки → гарантированно много чанков
    monkeypatch.setattr(raw_generator, "BROADCAST_CHUNK_BYTES", 1)
    proc2 = _processor()
    E_chunk, f_chunk = proc2.generate_raw_data()[:2]

    assert E_chunk.shape == E_full.shape
    assert np.array_equal(f_chunk, f_full)
    # бит-в-бит (не allclose!)
    assert np.array_equal(E_chunk.view(np.uint8), E_full.view(np.uint8))


def test_chunk_count_grows_when_budget_shrinks(monkeypatch):
    """Малый бюджет → чанков больше; результат тот же."""
    from simulation import raw_generator

    proc = _processor()
    E_ref, _ = proc.generate_raw_data()[:2]

    calls = {"n": 0}
    real_sum = np.sum

    def counting_sum(a, *args, **kw):
        calls["n"] += 1
        return real_sum(a, *args, **kw)

    monkeypatch.setattr(raw_generator, "BROADCAST_CHUNK_BYTES", 64 * 1024)
    monkeypatch.setattr(raw_generator.np, "sum", counting_sum)
    proc2 = _processor()
    E_chunk, _ = proc2.generate_raw_data()[:2]

    assert calls["n"] > 1, "ожидалось чанкование при малом бюджете"
    assert np.array_equal(E_chunk.view(np.uint8), E_ref.view(np.uint8))


def test_broadcast_work_limit_raises_config_error(monkeypatch):
    """Слишком большой broadcast → ValueError с рецептом (не OOM)."""
    from simulation import raw_generator

    monkeypatch.setattr(raw_generator, "BROADCAST_WORK_LIMIT_BYTES", 1024)
    proc = _processor()
    emitted = []
    proc.progress_message.connect(emitted.append)

    with pytest.raises(ValueError, match="broadcast-матрица"):
        proc.generate_raw_data()

    assert any("broadcast-матрица" in m for m in emitted), emitted[:3]
    assert any("Уменьшите" in m for m in emitted), emitted[:3]


def test_broadcast_limit_allows_normal_configs():
    """Дефолтный конфиг (Nf=134, N=256, I=182) проходит с запасом."""
    from simulation import raw_generator

    proc = _processor()
    E, f_r = proc.generate_raw_data()[:2]
    assert E.shape == (134, 256)

    # объём broadcast ≈ 100 МБ — существенно меньше лимита 8 ГБ
    I = len(proc._make_target()._load_scatterers()["intens"])
    work = 134 * 256 * I * 16
    assert work < raw_generator.BROADCAST_WORK_LIMIT_BYTES
