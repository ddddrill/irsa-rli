"""Тесты демо R8b «маневрирующая цель» (план §6.5).

Критерий приёмки: энтропия RID-кадра < энтропии RD-изображения минимум на
15 % — иначе параметры сценария слабые (план §6.5; подкрутить γ/T). Все
расчёты детерминированы: без шума (snr_db=inf) и без случайных эффектов,
SimEffects(seed=42) для явной фиксации seed. Физика: план §6.2.1–(6.2.6),
[TF, §5.4], [C&M, §3.2/(1.13)].

Измерения проводятся так же, как в демо-скрипте:
    - e_rd      — энтропия RD-изображения (|I|^2, сетка M×N);
    - e_rd_stft — энтропия RD на STFT-сетке (M×F, группировка |FFT|^2 —
                  конвенция R5a / demo_rid_imaging [TF, (5.18)]);
    - e_marginal— энтропия маргинала куба RID Σ_t |Q|^2 [TF, (5.18)];
    - e_frame   — энтропия отдельного RID-кадра в середине апертуры.
Все четыре — через существующий compute_image_entropy (polar_reformat §7.4).
"""
import os

import numpy as np

from examples import demo_maneuvering_target as demo


def test_r8b_demo_entropy_gap():
    """R8b: энтропия RID-кадра < энтропии RD-изображения минимум на 15 %.

    Критерий §6.5. Проверяем разрыв как по RD на полной сетке (M×N), так и
    на общей STFT-сетке (M×F) — в обоих случаях кадр «острее» минимум на
    15 % (физика: чирп (6.2.3)–(6.2.5) размазывает RD, окно T_w «замораживает»
    вращение, план §6.2.3).
    """
    m = demo.compute_maneuvering_metrics()
    assert m["e_frame"] < m["e_rd"]
    assert m["gap_pct"] >= 15.0, \
        f"разрыв RID-кадр vs RD = {m['gap_pct']:.1f}% < 15% (критерий §6.5)"
    assert m["gap_stftgrid_pct"] >= 15.0, \
        f"разрыв на STFT-сетке = {m['gap_stftgrid_pct']:.1f}% < 15%"
    assert np.isfinite(m["e_marginal"]) and m["e_marginal"] > 0.0


def test_r8b_demo_runs(tmp_path):
    """R8b: main()/run() демо завершается без исключений и создаёт PNG.

    Полный прогон скрипта (`python -m examples.demo_maneuvering_target`) —
    в run-фазе; здесь тот же путь run() с выходным файлом во временной папке.
    """
    out = tmp_path / "demo_maneuvering_target.png"
    m = demo.run(verbose=False, out_path=str(out))
    assert os.path.exists(str(out)), "скрипт не создал PNG"
    assert os.path.getsize(str(out)) > 0, "PNG пуст"
    assert m["gap_pct"] >= 15.0


def test_rid_frame_sharper_than_rd():
    """§6.2.5: RID-кадр — «мгновенный снимок»: доля энергии в топ-20 ячейках
    кадра (норм. на полную энергию кадра) выше, чем у RD на той же сетке."""
    m = demo.compute_maneuvering_metrics()
    conc_frame = m["top_energy_frac"][0]
    conc_rd = m["top_energy_frac"][1]
    assert conc_frame > 1.1 * conc_rd, \
        (f"концентрация кадра ({conc_frame:.3f}) не выше RD "
         f"({conc_rd:.3f}) в топ-20 ячейках")