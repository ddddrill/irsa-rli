"""Тесты config.py для инверсии параметризации (задача RP2).

Приёмка: новый конфиг → save → load → поля на месте; конфиг старой
версии (v2.5.0, без param_mode) → param_mode="обзор" (legacy-режим).
"""
import json

from config import AppConfig


def test_resolution_roundtrip_save_load(tmp_path):
    """Новый конфиг: param_mode/res_* переживают save→load без потерь."""
    cfg = AppConfig()
    cfg.param_mode = "разрешения"
    cfg.res_range_m = 0.15
    cfg.res_azimuth_m = 0.72

    p = tmp_path / "config.json"
    cfg.save(str(p))

    loaded = AppConfig.load(str(p))
    assert loaded.param_mode == "разрешения"
    assert loaded.res_range_m == 0.15
    assert loaded.res_azimuth_m == 0.72
    # legacy-поля сохранены в обоих направлениях (не удалены)
    for name in ("survey_length", "survey_width", "spectrum_width", "omega"):
        assert hasattr(loaded, name)
    assert loaded.survey_length == 40.0


def test_legacy_config_loads_in_survey_mode(tmp_path):
    """Старый config.json без param_mode → param_mode='обзор'."""
    legacy = {
        "method": "стандартный",
        "center_frequency": 8.0,
        "spectrum_width": 0.4,
        "survey_length": 60.0,
        "survey_width": 50.0,
        "omega": 1.745e-4,
        "num_pulses": 256,
    }
    p = tmp_path / "config.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(legacy, f, ensure_ascii=False)

    cfg = AppConfig.load(str(p))
    assert cfg.param_mode == "обзор"
    assert cfg.spectrum_width == 0.4
    assert cfg.survey_length == 60.0
    assert cfg.survey_width == 50.0
    assert cfg.omega == 1.745e-4
    # новые поля — дефолтные (не путать с legacy-вводом)
    assert cfg.res_range_m == 0.30
    assert cfg.res_azimuth_m == 1.34


def test_missing_file_defaults_to_resolution_mode():
    """Отсутствующий config.json → дефолты, режим «разрешения»."""
    cfg = AppConfig.load("нет_такого_файла_config.json")
    assert cfg.param_mode == "разрешения"
    assert cfg.res_range_m == 0.30 and cfg.res_azimuth_m == 1.34
