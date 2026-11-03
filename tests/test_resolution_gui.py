"""Тесты GUI инверсии параметризации (задача RP3).

Приёмка (план RP §3 RP3):
    - секция «Разрешение и обзор» на месте, радио «Разрешения (реком.)»
      выбрано по умолчанию (cfg.param_mode='разрешения');
    - переключение режима меняет enable/disable полей (derived — не ввод,
      урок F8);
    - живая деривация: Δr/Δx → read-only вывод B/Δθ/ω/обзор;
    - _calc_single_rli передаёт res_* в DataProcessor (заглушкой).
config.save() замокан (monkeypatch) — тесты не трогают реальный config.json.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from config import AppConfig


def _qapp():
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture()
def win(monkeypatch):
    app = _qapp()
    monkeypatch.setattr(AppConfig, "save", lambda *a, **k: None)
    # тесты не зависят от реального config.json (может быть legacy v2.5.0)
    monkeypatch.setattr(
        AppConfig, "load", classmethod(lambda cls, path=None: cls()))
    from ui.main_window import MainWindow

    w = MainWindow()
    yield w
    w.close()
    app.processEvents()


# ------------------------------------------------ 1. smoke: виджеты, режим
def test_resolution_section_widgets_and_mode(win):
    assert hasattr(win, "rb_res") and hasattr(win, "rb_survey")
    assert win.rb_res.isChecked()          # cfg.param_mode default «разрешения»
    assert not win.rb_survey.isChecked()
    # ввод разрешений активен
    assert win.le_res_range[1].isEnabled()
    assert win.le_res_azimuth[1].isEnabled()
    # derived-поля отключены (честные derived, урок F8)
    for pair in (win.le_survey_length, win.le_survey_width,
                 win.le_spectrum, win.le_omega):
        assert not pair[1].isEnabled(), pair
    # живая деривация уже отработала на дефолтах (Δr=0.30/Δx=1.34)
    text = win.lbl_derived.text()
    assert "B =" in text and "ГГц" in text
    assert "Δθ" in text and "ω" in text
    assert "40" in text                    # авто-обзор (пол 40 м)


def test_derived_labels_values_demo1(win):
    """Ввод demo-1 (N=256, PRI=0.05): B=0.5 ГГц, Nf=134, ω = Δθ/(N·PRI)."""
    import math
    win.le_pri[1].setText("0.05")
    win.le_num_pulses[1].setText("256")
    _qapp().processEvents()
    text = win.lbl_derived.text()
    assert "0.500 ГГц" in text             # B = c/(2·0.30)
    assert "0.80°" in text                 # Δθ снэп к сетке 0.05°
    assert win._recon_plan is not None
    assert win._recon_plan.Nf == 134
    assert win._recon_plan.omega_rad_s == math.radians(0.8) / (256 * 0.05)


# ------------------------------------------------ 2. переключение режима
def test_switch_to_survey_mode_toggles_fields(win):
    win.rb_survey.setChecked(True)
    app = _qapp()
    app.processEvents()
    assert win.cfg.param_mode == "обзор"   # записан в конфиг через collect
    assert win.cfg.param_mode != "разрешения"
    # legacy-ввод снова активен
    for pair in (win.le_survey_length, win.le_survey_width,
                 win.le_spectrum, win.le_omega):
        assert pair[1].isEnabled(), pair
    # поля разрешений отключены
    assert not win.le_res_range[1].isEnabled()
    assert win.lbl_derived.text() == ""    # деривация не в legacy-режиме
    assert win._recon_plan is None


def test_switch_back_to_resolution_mode(win):
    win.rb_survey.setChecked(True)
    _qapp().processEvents()
    win.rb_res.setChecked(True)
    _qapp().processEvents()
    assert win.cfg.param_mode == "разрешения"
    assert win.le_res_range[1].isEnabled()
    assert not win.le_spectrum[1].isEnabled()
    assert "B =" in win.lbl_derived.text()


# ------------------------------------------------ 3. живая деривация
def test_live_derivation_updates_on_input(win):
    win.le_res_range[1].setText("0.05")
    _qapp().processEvents()
    # B = c/(2·0.05) = 3.000 ГГц
    assert "3.000 ГГц" in win.lbl_derived.text()
    assert win._recon_plan.Nf > 134        # обзор вырос? Nf = ceil(2B·40/c)
    assert win.cfg.res_range_m == 0.05


def test_derivation_warning_shown(win):
    win.le_res_azimuth[1].setText("0.10")  # очень мелкое Δx → C1a warning
    _qapp().processEvents()
    assert "предупреждения" in win.lbl_derived.text()


# ------------------------------------------------ 4. сигнал в таблицу
def test_signal_tab_theta_row():
    """Вкладка «Сигнал»: строка «Δθ (полный угол)» добавлена в обоих режимах."""
    app = _qapp()
    monkey_save = AppConfig.save
    AppConfig.save = lambda *a, **k: None
    try:
        from ui.main_window import MainWindow

        w = MainWindow()
        Nf, Np = 134, 256
        f_r = np.linspace(8e9 - 0.25e9, 8e9 + 0.25e9, Nf)
        E = np.zeros((Nf, Np), dtype=complex)
        for mode in ("разрешения", "обзор"):
            w.cfg.param_mode = mode
            w.processor = None
            w._update_signal_tab(E, f_r, Np)
            names = [w.signal_table.item(r, 0).text()
                     for r in range(w.signal_table.rowCount())
                     if w.signal_table.item(r, 0)]
            assert "Δθ (полный угол)" in names, (mode, names)
        w.close()
        app.processEvents()
    finally:
        AppConfig.save = monkey_save


# ------------------------------------------------ 5. проброс в DataProcessor
def test_calc_rli_passes_res_kwargs(win, monkeypatch):
    """«Разрешения»: DataProcessor получает res_range_m/res_azimuth_m."""
    from PyQt5.QtCore import QObject, pyqtSignal

    class _StubProc(QObject):
        range_profiles_ready = pyqtSignal(object, object)
        mocomp_envelope_ready = pyqtSignal(object, int)
        mocomp_phase_ready = pyqtSignal(object)
        mocomp_before_after = pyqtSignal(object, object, object)
        isar_ready = pyqtSignal(object, object, object)
        signal_data_ready = pyqtSignal(object, object)
        tf_ready = pyqtSignal(object)
        progress_message = pyqtSignal(str)
        error_occurred = pyqtSignal(str)
        finished = pyqtSignal()

        def __init__(self, **kw):
            super().__init__()
            self.kw = kw

        def start(self):
            pass

    captured = {}
    monkeypatch.setattr("ui.main_window.DataProcessor", _StubProc)
    win._calc_single_rli()

    captured = win.processor.kw if isinstance(win.processor, _StubProc) else None
    assert captured is not None, "заглушка не установлена"
    # cfg по умолчанию — режим «разрешения»
    assert captured["res_range_m"] == 0.30
    assert captured["res_azimuth_m"] == 1.34
    # legacy-аргументы по-прежнему передаются (DataProcessor сам подменяет)
    assert captured["spectr_w"] == win.cfg.spectrum_width

    win.rb_survey.setChecked(True)
    _qapp().processEvents()
    win._calc_single_rli()
    assert isinstance(win.processor, _StubProc)
    assert win.processor.kw["res_range_m"] is None      # legacy → None
    assert win.processor.kw["res_azimuth_m"] is None
