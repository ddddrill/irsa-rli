"""Тесты задачи R5b: GUI-вкладка «Сигнатуры» и сигнал tf_ready (TF, гл. 2/5).

Инвариант (§7.6): при tf_enabled=False (дефолт) конвейер битово идентичен
прежнему, сигнал tf_ready НЕ эмитится; при tf_enabled=True эмитится dict
с корректной RID-структурой (Q_mag, frame_idx, approximate, top_bins, ...).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from processing.tf_analysis import rid_cube, rid_frame, rid_marginal


def _make_processor(**kwargs):
    from processing.data_processor import DataProcessor

    opts = dict(
        filename="matrices/matrices_solarB_high.pkl",
        method="стандартный",
        f_c=8.0,
        Xmax=20.0,
        Ymax=20.0,
        spectr_w=0.5,
        ang=9.0,
        range_m=500000.0,
        nifft_size=256,
        V=0.0,
        alpha=0.0,
        omega=0.004,
        pri=0.01,
        num_pulses=128,
        use_mocomp=True,
        window="none",
    )
    opts.update(kwargs)
    return DataProcessor(**opts)


def _run_standard(proc, E, f_r, target):
    return proc._compute_isar_standard(E, f_r, target, "dB")


# ------------------------------------------- 1. Битовая совместимость off
def test_tf_disabled_bitwise_identical():
    """§7.6: tf_enabled=False / без поля → выход битово равен, tf_ready не
    эмитится (главный инвариант обратной совместимости)."""
    from processing.azimuth_compress import azimuth_compress, compute_amplitude
    from processing.mocomp import mocomp
    from processing.range_compress import range_compress

    proc_default = _make_processor()                    # старое поле отсутствует
    proc_off = _make_processor(tf_enabled=False, tf_n_frames=32, tf_window="hamming")
    n_default, n_off = [], []
    proc_default.tf_ready.connect(n_default.append)
    proc_off.tf_ready.connect(n_off.append)

    E, f_r, target = proc_off.generate_raw_data()
    I1, amp1, ra1, az1, _ = _run_standard(proc_default, E, f_r, target)
    I2, amp2, ra2, az2, _ = _run_standard(proc_off, E, f_r, target)

    # Явный прежний конвейер (range -> mocomp -> azimuth) с теми же вызовами.
    P, _, range_axis, _ = range_compress(
        E, f_r, R0=proc_off.range_m, window=proc_off.range_window,
        window_param=proc_off.window_param,
    )
    P_proc, _, _, _ = mocomp(P, range_axis)
    I3, az3, _ = azimuth_compress(
        P_proc, proc_off.omega, proc_off.f_c, pri=proc_off.pri,
        window=proc_off.window, window_param=proc_off.window_param,
    )
    amp3 = compute_amplitude(I3, mode="dB")

    assert np.array_equal(I1, I2) and np.array_equal(I2, I3)
    assert np.array_equal(amp1, amp2) and np.array_equal(amp2, amp3)
    assert np.array_equal(ra1, ra2)
    assert np.array_equal(az1, az2) and np.array_equal(az2, az3)
    assert n_default == [] and n_off == [], "tf_ready не должен эмититься при off"


# --------------------------------------------- 2. tf_enabled=True -> куб
def test_tf_enabled_emits_cube():
    """§7.6: tf_ready эмитит dict с Q_mag ((M,F,T)), frame_idx длины
    tf_n_frames, top_bins >= 5, approximate и сигнатурами."""
    proc = _make_processor(tf_enabled=True, tf_n_frames=16, tf_window="hann")
    emitted = []
    proc.tf_ready.connect(emitted.append)

    E, f_r, target = proc.generate_raw_data()
    _run_standard(proc, E, f_r, target)

    assert len(emitted) == 1, "ровно один tf_ready за прогон"
    tf = emitted[0]
    Q = tf["Q_mag"]
    assert Q.ndim == 3 and Q.dtype == np.float32
    M, F, T = Q.shape
    assert M > 0 and F > 0 and T > 0
    assert len(tf["frame_idx"]) == proc.tf_n_frames == 16
    assert "approximate" in tf and isinstance(tf["approximate"], bool)
    assert len(tf["top_bins"]) >= 5
    assert len(set(tf["top_bins"])) == len(tf["top_bins"])
    assert tf["fs"] == pytest.approx(1.0 / proc.pri)
    for key in ("signatures", "total_echo", "rows"):
        assert key in tf
    for b in tf["top_bins"]:
        sig = tf["signatures"][str(b)]
        assert np.asarray(sig["S_db"]).shape[0] == F
    assert np.asarray(tf["rows"][str(tf["top_bins"][0])]).shape[0] == 128


# ------------------------------------------- 3. GUI smoke (offscreen)
def _qapp():
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_gui_smoke_signal_tab():
    """§7.6: вкладка «Сигнатуры» на позиции 4, все элементы на месте."""
    app = _qapp()
    from ui.main_window import MainWindow

    win = MainWindow()
    try:
        tabs = win.tabs
        assert tabs.count() == 6
        assert tabs.tabText(3) == "РЛИ"
        assert tabs.tabText(4) == "Сигнатуры"
        assert tabs.tabText(5) == "Поток кадров"
        for attr in ("tf_bin_combo", "tf_transform_combo",
                     "tf_frame_slider", "tf_marginal_check", "pg_tf"):
            assert hasattr(win, attr), attr
        assert win.tf_transform_combo.currentText() == "STFT"
        assert win.tf_bin_combo.count() == 1     # данных ещё нет
    finally:
        win.close()
        app.processEvents()


def _fake_tf_data(M=32, F=16, T=8, n_frames=4):
    rng = np.random.default_rng(3)
    Q = rng.random((M, F, T)).astype(np.float32)
    f = np.fft.fftshift(np.fft.fftfreq(F, d=0.01))
    t = np.arange(T) * 0.02
    frame_idx = np.linspace(0, T - 1, n_frames).astype(int)
    top = [M - 1, M - 2, M - 3, 0, 1]
    sigs = {
        str(b): {"f": f, "t": t, "S_db": -rng.random((F, T)) * 40}
        for b in top
    }
    return {
        "Q_mag": Q, "f": f, "t": t, "frame_idx": frame_idx,
        "approximate": False,
        "meta": {"nperseg": F, "hop": 1, "hop_eff": 1, "window": "hann",
                 "scale": 1.0, "bins_used": list(range(M)), "n_frames": n_frames},
        "top_bins": top,
        "signatures": sigs,
        "total_echo": {"f": f, "t": t,
                       "S_db": -rng.random((F, T)) * 40,
                       "S_db_sum": -rng.random((F, T)) * 40},
        "rows": {str(b): rng.normal(size=8) + 1j * rng.normal(size=8)
                 for b in top},
        "fs": 100.0, "n_frames": n_frames,
    }


def test_gui_updates_controls_and_redraw():
    """§7.6: после tf_ready комбо/слйдер наполняются, все режимы рисуются."""
    app = _qapp()
    from ui.main_window import MainWindow

    win = MainWindow()
    try:
        win._on_tf_ready(_fake_tf_data())
        assert win.tf_bin_combo.count() == 6        # «Суммарное эхо» + 5 бинов
        assert win.tf_frame_slider.maximum() == 3

        win._redraw_tf()                            # STFT + суммарное эхо
        assert win.tf_img.image.shape[0] == 16      # (F, T)

        win.tf_marginal_check.setChecked(True)      # маргинал RID
        win._redraw_tf()
        assert win.tf_img.image.shape == (32, 16)   # (M, F)

        win.tf_marginal_check.setChecked(False)
        win.tf_transform_combo.setCurrentIndex(1)   # WVD топ-бина
        win._redraw_tf()
        img = win.tf_img.image
        assert img.shape[0] == 8 and img.shape[1] == 8

        win.tf_frame_slider.setValue(2)             # RID-кадр куба
        win._redraw_tf()
        assert win.tf_img.image.shape == (32, 16)
    finally:
        win.close()
        app.processEvents()


# ------------------------------------------------- 4. rid_frame dB smoke
def test_rid_frame_db():
    """§7.6: rid_frame/rid_marginal на малых данных — dB-кадр нормирован к
    0 дБ, маргинал — энергия (>= 0), всё конечно."""
    rng = np.random.default_rng(5)
    P = (rng.normal(size=(4, 64)) + 1j * rng.normal(size=(4, 64)))
    rid = rid_cube(P, fs=1.0, nperseg=16, n_frames=8)
    frame = rid_frame(rid, 0)
    assert frame.shape == (4, 16)
    assert np.isclose(frame.max(), 0.0)             # нормировка 0 дБ
    assert np.isfinite(frame).all()
    marg = rid_marginal(rid)
    assert marg.shape == (4, 16)
    assert np.isfinite(marg).all()
    assert marg.min() >= -1e-6