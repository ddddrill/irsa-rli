"""Tests for processing/cross_range_scaling.py (Task R3).

Mu estimation, synthetic full-regression, simulator e2e, fallback on noise,
bitwise identity when off.
"""
import os, math, tempfile

import numpy as np
import pytest

SPEED_OF_LIGHT = 3e8


# ── helpers ──────────────────────────────────────────────────────────────


def _chirp(mu, N=64, fs=1.0):
    t = np.arange(N) / fs
    return np.exp(1j * np.pi * mu * t ** 2)


# ── 1. μ estimation on synthetic chirps ──────────────────────────────────

class TestMuEstimation:
    N = 64
    fs = 1.0

    @pytest.mark.parametrize("mu", [0.01, 0.1, 0.25])
    def test_phasefit(self, mu):
        from processing.cross_range_scaling import estimate_chirp_rate
        s = _chirp(mu, self.N, self.fs)
        mu_hat, q = estimate_chirp_rate(s, self.fs, method="phasefit")
        assert abs(mu_hat - mu) / mu < 0.05, f"phasefit μ̂={mu_hat:.6e}, μ={mu:.6e}"
        assert 0.0 <= q <= 1.0 + 1e-10

    @pytest.mark.parametrize("mu", [0.01, 0.1, 0.25])
    def test_lpft(self, mu):
        from processing.cross_range_scaling import estimate_chirp_rate
        s = _chirp(mu, self.N, self.fs)
        mu_hat, q = estimate_chirp_rate(s, self.fs, method="lpft")
        assert abs(mu_hat - mu) / mu < 0.05, f"lpft μ̂={mu_hat:.6e}, μ={mu:.6e}"
        assert q > 0

    def test_lpft_extended_domain(self):
        # |μ|·dt² = 0.5 — граница однозначности: phasefit алиасится, lpft
        # корректно работает с корректной стартовой точкой сетки.
        from processing.cross_range_scaling import estimate_chirp_rate
        mu = 0.5
        s = _chirp(mu, self.N, self.fs)
        mu_hat, q = estimate_chirp_rate(s, self.fs, method="lpft", mu_init=0.4)
        assert abs(mu_hat - mu) / mu < 0.05, f"lpft(ext) μ̂={mu_hat:.6e}, μ={mu:.6e}"


# ── 2. Full synthetic regression (≥6 scatterers) ────────────────────────


def _build_synth_image(rs, us, om, f_c, N=64, M=256, fs=1.0):
    rs, us = np.asarray(rs, float), np.asarray(us, float)
    t = np.arange(N) / fs
    ra = np.linspace(-(np.max(np.abs(rs)) + 3), np.max(np.abs(rs)) + 3, M)
    da = (np.arange(N) - N // 2) * (fs / N)
    I = np.zeros((M, N), dtype=complex)
    mu_k = (2.0 * f_c / SPEED_OF_LIGHT) * rs * om ** 2
    fd_k = (2.0 * f_c / SPEED_OF_LIGHT) * om * us
    for k in range(len(rs)):
        yk = int(np.argmin(np.abs(ra - rs[k])))
        dk = int(np.argmin(np.abs(da - fd_k[k])))
        sig = np.exp(1j * (2 * np.pi * fd_k[k] * t + np.pi * mu_k[k] * t ** 2))
        I[yk, :] += np.fft.fftshift(np.fft.fft(sig))
    return I, ra, da


class TestSyntheticRegression:
    def test_ratio_within_tolerance(self):
        from processing.cross_range_scaling import cross_range_scale
        f_c = 10e9
        om = 0.0012
        rs = [-9, -6, -3, 0, 3, 6, 9]
        us = [-3, -2, -1, 0, 1, 2, 3]
        I, ra, da = _build_synth_image(rs, us, om, f_c)
        cr = cross_range_scale(I, ra, da, f_c, 1.0, method="phasefit")
        assert not cr["fallback"], f"unexpected fallback; r2={cr['r2']}"
        assert cr["n_used"] >= 3
        ratio = cr["omega_eff"] / om
        assert abs(ratio - 1) < 0.10, f"Ω̂/Ω={ratio:.4f}"
        assert cr["r2"] > 0.85, f"r2={cr['r2']:.3f}"


# ── 3. Simulator end-to-end ─────────────────────────────────────────────


def _e2e_pipeline(N=64, spec_w=0.5e9, rk=None, uk=None, f_c=8e9, ang=9.0,
                  pri=1.0, omega_true=None, snr_db=None, method="phasefit"):
    from models.radar import Radar
    from models.target import Target
    from simulation.raw_generator import generate_raw_matrix, SimEffects
    from processing.range_compress import range_compress
    from processing.mocomp import mocomp
    from processing.azimuth_compress import azimuth_compress
    from processing.cross_range_scaling import cross_range_scale

    wt = omega_true if omega_true is not None else math.radians(ang) / N
    radar = Radar(c=SPEED_OF_LIGHT, f_c=f_c, Xmax=60.0, Ymax=60.0,
                  spectr_w=spec_w, ph_c=0.0, ang=ang, nifft_size=1024)
    if snr_db is not None:
        radar.snr_db = snr_db
    xs = np.asarray(rk, float)
    ys = np.asarray(uk[:len(rk)], float)
    ints = np.linspace(1, 5, len(rk))
    sp = os.path.join(tempfile.gettempdir(), "r3_test_e2e.npy")
    np.save(sp, np.array([[ints, xs, ys]], dtype=object), allow_pickle=True)
    tg = Target(sp, pri=pri, num_pulses=N, R0=500000.0, V=0.0, alpha=0.0, omega=wt)
    E, f_r = generate_raw_matrix(radar, tg, effects=SimEffects())
    P, Pa, ra, dr = range_compress(E, f_r, R0=500000.0, window="none", window_param=None)
    Pc, s, rb, pe = mocomp(P, ra)
    I, az, da = azimuth_compress(Pc, wt, f_c, pri=pri, window="none", window_param=None)
    cr = cross_range_scale(I, ra, da, f_c, 1.0 / pri, method=method)
    return cr, wt


class TestSimulatorEndToEnd:
    RK = [-4.4, -2.2, 0, 2.2, 4.4]
    UK = [-2, -1, 0, 1, 2]

    def test_ratio_within_tolerance(self):
        cr, wt = _e2e_pipeline(N=64, rk=self.RK, uk=self.UK)
        assert not cr["fallback"], f"unexpected fallback; r2={cr['r2']}"
        assert cr["n_used"] >= 3
        ratio = cr["omega_eff"] / wt
        assert abs(ratio - 1) < 0.10, f"Ω̂/ω={ratio:.4f}"
        assert cr["r2"] > 0.85, f"r2={cr['r2']:.3f}"


# ── 4. Fallback on noise ────────────────────────────────────────────────


class TestFallbackOnNoise:
    RK = [-4.4, -2.2, 0, 2.2, 4.4]
    UK = [-2, -1, 0, 1, 2]

    def test_fallback_or_degraded(self):
        np.random.seed(0)   # add_complex_awgn использует глобальный RNG
        cr, wt = _e2e_pipeline(N=64, rk=self.RK, uk=self.UK, snr_db=-5.0)
        if cr["fallback"]:
            return  # acceptable
        ratio = cr["omega_eff"] / wt
        assert abs(ratio - 1) < 0.30, f"degraded ratio={ratio:.3f}"


# ── 5. Bitwise identity when off ────────────────────────────────────────


class TestBitwiseOff:
    def test_off_pipeline_matches_default(self):
        pytest.importorskip("PyQt5")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from processing.data_processor import DataProcessor

        common = dict(
            filename="matrices/matrices_solarB_high.pkl",
            method="стандартный",
            f_c=8.0, Xmax=20.0, Ymax=20.0, spectr_w=0.5,
            ang=9.0, range_m=500000.0, nifft_size=256,
            V=0.0, alpha=0.0, omega=0.004, pri=0.01,
            num_pulses=128, use_mocomp=True, window="none",
        )
        proc_def = DataProcessor(**common)
        proc_off = DataProcessor(**common, cross_range_scaling="off")

        E, f_r, target = proc_def.generate_raw_data()

        amp_results = []
        def on_isar(msg):
            amp_results.append(msg)
        proc_def.isar_ready.connect(on_isar)

        from processing.azimuth_compress import azimuth_compress, compute_amplitude
        from processing.mocomp import mocomp
        from processing.range_compress import range_compress

        I1, amp1, ra1, az1, _ = proc_def._compute_isar_standard(E, f_r, target, "dB")
        I2, amp2, ra2, az2, _ = proc_off._compute_isar_standard(E, f_r, target, "dB")
        assert np.array_equal(amp1, amp2), "amplitude differs when cross_range_scaling=off"
        assert np.array_equal(I1, I2), "complex image differs when cross_range_scaling=off"
