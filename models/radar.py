import math
import cmath
import numpy as np


class Radar:
    """Параметры радара SFCW (Stepped Frequency Continuous Wave).

    Инкапсулирует все параметры сигнала и геометрии наблюдения,
    необходимые для формирования поля обратного рассеяния и РЛИ.
    """

    def __init__(self, c, f_c, Xmax, Ymax, spectr_w, ph_c, ang, nifft_size=1024, snr_db=np.inf):
        self.c = c
        self.f_c = f_c
        self.Xmax = Xmax
        self.Ymax = Ymax
        self.spectr_w = spectr_w
        self.ph_c = ph_c
        self.ang = ang
        self.nifft_size = nifft_size
        self.snr_db = snr_db

        self.k_c = (2 * math.pi * f_c) / self.c
        self.ph_w = math.radians(self.ang)

        self.resolution_x = self.c / (self.spectr_w * 2)
        self.resolution_y = (math.pi / self.k_c) / self.ph_w

        self.Nf = math.ceil((self.Xmax * self.spectr_w * 2) / self.c)
        self.Nph = math.ceil((2 * Ymax * self.ph_w * f_c) / self.c)

        self.df = self.spectr_w / self.Nf
        self.dph = self.ph_w / self.Nph

        self.cos = np.vectorize(math.cos)
        self.sin = np.vectorize(math.sin)
        self.exp = np.vectorize(cmath.exp)
        self.complex_v = np.vectorize(complex)
        self.floor = np.vectorize(math.floor)

    def vectorize_functions(self):
        return self.cos, self.sin, self.exp, self.complex_v, self.floor

    def base_img_params(self):
        """Параметры для формирования РЛИ стандартным методом."""
        self.Nf = math.ceil((self.Xmax * self.spectr_w * 2) / self.c)
        self.Nph = math.ceil((2 * self.Ymax * self.ph_w * self.f_c) / self.c)

        self.f_r = np.linspace(
            self.f_c - self.df * (self.Nf / 2),
            self.f_c + self.df * (self.Nf / 2) - self.df,
            self.Nf,
        )
        self.ph_r = np.linspace(
            self.ph_c - self.dph * (self.Nph / 2),
            self.ph_c + self.dph * (self.Nph / 2) - self.dph,
            self.Nph,
        )
        self.k_r = 2 * math.pi * self.f_r / self.c

        self.Nifft_fr = self.nifft_size
        self.Nifft_ph = self.nifft_size

        self.df = self.spectr_w / self.Nf
        self.dph = self.ph_w / self.Nph

        self.FR = max(self.f_r) - min(self.f_r)
        self.PH = max(self.ph_r) - min(self.ph_r)

        return (
            self.Nf, self.Nph,
            self.f_r, self.ph_r, self.k_r,
            self.Nifft_fr, self.Nifft_ph,
            self.df, self.dph,
            self.FR, self.PH,
        )

    def polar_img_params(self):
        """Параметры для формирования РЛИ с полярным переформатированием."""
        self.f_r = np.linspace(
            self.f_c - self.df * (self.Nf / 2),
            self.f_c + self.df * (self.Nf / 2) - self.df,
            self.Nf,
        )
        self.ph_r = np.linspace(
            self.ph_c - self.dph * (self.Nph / 2),
            self.ph_c + self.dph * (self.Nph / 2) - self.dph,
            self.Nph,
        )
        self.k_r = 2 * math.pi * self.f_r / self.c

        self.Nf = math.ceil((self.Xmax * self.spectr_w * 2) / self.c)
        self.Nph = math.ceil((2 * self.Ymax * self.ph_w * self.f_c) / self.c)

        self.kx = np.outer(self.k_r, self.cos(self.ph_r))
        self.ky = np.outer(self.k_r, self.sin(self.ph_r))
        self.kxMax = np.amax(self.kx)
        self.kxMin = np.amin(self.kx)
        self.kyMax = np.amax(self.ky)
        self.kyMin = np.amin(self.ky)

        self.Nifft_fr = self.nifft_size
        self.Nifft_ph = self.nifft_size
        self.M = 4

        return (
            self.Nf, self.Nph,
            self.kx, self.ky,
            self.kxMax, self.kyMax, self.kxMin, self.kyMin,
            self.Nifft_fr, self.Nifft_ph, self.M,
        )
