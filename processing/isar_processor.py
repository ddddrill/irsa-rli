import numpy as np
from scipy.fft import ifft2, fftshift
from scipy.interpolate import griddata


class StandardISARProcessor:
    """Процессор РЛИ стандартным методом (без полярного переформатирования)."""

    def __init__(self, Nf, Nph, f_r, ph_r, k_r, Nifft_fr, Nifft_ph, df, dph, FR, PH,
                 f_c, complex_v, exp_v):
        self.Nf = Nf
        self.Nph = Nph
        self.f_r = f_r
        self.ph_r = ph_r
        self.k_r = k_r
        self.Nifft_fr = Nifft_fr
        self.Nifft_ph = Nifft_ph
        self.df = df
        self.dph = dph
        self.FR = FR
        self.PH = PH
        self.f_c = f_c
        self.complex_v = complex_v
        self.exp_v = exp_v

    def compute_field(self, intens, x, y):
        """Вычислить поле обратного рассеяния Es для набора рассеивателей."""
        Es = np.zeros(
            (self.Nf, self.Nph), dtype="complex128"
        )
        ms_cos = np.multiply.outer(self.k_r, np.cos(self.ph_r))
        ms_sin = np.multiply.outer(self.k_r, np.sin(self.ph_r))

        for i in range(len(intens)):
            k_col = self.k_r[:, np.newaxis]
            ij = intens[i] * self.exp_v(
                self.complex_v(
                    0, 2 * k_col * y[i] + 2 * ms_cos * x[i] + 2 * ms_sin * y[i]
                )
            )
            Es += ij

        return Es

    def compute_image(self, Es):
        """Вычислить РЛИ из поля обратного рассеяния (2D IFFT)."""
        ifft_x = (self.Nifft_fr * self.FR) / self.Nf
        fz = self.Nifft_fr * self.df
        dt = 1 / fz
        t = np.arange(0, self.Nifft_fr) * dt
        xz = (t * 3e8) / 2
        xz = xz - xz[self.Nifft_fr - 1] / 2

        ifft_y = (self.Nifft_ph * self.PH) / self.Nph
        kz = self.Nifft_fr * self.dph
        dlen = 1 / kz
        leng = np.arange(0, self.Nifft_ph) * dlen
        leng = leng - leng[self.Nifft_ph - 1] / 2

        isar = (
            ifft_x * ifft_y * fftshift(ifft2(Es, s=[self.Nifft_fr, self.Nifft_ph]))
        ) / (self.FR * self.PH)

        return isar, xz, leng


class PolarISARProcessor:
    """Процессор РЛИ с полярным переформатированием."""

    def __init__(self, Nf, Nph, kx, ky, kxMax, kyMax, kxMin, kyMin,
                 Nifft_fr, Nifft_ph, M):
        self.Nf = Nf
        self.Nph = Nph
        self.kx = kx
        self.ky = ky
        self.kxMax = kxMax
        self.kyMax = kyMax
        self.kxMin = kxMin
        self.kyMin = kyMin
        self.Nifft_fr = Nifft_fr
        self.Nifft_ph = Nifft_ph
        self.M = M

    def polar_reformat(self, Es):
        """Полярное переформатирование поля обратного рассеяния."""
        xs = complex(0, self.Nf * self.M)
        ys = complex(0, self.Nph * self.M)
        grid_x, grid_y = np.mgrid[self.kxMin:self.kxMax:xs, self.kyMin:self.kyMax:ys]

        k_x = np.ravel(self.kx)
        k_y = np.ravel(self.ky)
        ee = np.ravel(Es)
        po = np.column_stack([k_x, k_y])

        Es_pol = griddata(po, ee, (grid_x, grid_y), method="linear", fill_value=0)
        return Es_pol, grid_x, grid_y

    def compute_image(self, Es_pol, grid_x, grid_y):
        """Вычислить РЛИ после полярного переформатирования."""
        K_x = np.amax(grid_x) - np.amin(grid_x)
        dkx = grid_x[1][0] - grid_x[0][0]
        ifft_x_p = (self.Nifft_fr * K_x) / (self.Nf * self.M)
        kz_xp = self.Nifft_fr * dkx
        dlen_xp = np.pi / kz_xp
        len_xp = np.arange(0, self.Nifft_fr) * dlen_xp
        len_xp = len_xp - len_xp[self.Nifft_fr - 1] / 2

        K_y = np.amax(grid_y) - np.amin(grid_y)
        dky = grid_y[0][1] - grid_y[0][0]
        ifft_y_p = (self.Nifft_fr * K_y) / (self.Nph * self.M * np.pi)
        kz_yp = self.Nifft_fr * dky
        dlen_yp = np.pi / kz_yp
        len_yp = np.arange(0, self.Nifft_ph) * dlen_yp
        len_yp = len_yp - len_yp[self.Nifft_ph - 1] / 2

        isar_p = (ifft_x_p * ifft_y_p) * fftshift(
            ifft2(Es_pol, s=[self.Nifft_fr, self.Nifft_ph])
        )

        return isar_p, K_x, K_y, len_xp, len_yp
