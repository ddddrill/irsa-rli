import math
import os
import numpy as np

from paths import resource_path

NUM_IMAGES = 16
SPEED_OF_LIGHT = 3e8


class Target:
    """Физическая модель цели с кинематикой (поступательное + вращательное движение).

    Система координат:
      Ось X — направление от радара к цели (Line of Sight, RLOS).
      Ось Y — перпендикулярно оси X (горизонтальное азимутальное направление).
      Все координаты точек (x_i, y_i) задаются относительно ЦМ цели.

    Медленное время: t_n = pulse_index * PRI
    Угол поворота:  theta(t_n) = omega * t_n + 0.5 * gamma * t_n**2     — C&M, (1.13)
    Координаты ЦМ:  X_cm(t_n) = R0 + Vx * t_n,  Y_cm(t_n) = Vy * t_n

    При gamma != 0 (маневрирующая цель) полный фокус РЛИ даёт только
    TF-конвейер (R5): чирп-член a2*t^2 в фазе размазывает RD-изображение,
    «слепой» PFA расфокусируется (это ожидаемое поведение, см. C&M, §3.2).
    """

    def __init__(self, file, pri=1.0, num_pulses=NUM_IMAGES,
                 R0=0.0, V=0.0, alpha=0.0, omega=0.0, gamma=0.0,
                 accel=0.0, vib_amplitude=0.0, vib_frequency=1.0):
        """
        Args:
            file: путь к файлу .npy с данными рассеивателей.
            pri: период повторения импульсов (с).
            num_pulses: количество импульсов (медленное время).
            R0: начальная дальность до ЦМ (м).
            V: модуль поступательной скорости (м/с).
            alpha: угол направления движения (град) относительно RLOS.
            omega: угловая скорость вращения вокруг ЦМ (рад/с).
            gamma: угловое ускорение вращения (рад/с²).
                   Угол поворота: theta(t) = omega*t + 0.5*gamma*t^2
                   (C&M, (1.13)). При gamma=0 — равномерное вращение.
            accel: ускорение ЦМ вдоль направления alpha (м/с²).
                   Даёт квадратичную фазовую ошибку по медленному времени.
            vib_amplitude: амплитуда микровибрации ЦМ по дальности (м).
                   Гармоническое микродоплеровское движение (по умолчанию 0).
            vib_frequency: частота микровибрации (Гц).
        """
        self.file = file
        self.pri = pri
        self.num_pulses = num_pulses
        self.R0 = R0
        self.V = V
        self.alpha = alpha
        self.omega = omega
        self.gamma = gamma
        self.accel = accel
        self.vib_amplitude = vib_amplitude
        self.vib_frequency = vib_frequency

        self._raw_tensor = None
        self._scatterers = None

    def _load_data(self):
        if self._raw_tensor is None:
            detail_path = resource_path(self.file)
            self._raw_tensor = np.load(detail_path, allow_pickle=True)
        return self._raw_tensor

    def _load_scatterers(self):
        """Загрузить рассеиватели и отцентрировать координаты относительно ЦМ."""
        if self._scatterers is None:
            data = self._load_data()
            intens, x, y = data[0]
            self._scatterers = {
                "intens": np.array(intens, dtype=float),
                "x0": np.array(x, dtype=float) - np.mean(x),
                "y0": np.array(y, dtype=float) - np.mean(y),
            }
        return self._scatterers

    def targets_matrix(self):
        """Возвращает весь тензор данных (для обратной совместимости)."""
        return self._load_data()

    def slow_time(self, pulse_index):
        """Медленное время: t_n = pulse_index * PRI."""
        return pulse_index * self.pri

    def get_frame(self, pulse_index):
        """Получить данные цели для заданного номера импульса.

        Возвращает (intens, x, y) — интенсивности и координаты рассеивателей
        в системе координат радара (абсолютные).
        """
        t_n = self.slow_time(pulse_index)
        sc = self._load_scatterers()
        intens = sc["intens"]
        x_abs, y_abs = self._absolute_positions(t_n, sc["x0"], sc["y0"])
        return intens, x_abs, y_abs

    def get_ranges(self, pulse_index):
        """Вычислить дальности до всех рассеивателей для заданного импульса.

        Возвращает массив R_i(t_n) — расстояния от радара до каждой точки.
        """
        t_n = self.slow_time(pulse_index)
        sc = self._load_scatterers()
        x_abs, y_abs = self._absolute_positions(t_n, sc["x0"], sc["y0"])
        ranges = np.sqrt(x_abs ** 2 + y_abs ** 2)
        return ranges

    def get_ranges_all(self):
        """Вычислить матрицу дальностей для всех импульсов.

        Возвращает матрицу (num_pulses, num_scatterers).
        """
        sc = self._load_scatterers()
        ranges = np.zeros((self.num_pulses, len(sc["intens"])))
        for n in range(self.num_pulses):
            t_n = self.slow_time(n)
            x_abs, y_abs = self._absolute_positions(t_n, sc["x0"], sc["y0"])
            ranges[n] = np.sqrt(x_abs ** 2 + y_abs ** 2)
        return ranges

    def get_cross_range_all(self):
        """Матрица поперечных координат рассеивателей (num_pulses, num_scatterers).

        Поперечная координата Y_abs — проекция положения точки на ось,
        перпендикулярную линии визия. Нужна для модели затенения.
        """
        sc = self._load_scatterers()
        y_all = np.zeros((self.num_pulses, len(sc["intens"])))
        for n in range(self.num_pulses):
            t_n = self.slow_time(n)
            _, y_abs = self._absolute_positions(t_n, sc["x0"], sc["y0"])
            y_all[n] = y_abs
        return y_all

    def get_aspect_angles(self):
        """Углы наблюдения рассеивателей (радиан) — C&M, (1.13).

        theta_n = omega * t_n + 0.5 * gamma * t_n^2,  t_n = n * PRI
        Возвращает массив shape (num_pulses,). При omega=0 и gamma=0 все углы
        нулевые. Используется для модели анизотропии рассеивателей; единый
        закон угла согласует анизотропию/затенение с кинематикой вращения.
        """
        n = np.arange(self.num_pulses)
        t_n = n * self.pri
        return self.omega * t_n + 0.5 * self.gamma * t_n ** 2

    def _absolute_positions(self, t_n, x0, y0):
        """Вычислить абсолютные координаты рассеивателей в момент t_n.

        Шаг 2.3: Поступательное движение ЦМ (с опциональным ускорением).
            Vx = V * cos(alpha),  Vy = V * sin(alpha)
            X_cm = R0 + Vx * t_n + 0.5 * a * cos(alpha) * t_n^2
            Y_cm = Vy * t_n + 0.5 * a * sin(alpha) * t_n^2

        Шаг 2.3а: Микровибрация ЦМ по дальности (микродоплер).
            X_cm += A_vib * sin(2*pi*f_vib*t_n)

        Шаг 2.4: Вращательное движение вокруг ЦМ.
            theta = omega * t_n + 0.5 * gamma * t_n^2        — C&M, (1.13)
            delta_x = x0 * cos(theta) - y0 * sin(theta)
            delta_y = x0 * sin(theta) + y0 * cos(theta)

        Шаг 2.5: Абсолютные координаты и дальность.
            X_abs = X_cm + delta_x,  Y_abs = Y_cm + delta_y

        При accel=0 и vib_amplitude=0 формулы совпадают с классической
        линейной моделью (обратная совместимость).
        """
        alpha_rad = math.radians(self.alpha)
        Vx = self.V * math.cos(alpha_rad)
        Vy = self.V * math.sin(alpha_rad)

        X_cm = self.R0 + Vx * t_n
        Y_cm = Vy * t_n

        if self.accel != 0.0:
            X_cm += 0.5 * self.accel * math.cos(alpha_rad) * t_n ** 2
            Y_cm += 0.5 * self.accel * math.sin(alpha_rad) * t_n ** 2

        if self.vib_amplitude != 0.0:
            X_cm += self.vib_amplitude * math.sin(2.0 * math.pi * self.vib_frequency * t_n)

        theta = self.omega * t_n + 0.5 * self.gamma * t_n ** 2   # C&M, (1.13)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        delta_x = x0 * cos_t - y0 * sin_t
        delta_y = x0 * sin_t + y0 * cos_t

        X_abs = X_cm + delta_x
        Y_abs = Y_cm + delta_y

        return X_abs, Y_abs
