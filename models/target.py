import math
import os
import numpy as np

NUM_IMAGES = 16
SPEED_OF_LIGHT = 3e8


class Target:
    """Физическая модель цели с кинематикой (поступательное + вращательное движение).

    Система координат:
      Ось X — направление от радара к цели (Line of Sight, RLOS).
      Ось Y — перпендикулярно оси X (горизонтальное азимутальное направление).
      Все координаты точек (x_i, y_i) задаются относительно ЦМ цели.

    Медленное время: t_n = pulse_index * PRI
    Угол поворота:  theta(t_n) = omega * t_n
    Координаты ЦМ:  X_cm(t_n) = R0 + Vx * t_n,  Y_cm(t_n) = Vy * t_n
    """

    def __init__(self, file, pri=1.0, num_pulses=NUM_IMAGES,
                 R0=0.0, V=0.0, alpha=0.0, omega=0.0):
        """
        Args:
            file: путь к файлу .npy с данными рассеивателей.
            pri: период повторения импульсов (с).
            num_pulses: количество импульсов (медленное время).
            R0: начальная дальность до ЦМ (м).
            V: модуль поступательной скорости (м/с).
            alpha: угол направления движения (рад) относительно RLOS.
            omega: угловая скорость вращения вокруг ЦМ (рад/с).
        """
        self.file = file
        self.pri = pri
        self.num_pulses = num_pulses
        self.R0 = R0
        self.V = V
        self.alpha = alpha
        self.omega = omega

        self._raw_tensor = None
        self._scatterers = None

    def _load_data(self):
        if self._raw_tensor is None:
            detail_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", self.file
            )
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

    def _absolute_positions(self, t_n, x0, y0):
        """Вычислить абсолютные координаты рассеивателей в момент t_n.

        Шаг 2.3: Поступательное движение ЦМ.
            Vx = V * cos(alpha),  Vy = V * sin(alpha)
            X_cm = R0 + Vx * t_n,  Y_cm = Vy * t_n

        Шаг 2.4: Вращательное движение вокруг ЦМ.
            theta = omega * t_n
            delta_x = x0 * cos(theta) - y0 * sin(theta)
            delta_y = x0 * sin(theta) + y0 * cos(theta)

        Шаг 2.5: Абсолютные координаты и дальность.
            X_abs = X_cm + delta_x,  Y_abs = Y_cm + delta_y
        """
        Vx = self.V * math.cos(math.radians(self.alpha))
        Vy = self.V * math.sin(math.radians(self.alpha))

        X_cm = self.R0 + Vx * t_n
        Y_cm = Vy * t_n

        theta = self.omega * t_n
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        delta_x = x0 * cos_t - y0 * sin_t
        delta_y = x0 * sin_t + y0 * cos_t

        X_abs = X_cm + delta_x
        Y_abs = Y_cm + delta_y

        return X_abs, Y_abs
