import math
import os
import numpy as np

NUM_IMAGES = 16


class Target:
    """Модель цели с интерфейсом медленного времени.

    Загружает предварительно рассчитанные данные о рассеивателях
    и предоставляет интерфейс через номер импульса (медленное время).

    Медленное время: t_n = pulse_index * PRI
    Угол наблюдения: angle = omega * t_n
    """

    def __init__(self, file, pri=1.0, omega=None, ang_rad=None, num_pulses=NUM_IMAGES):
        self.file = file
        self.pri = pri
        self.num_pulses = num_pulses

        if omega is not None:
            self.omega = omega
        elif ang_rad is not None:
            self.omega = ang_rad / (num_pulses * pri)
        else:
            self.omega = 0.0

        self._raw_tensor = None

    def _load_data(self):
        if self._raw_tensor is None:
            detail_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", self.file
            )
            self._raw_tensor = np.load(detail_path, allow_pickle=True)
        return self._raw_tensor

    def targets_matrix(self):
        """Возвращает весь тензор данных (для обратной совместимости)."""
        return self._load_data()

    def slow_time(self, pulse_index):
        """Медленное время для данного номера импульса: t_n = pulse_index * PRI."""
        return pulse_index * self.pri

    def angle(self, pulse_index):
        """Угол наблюдения для данного номера импульса: angle = omega * t_n."""
        return self.omega * self.slow_time(pulse_index)

    def get_frame(self, pulse_index):
        """Получить данные цели для заданного номера импульса.

        Возвращает (intens, x, y) — интенсивности и координаты рассеивателей.
        """
        data = self._load_data()
        intens, x, y = data[pulse_index]
        return intens, x, y
