from dataclasses import dataclass, asdict, fields
import json
import os

from paths import app_dir

CONFIG_PATH = os.path.join(app_dir(), "config.json")


# Пресеты эффектов реализма для GUI.
# «Идеальный» — все эффекты выключены (классическая точечная модель).
# «Реалистичный» — рекомендуемый набор для демонстрации реальных условий.
REALISM_PRESETS = {
    "Идеальный": {
        "use_complex_sigma": False, "use_anisotropy": False,
        "use_shadowing": False, "use_antenna_pattern": False,
        "use_phase_noise": False, "accel": 0.0, "vib_amplitude_mm": 0.0,
    },
    "Реалистичный": {
        "use_complex_sigma": True, "use_anisotropy": True,
        "use_shadowing": True, "use_antenna_pattern": True,
        "use_phase_noise": True, "accel": 0.5, "vib_amplitude_mm": 2.0,
    },
}


@dataclass
class AppConfig:
    method: str = "стандартный"
    center_frequency: float = 8.0       # ГГц
    spectrum_width: float = 0.5         # ГГц
    nifft_size: int = 1024              # размер FFT (степень 2)
    beam_width: float = 9.0             # град
    survey_length: float = 20.0         # м
    survey_width: float = 20.0          # м
    satellite: str = "solarB"
    range_km: float = 500.0             # дальность до объекта, км

    # Модель цели и ось вращения (GUI передаёт в DataProcessor;
    # 3D-обработка форм — в ветке feature/scatterer-3d, вне R2-R8)
    target_model: str = "точечная"      # точечная / формы (json)
    omega_axis_y: float = 0.0           # Y-компонента оси вращения (X=0)
    omega_axis_z: float = 0.0           # Z-компонента оси вращения
    precession_rate: float = 0.0        # частота прецессии оси, рад/с
    precession_angle_deg: float = 0.0   # полуразворот конуса прецессии, град

    # Кинематика цели
    V: float = 0.0                      # скорость трансляции, м/с
    alpha: float = 0.0                  # угол направления движения, град
    omega: float = 0.0                  # угловая скорость, рад/с (0 = auto из beam_width)
    gamma: float = 0.0                  # угловое ускорение, рад/с² (0 = равномерное вращение)
    pri: float = 1.0                    # период повторения импульсов, с
    num_pulses: int = 1000              # число импульсов (= размер азимутальной оси РЛИ)

    # Обработка
    use_mocomp: bool = True             # включить компенсацию движения
    window: str = "none"                # азимутальное окно: none, hamming, hann, kaiser, chebwin, taylor
    window_param: float = None          # параметр окна: beta (kaiser, дефолт 8.0) или at-дБ (chebwin, дефолт 60.0)
    range_window: str = "none"          # дальностное окно (тот же набор); "none" = честная PSF Дирихле
    display_mode: str = "dB"            # режим отображения: linear, dB
    snr_db: float = float('inf')        # SNR в дБ (inf = без шума)

    # TF-анализ (вкладка «Сигнатуры»; дефолт выключен — битовая совместимость)
    tf_enabled: bool = False            # расчёт RID-куба и сигнатур в стандартном конвейере
    tf_n_frames: int = 16               # число RID-кадров куба (16–32, [TF, §5.3])
    tf_window: str = "hann"             # окно STFT/RID (processing.windows.WINDOW_NAMES)

    # Кросс-масштабирование по данным (chirp-rate): off / auto
    cross_range_scaling: str = "off"    # off = без изменений конвейера; auto = заменить ось азимута оценкой Omega_eff

    # Эффекты реализма (все выключены = классическая модель)
    realism_preset: str = "Идеальный"   # Идеальный / Реалистичный / Пользовательский
    seed: int = 42                      # базовый seed генераторов случайности
    use_complex_sigma: bool = False     # случайные фазы рассеивателей
    use_anisotropy: bool = False        # угловая зависимость sigma_i(theta)
    anisotropy_width_deg: float = 10.0  # ширина лепестка анизотропии, град
    use_shadowing: bool = False         # геометрическое затенение
    shadow_radius_m: float = 0.4        # радиус затенения, м
    use_antenna_pattern: bool = False   # двухпутевая ДН антенны
    pointing_error_deg: float = 0.0     # ошибка наведения, град
    antenna_drift_deg: float = 0.0      # дрейф луча за апертуру, град
    use_phase_noise: bool = False       # фазовый шум опорного генератора
    phase_noise_std_deg: float = 10.0   # RMS фазового набега за апертуру, град
    accel: float = 0.0                  # ускорение цели, м/с²
    vib_amplitude_mm: float = 0.0       # амплитуда микровибрации, мм
    vib_frequency_hz: float = 1.0       # частота микровибрации, Гц

    @property
    def range_m(self) -> float:
        return self.range_km * 1000.0

    def save(self, path: str = CONFIG_PATH) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str = CONFIG_PATH) -> "AppConfig":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            valid_keys = {f.name for f in fields(cls)}
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            return cls(**filtered)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return cls()
