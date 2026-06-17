from dataclasses import dataclass, asdict, fields
import json
import os

from paths import app_dir

CONFIG_PATH = os.path.join(app_dir(), "config.json")


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

    # Кинематика цели
    V: float = 0.0                      # скорость трансляции, м/с
    alpha: float = 0.0                  # угол направления движения, град
    omega: float = 0.0                  # угловая скорость, рад/с (0 = auto из beam_width)
    pri: float = 1.0                    # период повторения импульсов, с
    num_pulses: int = 1000              # число импульсов (= размер азимутальной оси РЛИ)

    # Обработка
    use_mocomp: bool = True             # включить компенсацию движения
    window: str = "none"                # оконная функция: none, hamming, hann
    display_mode: str = "dB"            # режим отображения: linear, dB
    snr_db: float = float('inf')        # SNR в дБ (inf = без шума)

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
