# IRSA — Физически корректный симулятор ISAR

<p align="center">
  <img src="assets/icon.ico" alt="IRSA Logo" width="64"/>
</p>

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

**IRSA** — симулятор радиолокационных изображений (РЛИ) с физически корректной моделью движения и компенсацией поступательного движения (MOCOMP).

Проект реализует классический алгоритм **Range-Doppler** для формирования изображений типа ISAR (Inverse Synthetic Aperture Radar), описанный в книге C. Özdemir *"Inverse Synthetic Aperture Radar Imaging with MATLAB Algorithms"* (Главы 4, 6, 8).

## Теория

ISAR-изображение формируется за счёт вращения цели относительно радара. Разные точки цели имеют разную лучевую скорость, а значит — разный доплеровский сдвиг частоты. Это позволяет разделить точки по азимуту (Cross-Range) с помощью БПФ.

### Конвейер обработки

```
Raw SFCW Data → Range Compression → MOCOMP → Azimuth Compression → ISAR Image
     (E)              (P)           (P_comp)        (I)              (|I|)
```

| Этап | Модуль | Описание |
|------|--------|----------|
| Генерация | `simulation/raw_generator.py` | Формирование матрицы сырых данных SFCW: E[m,n] = Σ σᵢ · exp(-j4πfₘRᵢ(tₙ)/c) |
| Сжатие по дальности | `processing/range_compress.py` | 1D IFFT по оси частот + дешифровка (dechirp) |
| MOCOMP | `processing/mocomp.py` | Компенсация поступательного движения: выравнивание огибающей + коррекция фазы |
| Сжатие по азимуту | `processing/azimuth_compress.py` | 1D FFT по оси медленного времени + физическое масштабирование |

### Кинематическая модель

Цель задаётся набором точечных рассеивателей с координатами (xᵢ, yᵢ) относительно центра масс (ЦМ). Для каждого импульса tₙ вычисляется:

- **Поступательное движение ЦМ**: X_cm = R₀ + V·cos(α)·tₙ, Y_cm = V·sin(α)·tₙ
- **Вращательное движение**: θ(tₙ) = ω·tₙ → поворот рассеивателей вокруг ЦМ
- **Абсолютная дальность**: Rᵢ(tₙ) = √(X_abs² + Y_abs²)

### MOCOMP (Motion Compensation)

Компенсация состоит из двух шагов:

1. **Выравнивание огибающей** (Envelope Alignment) — метод взаимной корреляции (Раздел 8.3.1 Özdemir)
2. **Коррекция фазы** (Phase Autofocus) — метод опорной точки (Раздел 8.3.3 Özdemir)

## Архитектура

```
IRSA/
├── models/
│   ├── radar.py          # Параметры радара SFCW
│   └── target.py         # Кинематическая модель цели
├── simulation/
│   └── raw_generator.py  # Генерация сырых данных SFCW
├── processing/
│   ├── range_compress.py # Сжатие по дальности (IFFT)
│   ├── mocomp.py         # Компенсация движения (MOCOMP)
│   ├── azimuth_compress.py # Сжатие по азимуту (FFT)
│   └── isar_processor.py # Старый метод (обратная совместимость)
├── examples/
│   ├── demo_ideal_turntable.py    # Демо 1: Идеальный разворот
│   ├── demo_motion_blur.py        # Демо 2: Размытие движением
│   └── demo_full_pipeline_mocomp.py # Демо 3: Полный конвейер
├── ui/                   # PyQt5 интерфейс
├── matrices/             # Данные моделей (.pkl)
└── main.py               # Точка входа
```

## Демонстрация

### Демо 1: Идеальный разворот (V=0, ω≠0)

Чистое вращение без поступательного движения. MOCOMP не нужен — РЛИ формируется идеально.

### Демо 2: Размытие движением (V≠0, ω≠0, без MOCOMP)

Поступательное движение вызывает «косые линии» на профилях дальности (Range Walk) и полное размытие РЛИ.

### Демо 3: Полный конвейер с MOCOMP (V≠0, ω≠0)

MOCOMP выпрямляет косые линии и корректирует фазу → чёткое РЛИ с яркими точками.

```bash
python -m examples.demo_ideal_turntable
python -m examples.demo_motion_blur
python -m examples.demo_full_pipeline_mocomp
```

## Установка

```bash
git clone https://github.com/yourusername/irsa.git
cd irsa
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск

### GUI (PyQt5)
```bash
python main.py
```

### Демо-скрипты
```bash
python -m examples.demo_ideal_turntable
python -m examples.demo_motion_blur
python -m examples.demo_full_pipeline_mocomp
```

## Параметры по умолчанию

| Параметр | Значение | Описание |
|----------|----------|----------|
| f_c | 8.0 ГГц | Центральная частота |
| B | 0.5 ГГц | Полоса сигнала |
| beam_width | 9.0° | Ширина ДН |
| R₀ | 500 км | Начальная дальность |
| V | 0–100 м/с | Поступательная скорость |
| ω | ~0.01 рад/с | Угловая скорость вращения |
| N_pulses | 16 | Количество импульсов |

## Требования

- Python 3.10+
- NumPy, SciPy, Matplotlib
- PyQt5 (для GUI)

## Лицензия

[MIT](LICENSE)
