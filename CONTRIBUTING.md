# Contributing to IRSA

Спасибо за интерес к проекту IRSA! Этот документ объясняет, как внести свой вклад.

## Как начать

1. **Fork** репозитория
2. **Clone** своего форка:
   ```bash
   git clone https://github.com/YOUR_USERNAME/irsa.git
   cd irsa
   ```
3. Создай ветку для изменений:
   ```bash
   git checkout -b feature/your-feature
   # или
   git checkout -b fix/bug-description
   ```

## Требования к коду

### Стиль
- Используй 4 пробела для отступов (не табы)
- Максимальная длина строки — 120 символов
- Называй переменные понятно: `center_frequency` вместо `cf` или `cfr`

### Python
- Минимальная версия — Python 3.10
- Все функции должны иметь docstrings
- Используй аннотации типов где возможно

### Venv
- Используй `.build_venv` (Python 3.12) — он уже настроен в `.vscode/settings.json`
- Не коммить файлы из `.build_venv/`

### Коммиты

Пиши понятные сообщения коммитов:
```
# Хорошо:
add: выбор спутника в интерфейсе
fix: исправлена ошибка с размером FFT

# Плохо:
fix bug
update
```

## Как отправить изменения

1. **Добавь** изменённые файлы:
   ```bash
   git add .
   ```

2. **За commit** изменения:
   ```bash
   git commit -m "add: описание изменений"
   ```

3. **За push** в свой форк:
   ```bash
   git push origin feature/your-feature
   ```

4. Создай **Pull Request** на GitHub

## Что проверять перед PR

- [ ] Код работает без ошибок
- [ ] Нет лишних файлов в коммите (`__pycache__`, `.pyc`)
- [ ] Добавлено описание изменений
- [ ] Приложение запускается: `run.bat`
- [ ] Все 5 вкладок GUI заполняются
- [ ] Демо-скрипты работают: `python -m examples.demo_ideal_turntable`

## Структура проекта

```
VKR/
├── main.py                          # Точка входа
├── config.py                        # Конфигурация (AppConfig)
├── filenames.py                     # Маппинг спутников на файлы
├── models/
│   ├── radar.py                     # Параметры радара SFCW (Radar)
│   └── target.py                    # Кинематическая модель цели (Target)
├── simulation/
│   ├── raw_generator.py             # Генерация сырых данных SFCW
│   ├── validate_range_compression.py
│   ├── validate_mocomp.py
│   └── validate_isar.py
├── processing/
│   ├── data_processor.py            # Оркестратор (DataProcessor, QThread)
│   ├── range_compress.py            # Сжатие по дальности (IFFT + dechirp)
│   ├── mocomp.py                    # Компенсация движения (MOCOMP)
│   ├── azimuth_compress.py          # Азимутальное сжатие (FFT)
│   ├── polar_reformat.py            # Полярное переформатирование (k-space)
│   └── isar_processor.py            # StandardISARProcessor, PolarISARProcessor
├── ui/
│   ├── main_window.py               # Главное окно (QTabWidget, 5 вкладок)
│   └── styles.py                    # Стили PyQt5
├── examples/
│   ├── demo_ideal_turntable.py               # Демо 1: Идеальный разворот (V=0)
│   ├── demo_motion_blur.py                   # Демо 2: Размытие движением
│   ├── demo_full_pipeline_mocomp.py          # Демо 3: Полный конвейер с MOCOMP
│   ├── demo_large_angle.py                   # Демо 4: Проблема больших углов
│   ├── demo_polar_reformatting_validation.py # Демо 5: Валидация полярки
│   └── test_data_processor_polar.py          # Интеграционный тест + метрики
├── matrices/                        # Данные рассеяния спутников (.pkl)
├── sat_info_p/                      # Изображения и описания спутников
├── assets/                          # Иконки, скриншоты
├── radioimage/                      # Выходные директории с РЛИ
├── requirements.txt                 # Зависимости
└── LICENSE                          # MIT
```

### Конвейер обработки

**Стандартный** (малые углы, Δθ < 3°):

```
Raw SFCW → Range Compression → MOCOMP → Azimuth Compression → ISAR Image
    ↓              ↓              ↓              ↓                ↓
 E[M×N]         P[M×N]       P_comp[M×N]     I[M×N]          |I|[M×N]
```

**Полярное переформатирование** (большие углы, Δθ ≥ 3°):

```
Raw SFCW → Range Compression → MOCOMP → Polar Reformat → 2D IFFT → ISAR Image
    ↓              ↓              ↓          ↓              ↓          ↓
 E[M×N]         P[M×N]       P_comp[M×N]  E_cart[kx,ky]   I[M×N]    |I|[M×N]
                                        (k,θ)→(kx,ky)
                                        + Taylor window
```

Маршрутизация выбирается в `DataProcessor.compute_isar_notified()` по полю `self.method` или автоматически через `should_use_polar_reformat(omega, pri, N, threshold_deg=3.0)`.

## Тестирование

Перед отправкой проверь что приложение:
- Запускается без ошибок: `run.bat`
- Одиночный РЛИ рассчитывается (кнопка «Рассчитать РЛИ»)
- Все 5 вкладок заполняются (Сигнал, Профили дальности, Компенсация движения, РЛИ, Поток кадров)
- Метод «с полярным переформатированием» даёт сфокусированное изображение (energy в top-1% пикселей > 30%)
- Демо-скрипты работают без ошибок:

```bash
python -m examples.demo_ideal_turntable
python -m examples.demo_motion_blur
python -m examples.demo_full_pipeline_mocomp
python -m examples.demo_large_angle
python -m examples.demo_polar_reformatting_validation
python -m examples.test_data_processor_polar
```

### Что считать «поломанным»

- При стандартном методе с `Δθ < 3°` — точки должны быть сфокусированы
- При стандартном методе с `Δθ ≥ 10°` — ожидаемы дуги/размазня (arc smearing)
- При полярном методе с `Δθ ≥ 3°` — точки должны быть сфокусированы (это и есть назначение переформатирования)
- В dB-режиме пиксели в фоне должны быть в диапазоне −30…−50 dB, точки — около 0 dB
- В linear-режиме большинство значений близко к 0, яркие пятна — единицы (это нормально, изображение выглядит «тёмным»)

## Вопросы

- **Issue** — для багов и предложений
- **Discussions** — для вопросов

---

Спасибо за участие!
