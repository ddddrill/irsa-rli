# Журнал изменений

Все значимые изменения проекта документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/),
проект придерживается [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Baseline (ревизия `07a63dc`, перед задачами R2–R8)

Зафиксировано до внесения изменений плана исправлений (см. «План_исправлений_IRSA_R2-R8.md», §10.3). С этими значениями сравниваются регрессионные проверки каждой задачи.

- `pytest tests/ -q`: **15 passed** (13 warnings: matplotlib/pyparsing deprecation).
- `examples/test_data_processor_polar`: **ALL CHECKS PASSED**; энтропия Standard = 16.26 бит, Polar = 13.48 бит; энергия в топ-1% пикселей = 93.3% (порог > 30%).
- `examples/demo_full_pipeline_mocomp` (V=50 м/с, α=0.30): до MOCOMP энтропия 6.82, пик 1948, drift 11 бинов; после MOCOMP энтропия 6.77, пик 3897, drift −9 бинов; улучшение пика ×2.0.

### Задача R2: выравнивание огибающей v2 (усреднённая опора + суббина + pad-and-shift)

Реализовано выравнивание огибающей профилей дальности v2 — **C&M, §4.1.1, (4.12)–(4.14)** (план §5). Дефолт изменён, `shifts` теперь **float** (осознанный фикс P0):

- `processing/mocomp.py` — `envelope_alignment(P, method="correlation", subbin=True, ref_mode="average", guard_bins=None)`: (1) опорный профиль по умолчанию — **усреднённый** (двухпроходная схема: грубый проход на первый профиль → среднее `|P₁|` → точный проход на опору), а не шумный первый столбец; (2) суббиновая точность — параболическая интерполяция пика корреляции огибающих `|P|` (`correlate` над `|P|`, не над комплексными столбцами) + дробный сдвиг FFT phase-ramp'ом; (3) **pad-and-shift** вместо `np.roll` — нулевой буфер `M2 = M + 2·guard`, окно извлечения `[guard : guard+M]` (заворот энергии поглощается guard-зонами). Guard по умолчанию `max|shifts| + 8`, явный `guard_bins` валидируется (`≥ max|shifts| + 2`, иначе `ValueError`).
- `ref_mode="first"` — legacy-режим один проход на первый профиль; `method="centroid"` — сохранён, сдвиг тоже через pad-and-shift. При `subbin=False` сдвиги целочисленные (равны старым `np.roll`-лагам).
- `mocomp(P, range_axis=None, align_kwargs=None)` — `align_kwargs` пробрасывается в `envelope_alignment`; старые вызовы `mocomp(P, range_axis)` не меняются, новые дефолты подхватываются автоматически. `phase_autofocus` не тронута.
- Дробный сдвиг вносит в E-домен предсказуемый линейный фазовый тренд `4π·Δrₙ·f/c` — PPP (фазовый автофокус) компенсирует его сам (план §5.2.5).
- Тесты `tests/test_mocomp_v2.py` (TDD, красный→зелёный) — суббиновая точность (RMSE < 0.05 бина, kaiser(8)), отсутствие заворота (рассеиватель у края сетки, пиковая линия инвариантна + guard-нечувствительность), устойчивость `average` против `first` при AWGN (0 дБ) с нетипичным первым импульсом, legacy-совместимость лагов.
- Регрессия (тест 4): `examples/test_data_processor_polar` — **ALL CHECKS PASSED**; энтропия Standard 16.26 → **13.59 бит** (лучше), Polar 13.48 бит (без изменений, mocomp не участвует), топ-1% 93.3% (без изменений). `examples/demo_full_pipeline_mocomp` — до MOCOMP 6.82/пик 1948; после MOCOMP энтропия **6.73**, пик **4527**, улучшение пика **×2.3** (baseline: 6.77/3897/×2.0). `pytest tests/ -q`: **46 passed** (42 baseline + 4 новых).

### Задача R4: физический порог полярного конвейера

Порог переключения на полярное переформатирование теперь вычисляется из физического критерия применимости малой угловой аппроксимации **C&M, §3.1: Δθ_max ≈ √(λ/L_LOS)/2** (QPE кромкового рассеивателя ≤ π/16), а не равен эмпирической константе 3°.

- `processing/polar_reformat.py` — новая чистая функция `polar_threshold_deg(wavelength, target_size_los)`; `should_use_polar_reformat` расширена до кортежа из 5 элементов `(use_polar, Δθ_рад, Δθ_град, threshold_used_deg, threshold_source)` с правилом выбора порога: явный `threshold_deg` → `"legacy"`; `threshold_deg=None` c `wavelength`+`target_size_los` → `"physical"`; без данных → фолбэк 3° (`"legacy"`). Старый позиционный вызов с `threshold_deg` сохраняет поведение (обратная совместимость).
- `processing/polar_reformat.py` — `smart_pipeline` принимает `wavelength`/`target_size_los`, возвращает `threshold_used_deg`/`threshold_source` (+`physical_threshold_deg` при наличии данных) и логирует оба порога (физический и legacy 3°).
- `processing/data_processor.py` — `compute_isar_notified()` при выбранном стандартном методе и `Δθ > √(λ/L_LOS)/2` (λ = c/f_c, L_LOS ≈ `self.Xmax`) эмитит в `progress_message` только диагностическое предупреждение (формат `WARN: Δθ = … > физический порог … (C&M §3.1) — рекомендуется метод с полярным переформатированием (legacy-порог 3.00°)`). Поведение конвейера не меняется.
- Тесты `tests/test_polar_threshold.py` — опорные значения порога (X-band/L=20 м → 1.24°), «серая зона» меняет решение, фолбэк legacy, позиционная совместимость, некорректные входы, воспроизводимость GUI-предупреждения на пресете ω = 2°/16 импульсов.

### Задача R7: взвешивание по дальности и PSF-тесты разрешения

Дальностное сжатие получило опциональное взвешивание спектра, азимутальный набор окон расширен до kaiser/chebwin/taylor; добавлены автогесты PSF (ширина главного лепестка и боковые лепестки) — C&M, §6.3.1 (взвешивание), §3.3 (окно перед IFFT/FFT).

- `processing/windows.py` (новое) — единый словарь `WINDOW_NAMES`/`WINDOW_FUNCTIONS` и хелпер `get_weighting(name, N, param=None)` с нормировкой `win/win.mean()` (восстановление когерентного усиления). Для `"none"` (boxcar) нормировка не применяется — битовая идентичность старому коду. `chebwin` при N<4 вырождается → boxcar (guard). `taylor` переиспользует `taylor_window_1d` полярного конвейера (импорт ленивый, без цикла).
- `processing/range_compress.py` — новая сигнатура `range_compress(E, f_r, R0=0.0, window="none", window_param=None)`: окно применяется к столбцам E (axis=0) ДО IFFT. Дефолт `"none"` → побитово прежний результат.
- `processing/azimuth_compress.py` — `window_param` в сигнатуре; взвешивание через общий `get_weighting` (нормировка vs старая ненормированная hamming/hann — амплитуда пиков оконных изображений теперь сопоставима между окнами).
- `processing/psf_metrics.py` (новое) — `psf_metrics(profile_db, main_lobe_halfwidth_bins) → (width_−3dB_bins, max_sll_db)` для переиспользования в R2/R3/R6.
- `processing/data_processor.py` — новые поля конфига проброшены во все вызовы: `range_window` в 8 вызовов `range_compress`, `window_param` в 3 вызова `azimuth_compress` и в `range_compress`.
- `config.py` — поля `window_param` (None) и `range_window` ("none").
- `ui/main_window.py` — комбобокс окон расширен до `WINDOW_NAMES` (kaiser/chebwin/taylor), добавлено поле «Параметр окна (β/at)».
- Тесты `tests/test_psf.py` — битовая идентичность `"none"` (range/azimuth), параметризованные SLL (none/hann/hamming/kaiser/chebwin, допуск ≤ ожидание+3 дБ), уширение ГЛ hann ≤1.7× от boxcar, разрешение по Рэлею (boxcar различим, kaiser сливается), симметричный азимутальный PSF.
- `PARAMETERS_GUIDE.md` — честная таблица компромисса взвешивания (§6.3.1) и параметры `window_param`/`range_window` (дефолт `"none"`, демо не меняются).

### Задача R8a: угловое ускорение γ (маневрирующая цель)

В целевую модель добавлено угловое ускорение вращения — **C&M, (1.13): `θ(t) = θ₀ + Ω·t + γ·t²/2`**. Дефолт `gamma=0.0` → поведение по умолчанию не меняется (побитовая совместимость проверена тестами).

- `models/target.py` — параметр `gamma=0.0` (рад/с²) в `Target.__init__`; закон угла применён в `_absolute_positions` (`theta = omega*t_n + 0.5*gamma*t_n**2`) и `get_aspect_angles` (единый закон для анизотропии/затенения). В докстринге отмечено: при γ ≠ 0 полный фокус даёт только TF-конвейер (R5), «слепой» PFA расфокусируется (ожидаемо).
- `processing/data_processor.py` — kwarg `gamma=0.0`, проброс в `Target` через `_make_target()`.
- `config.py` — поле `gamma: float = 0.0` в блоке «Кинематика цели».
- `ui/main_window.py` — числовое поле «Угл. ускорение, рад/с²» (`le_gamma`, рядом с `le_accel`), чтение в `_on_config_changed`, передача `gamma=self.cfg.gamma` в оба конструктора `DataProcessor`. Дефолт 0 — демо-пресеты не меняются (smoke offscreen).
- `processing/polar_reformat.py` — `build_kspace_grid(f_r, omega, pri, num_pulses, theta_array=None)`: при явной передаче вектора углов (например `target.get_aspect_angles()`) используется нелинейная сетка вместо `omega*n*pri` (нужно R6); иначе поведение как раньше.
- Тесты `tests/test_target_gamma.py` — битовая совместимость `gamma=0`, формула `θ = ω·t + 0.5·γ·t²` (в т.ч. через `_absolute_positions`/`slow_time`), полярный конвейер выживает при `gamma=1e-4` (энтропия конечна, оси в метрах, `np.isfinite`), нелинейная сетка `theta_array`.

### Задача R5a: TF-ядро tf_analysis (STFT, маргиналы, RID, WVD) и сигнатуры микро-Доплера

Новый модуль `processing/tf_analysis.py` — временно-частотное ядро на STFT (**TF, гл. 2**: определение, сетка, marg: (2.2)–(2.9), массовое сглаживание §2.5; **MD, §1.8**: RID, спектрограмма-ядро) с физически выверенной калибровкой маргиналов (**MD, §6.2.6**) и автогестами по TDD.

- `stft_spectrogram(x, fs=1.0, nperseg=None, noverlap=None, window="hann", scaling_calibrated=True)` → `dict(f, t, Z, S_db)`: оконный STFT через `get_weighting` (`processing/windows.py`), реконструкция Коэна—Нуссбаумом (Nuttall 1981) комплексных доплеровских гистограмм, односторонний доплеровский спектр, `S_db` c динамическим диапазоном 80 дБ и rel-нормировкой к максимуму.
- `strongest_range_bins(P, k=5, min_frac=1e-3)` — топ-k дальностных строк по средней энергии.
- `micro_doppler_signature(P, bin_index, fs=1.0, **kw)` и `total_echo_signature(P, fs=1.0, **kw)` → кортеж/словарь T-F-данных (спектрограмма, солапс-частоты, полоса, джиттер/центроид).
- `rid_cube(P, fs=1.0, n_frames=16, nperseg=None, window="hann", hop=1, dtype_mag=np.float32)` → **RID-куб** `Q_mag[f, t(marg), n]` через спектрограмму с коротким окном поверх скользящих коротких наблюдений (массовое сглаживание TF, §2.5). Автоподбор бюджета памяти: при превышении 100 МБ — прореживание до топ-256 доплер-фильтров (`RID_MAX_BINS`) и уширение hop (`hop_eff`), флаг `approximate=True`. `rid_frame(rid, i)` — dB-кадр с нормировкой; `rid_marginal(rid)` — `Σ_t |Q|²`.
- `wigner_ville(x, fs=1.0)`, `pseudo_wigner_ville(x, fs=1.0, nperseg=128, hop=1)`, `hilbert_analytic(x)`, `wvd_like_instantaneous_from_S_db(...)` — WVD по чётнолаговой схеме (нечётная автокорреляция → оси `f = k·fs/(2N)`), аналитический сигнал через `scipy.signal.hilbert` для вещественного входа.
- **Калибровка маргинала** — аналитическая постояннонаблюдаемость для чистого тона на дискретной доплеровской частоте: `marginal·SCALE(N, L, window) = |FFT|²` с точностью сглаживания окном (**MD, §6.2.6**). SCALE выводится из суммы квадратов нечётного ядра со сдвигом окна `w[n]` и постоянна в конфиг-области. Модульная константа `SCALE = _marginal_scale(256, 128, "hann") = 292.25417`.
- Тесты `tests/test_tf_analysis.py` + фикстуры `radar_tf`/`target_tf` в `tests/conftest.py`: тон на 128-точечном STFT в пределах 0.2 бина с cv₂ ≈ 0 при чистом тоне; полоса виброавтоклавного сценария (Carson: ≥0.95 при измеренном 0.976; ±15% мгновенной полосы: ≥0.90 при 0.951); маргинал-spans-энергии: mem ≈ 64.7 МБ (256×126×501 кадр) ≤ 100 МБ при автоматическом прореживании; маргинал-идентичность SCALE к |FFT|² (отн. ошибка ~1e-16 при hop=1) с флагом `approximate` при hop>1; WVD-концентрация LFM-тона (±0.01 Гц: 0.92 против 0.37 STFT) и задокументированный булletin-крест-терм двух тонов (0 дБ > −30 дБ); детерминизм.
- Демо `examples/demo_micro_doppler_stft.py` (реальный автоклавный сценарий: 99.9% энергии в Carson-полосе, период ВДВ 0.333 с = 1/f_vib через WVD-панель) и `examples/demo_rid_imaging.py` (RID-кадры + маргинал против |FFT|², err_dB = 0.146, «на глаз» совпадение форм).
- `pytest tests/ -q`: **53 passed** (46 + 7 новых).

## [2.4.0] — 2026-06-30

### Добавлено
- **Физические эффекты реализма** в генератор сырых данных (`simulation/raw_generator.py`):
  1. *Комплексные амплитуды рассеивателей* — случайные фазы `σᵢ·exp(jφᵢ)` из seeded RNG (`simulation/scatterer_effects.py:make_complex_amplitudes`).
  2. *Анизотропия* — гауссова угловая зависимость `σᵢ(θₙ)` с случайными пиками по сектору съёмки (`make_scatterer_patterns`, `anisotropy_weights`).
  3. *Затенение (occlusion)* — геометрическая окклюзия рассеивателей по дальности и поперечному смещению (`occlusion_mask`).
  4. *Фазовый шум опорного генератора* — винеровский процесс с нормированным RMS (`simulation/system_effects.py:phase_noise_process`).
  5. *Двухпутевая ДН антенны* — гауссова модель `G²(θ)` с ошибкой наведения и дрейфом (`two_way_antenna_weights`).
- **Нелинейная кинематика цели** (`models/target.py`):
  - Ускорение `a` вдоль направления `α` — квадратичная фазовая ошибка.
  - Гармоническая микровибрация по дальности — микро-доплер.
- `SimEffects` dataclass в `simulation/raw_generator.py` — все флаги по умолчанию выключены, полная обратная совместимость.
- Модуль `simulation/rng.py` — централизованный seeded RNG с изоляцией под-генераторов (`spawn_rng`).
- Секция **«Реализм»** в GUI (`ui/main_window.py`): пресеты «Идеальный» / «Реалистичный» / «Пользовательский», чекбоксы и поля ввода для каждого эффекта.
- `REALISM_PRESETS` в `config.py` — рекомендуемые наборы параметров.
- Демо-скрипт `examples/demo_realism_effects.py` — 6-панельное сравнение «до/после».
- Набор тестов `tests/` (pytest):
  - `test_backward_compat.py` — регрессия: effects-off ≡ классическая формула.
  - `test_determinism.py` — воспроизводимость по seed.
  - `test_effects_unit.py` — юнит-проверки каждого эффекта.
  - `test_pipeline_integration.py` — полный конвейер (стандартный + полярный) со всеми эффектами.

### Изменено
- `processing/data_processor.py` — приём `realism`-kwargs (дефолты off), проброс в `Target` и `generate_raw_matrix`.
- `requirements.txt` — добавлен `pytest`.

## [2.2.0] — 2026-06-17

### Исправлено
- **CRITICAL**: `mocomp.py` — опорный бин в `phase_autofocus` выбирался по среднему комплексных значений вместо среднего модулей (`np.mean(P)` → `np.mean(np.abs(P))`). Без этого MOCOMP не работал на движущихся целях.
- **CRITICAL**: `isar_processor.py` — `compute_field` содержала лишний член `2·k·y` в фазе и положительный знак экспоненты. Формула приведена к канонической: `exp(-1j·2·(k·cosθ·x + k·sinθ·y))`.
- **CRITICAL**: `polar_reformat.py` — окно Тейлора `taylor_window_1d`: `num` и `denom` вычисляли одинаковый продукт, что давало F[i]≡1 и уровень боковых лепестков ~−13 дБ вместо −35 дБ. Исправлена формула коэффициентов Тейлора.
- **CRITICAL**: `models/radar.py` — `polar_img_params` использовал `k_r = 2π·f/c` (одностороннее), тогда как `polar_reformat.py` использует `4π·f/c` (round-trip). Выровнено до `4π·f/c`.
- `isar_processor.py` — `compute_image`: нормировка `1/(FR·PH)` заменена на стандартную `fftshift(ifft2(...))`.
- `isar_processor.py` — `PolarISARProcessor.compute_image`: опечатка `Nifft_fr` → `Nifft_ph` в формуле `ifft_y_p`.
- `isar_processor.py` — `compute_field`: убрано использование `np.vectorize(math.cos)`, теперь `np.cos`/`np.sin` через `np.multiply.outer` (в 10–100× быстрее).

### Добавлено
- `polar_reformat.py` — `smart_pipeline` принимает параметр `use_mocomp` и вызывает MOCOMP в стандартной ветке (ранее пропускал компенсацию движения).
- `data_processor.py` — `_compute_isar_polar_sync` теперь включает range_compress + MOCOMP (ранее дублировал `_compute_isar_polar` без компенсации).
- `models/target.py` — исправлен docstring: `alpha` задаётся в градусах, не в радианах.
- `run.bat` — скрипт-лаунчер для запуска приложения двойным кликом.
- `.vscode/settings.json` — настройка Python-интерпретатора и автозапуска venv.

### Изменено
- `processing/polar_reformat.py` — оси `x_axis`/`y_axis` в `form_image_2d` теперь используют `(arange - center) * step` вместо `linspace` (корректный нуль в центре для чётных размеров).
- `processing/range_compress.py` — `range_axis` упрощён для читаемости.

## [2.1.0] — 2026-06-05

### Добавлено
- **Полярное переформатирование** для широкоугольной съёмки ISAR: новый модуль `processing/polar_reformat.py`
  - Формирование полярной сетки в k-пространстве: `build_kspace_grid`, `compute_wavenumber_axis`, `compute_angle_axis`, `polar_to_cartesian`
  - Проектирование декартовой сетки по теореме Котельникова: `design_cartesian_grid`
  - Билинейная интерполяция полярная → декартова: `interpolate_polar_to_cartesian` (scipy RegularGridInterpolator)
  - Обработка слепых зон + окно Тейлора + zero-padding: `process_blind_zones`, `taylor_window_1d`, `apply_window_2d`
  - 2D IFFT + физическое масштабирование осей: `form_image_2d`, `compute_amplitude_db`
- Умная маршрутизация конвейера в `DataProcessor`: автоматическое переключение стандартного и полярного методов по порогу Δθ (`should_use_polar_reformat`, порог 3°)
- Метрики качества фокусировки: `compute_image_entropy`, `compute_isnr`
- Вкладка **«Сигнал»** в GUI: визуализация SFCW-лесенки во временной и частотной областях, таблица параметров сигнала
- Параметры обзора в GUI: `survey_length`, `survey_width` (для антиалиасинга)
- Параметры кинематики в GUI: `V`, `α`, `ω`, `PRI`, `num_pulses`
- Подсказки (tooltips) ко всем полям ввода с физическим смыслом и формулами
- Кнопка «Рассчитать РЛИ» для одиночного расчёта с уведомлениями на каждом этапе
- Qt-сигнал `signal_data_ready` для проброса сырых данных во вкладку «Сигнал»
- Демо-скрипты:
  - `examples/demo_large_angle.py` — демонстрация проблемы «дуг» при больших Δθ
  - `examples/demo_polar_reformatting_validation.py` — тройное сравнение (наивный 2D FFT / ближайший сосед / билинейная интерполяция + окно)
  - `examples/test_data_processor_polar.py` — интеграционный тест конвейера с метриками

### Изменено
- GUI переработан: 4 → 5 вкладок (добавлена «Сигнал»), заголовок окна «IRSA — ISAR Radar Image Simulator», размер 1400×800
- Верхняя панель разделена на три секции (спутник | параметры | кнопки) через горизонтальный QSplitter
- Кнопка «Вывести результаты» заменена на «Запустить поток»/«Остановить» + таймер покадрового вывода
- `AppConfig` дополнен полями `V`, `alpha`, `omega`, `pri`, `num_pulses`, `use_mocomp`, `window`, `display_mode`
- Сигнатура `DataProcessor.__init__` расширена параметрами кинематики и обработки
- `compute_isar_notified` маршрутизирует на `_compute_isar_standard` или `_compute_isar_polar` в зависимости от `self.method`
- `range_compress` стал чистой функцией (без зависимости от `Radar`)

### Исправлено
- `DataProcessor.__init__`: значение `omega=0.0` (обозначающее «авто из beam_width») теперь корректно приводит к авто-расчёту; ранее из-за проверки `if omega is not None` нулевое значение трактовалось как «без вращения», и полярный конвейер молча откатывался на стандартный

## [2.0.0] — 2026-06-04

### Добавлено
- Физическая кинематическая модель цели (поступательное + вращательное движение)
- Генерация сырых данных SFCW (модуль `simulation/raw_generator.py`)
- Сжатие по дальности: IFFT + дешифровка (dechirp) (`processing/range_compress.py`)
- Компенсация движения MOCOMP: выравнивание огибающей + автофокус по фазе (`processing/mocomp.py`)
- Азимутальное сжатие: FFT + оконные функции (`processing/azimuth_compress.py`)
- Оркестратор `DataProcessor` с Qt-сигналами для промежуточных результатов
- Метод `compute_isar_notified()` — полный конвейер с уведомлениями на каждом этапе
- Интерфейс с 4 вкладками: профили дальности, MOCOMP, РЛИ, поток кадров
- Параметры кинематики в UI: V, α, ω, PRI, число импульсов
- Переключатель MOCOMP вкл/выкл, выбор оконной функции, режим отображения (linear/dB)
- 3 демо-скрипта: идеальный разворот, размытие движением, полный конвейер с MOCOMP
- 3 скрипта валидации: сжатие по дальности, MOCOMP, ISAR
- Модульная архитектура: `models/`, `processing/`, `simulation/`

### Изменено
- Полная переработка UI: QTabWidget с вкладками вместо плоского layout
- `DataProcessor` расширен параметрами `use_mocomp`, `window`, кинематикой
- `AppConfig` расширен полями кинематики и обработки
- Оси графиков отображают физические величины (м, Гц, рад/м)
- Конвейер ISAR: Raw → Сжатие по дальности → MOCOMP → Азимутальное сжатие → Изображение

### Удалено
- Удалены устаревшие файлы: `parametrs.py`, `RLI.py`, `target.py` (корневой)
- Удалены неиспользуемые UI-модули: `ui/dialogs.py`, `ui/widgets.py`

## [1.0.0] — 2026-04-20

### Добавлено
- Основной интерфейс приложения на PyQt5
- Конфигурация параметров радара (частота, спектр, размер FFT, ширина ДН, обзорная область, дальность)
- Поддержка 5 спутников: cloudSAT, calipso, ICESat2, LRO, solarB
- Два метода обработки: стандартный и с полярным переформатированием
- Визуализация поля рассеяния и РЛИ в реальном времени
- Пакетная обработка 16 кадров
- Экспорт результатов в PNG
- Сохранение конфигурации в JSON

### Изменено
- Рефакторинг файловой структуры для удобства сопровождения
- Оптимизация памяти: сохранение результатов напрямую на диск
- Изменён размер FFT по умолчанию: 4096 → 1024

### Удалено
- Параметр «Совмещённая цель» (данные недоступны)

## [0.0.1] — 2026-01-01

### Добавлено
- Первоначальный релиз
- Базовая обработка РЛИ
- Простой графический интерфейс
