# Журнал изменений

Все значимые изменения проекта документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/),
проект придерживается [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.5.0] — 2026-09-14

### Добавлено
- Редизайн README.md по рекомендациям: hero-изображение, галерея «до/после», Quick Start, программный API; обновлённые возможности и ограничения R2–R8; ассеты `assets/readme/`.

### Baseline (ревизия `07a63dc`, перед задачами R2–R8)

Зафиксировано до внесения изменений плана исправлений (см. «План_исправлений_IRSA_R2-R8.md», §10.3). С этими значениями сравниваются регрессионные проверки каждой задачи.

- `pytest tests/ -q`: **15 passed** (13 warnings: matplotlib/pyparsing deprecation).
- `examples/test_data_processor_polar`: **ALL CHECKS PASSED**; энтропия Standard = 16.26 бит, Polar = 13.48 бит; энергия в топ-1% пикселей = 93.3% (порог > 30%).
- `examples/demo_full_pipeline_mocomp` (V=50 м/с, α=0.30): до MOCOMP энтропия 6.82, пик 1948, drift 11 бинов; после MOCOMP энтропия 6.77, пик 3897, drift −9 бинов; улучшение пика ×2.0.

### Задача R3: кросс-масштабирование по данным (chirp-rate)

Оценка эффективной угловой скорости `Ω_eff` по линейному тренду чирпа вдоль кросс-дальности на сжатом по дальности и азимуту изображении (план §8).

- `processing/cross_range_scaling.py` — новый модуль: `segment_dominant_scatterers` (выделение доминирующих рассеивателей заливкой, порог mean + σ·std, дедупликация пиков), `azimuth_decompress` (точный обратный оператор к `azimuth_compress`: `ifft(ifftshift(...))`, восстановление строки по полной доплеровской полосе), `estimate_chirp_rate` (методы `lpft` — локальный FFT с максимумом контраста std/mean по сетке `init±25%`, дефолт по брифу; `phasefit` — линейно-фазовая регрессия лаг-произведения, наклон `a1/(2π·dt²)`, качество = R²; устойчив к постоянному доплеру, область однозначности `|μ|·dt² < 0.5`), `cross_range_scale` (взвешенный МНК `μ = K·y + b` с двукратным отсевом выбросов 2σ, вес = min(амплитуда, p75), критерий `K > 0, R² ≥ 0.85, N ≥ 3`, иначе `fallback`; возвращает `omega_eff`, `azimuth_axis_m`).
  - Конвенция знаков: в симуляторе фаза `s(t) = exp(j(2π·f_D·t + π·μ·t²))`, `μ = (2·f_c/c)·x·Ω²`; положительная кросс-дальность → положительный чирп (план §8.2.1 записан для сопряжённой конвенции `−a2/π` — указано в docstring).
  - Автофокусировка (mocomp) делает чирпы относительными к опорному рассеивателю `μ = K·(y − y_ref)`; наклон K сохраняется, `y_ref` поглощается точкой пересечения b (план §8.2.3).
  - Восстановление строки из усечённой (±3 бина) полосы спектра систематически занижает μ на ~12–20 % (усечение спектральной протяжённости чирпа), поэтому `azimuth_decompress` использует полную полосу; `decompress_window` доступен, но не рекомендуется.
- `processing/data_processor.py` — kwargs `cross_range_scaling="off"` (дефолт = битовая совместимость). При `"auto"` в `_compute_isar_standard` после азимутального сжатия вызывается `cross_range_scale(..., method="phasefit")`; при успехе ось азимута заменяется метровой осью от `Ω_eff`, иначе — `[fallback → cfg]`; прогресс эмитится в `progress_message`.
- `config.py` — поле `cross_range_scaling: str = "off"` (off/auto) в блоке «Обработка».
- `ui/main_window.py` — чекбокс «Кросс-масштабирование по данным (chirp-rate)» на вкладке «РЛИ» (включён ⇔ `config.cross_range_scaling="auto"`), прокидывается в оба конструктора `DataProcessor`.
- `examples/demo_cross_range_scaling.py` + PNG — синтетическая цель 5 рассеивателей (r = ±4.4/±2.2/0 м, u = ±2/±1/0 м/с), конвейер Raw→Range→MOCOMP→Azimuth→scale: `Ω_eff/ω ≈ 1.004`, `R² ≈ 1.00`, граф РЛИ + регрессия `μ̂(y)`.
- Тесты `tests/test_cross_range_scaling.py` (TDD, красный→зелёный): оценка μ на чистых чирпах (`phasefit`/`lpft`, μ ∈ {0.01, 0.1, 0.25}, + расширенный домен 0.5 для `lpft` с корректным init — граница однозначности), полная синтетическая регрессия (7 рассеивателей, `|Ω̂/Ω−1| < 10 %`, `R² > 0.85`), сквозной конвейер симулятора (5 рассеивателей, N=64), fallback при AWGN (snr_db = −5), битовая идентичность при `cross_range_scaling="off"` (np.array_equal на I и амплитуде).
- Регрессия: `pytest tests/ -q` — **77 passed** (66 baseline + 11 новых).
- Регрессия §10.2 (R3-вход, `cross_range_scaling` не влияет на выход РЛИ): `examples/test_data_processor_polar` — **ALL CHECKS PASSED**, энтропия Standard 13.59 / Polar 13.48, топ-1% 93.3%; `examples/demo_full_pipeline_mocomp` — до MOCOMP энтропия 6.82/пик 1948, после 6.73/4527 (×2.3) — значения baseline R2 (для N=16 автофокус-сценарий «auto» даёт fallback, N точек = 0 — прогон идентичен off, ось из cfg).

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

### Задача R5a: TF-ядро tf_analysis (STFT, маргиналы, RID, WVD)

Новый модуль `processing/tf_analysis.py` — частотно-временной анализ микро-Доплера и мгновенных RD-кадров. Источники: [TF, гл. 2 и (5.18)], [MD, §1.8, (2.76)–(2.78)] (Carson-полоса Бесселевской гребёнки). Все функции чистые (детерминированные).

- `stft_spectrogram(x, fs=1.0, nperseg=None, noverlap=None, window="hann", scaling_calibrated=True)` → `dict(f, t, Z, S_db)`: оконный STFT (`scipy.signal.stft`, `return_onesided=False` + fftshift → ось Доплера как `azimuth_compress.doppler_axis`, Гц), дефолты `nperseg = max(4, N//8)`, `noverlap = round(0.75·nperseg)`.
- `strongest_range_bins(P_comp, k=5)` — топ-K бинов по суммарной энергии |P|²; `micro_doppler_signature(P_comp, bin_index, fs, **stft_kwargs)` — сигнатура выбранного бина; `total_echo_signature(P_comp, fs, **stft_kwargs)` — спектрограмма суммарного эха `s(t) = Σ_m P[m, t]` + энергетическое объединение `Σ_m |STFT(P[m,:])|²` [MD, §1.8].
- `rid_cube(P_comp, fs=1.0, n_frames=16, nperseg=None, window="hann", hop=1, dtype_mag=np.float32)` → `dict(Q_mag=(M,F,T) float32, f, t, frame_idx, approximate, meta)`: маргинальная калибровка по одиночному тону (`Σ_t|STFT|² = |FFT|² ⊛ |W|²` при hop=1), бюджет памяти ≤ 100 МБ (автопрореживание дальностных бинов до топ-K, `meta["bins_used"]`), `hop > 1 → approximate=True`. `rid_frame(rid, i)` — dB-кадр, готов к imshow; `rid_marginal(rid)` — `Σ_t Q` [TF, (5.18)].
- `wigner_ville(x, fs=1.0)` — WVD аналитического сигнала (lag-матрица N×N, FFT по лагам, ось `f = k·fs/(2N)`) [TF, §2.2.1]; кросс-термы — свойство билинейности, задокументировано тестом.
- Калибровочная константа модуля `SCALE = _marginal_scale(256, 128, "hann")` ≈ 292.25417: выведена аналитически для чистого тона (частотный маргинал спектрограммы = `|FFT|² ⊛ |W|²`), применяется через `_marginal_scale(N, nperseg, window)`.
- Тесты `tests/test_tf_analysis.py` + фикстуры `radar_tf`/`target_tf` (N=256) в `tests/conftest.py`: одиночный тон (гребень в пределах 0.2 бина); полоса вибрационной сигнатуры (95% энергии в Carson-полосе, допуск ±15% мгновенной полосы — ≥0.90); маргинал = |FFT|² (rel ~1e-16 при hop=1); формы RID-куба и 16 равноотстоящих кадров; WVD-концентрация ЛЧМ (0.92 против 0.37 у STFT); кросс-терм двух тонов на средней частоте (0 дБ); детерминизм `np.array_equal`.
- Демо `examples/demo_micro_doppler_stft.py` (вибрация: профили, STFT-спектрограмма + пунктиры ±6.28 Гц, WVD, суммарная сигнатура; PNG) и `examples/demo_rid_imaging.py` (16 RID-кадров, маргинал vs |FFT|², печать относительной ошибки; PNG).
- `pytest tests/ -q`: **53 passed** (46 + 7 новых).

### Задача R5b: GUI-вкладка «Сигнатуры» и сигнал `tf_ready`

TF-анализ из R5a подключён к GUI: новая вкладка «Сигнатуры» (позиция 4, между «РЛИ» и «Поток кадров») и сигнал `DataProcessor.tf_ready` для стандартного конвейера (гл. 2/5). Дефолт выключен — конвейер битово идентичен прежнему.

- `processing/data_processor.py` — сигнал `tf_ready = pyqtSignal(object)` (dict) рядом с `isar_ready`; kwargs `tf_enabled=False, tf_n_frames=16, tf_window="hann"`. В `_compute_isar_standard` ПОСЛЕ MOCOMP-блока (P_proc выровнен) при `tf_enabled` считаются `rid_cube(P_proc, fs=1/pri, n_frames, window)` + спектрограммы топ-5 бинов (`strongest_range_bins`) и суммарного эха; эмитится один dict за прогон (куб, `top_bins`, `signatures`, `total_echo`, строки P для WVD, `fs`, `n_frames`). Прогресс через `progress_message` («TF-анализ: STFT…»). Полярный путь не тронут.
- `ui/main_window.py` — вкладка «Сигнатуры» (между «РЛИ» и «Поток кадров», индексы 3/4/5 обновлены и в `_show_stream_results`): QComboBox «Дальний бин» («Суммарное эхо» + топ-5 бинов, обновляется на каждый `tf_ready`), QComboBox «Преобразование» (STFT/WVD; WVD «Суммарного эха» автоматически заменяется топ-бином), QSlider «Кадр t» с подписью времени кадра, чекбокс «Маргинальная сумма (≈ Фурье)». Канвас — pyqtgraph ImageItem (паттерн вкладок «Сигнал»/«РЛИ»), `imshow(S_db, vmin=-40, vmax=0)`, оси «Доплер, Гц» / «Время, с». Режимы: STFT-спектрограмма бина/суммарного эха, WVD (лениво + кэш из строк куба), RID-кадр (слйдер), маргинал `Σ_t Q` [TF, (5.18)].
- `config.py` — поля `tf_enabled: bool = False`, `tf_n_frames: int = 16`, `tf_window: str = "hann"` в блоке «Обработка».
- `PARAMETERS_GUIDE.md` — раздел «Частотно-временной анализ: вкладка „Сигнатуры“»: выбор `nperseg` по критерию (7.2.8) `T_w < 1/√|df_D/dt|` и кросс-рейндж-границе, число кадров 16–32 [TF, §5.3], единицы «Доплер, Гц» (`fs = 1/PRI`).
- Тесты `tests/test_tf_gui.py` (TDD, красный→зелёный): битовая идентичность при `tf_enabled=False`/без поля (± явный прежний конвейер, сигнал не эмитится), `tf_enabled=True` → dict с `Q_mag (M,F,T)`/`frame_idx`/`top_bins≥5`, GUI smoke offscreen (вкладка на позиции 4, элементы на месте), обновление контролов и всех режимов отрисовки, `rid_frame`/`rid_marginal` dB-smoke. Замер переключения бина/кадра на пресете 256×256 — макс. ~2 мс (≤ 200 мс).
- `pytest tests/ -q`: **59 passed** (54 + 5 новых).

### Задача R6: keystone-перешкалирование и флаг `use_keystone` в конвейере

Keystone-коррекция вращательной миграции ячеек дальности — C&M, §5.5, (5.30)/(5.37). Модуль `processing/keystone.py` реализован в R6-core; эта задача интегрирует его в стандартный конвейер флагом `use_keystone` (дефолт False — битовая идентичность прежнему пути).

- `processing/keystone.py` — `keystone_transform(E, f_r, f_c, method="sinc"|"cubic", pad=None)`: перешкалирование медленного времени `t_ks = (f_c/f_r)·t`; выходной отсчёт j берётся из строки по индексу `j·(f_c/f_r[m])` (⚠ не `j·(f_r/f_c)` — направление зафиксировано тестом §9.6.1). `sinc` — band-limited FFT-оверсемплинг ×16 + локальная 4-точечная кубическая интерполяция (ошибка < 1e-6 на «неутекающих» тонах); `cubic` — CubicSpline (fallback, ~1e-2). Нулевая приставка по медленному времени (авто `ceil(N·(f_c/min(f_r)−1))+2`).
- `processing/data_processor.py` — kwarg `use_keystone=False` (рядом с `cross_range_scaling`). В `_compute_isar_standard` ПОСЛЕ MOCOMP-блока и ДО TF-блока (3b), как и в `_standard_frame`, при `self.use_keystone` выполняется гибрид §9.4 п.2: `E_comp = np.fft.fft(np.fft.ifftshift(P_proc, axes=0), axis=0)` → `keystone_transform(E_comp, f_r, self.f_c)` → повторное `range_compress` → далее прежний путь (азимут, R3). Двойное сжатие дальности приемлемо — флаг off по умолчанию. Гибрид корректен: MOCOMP (поступательная компенсация) выполняется ДО keystone, как требует [C&M, §5.5]; γ-миграция остаётся для RID (R5) — так задумано.
- `config.py` — поле `use_keystone: bool = False` в блоке «Обработка».
- Критерии применимости (задокументированы в `PARAMETERS_GUIDE.md`): `y_max·ω·T > dr`; остаточный член (C&M (5.40), план (9.2.8)) `Ω²·T²·r_max < 4·dr`, иначе нужен PFA/TF; остаточный чирп (5.41) остаётся автофокусу/R3.
- Тесты `tests/test_keystone.py` (TDD, +1, всего 5): `test_disabled_is_bitwise_identical` (§9.6.4) — `DataProcessor` с `use_keystone=False` и конструкция без параметра дают битово равные комплексное I и амплитуду (`np.array_equal`), красный→зелёный (TypeError → pass).
- Регрессия §10.2 (честный замер, keystone off → on): solarB N=16, Δθ=9°, PRI=1 с — энтропия 8.8875 → 8.9420, пик 3308.6 → 2685.8, std|I| 236.4 → 233.1; demo_full_pipeline_mocomp (ICEsat2, V=50, N=16) — энтропия 8.7413 → 8.6963, пик 4527.1 → 3615.9; solarB N=128 (та же Δθ=9°) — энтропия 11.1594 → 11.0600 (−0.10), пик 23956 → 21686. На N=16 keystone **нейтрален**: миграция `y_max·ω·T ≈ 1.86 м ≈ 6.2 dr` при Δθ=9° формально превышает ячейку, но с 16 импульсами азимутальное БПФ не имеет разрешения извлечь выгоду, а двойное сжатие добавляет интерполяционную ошибку (ΔH ≤ ±0.05 бит, пик ×0.80–0.81). Реальный выигрыш фокусировки — на стенде §9.5 (N=256, миграция ≈ 8.5 dr): энтропия 6.984 → 3.393 бит (−51%), пик 35.5 → 130.7 (×3.68); при N≥128 на пресете заметно небольшое улучшение энтропии (−0.10 бит).
- Регрессия: `pytest tests/ -q` — **78 passed** (77 + 1); `examples/demo_keystone` — **ALL CHECKS PASSED** (см. числа выше); `examples/test_data_processor_polar` — **ALL CHECKS PASSED** (энтропия Standard 13.59 / Polar 13.48, топ-1% 93.3%).

## [2.4.0] — 2026-08-31

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
