# 🩺 RSNA Knee Abnormality Detection: A Diagnostic Quest

Welcome to the clinic of the future!  
You are a radiologist — but not an ordinary one. You are armed with neural networks.  
Your task is to make **12 key diagnoses** from knee MRI studies, using both images and, during training, the accompanying radiology reports.

---

## 🧩 The Core Challenge

Each study is a **puzzle** made of several MRI series — slices acquired in different planes and with different contrast weightings.  
You must assemble the full picture and answer 12 questions: Is there a ligament tear? Meniscal damage? Osteoarthritis? Inflammation? A cyst? A fracture?

**Metric:** macro-averaged ROC AUC across all 12 labels.  
**Key twist:** during training, you have access not only to the images but also to the **radiologists’ free-text reports**. At inference time, reports are absent — so you must learn to squeeze every bit of signal from them while training.

---

## 🧠 Analogy: “Training a Virtual Radiology Resident”

Imagine you are teaching a new resident:

- **DICOM series** — different examination tools: sagittal, coronal, and axial views.
- **Fluid-sensitive sequences** (T2, PD, STIR) — a flashlight that highlights fluid, edema, and tears.
- **train.csv** — patient charts with partially completed diagnoses.
- **train_series.csv** — a description of each scan: plane, fat suppression, fluid sensitivity.
- **Radiology reports** — notes from an experienced mentor who sometimes writes “ACL tear” and sometimes “intact menisci.”

Your resident (the model) must learn to make diagnoses even when the mentor is not there (at test time).

---

## 🏆 Difficulty Tiers of the Diagnoses

| Tier | Diagnoses | Why They Are Hard | What Helps |
|------|-----------|-------------------|------------|
| 🟢 **Easier** | Baker's cyst, Effusion | Clearly visible on fluid-sensitive sequences | A single sagittal series |
| 🟡 **Moderate** | MCL, Medial/Lateral Meniscus, PF OA | Need assessment across multiple planes | 2–3 series + attention |
| 🔴 **Hard** | ACL, Synovitis, Contusion, Fracture | May be subtle; require broader context | Multi-plane aggregation, 2.5D/3D inputs |
| 🟣 **Very Hard** | Medial OA, Lateral OA | Cartilage thickness assessment; need thin slices | Specialized models + post-processing |

---

## 💡 The Main Takeaway

**This competition is won by whoever does three things best:**

1. **Select the right series** — not all 500 GB of data is useful.
2. **Exploit the reports** to expand weak labels — they are free supervision.
3. **Aggregate carefully** across planes and slices.

No magic — only solid data engineering and thoughtful modeling.

---

**Let’s go!** 🚀  
We start with a simple diagnostic tool and gradually turn it into a radiology expert.

# 📊 Evolution of Public Solutions: 0.937 → 0.94 → 0.941

## 🥉 Work 1 — 0.937
**Head and Shoulders, Knees and Toes**

🔗 https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes

- CoAtNet branch expanded to **4 arms**: v5, v10, v5-reverse, v8.
- Reverse-flip adds diversity without new training.
- Outer CoAtNet weight becomes **global 0.60** instead of per-target.

---

## 🥈 Work 2 — 0.94
**RSNA Knee Hybrid Raptor — Lateral Meniscus R100**

🔗 https://www.kaggle.com/code/renta0426/rsna-knee-hybrid-raptor-lateral-meniscus-r100

- Starts from the exact 0.939 parent (reproduced 1-to-1).
- Changes **only one label**: Lateral Meniscus.
- Outer routing for Lateral Meniscus: Transformer/Rad **0.00** / strengthened Raptor **1.00**.
- All other 11 labels remain **bit-for-bit equal** to the parent at 0.40 / 0.60.

> **Key idea:** isolate one problematic target and give it the full weight of the strongest branch.

---

## 🥇 Work 3 — 0.941
**Bend the Knee to the Dinosaurs (ALL PUBLIC)**

🔗 https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs

- Keeps the 4-arm Raptor from Work 1.
- Adds **residual-gated CoAtNet** (e4/e6/e8) inside the Raptor branch.
- Inner blend: public Raptor 0.60 / residual-CoAt 0.40.
- Adds a public **CoAtNet MRI reader**: 384 px adjacent-slice triplets → CoAtNet RMLP-2 backbone.
- Finding-specific gated spatial head + second attention stage pools windows across the study.
- Updated CoAtNet arm weights: maxspan-v5 = **0.60**, reverse = **0.10**.
- Per-target outer routing: ACL **0.75**, Medial Meniscus **0.80**, Lateral Meniscus **1.00**, Lateral OA **0.75**, Fracture **0.75**.

> **Key idea:** strengthen the Raptor arm with a genuinely different visual inductive bias and let each finding lean on whichever branch predicts it best.

---

## Key Pattern

Each step isolates a single high-impact change:

1. **Diversity** — more CoAtNet arms + reverse-flip.
2. **Inner strengthening** — residual-gated CoAt.
3. **Target-specific routing** — give full weight to the best model for a problematic label.

This is how the public solutions moved from **0.937 → 0.94 → 0.941** — and where we build from.

---

## 📚 Credits

We stand on the shoulders of giants. Our work reproduces, extends, and improves on the public solutions above. Full credit to their authors and contributors.

- **Work 1 (0.937)** — [prvsiyan](https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes)
- **Work 2 (0.94)** — [renta0426](https://www.kaggle.com/code/renta0426/rsna-knee-hybrid-raptor-lateral-meniscus-r100)
- **Work 3 (0.941)** — [Mattia Angeli](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs) + credits: Pilkwang, Sofia Anjenje, Antoine G., prvsiyan, Marwan Mahmoud, dreaddevelopment/Roman Tamrazov, renta.k, Anvith Pothula

# 1. 🚀 Инициализация инференса и проверка GPU

## Что делает эта ячейка

Готовит окружение для предсказаний на тестовых данных:

- Настраивает количество потоков CPU.
- Проверяет доступность CUDA на каждой GPU.
- Создаёт список устройств `DEVS`, на которых будет выполняться инференс.
- Фиксирует seed для воспроизводимости.
- Задаёт все параметры предобработки, слотов и аугментаций.

Здесь не происходит чтения DICOM или загрузки моделей — только подготовка.

# 2. 🧭 Вспомогательные функции и планировщик кэша

## Что делает эта ячейка

Определяет функции, которые используются на протяжении всего инференса:

- `find_root()` — находит корень данных соревнования.
- `find_dinov2()` — находит DINOv2 модель нужного варианта.
- `plan_cache()` — рассчитывает, сколько срезов можно хранить в памяти.
- `read_labels()` — загружает метки из отчётов (не используется в инференсе, но оставлена для совместимости).

Также сразу вычисляется `N_GROUP` — сколько групп срезов будет загружено для каждой серии.

В инференсе эти функции обеспечивают правильные пути и распределение памяти.

# 3. 🦴 Чтение DICOM-заголовков и определение латеральности

## Что делает эта ячейка

Считывает метаданные DICOM для каждой серии без загрузки пикселей:

- `probe()` — открывает один срез из серии и извлекает ключевые теги.
- `walk()` — обходит все серии train/test и собирает метаданные в DataFrame.
- `annotate()` — восстанавливает тип последовательности (T1/T2/PD/GRE) и наличие fat suppression.
- `lat_of()` — определяет сторону колена (левое/правое) по тегу Laterality или геометрии.

Это необходимо для правильного выбора слотов и нормализации laterality перед инференсом.

# 4. 🗂️ Выбор серий для слотов

## Что делает эта ячейка

Для каждого исследования выбирает наиболее подходящую серию под каждый из шести слотов:

- Sagittal / Coronal / Axial × Fluid-Sensitive / Non-Fluid-Sensitive / T1.
- Если точного совпадения нет, используется fallback.

Результат — словарь `{study: {slot_name: series_row}}`, который потом используется для построения кэша изображений.

# 5. 📐 Геометрическая сортировка и чтение срезов

## Что делает эта ячейка

- `order_slices()` — сортирует DICOM-файлы по физическому положению среза в пространстве, а не по имени файла.
- `read_slot()` — загружает центральные срезы, кадрирует по 130 мм, нормализует интенсивность по 1–99 перцентилям и приводит к `IMG×IMG`.

Результат — тензор uint8 размером `(n_slices, IMG, IMG)`, готовый для модели.

# 6. 🔄 Нормализация латеральности

## Что делает эта ячейка

Приводит все колени к единой (левой) ориентации:

- Для корональных и аксиальных срезов правого колена — горизонтальный flip.
- Для сагиттальных — реверс порядка срезов.

Это важно для медиальных/латеральных структур, которые зависят от стороны.

# 7. 💾 Построение кэша изображений

## Что делает эта ячейка

Читает все выбранные серии и сохраняет их в памяти как uint8-массив.

- Сначала сортирует срезы геометрически.
- Затем декодирует и нормализует каждый слот.
- Возвращает `(studies, cache, mask)`.

Это самая тяжёлая по I/O часть инференса. После неё модели работают быстро, без повторного чтения DICOM.

# 8. 🧠 SlotHead: внимание по слотам

## Что делает эта ячейка

Определяет голову, которая агрегирует слоты с помощью внимания.

- Каждый слот проецируется в скрытое пространство.
- Для каждого диагноза обучается свой query.
- Внимание маскирует отсутствующие слоты.
- Итоговый вектор — взвешенная сумма слотов.

Это ключевая часть DINOv2 трансформера.

# 9. 🏗️ Основная модель DINOv2

## Что делает эта ячейка

Оборачивает DINOv2 backbone и SlotHead в единую модель.

- Принимает батч изображений из кэша.
- Прогоняет через DINOv2.
- Извлекает CLS-токен и средний патч.
- Передаёт в SlotHead для получения 12 логитов.

# 10. 📦 Загрузка DINOv2

## Что делает эта ячейка

Загружает предобученный DINOv2 из подключённой модели.

- Замораживает все параметры.
- Размораживает последние `unfreeze_last` блоков.
- Возвращает модель с головой.

# 11. 🔍 Проверка целостности, ансамблирование и инференс пакета

## Что делает эта ячейка

Содержит функции для:

- `fingerprint` / `check_fingerprint` — проверка, что веса модели соответствуют ожидаемым.
- `predict_member` — инференс одной модели с TTA-пулингом.
- `infer_from_package` — загрузка манифеста из пакета весов, построение кэша и прогон всех моделей.
- `_combine` — взвешенное ранговое усреднение предсказаний.

Это ядро DINOv2-ветки. Создаёт `submission.csv` после обработки всех 20 моделей.

# 12. 🔄 Аугментация и предсказание

## Что делает эта ячейка

- `augment()` — лёгкие случайные повороты, сдвиги, масштаб и изменение яркости.
- `predict()` — прогон модели по всем группам срезов и усреднение вероятностей.
- `macro_auc()` — метрика для валидации (в инференсе не используется, но нужна для совместимости).

# 15. 📝 Запись submission и валидация формата

## Что делает эта ячейка

- `write_submission()` — конвертирует предсказания в ранги и сохраняет CSV.
- `write_benchmark_submission()` — создаёт fallback-файл с 0.5.
- `_v37_validate_submission()` — проверяет порядок колонок, уникальность StudyInstanceUID и конечность значений.

**Важно:** функция `main()` в этой ячейке не нужна — она относится к обучению, а не к инференсу.

# 15b. 🦕 DINOv2: создание первого submission.csv

# 17. 🦖 DINOv3: инициализация

## Что делает эта ячейка

Готовит параметры для DINOv3 ветки:

- 6 слотов × 16 срезов.
- Ищет датасет с фолдами `knee-mri-fold-weights`.
- Определяет устройство (CUDA/CPU).

Сама загрузка моделей будет в следующей ячейке.

# 18. 🦖 DINOv3: подготовка исследований

## Что делает эта ячейка

Загружает тестовые исследования и строит volume для каждого:

- 6 слотов × 16 срезов.
- Сортировка срезов по InstanceNumber.
- Нормализация laterality.
- Кадрирование 130 мм, ресайз до 336×336.

Результат — массив `(N_SLOT, N_SLICE, SIZE, SIZE)` и маска слотов.

# 19. 🦖 DINOv3: архитектура и загрузка моделей

## Что делает эта ячейка

Определяет все компоненты DINOv3 ветки:

- Различные пулинги (MeanMax, Attention, TokenX, Codex).
- Механизмы смешивания слотов по глубине.
- Основную модель `Net`.
- Загружает 5 фолдов `m_f0..m_f4` из чекпоинтов.

После этой ячейки `models` содержит 5 готовых DINOv3 моделей.

# 20. 🦖 DINOv3: инференс и ранговое усреднение

## Что делает эта ячейка

- Выполняет инференс всех 5 фолдов DINOv3.
- Считает ранги предсказаний.
- Усредняет ранги фолдов.
- Сохраняет результат в `A5_PREDS`.

Это вторая ветка пайплайна.

# 21. 🦖 DINOv3: смешивание с DINOv2

## Что делает эта ячейка

Смешивает предсказания DINOv3 с текущим `submission.csv` (DINOv2):

- DINOv2: вес 0.55
- DINOv3: вес 0.45

Результат — обновлённый `submission.csv`.

# 22. 🩻 RadImageNet: инференс и калибровка

## Что делает эта ячейка

Выполняет третью ветку пайплайна:

- Загружает ResNet50 RadImageNet и две группы голов (public и reference).
- Строит кэш для трёх раскладок слотов (E10, E13, V48-pass2).
- Смешивает предсказания с текущим submission.
- Применяет калибровку для части диагнозов.

Обновляет `submission.csv`.

# 23. 🦖 CoAtNet 4 руки + residual-gated + outer-routing

## Что делает эта ячейка

Финальная и самая важная часть пайплайна:

1. Воспроизводит публичный Raptor (4 руки CoAtNet).
2. Добавляет residual-gated CoAt (e4/e6/e8) с alpha 0.40.
3. Смешивает с Transformer/Rad.
4. Для Lateral Meniscus меняет outer-routing: Raptor = 100%, Transformer/Rad = 0%.
5. Создаёт итоговый `submission.csv`.

Именно эта ячейка отличает решение 0.94 от 0.939.