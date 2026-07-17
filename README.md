# 🌿 Plant Disease Recognition — CNN + Transfer Learning

Классификация болезней растений по фотографии листа на 3 класса:
**Healthy** (здоровый), **Powdery** (мучнистая роса), **Rust** (ржавчина).

## Датасет

[Kaggle — Plant Disease Recognition Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset) (~1530 изображений):

| Выборка | Healthy | Powdery | Rust | Всего |
|---|---|---|---|---|
| Train | 458 | 430 | 434 | 1322 |
| Validation | 20 | 20 | 20 | 60 |
| Test | 50 | 50 | 50 | 150 |

Датасет скачивается автоматически через `kagglehub` при запуске ноутбука.

## Результаты

| Метрика | Значение |
|---|---|
| **Test Accuracy** | **94.67%** (142/150) |
| Precision (macro) | 0.9540 |
| Recall (macro) | 0.9467 |
| F1-score (macro) | 0.9475 |

| Класс | Precision | Recall | F1 |
|---|---|---|---|
| Healthy | 0.8621 | 1.0000 | 0.9259 |
| Powdery | 1.0000 | 0.9400 | 0.9691 |
| Rust | 1.0000 | 0.9000 | 0.9474 |

**Анализ ошибок:** все 8 ошибок — ложноотрицательные (больной лист определён как здоровый).
Ложных тревог нет. Для агрономии это самый дорогой тип ошибки — пропущенный очаг
распространяется по полю, поэтому в проде порог стоит смещать в сторону «болен».

**Методологическое замечание.** Штатная валидационная выборка датасета — 60 изображений,
чего недостаточно для отбора модели: `val_accuracy` меняется шагами по 1.7%, `val_loss` шумит,
и EarlyStopping фиксирует случайную эпоху. Мы выделили дополнительные 15% из обучающей
выборки (60 → 288 изображений); **тестовая выборка при этом не затрагивалась**.
Все гиперпараметры выбраны по валидации, а не по тесту.

## Метод

- **Архитектура:** MobileNetV2 (предобучена на ImageNet) + GlobalAveragePooling + Dropout + Dense(3, softmax)
- **Аугментация:** RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomContrast
- **Обучение в 2 этапа:**
  1. *Feature extraction* — база заморожена, обучается только голова (Adam, lr=1e-3)
  2. *Fine-tuning* — разморожены верхние слои базы (Adam, lr=1e-5, EarlyStopping, ReduceLROnPlateau)
- **Оценка:** Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC/AUC, Grad-CAM

## Запуск

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook plant.ipynb
```

## Структура проекта

```
plant.ipynb                        — основной ноутбук (весь пайплайн + графики)
plant_src.py                       — исходник ноутбука (формат jupytext py:percent)
plant_disease_mobilenetv2.keras    — обученная модель (создаётся при запуске)
requirements.txt                   — зависимости
```
