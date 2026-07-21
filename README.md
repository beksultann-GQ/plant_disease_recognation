# 🌿 Өсімдік ауруларын анықтау — CNN + Transfer Learning

Жапырақ суреті бойынша өсімдік ауруын 3 классқа жіктеу:
**Healthy** (сау), **Powdery** (ұнтақ шық), **Rust** (тат ауруы).

## Нәтиже

| Метрика | Мәні |
|---|---|
| **Test Accuracy** | **94.67%** (150-ден 142) |
| Precision (macro) | 0.9540 |
| Recall (macro) | 0.9467 |
| F1-score (macro) | 0.9475 |

8 қатенің бәрі — ауру жапырақты «сау» деп тану (False Negative). Жалған дабыл жоқ.

## Әдіс

- **Модель:** MobileNetV2 (ImageNet-те алдын ала оқытылған) + Dropout + Dense(3, softmax)
- **Augmentation:** RandomFlip, RandomRotation, RandomZoom, RandomContrast
- **Екі кезеңді оқыту:** алдымен «бас» (lr=1e-3), содан кейін fine-tuning (lr=1e-5)
- **Бағалау:** Accuracy, Precision, Recall, F1, Confusion Matrix, ROC/AUC, Grad-CAM

## Деректер

[Kaggle — Plant Disease Recognition](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset) (~1530 сурет). Ноутбук іске қосылғанда `kagglehub` арқылы автоматты жүктеледі.

> **Ескерту:** штаттық валидация тым аз (60 сурет) болғандықтан, оны 288-ге дейін кеңейттік (Train-нен 15%). Тест выборкасы (150 сурет) қозғалмады — барлық параметр валидация бойынша таңдалды.

## Іске қосу

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/jupyter notebook plant.ipynb
```

## Файлдар

```
plant.ipynb                     — негізгі ноутбук (код + графиктер)
plant_src.py                    — ноутбуктің бастапқы коды (jupytext)
plant_disease_mobilenetv2.keras — оқытылған модель (іске қосқанда жасалады)
requirements.txt                — кітапханалар
```
