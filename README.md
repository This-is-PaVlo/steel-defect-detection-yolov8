# Steel Defect Detection with YOLOv8

## English

### Overview
This project implements automatic steel surface defect detection using the YOLOv8 object detection model.

The model was trained on the NEU Surface Defect Database and can detect multiple types of industrial steel defects.

The project includes:
- dataset preparation
- XML to YOLO annotation conversion
- dataset validation
- annotation visualization
- YOLOv8 model training
- validation and evaluation
- prediction visualization
- metrics analysis

### Technologies
- Python
- PyTorch
- YOLOv8
- OpenCV
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Albumentations

### Dataset
Dataset: NEU Surface Defect Database

Defect classes:
- crazing
- inclusion
- patches
- pitted_surface
- rolled-in_scale
- scratches

Dataset download:
https://drive.google.com/file/d/1ID32r-OK2syUrV_RQr-Bq0l_JCjGwV87/view

After downloading, place the archive in:
```text
data/raw/NEU-DET.zip
```

### Features
- automatic dataset download
- XML annotation parsing
- YOLO format conversion
- dataset structure validation
- class distribution analysis
- annotation visualization
- YOLOv8 training pipeline
- prediction rendering
- training metrics visualization

### Model
Model architecture:
- YOLOv8n

Evaluation metrics:
- Precision
- Recall
- mAP50
- mAP50-95

### How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python src/main.py
```

### Project Structure

```text
steel-defect-detection-yolov8/
│
├── data/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── results/
│   └── .gitkeep
│
├── src/
│   └── main.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Output
The project generates:
- trained YOLOv8 model weights
- validation metrics
- confusion matrix
- training graphs
- prediction visualizations

---

## Русский

### Описание
Проект реализует автоматическую детекцию дефектов поверхности стали с использованием модели YOLOv8.

Модель обучается на датасете NEU Surface Defect Database и способна обнаруживать несколько типов промышленных дефектов поверхности металла.

Проект включает:
- подготовку датасета
- конвертацию XML-аннотаций в формат YOLO
- проверку датасета
- визуализацию разметки
- обучение YOLOv8
- валидацию и оценку качества
- визуализацию предсказаний
- анализ метрик

### Технологии
- Python
- PyTorch
- YOLOv8
- OpenCV
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Albumentations

### Датасет
Датасет: NEU Surface Defect Database

Классы дефектов:
- crazing
- inclusion
- patches
- pitted_surface
- rolled-in_scale
- scratches

Скачать датасет:
https://drive.google.com/file/d/1ID32r-OK2syUrV_RQr-Bq0l_JCjGwV87/view

После скачивания поместите архив в:
```text
data/raw/NEU-DET.zip
```

### Возможности
- автоматическая загрузка датасета
- обработка XML-аннотаций
- конвертация в формат YOLO
- проверка структуры датасета
- анализ распределения классов
- визуализация разметки
- обучение YOLOv8
- визуализация предсказаний
- построение графиков обучения

### Модель
Архитектура модели:
- YOLOv8n

Метрики качества:
- Precision
- Recall
- mAP50
- mAP50-95

### Запуск

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Запуск обучения:

```bash
python src/main.py
```

### Структура проекта

```text
steel-defect-detection-yolov8/
│
├── data/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── results/
│   └── .gitkeep
│
├── src/
│   └── main.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Результат
В результате работы проекта создаются:
- обученные веса YOLOv8
- метрики качества
- confusion matrix
- графики обучения
- визуализация предсказаний