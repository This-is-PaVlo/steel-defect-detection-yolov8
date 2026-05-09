# -*- coding: utf-8 -*-

import os
import cv2
import yaml
import shutil
import zipfile
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from tqdm.auto import tqdm
from xml.etree import ElementTree as ET

import torch
import gdown
from ultralytics import YOLO

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)


def print_system_info():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# ОСНОВНЫЕ ПУТИ И ПАРАМЕТРЫ
# ============================================================

PROJECT_DIR = Path(".")
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
YOLO_DATASET_DIR = PROCESSED_DIR / "neu_yolo"
MODELS_DIR = PROJECT_DIR / "models"

RUN_NAME = "yolov8n_baseline"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EPOCHS = 30
BATCH_SIZE = 16
YOLO_IMGSZ = 640
CONF_THRES = 0.25
IOU_THRES = 0.50

for path in [PROJECT_DIR, DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# ЗАГРУЗКА ДАТАСЕТА
# ============================================================

FILE_ID = "1ID32r-OK2syUrV_RQr-Bq0l_JCjGwV87"

ZIP_PATH = RAW_DIR / "NEU-DET.zip"
EXTRACT_DIR = RAW_DIR
NEU_DIR = EXTRACT_DIR / "NEU-DET"


def download_dataset():

    if not ZIP_PATH.exists():
        url = f"https://drive.google.com/uc?id={FILE_ID}"

        print("Downloading dataset...")

        gdown.download(
            url,
            str(ZIP_PATH),
            quiet=False
        )

    else:
        print(f"Dataset already exists: {ZIP_PATH}")

    if not NEU_DIR.exists():

        print("Extracting archive...")

        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

        print(f"Extracted to: {EXTRACT_DIR}")

    else:
        print(f"Dataset already extracted: {NEU_DIR}")


# ============================================================
# ПОИСК ИЗОБРАЖЕНИЙ
# ============================================================

RAW_IMAGES_TRAIN_DIR = NEU_DIR / "train" / "images"
RAW_IMAGES_VAL_DIR = NEU_DIR / "validation" / "images"

RAW_XML_TRAIN_DIR = NEU_DIR / "train" / "annotations"
RAW_XML_VAL_DIR = NEU_DIR / "validation" / "annotations"


# ============================================================
# СОЗДАНИЕ YOLO СТРУКТУРЫ
# ============================================================

def prepare_yolo_dirs():

    if YOLO_DATASET_DIR.exists():
        shutil.rmtree(YOLO_DATASET_DIR)

    (YOLO_DATASET_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
    (YOLO_DATASET_DIR / "images" / "validation").mkdir(parents=True, exist_ok=True)

    (YOLO_DATASET_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (YOLO_DATASET_DIR / "labels" / "validation").mkdir(parents=True, exist_ok=True)

    print("YOLO dataset structure created")


# ============================================================
# XML -> YOLO
# ============================================================

def collect_all_images(images_dir: Path):

    return sorted([
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])


def build_image_index(images_dir: Path):

    image_files = collect_all_images(images_dir)

    index = {}

    for img_path in image_files:
        index[img_path.stem] = img_path

    return index


def convert_bbox_to_yolo(size, box):

    width, height = size
    xmin, ymin, xmax, ymax = box

    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height

    bbox_w = (xmax - xmin) / width
    bbox_h = (ymax - ymin) / height

    return x_center, y_center, bbox_w, bbox_h


def process_split(split_name, xml_dir, image_dir, yolo_dir):

    stats = {
        "images": 0,
        "labels": 0,
        "objects": 0
    }

    xml_files = sorted(xml_dir.glob("*.xml"))

    image_index = build_image_index(image_dir)

    for xml_path in tqdm(xml_files, desc=f"Processing {split_name}"):

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            stem = xml_path.stem

            image_path = image_index.get(stem)

            if image_path is None:
                continue

            size_tag = root.find("size")

            width = int(size_tag.find("width").text)
            height = int(size_tag.find("height").text)

            yolo_lines = []

            for obj in root.findall("object"):

                class_name = obj.find("name").text.strip()

                if class_name not in CLASS_TO_ID:
                    continue

                bndbox = obj.find("bndbox")

                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                x_center, y_center, bbox_w, bbox_h = convert_bbox_to_yolo(
                    (width, height),
                    (xmin, ymin, xmax, ymax)
                )

                class_id = CLASS_TO_ID[class_name]

                yolo_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}"
                )

                stats["objects"] += 1

            target_img = yolo_dir / "images" / split_name / image_path.name
            target_lbl = yolo_dir / "labels" / split_name / f"{image_path.stem}.txt"

            shutil.copy2(image_path, target_img)

            with open(target_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

            stats["images"] += 1
            stats["labels"] += 1

        except Exception as e:
            print(f"Error processing {xml_path.name}: {e}")

    return stats


# ============================================================
# DATASET.YAML
# ============================================================

def create_dataset_yaml():

    dataset_yaml = {
        "path": str(YOLO_DATASET_DIR),
        "train": "images/train",
        "val": "images/validation",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)}
    }

    with open(YOLO_DATASET_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, allow_unicode=True, sort_keys=False)

    print("dataset.yaml created")


# ============================================================
# ПРОВЕРКА ДАТАСЕТА
# ============================================================

def validate_yolo_split(split_name):

    images_dir = YOLO_DATASET_DIR / "images" / split_name
    labels_dir = YOLO_DATASET_DIR / "labels" / split_name

    image_files = sorted(images_dir.glob("*"))
    label_files = sorted(labels_dir.glob("*.txt"))

    print(f"\n[{split_name}]")
    print("Images:", len(image_files))
    print("Labels:", len(label_files))


# ============================================================
# ВИЗУАЛИЗАЦИЯ РАЗМЕТКИ
# ============================================================

def draw_yolo_boxes(image_path, label_path):

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]

    if label_path.exists():

        lines = label_path.read_text(encoding="utf-8").splitlines()

        for line in lines:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls_id, x_center, y_center, bw, bh = parts

            cls_id = int(cls_id)

            x_center, y_center, bw, bh = map(
                float,
                [x_center, y_center, bw, bh]
            )

            x1 = int((x_center - bw / 2) * w)
            y1 = int((y_center - bh / 2) * h)

            x2 = int((x_center + bw / 2) * w)
            y2 = int((y_center + bh / 2) * h)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                image,
                CLASS_NAMES[cls_id],
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

    return image


def visualize_samples():

    train_images_dir = YOLO_DATASET_DIR / "images" / "train"
    train_labels_dir = YOLO_DATASET_DIR / "labels" / "train"

    sample_images = sorted(train_images_dir.glob("*"))

    if len(sample_images) == 0:
        return

    sample_images = random.sample(
        sample_images,
        min(6, len(sample_images))
    )

    plt.figure(figsize=(15, 10))

    for i, img_path in enumerate(sample_images, start=1):

        lbl_path = train_labels_dir / f"{img_path.stem}.txt"

        drawn = draw_yolo_boxes(img_path, lbl_path)

        plt.subplot(2, 3, i)

        plt.imshow(drawn)

        plt.title(img_path.name)

        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# ОБУЧЕНИЕ
# ============================================================

def train_model():

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(YOLO_DATASET_DIR / "dataset.yaml"),
        epochs=EPOCHS,
        imgsz=YOLO_IMGSZ,
        batch=BATCH_SIZE,
        project=str(MODELS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        plots=True,
        seed=SEED,
        deterministic=True,
        workers=2
    )

    return model, results


# ============================================================
# ГРАФИКИ ОБУЧЕНИЯ
# ============================================================

def plot_training_results():

    results_csv_path = MODELS_DIR / RUN_NAME / "results.csv"

    if not results_csv_path.exists():
        return

    results_df = pd.read_csv(results_csv_path)

    print(results_df.tail())

    plt.figure(figsize=(8, 5))

    plt.plot(
        results_df["epoch"],
        results_df["train/box_loss"],
        label="train/box_loss"
    )

    plt.plot(
        results_df["epoch"],
        results_df["val/box_loss"],
        label="val/box_loss"
    )

    plt.title("Box Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(True)

    plt.legend()

    plt.show()

    plt.figure(figsize=(8, 5))

    plt.plot(
        results_df["epoch"],
        results_df["metrics/mAP50(B)"],
        label="mAP50"
    )

    plt.plot(
        results_df["epoch"],
        results_df["metrics/mAP50-95(B)"],
        label="mAP50-95"
    )

    plt.title("Metrics")

    plt.xlabel("Epoch")
    plt.ylabel("Value")

    plt.grid(True)

    plt.legend()

    plt.show()


# ============================================================
# ВАЛИДАЦИЯ
# ============================================================

def validate_model():

    best_model_path = MODELS_DIR / RUN_NAME / "weights" / "best.pt"

    if not best_model_path.exists():
        raise FileNotFoundError(best_model_path)

    best_model = YOLO(str(best_model_path))

    metrics = best_model.val(
        data=str(YOLO_DATASET_DIR / "dataset.yaml"),
        split="val",
        imgsz=YOLO_IMGSZ,
        conf=CONF_THRES,
        iou=IOU_THRES,
        plots=True,
        project=str(MODELS_DIR),
        name=f"{RUN_NAME}_val_eval",
        exist_ok=True
    )

    metrics_table = pd.DataFrame({
        "metric": [
            "precision",
            "recall",
            "mAP50",
            "mAP50-95"
        ],
        "value": [
            metrics.results_dict["metrics/precision(B)"],
            metrics.results_dict["metrics/recall(B)"],
            metrics.results_dict["metrics/mAP50(B)"],
            metrics.results_dict["metrics/mAP50-95(B)"]
        ]
    })

    print(metrics_table)

    return best_model


# ============================================================
# ПРЕДСКАЗАНИЯ
# ============================================================

def visualize_predictions(best_model):

    validation_images_dir = YOLO_DATASET_DIR / "images" / "validation"

    validation_image_files = sorted(validation_images_dir.glob("*"))

    if len(validation_image_files) == 0:
        return

    sample_validation_images = random.sample(
        validation_image_files,
        min(6, len(validation_image_files))
    )

    pred_results = best_model.predict(
        source=[str(p) for p in sample_validation_images],
        imgsz=YOLO_IMGSZ,
        conf=CONF_THRES,
        iou=IOU_THRES,
        save=False,
        verbose=False
    )

    for img_path, result in zip(sample_validation_images, pred_results):

        plotted = result.plot()

        plt.figure(figsize=(6, 6))

        plt.imshow(plotted[..., ::-1])

        plt.title(f"Prediction: {img_path.name}")

        plt.axis("off")

        plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print_system_info()

    download_dataset()

    prepare_yolo_dirs()

    print("\nProcessing train split...")

    process_split(
        "train",
        RAW_XML_TRAIN_DIR,
        RAW_IMAGES_TRAIN_DIR,
        YOLO_DATASET_DIR
    )

    print("\nProcessing validation split...")

    process_split(
        "validation",
        RAW_XML_VAL_DIR,
        RAW_IMAGES_VAL_DIR,
        YOLO_DATASET_DIR
    )

    create_dataset_yaml()

    validate_yolo_split("train")
    validate_yolo_split("validation")

    visualize_samples()

    _, _ = train_model()

    plot_training_results()

    best_model = validate_model()

    visualize_predictions(best_model)

    print("\nProject completed successfully")


if __name__ == "__main__":
    main()