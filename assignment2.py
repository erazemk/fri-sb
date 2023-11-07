#!/usr/bin/env python3
# Assignment 2 - Erazem Kokot
import json
import os
import sys

import cv2
import yolov5
import numpy as np

from itertools import repeat
from pybboxes import BoundingBox
from multiprocessing import Pool

DATA_DIR = "data/ass2"
IMG_DIR = "data/ass2/images"
CPU_CORES = 8  # Number of CPU cores to use for multiprocessing (number of spawned subprocesses)

YOLO_MODEL = DATA_DIR + "/yolo5s.pt"
HAAR_CLASSIFIERS = [DATA_DIR + "/haarcascade_mcs_leftear.xml",
                    DATA_DIR + "/haarcascade_mcs_rightear.xml"]


# Extracts image and text file paths
def find_images(img_dir: str) -> list[tuple[str, str]]:
    images = []
    for (root, _, files) in os.walk(img_dir):
        for f in files:
            if f.endswith(".png"):
                img_file = os.path.join(root, f)
                txt_file = os.path.splitext(img_file)[0] + ".txt"
                images.append((img_file, txt_file))

    return images


# Extracts info from ground truth file and converts it to bounding box coordinates
def get_gt(file_path: str, img_size: tuple[int, int]) -> tuple[int, int, int, int]:
    with open(file_path, 'r') as f:
        # Fields: (x-center, y-center, width, height)
        (xc, yc, w, h) = tuple([f(i) for f, i in zip((float, float, float, float),
                                                     f.read().split()[1:])])

        # Convert ground truth data into proper coordinates
        yolo_bbox = BoundingBox.from_yolo(xc, yc, w, h, img_size)
        coordinates = yolo_bbox.to_voc(return_values=True)

    return coordinates


# Calculates IOU using ground truth coordinates as a reference
# Stolen from https://pyimagesearch.com/2022/05/02/mean-average-precision-map-using-the-coco-evaluator/
def calculate_iou(gt: tuple[int, int, int, int],
                  pred: tuple[int, int, int, int]) -> float:
    x1 = max(gt[0], pred[0])
    y1 = max(gt[1], pred[1])
    x2 = min(gt[2], pred[2])
    y2 = min(gt[3], pred[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area1 = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    area2 = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)

    return intersection / float(area1 + area2 - intersection)


# Uses Haar cascade detector for ear detection
def haar_cascade_detector(images, classifier_paths, scale_factor=1.05, min_neighbors=2):
    classifiers = [cv2.CascadeClassifier(c) for c in classifier_paths]
    data = {}

    for tup in images:
        img_file = tup[0]
        txt_file = tup[1]
        img_name = os.path.splitext(os.path.basename(img_file))[0]

        # Image preprocessing
        image = cv2.imread(img_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get ground truth coordinates for image
        (h, w) = image.shape[:2]
        gt = get_gt(txt_file, (w, h))

        # Get Haar-detected coordinates for image (only use last match)
        coordinates = (-1, -1, -1, -1)
        for ear in [c.detectMultiScale(gray, scale_factor, min_neighbors) for c in classifiers]:
            for (x, y, w, h) in ear:
                coordinates = BoundingBox.from_coco(x, y, w, h).to_voc(return_values=True)

        iou = calculate_iou(gt, coordinates)

        data[img_name] = {}
        data[img_name]["gt"] = gt
        data[img_name]["iou"] = iou
        data[img_name]["pred"] = coordinates
        data[img_name]["scaleFactor"] = scale_factor
        data[img_name]["minNeighbors"] = min_neighbors

        # cv2.rectangle(image, gt[:2], gt[2:], (255, 0, 0), 2)
        # cv2.rectangle(image, coordinates[:2], coordinates[2:], (0, 255, 0), 2)
        # cv2.putText(image, "IoU: %.4f" % iou, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        # cv2.namedWindow("Image: %s" % img_name, cv2.WINDOW_NORMAL)
        # cv2.imshow("Image: %s" % img_name, image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return data


# Uses YOLOv5 object detector for ear detection
def yolo_detector(images, model_path):
    model = yolov5.load(model_path)

    data = {}
    for tup in images:
        img_file = tup[0]
        txt_file = tup[1]
        img_name = os.path.splitext(os.path.basename(img_file))[0]

        # Get ground truth coordinates for image
        image = cv2.imread(img_file)
        (h, w) = image.shape[:2]
        gt = get_gt(txt_file, (w, h))

        results = model(img_file)

        data[img_name] = {}
        data[img_name]["gt"] = gt
        if results.pred[0].size()[0] == 0:
            data[img_name]["iou"] = 0
            data[img_name]["type"] = 2
            continue

        predictions = results.pred[0][0].detach().cpu().numpy()

        x1, y1, x2, y2 = predictions[:4]
        coordinates = (round(x1), round(y1), round(x2), round(y2))

        iou = calculate_iou(gt, coordinates)
        data[img_name]["iou"] = iou
        data[img_name]["pred"] = coordinates
        data[img_name]["confidence"] = predictions[4]

        # cv2.rectangle(image, gt[:2], gt[2:], (255, 0, 0), 2)
        # cv2.rectangle(image, coordinates[:2], coordinates[2:], (0, 255, 0), 2)
        # cv2.putText(image, "IoU: %.4f" % iou, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        # cv2.namedWindow("Image: %s" % img_name, cv2.WINDOW_NORMAL)
        # cv2.imshow("Image: %s" % img_name, image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return data


# Run the specified function in parallel, with images and data being its arguments
def run_in_parallel(func, images, classifiers, scale_factor=None, min_neighbors=None, n_jobs=CPU_CORES):
    # Split images into smaller lists for multiprocessing
    chunks = np.array_split(images, n_jobs)

    # Run function using multiple subprocesses
    with Pool() as pool:
        if scale_factor is not None and min_neighbors is not None:
            data_dicts = pool.starmap(func, zip(chunks, repeat(classifiers),
                                                repeat(scale_factor), repeat(min_neighbors)))
        else:
            data_dicts = pool.starmap(func, zip(chunks, repeat(classifiers)))

    # Combine list of data dictionaries into one dictionary
    return {k: v for d in data_dicts for (k, v) in d.items()}


# Sums up the number of each detection type and calculates precision-recall
def analyze_data(data, threshold):
    types = {"TP": 0, "FP": 0, "FN": 0}

    results = []
    for images in data:  # Dictionaries of images
        for params in images.values():
            iou = params["iou"]
            pred = params["pred"]

            if pred == [-1, -1, -1, -1]:  # No ear detected
                types["FN"] += 1
            elif iou >= threshold:  # Ear correctly detected
                types["TP"] += 1
            elif iou < threshold:  # Ear incorrectly detected
                types["FP"] += 1
            else:
                sys.exit("PANIC: This shouldn't have happened")

        precision = types["TP"] / (types["TP"] + types["FP"])
        recall = types["TP"] / (types["TP"] + types["FN"] + np.nextafter(0, 1))  # Prevent division by zero
        results.append({"precision": precision, "recall": recall,
                        "scaleFactor": images[list(images.keys())[0]]["scaleFactor"],
                        "minNeighbors": images[list(images.keys())[0]]["minNeighbors"]})
    return results


def main():
    images = find_images(IMG_DIR)

    # Initial run over all parameter combinations
    # haar_data = []
    # for scale_factor in [1.05, 1.1, 1.15, 1.2, 1.25, 1.3]:
    #     for min_neighbors in [1, 2, 3, 4]:
    #         haar_data.append(run_in_parallel(haar_cascade_detector, images, HAAR_CLASSIFIERS,
    #                                          scale_factor, min_neighbors))
    #
    #         # Cache results for each iteration, so it can be stopped in the middle
    #         with open("data/ass2/haar_data.json", "w") as f:
    #             json.dump(haar_data, f)
    #
    #         print("Finished params: sf=%.2f, mn=%.2f" % (scale_factor, min_neighbors))

    with open("data/ass2/haar_data.json", "r") as f:
        haar_data = json.load(f)

    with open("iou-table.txt", "w") as f:
        f.write("| Scale factor | Min. neighbors | IoU |\n")
        for d in haar_data:  # d = each combination of parameters
            ap = 0.0
            for img in d.values():
                ap += img["iou"]

            f.write("| %.2f | %.2f | %.4f |\n" %
                    (d[list(d.keys())[0]]["scaleFactor"], d[list(d.keys())[0]]["minNeighbors"], ap / len(d)))

    # Best parameters
    # scale_factor = 1.05
    # min_neighbors = 2

    with open("prec-rec-table.txt", "w") as f:
        f.write("| Threshold | Scale factor | Min. neighbors | Precision | Recall |\n")
        for i in range(0, 101, 1):
            threshold = i * 0.01
            res = analyze_data(haar_data, threshold)

            # | Scale factor | Min. neighbors | IoU

            # Res: list of dictionaries
            for r in res:
                f.write("| %.2f | %.2f | %.2f | %.2f | %.2f |\n" %
                        (threshold, r["scaleFactor"], r["minNeighbors"], r["precision"], r["recall"]))

    # 3 jobs is the max my GPU can take
    yolo_data = run_in_parallel(yolo_detector, images[:10], YOLO_MODEL, n_jobs=1)
    ap = 0.0
    for d in yolo_data.values():
        ap += d["iou"]

    print("YOLOv5 AP:", ap / len(yolo_data))


if __name__ == "__main__":
    main()
