#!/usr/bin/env python
# Assignment 1 - Erazem Kokot

import os
import numpy as np
from multiprocessing import Pool
from PIL import Image, ImageOps
from typing import Callable, List, Tuple
from sklearn.metrics import pairwise_distances

# Update if needed
DATA_DIR = "data/ass1"

# Cache for images (no need to convert them every time)
pixels: List[np.ndarray] = []


# Reads images and calls a feature extractor for each one
def image_extractor(directory: str, feature_extractor: Callable) -> Tuple[List[np.ndarray], List[int]]:
    images: List[Image.Image] = []  # PIL Image objects
    classes: List[int] = []  # Image "classes" (directories)

    for (root, _, files) in os.walk(directory):
        for f in files:
            if f.endswith(".png"):
                path = os.path.join(root, f)
                classes.append(int(root.split("/")[1]))  # Converts dir name to integer

                # Read images in greyscale and as 128x128 pixels
                image = Image.open(path).resize((128, 128))
                image = ImageOps.grayscale(image)
                images.append(image)

    return feature_extractor(images), classes


# Plain feature extractor that just converts images into 1D NumPy arrays
def plain_extractor(images: List[np.ndarray]) -> List[np.ndarray]:
    # Only convert images to pixels if not already cached
    global pixels
    if len(pixels) != len(images):
        pixels = [np.asarray(image) for image in images]

    return pixels


# A more advanced Local Binary Pattern (LBP) feature extractor
def lbp_extractor(images: List[np.ndarray], histogram: bool = False, uniform: bool = False,
                  overlap: bool = False, radius: int = 1, lengths: int = 8) -> List[np.ndarray]:
    features = []

    # Only convert images to pixels if not already cached
    # (assume images haven't changed since last run)
    global pixels
    if pixels and len(pixels) == len(images):
        images = pixels
    else:
        images = [np.asarray(image) for image in images]

    i = 0
    for image in images:
        # Pretty-print progress
        i += 1
        if i % 50 == 0:
            print(".", end="")

        height, width = image.shape
        values = np.empty((height, width))

        neighbors = {
            0: (-1, +0),  # Top
            1: (-1, +1),  # Upper-right
            2: (+0, +1),  # Right
            3: (+1, +1),  # Bottom-right
            4: (+1, +0),  # Bottom
            5: (+1, -1),  # Bottom-left
            6: (+0, -1),  # Left
            7: (-1, -1),  # Upper-left
        }

        for h in range(height):
            for w in range(width):
                value = 0
                for n in neighbors:
                    if pixel(image, h + neighbors[n][0], w + neighbors[n][1]) >= image[h][w]:
                        value += 2 ** n

                values[h, w] = value

        features.append(values)

    # Add newline to progress
    print("")

    return features


# Returns the value of a pixel or 0 if out of bounds
def pixel(image: np.ndarray, h: int, w: int) -> int:
    if (h < 0 or h >= image.shape[0]) or (w < 0 or w >= image.shape[1]):
        return 0

    return image[h][w]


# Calculates rank-1 for a list of feature vectors
def calculate_rank_1(args):
    features = args[0]
    classes = args[1]
    metric = args[2]

    # Flatten all feature matrices
    features = [feature.flatten() for feature in features]

    # Convert lists to NumPy arrays for comparison
    features = np.asarray(features)
    classes = np.asarray(classes)

    # Calculate distances between all features
    distances = pairwise_distances(features, metric=metric)

    # Artificially inflate distance of a feature vector to itself (since they're always 0)
    np.fill_diagonal(distances, np.inf)

    # Get smallest distances between feature vectors
    matches = np.argmin(distances, axis=1)

    # Get number of correct class matches
    hits = np.sum(classes[matches] == classes)

    rank = hits / len(features)

    print("   %s metric: %.3f" % (metric, rank))

    return rank


def main():
    for extractor in [('plain', plain_extractor), ('basic_lbp', lbp_extractor)]:
        print("Testing %s feature extractor:" % extractor[0])
        features, classes = image_extractor(DATA_DIR, extractor[1])

        args = (
            (features, classes, 'cityblock'),
            (features, classes, 'cosine'),
            (features, classes, 'euclidean'),
            (features, classes, 'manhattan'),
        )

        # Run rank-1 tests in parallel
        with Pool() as pool:
            pool.map(calculate_rank_1, args)


if __name__ == "__main__":
    main()
