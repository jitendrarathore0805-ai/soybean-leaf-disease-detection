import cv2
import numpy as np


IMG_SIZE = (224, 224)


def segment_leaf(rgb):
    rgb = rgb.astype(np.uint8)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    _, mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    if np.mean(mask > 0) < 0.03:
        return rgb

    return cv2.bitwise_and(rgb, rgb, mask=mask)


def statistical_features(channel):
    channel = channel.astype(np.float32)

    return [
        np.mean(channel),
        np.std(channel),
        np.min(channel),
        np.max(channel),
        np.percentile(channel, 10),
        np.percentile(channel, 25),
        np.percentile(channel, 50),
        np.percentile(channel, 75),
        np.percentile(channel, 90),
    ]


def lbp_features(gray):
    lbp = np.zeros_like(gray, dtype=np.uint8)

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1),
        (1, 1), (1, 0), (1, -1),
        (0, -1),
    ]

    for bit, (dy, dx) in enumerate(neighbors):
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        lbp |= ((shifted >= gray).astype(np.uint8) << bit)

    hist, _ = np.histogram(
        lbp, bins=32, range=(0, 256), density=True
    )
    return hist.tolist()


def handcrafted_features_from_image(image_rgb):
    image = cv2.resize(image_rgb, IMG_SIZE)
    segmented = segment_leaf(image)

    hsv = cv2.cvtColor(segmented, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(segmented, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)

    features = []

    for channel in cv2.split(segmented):
        features.extend(statistical_features(channel))

    for channel in cv2.split(hsv):
        features.extend(statistical_features(channel))

    for channel in cv2.split(lab):
        features.extend(statistical_features(channel))

    features.extend(statistical_features(gray))
    features.extend(lbp_features(gray))

    edges = cv2.Canny(gray, 50, 150)
    features.extend([
        np.mean(edges),
        np.std(edges),
        np.sum(edges > 0) / edges.size,
    ])

    return np.nan_to_num(
        np.array(features, dtype=np.float32)
    )


def segmented_image(image_rgb):
    image = cv2.resize(image_rgb, IMG_SIZE)
    return segment_leaf(image)
