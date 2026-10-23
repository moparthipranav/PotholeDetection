import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.utils import (
    select_device,
    scale_coords,
    non_max_suppression,
    split_for_trace_model,
    driving_area_mask,
    lane_line_mask,
    LoadImages
)


# ============================================================
# CONFIGURATION
# ============================================================

WEIGHTS = "data/weights/yolopv2.pt"

IMAGE_DIR = "data/test/images"

# YOLO-format detection labels:
# class x_center y_center width height
DETECTION_LABEL_DIR = "data/test/labels"

# Ground-truth drivable-area masks
DRIVABLE_MASK_DIR = "data/test/drivable"

# Ground-truth lane-line masks
LANE_MASK_DIR = "data/test/lane"

IMG_SIZE = 640

CONF_THRESHOLD = 0.001
IOU_THRESHOLD = 0.5

DEVICE = "0"


# ============================================================
# DEVICE
# ============================================================

device = select_device(DEVICE)

print("Device:", device)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLOPv2...")

model = torch.jit.load(WEIGHTS)

model = model.to(device)

half = device.type != "cpu"

if half:
    model.half()

model.eval()

print("Model loaded successfully.")


if device.type != "cpu":

    dummy = torch.zeros(
        1,
        3,
        IMG_SIZE,
        IMG_SIZE
    ).to(device)

    dummy = dummy.type_as(
        next(model.parameters())
    )

    model(dummy)


def calculate_binary_metrics(pred_mask, gt_mask):

    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    TP = np.logical_and(
        pred_mask,
        gt_mask
    ).sum()

    TN = np.logical_and(
        ~pred_mask,
        ~gt_mask
    ).sum()

    FP = np.logical_and(
        pred_mask,
        ~gt_mask
    ).sum()

    FN = np.logical_and(
        ~pred_mask,
        gt_mask
    ).sum()

    precision = TP / (TP + FP + 1e-9)

    recall = TP / (TP + FN + 1e-9)

    f1 = (
        2 * precision * recall /
        (precision + recall + 1e-9)
    )

    iou = TP / (
        TP + FP + FN + 1e-9
    )

    accuracy = (
        TP + TN
    ) / (
        TP + TN + FP + FN + 1e-9
    )

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy
    }


def box_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(
        0,
        x2 - x1
    ) * max(
        0,
        y2 - y1
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    return intersection / (union + 1e-9)


def load_detection_labels(label_path, image_width, image_height):

    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:

        lines = f.readlines()

    for line in lines:

        values = line.strip().split()

        if len(values) < 5:
            continue

        cls = int(values[0])

        xc = float(values[1])
        yc = float(values[2])

        w = float(values[3])
        h = float(values[4])

        x1 = (xc - w / 2) * image_width
        y1 = (yc - h / 2) * image_height

        x2 = (xc + w / 2) * image_width
        y2 = (yc + h / 2) * image_height

        boxes.append(
            [
                cls,
                x1,
                y1,
                x2,
                y2
            ]
        )

    return boxes

detection_stats = []

class_names = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle"
]


lane_tp = 0
lane_tn = 0
lane_fp = 0
lane_fn = 0

drivable_tp = 0
drivable_tn = 0
drivable_fp = 0
drivable_fn = 0



dataset = LoadImages(
    IMAGE_DIR,
    img_size=IMG_SIZE,
    stride=32
)


print("\nStarting evaluation...\n")


with torch.no_grad():

    for path, img, im0s, vid_cap in tqdm(dataset):


        im0 = im0s.copy()

        image_name = Path(path).stem

        image_height, image_width = im0.shape[:2]


        img_tensor = torch.from_numpy(img).to(device)

        img_tensor = (
            img_tensor.half()
            if half
            else img_tensor.float()
        )

        img_tensor /= 255.0

        if img_tensor.ndimension() == 3:

            img_tensor = img_tensor.unsqueeze(0)


        outputs = model(img_tensor)

        pred = outputs[0]

        seg = outputs[1]

        ll = outputs[2]


        pred, anchor_grid = pred

        pred = split_for_trace_model(
            pred,
            anchor_grid
        )

        pred = non_max_suppression(
            pred,
            CONF_THRESHOLD,
            IOU_THRESHOLD
        )


        detections = pred[0]


        if len(detections):

            detections[:, :4] = scale_coords(
                img_tensor.shape[2:],
                detections[:, :4],
                im0.shape
            ).round()

        label_path = os.path.join(
            DETECTION_LABEL_DIR,
            image_name + ".txt"
        )

        gt_boxes = load_detection_labels(
            label_path,
            image_width,
            image_height
        )



        matched_gt = set()

        if len(detections):

            for detection in detections.cpu().numpy():

                px1, py1, px2, py2 = detection[:4]

                confidence = detection[4]

                predicted_class = int(
                    detection[5]
                )

                best_iou = 0

                best_gt = -1

                for gt_index, gt in enumerate(gt_boxes):

                    if gt_index in matched_gt:
                        continue

                    gt_class = int(gt[0])

                    if predicted_class != gt_class:
                        continue

                    gt_box = gt[1:5]

                    iou = box_iou(
                        [px1, py1, px2, py2],
                        gt_box
                    )

                    if iou > best_iou:

                        best_iou = iou

                        best_gt = gt_index


                if best_iou >= 0.5 and best_gt >= 0:

                    matched_gt.add(best_gt)

                    detection_stats.append(
                        (
                            predicted_class,
                            confidence,
                            1
                        )
                    )

                else:

                    detection_stats.append(
                        (
                            predicted_class,
                            confidence,
                            0
                        )
                    )


        da_pred = driving_area_mask(seg)



        da_pred = cv2.resize(
            da_pred.astype(np.uint8),
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST
        )


        da_gt_path = os.path.join(
            DRIVABLE_MASK_DIR,
            image_name + ".png"
        )

        if os.path.exists(da_gt_path):

            da_gt = cv2.imread(
                da_gt_path,
                cv2.IMREAD_GRAYSCALE
            )

            da_gt = cv2.resize(
                da_gt,
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST
            )

            da_gt = da_gt > 0

            da_pred_binary = da_pred > 0


            metrics = calculate_binary_metrics(
                da_pred_binary,
                da_gt
            )


            drivable_tp += metrics["TP"]
            drivable_tn += metrics["TN"]
            drivable_fp += metrics["FP"]
            drivable_fn += metrics["FN"]


        lane_pred = lane_line_mask(ll)


        lane_pred = cv2.resize(
            lane_pred.astype(np.uint8),
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST
        )



        lane_gt_path = os.path.join(
            LANE_MASK_DIR,
            image_name + ".png"
        )


        if os.path.exists(lane_gt_path):

            lane_gt = cv2.imread(
                lane_gt_path,
                cv2.IMREAD_GRAYSCALE
            )

            lane_gt = cv2.resize(
                lane_gt,
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST
            )

            lane_gt = lane_gt > 0

            lane_pred_binary = lane_pred > 0


            metrics = calculate_binary_metrics(
                lane_pred_binary,
                lane_gt
            )


            lane_tp += metrics["TP"]
            lane_tn += metrics["TN"]
            lane_fp += metrics["FP"]
            lane_fn += metrics["FN"]

def print_segmentation_results(
    name,
    tp,
    tn,
    fp,
    fn
):

    precision = tp / (tp + fp + 1e-9)

    recall = tp / (tp + fn + 1e-9)

    f1 = (
        2 * precision * recall /
        (precision + recall + 1e-9)
    )

    iou = tp / (
        tp + fp + fn + 1e-9
    )

    accuracy = (
        tp + tn
    ) / (
        tp + tn + fp + fn + 1e-9
    )


    print("\n" + "=" * 60)

    print(name)

    print("=" * 60)

    print(
        f"Precision       : {precision:.4f}"
    )

    print(
        f"Recall          : {recall:.4f}"
    )

    print(
        f"F1 Score        : {f1:.4f}"
    )

    print(
        f"IoU             : {iou:.4f}"
    )

    print(
        f"Pixel Accuracy  : {accuracy:.4f}"
    )

    print("=" * 60)


print_segmentation_results(
    "DRIVABLE AREA SEGMENTATION",
    drivable_tp,
    drivable_tn,
    drivable_fp,
    drivable_fn
)



print_segmentation_results(
    "LANE-LINE SEGMENTATION",
    lane_tp,
    lane_tn,
    lane_fp,
    lane_fn
)



print("\n" + "=" * 60)

print("TRAFFIC OBJECT DETECTION")

print("=" * 60)


if len(detection_stats) == 0:

    print("No detections found.")

else:

    tp = sum(
        1
        for _, _, matched
        in detection_stats
        if matched == 1
    )

    fp = sum(
        1
        for _, _, matched
        in detection_stats
        if matched == 0
    )

    total_gt = sum(
        1
        for path, _, _, _ in []
    )

    precision = tp / (
        tp + fp + 1e-9
    )

    print(
        f"Detection Precision : {precision:.4f}"
    )

    print(
        f"True Positives      : {tp}"
    )

    print(
        f"False Positives     : {fp}"
    )


print("\nEvaluation complete.")