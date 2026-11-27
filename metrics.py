import numpy as np
import torch

from yolo import CLASS_COUNT, box_iou


class VocMeanAP:
    """VOC 2012 interpolated AP at IoU 0.5, averaged over classes with ground truth."""

    def __init__(self):
        self.positives = np.zeros(CLASS_COUNT, dtype=int)
        self.matches = [[] for _ in range(CLASS_COUNT)]

    def update(self, detections, objects):
        boxes, scores, labels = detections
        for label in range(CLASS_COUNT):
            gt_mask = objects.labels == label
            gt_boxes = objects.boxes[gt_mask]
            difficult = objects.difficult[gt_mask]
            self.positives[label] += (~difficult).sum().item()
            indices = torch.where(labels == label)[0]
            if len(gt_boxes) == 0:
                self.matches[label].extend((score, False) for score in scores[indices].tolist())
                continue
            overlaps = box_iou(boxes[indices, None], gt_boxes[None, :])
            best_ious, best_indices = overlaps.max(dim=1)
            matched = set()
            for score, iou, gt_index in zip(
                scores[indices].tolist(), best_ious.tolist(), best_indices.tolist()
            ):
                if iou >= 0.5 and difficult[gt_index]:
                    continue
                true_positive = iou >= 0.5 and gt_index not in matched
                if true_positive:
                    matched.add(gt_index)
                self.matches[label].append((score, true_positive))

    def mean_ap(self):
        ap_values = []
        for positives, matches in zip(self.positives, self.matches):
            if positives == 0:
                continue
            matches.sort(key=lambda item: item[0], reverse=True)
            true_positives = np.cumsum([match for _, match in matches], dtype=float)
            recall = np.r_[0.0, true_positives / positives, 1.0]
            precision = np.r_[0.0, true_positives / np.arange(1, len(matches) + 1), 0.0]
            envelope = np.maximum.accumulate(precision[::-1])[::-1]
            ap_values.append(np.sum(np.diff(recall) * envelope[1:]))
        return float(np.mean(ap_values)) if ap_values else float("nan")
