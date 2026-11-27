import torch
from torch import nn
from torch.nn import functional as F

IMAGE_SIZE = 448
GRID_SIZE = 7
BOXES_PER_CELL = 2
VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
CLASS_COUNT = len(VOC_CLASSES)
CELL_OUTPUTS = 5 * BOXES_PER_CELL + CLASS_COUNT


class YOLOv1(nn.Module):
    def __init__(self, fc_size=4096):
        super().__init__()
        # (kernel, filters, stride), with None marking 2x2 max pooling.
        architecture = [
            (7, 64, 2), None,
            (3, 192, 1), None,
            (1, 128, 1), (3, 256, 1), (1, 256, 1), (3, 512, 1), None,
            *[(1, 256, 1), (3, 512, 1)] * 4,
            (1, 512, 1), (3, 1024, 1), None,
            *[(1, 512, 1), (3, 1024, 1)] * 2,
            (3, 1024, 1), (3, 1024, 2),
            (3, 1024, 1), (3, 1024, 1),
        ]
        layers = []
        channels = 3
        for spec in architecture:
            if spec is None:
                layers.append(nn.MaxPool2d(2, 2))
                continue
            kernel, filters, stride = spec
            conv = nn.Conv2d(channels, filters, kernel, stride, kernel // 2)
            nn.init.kaiming_normal_(conv.weight, a=0.1)
            nn.init.zeros_(conv.bias)
            layers.extend((conv, nn.LeakyReLU(0.1, inplace=True)))
            channels = filters
        self.features = nn.Sequential(*layers)
        self.fc_size = fc_size
        hidden = nn.Linear(1024 * GRID_SIZE ** 2, fc_size)
        output = nn.Linear(fc_size, GRID_SIZE ** 2 * CELL_OUTPUTS)
        nn.init.kaiming_normal_(hidden.weight, a=0.1)
        nn.init.zeros_(hidden.bias)
        nn.init.normal_(output.weight, std=0.001)
        nn.init.zeros_(output.bias)
        with torch.no_grad():
            # Start with two box sizes so both predictors can win IoU assignments.
            box_bias, _ = split_predictions(output.bias.view(-1, CELL_OUTPUTS))
            box_bias[..., :2] = 0.5
            box_bias[:, 0, 2:4] = 0.15 ** 0.5
            box_bias[:, 1, 2:4] = 0.35 ** 0.5
        self.head = nn.Sequential(
            nn.Flatten(), hidden, nn.LeakyReLU(0.1, inplace=True), nn.Dropout(0.5), output,
        )

    def forward(self, images):
        return self.head(self.features(images)).reshape(-1, GRID_SIZE, GRID_SIZE, CELL_OUTPUTS)


def split_predictions(predictions):
    boxes = predictions[..., :5 * BOXES_PER_CELL].unflatten(-1, (BOXES_PER_CELL, 5))
    classes = predictions[..., 5 * BOXES_PER_CELL:]
    return boxes, classes

# xy和wh是预测输出的边界框参数，分别表示边界框的中心坐标和宽高
def cell_boxes_to_xyxy(xy, wh):
    """Convert grid offsets and normalized sizes to normalized image corners."""
    positions = torch.arange(GRID_SIZE, device=xy.device, dtype=xy.dtype)
    row, col = torch.meshgrid(positions, positions, indexing="ij")
    offsets = torch.stack((col, row), dim=-1).unsqueeze(-2)
    center = (xy + offsets) / GRID_SIZE
    return torch.cat((center - wh / 2, center + wh / 2), dim=-1)


def box_iou(boxes_a, boxes_b):
    """Compute IoU for corner-coordinate boxes with broadcastable shapes."""
    top_left = torch.maximum(boxes_a[..., :2], boxes_b[..., :2])
    bottom_right = torch.minimum(boxes_a[..., 2:], boxes_b[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area_a = (boxes_a[..., 2:] - boxes_a[..., :2]).prod(dim=-1)
    area_b = (boxes_b[..., 2:] - boxes_b[..., :2]).prod(dim=-1)
    union = area_a + area_b - intersection
    return intersection / torch.where(union > 0, union, 1.0)


class YOLOv1Loss(nn.Module):
    def forward(self, predictions, targets):
        boxes, classes = split_predictions(predictions.float())
        target_box = targets[..., None, :4]
        has_object = targets[..., 4].bool()
        with torch.no_grad():
            predicted_corners = cell_boxes_to_xyxy(boxes[..., :2], boxes[..., 2:4].square())
            target_corners = cell_boxes_to_xyxy(target_box[..., :2], target_box[..., 2:4])
            iou = box_iou(predicted_corners, target_corners)
            responsible = F.one_hot(iou.argmax(dim=-1), BOXES_PER_CELL).bool()
            responsible = responsible & has_object.unsqueeze(-1)

        xy_error = (boxes[..., :2] - target_box[..., :2]).square().sum(dim=-1)
        wh_error = (boxes[..., 2:4] - target_box[..., 2:4].sqrt()).square().sum(dim=-1)
        coordinates = ((xy_error + wh_error) * responsible).sum()
        confidence = boxes[..., 4]
        object_error = ((confidence - iou).square() * responsible).sum()
        background_error = (confidence.square() * ~responsible).sum()
        class_error = ((classes - targets[..., 5:]).square().sum(dim=-1) * has_object).sum()
        return (5 * coordinates + object_error + 0.5 * background_error + class_error) / len(targets)


def decode_detections(prediction, score_threshold=0.25, iou_threshold=0.5):
    """Decode a single CPU prediction, applying NMS separately for each class."""
    boxes, classes = split_predictions(prediction.float())
    corners = cell_boxes_to_xyxy(boxes[..., :2], boxes[..., 2:4].square()).reshape(-1, 4)
    corners = corners.clamp(0, 1)
    valid = (corners[:, 2:] > corners[:, :2]).all(dim=-1)
    scores = boxes[..., 4:5].clamp(0, 1) * classes.unsqueeze(-2).clamp(0, 1)
    corners = corners[valid]
    scores = scores.reshape(-1, CLASS_COUNT)[valid]
    overlaps = box_iou(corners[:, None], corners[None, :])
    kept_boxes, kept_scores, kept_labels = [], [], []
    for label, class_scores in enumerate(scores.T):
        order = torch.where(class_scores > score_threshold)[0]
        order = order[class_scores[order].argsort(descending=True, stable=True)]
        while order.numel():
            best = order[0]
            kept_boxes.append(corners[best])
            kept_scores.append(class_scores[best])
            kept_labels.append(label)
            order = order[1:][overlaps[best, order[1:]] <= iou_threshold]
    if not kept_boxes:
        return corners.new_empty((0, 4)), scores.new_empty(0), torch.empty(0, dtype=torch.long)
    kept_scores = torch.stack(kept_scores)
    order = kept_scores.argsort(descending=True, stable=True)
    return torch.stack(kept_boxes)[order], kept_scores[order], torch.tensor(kept_labels)[order]


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Install a CUDA-enabled PyTorch build or use --device cpu.")
    return torch.device(name)
