import argparse
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from voc import image_tensor
from yolo import VOC_CLASSES, YOLOv1, decode_detections, resolve_device


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Detect objects in one image with YOLOv1.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("runs/yolov1/last.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/prediction.jpg"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--score", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    if not (0 <= args.score <= 1 and 0 <= args.iou <= 1):
        parser.error("--score and --iou must be in [0, 1]")
    if args.image.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input image.")

    device = resolve_device(args.device)
    state = torch.load(args.weights, map_location="cpu", mmap=True, weights_only=True)
    model = YOLOv1(state["fc_size"]).to(device)
    model.load_state_dict(state["model"])
    del state
    model.eval()

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    prediction = model(image_tensor(image).unsqueeze(0).to(device))[0].cpu()
    boxes, scores, classes = decode_detections(prediction, args.score, args.iou)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for box, score, index in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
        left, top, right, bottom = box
        bounds = (
            min(int(left * width), width - 1), min(int(top * height), height - 1),
            min(int(right * width), width - 1), min(int(bottom * height), height - 1),
        )
        draw.rectangle(bounds, outline="#13ddae", width=2)
        label = f"{VOC_CLASSES[index]} {score:.2f}"
        x, y = bounds[:2]
        text_right, text_bottom = draw.textbbox((x, y), label, anchor="lt")[2:]
        draw.rectangle((x, y, text_right + 3, text_bottom + 2), fill="#13ddae")
        draw.text((x + 1, y + 1), label, fill="black", anchor="lt")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Detections: {len(boxes)} | Saved: {args.output}")


if __name__ == "__main__":
    main()
