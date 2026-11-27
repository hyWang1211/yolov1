import argparse
import math
import os
from pathlib import Path
import random
import shutil
from uuid import uuid4

import torch
from torch.utils.data import DataLoader, Subset

from darknet_weights import load_extraction_weights
from metrics import VocMeanAP
from voc import DEFAULT_VOC_ROOT, VocDetection, collate_detections
from yolo import YOLOv1, YOLOv1Loss, decode_detections, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv1 on PASCAL VOC 2012.")
    parser.add_argument("--data", type=Path, default=DEFAULT_VOC_ROOT)
    parser.add_argument("--epochs", type=int, default=135)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fc-size", type=int, help="Width of the first FC layer (default: 4096).")
    parser.add_argument("--lr", type=float, help="Peak learning rate (default: 0.001).")
    parser.add_argument("--limit", type=int, help="Use at most this many images per split.")
    parser.add_argument("--map", action="store_true", help="Also measure validation mAP@0.5.")
    init = parser.add_mutually_exclusive_group()
    init.add_argument("--resume", type=Path, help="Continue from a training checkpoint.")
    init.add_argument(
        "--pretrained", type=Path, help="Initialize the first 20 convolutions with Darknet extraction.weights.",
    )
    parser.add_argument("--output", type=Path, help="Path to the latest training checkpoint.")
    args = parser.parse_args()
    if args.output is None:
        run_dir = "yolov1-pretrained" if args.pretrained is not None else "yolov1"
        args.output = Path("runs") / run_dir / "last.pt"
    for name in ("epochs", "batch_size", "fc_size", "lr", "limit"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    return args


def make_loader(args, split, device):
    training = split == "train"
    dataset = VocDetection(args.data, split, augment_images=training)
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=training,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        collate_fn=collate_detections,
    )


def learning_rate(peak, epoch, step, steps_per_epoch):
    progress = epoch + step / steps_per_epoch
    if progress < 5:
        return peak * 10 ** (progress / 5 - 1)
    if epoch < 75:
        return peak
    if epoch < 105:
        return peak * 0.1
    return peak * 0.01


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, peak_lr, writer, global_step):
    model.train()
    total = 0.0
    batches = len(loader)
    print(f"epoch {epoch + 1} training: 0/{batches} batches (starting first batch)", flush=True)
    for step, (images, targets, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        lr = learning_rate(peak_lr, epoch, step, batches)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=scaler.is_enabled()):
            predictions = model(images)
        loss = criterion(predictions, targets)
        value = loss.item()
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}, batch {step + 1}.")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += value * len(images)
        completed = step + 1
        writer.add_scalar("loss/train_batch", value, global_step + completed)
        writer.add_scalar("optimizer/lr", lr, global_step + completed)
        if completed == 1:
            writer.flush()
        if completed in (1, batches) or completed % 50 == 0:
            print(f"epoch {epoch + 1} training: {completed}/{batches} batches loss={value:.4f} lr={lr:.2g}", flush=True)
    return total / len(loader.dataset)


@torch.inference_mode()
def validate(model, loader, criterion, device, measure_map):
    model.eval()
    total = 0.0
    metric = VocMeanAP() if measure_map else None
    batches = len(loader)
    print(f"validation: 0/{batches} batches", flush=True)
    for completed, (images, targets, annotations) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            predictions = model(images)
        total += criterion(predictions, targets).item() * len(images)
        if metric is not None:
            for prediction, annotation in zip(predictions.cpu(), annotations):
                metric.update(decode_detections(prediction, score_threshold=0.0), annotation)
        if completed in (1, batches) or completed % 50 == 0:
            print(f"validation: {completed}/{batches} batches", flush=True)
    return total / len(loader.dataset), metric.mean_ap() if metric is not None else None


def save_checkpoint(state, output):
    epoch = state["epoch"]
    staging = output.parent / f".checkpoint-{uuid4().hex}"
    staging.mkdir()
    try:
        temporary = staging / "last.pt"
        torch.save(state, temporary)
        if epoch % 5 != 0:
            temporary.replace(output)
            return

        snapshot = output.parent / "checkpoints" / f"epoch_{epoch:04d}.pt"
        snapshot.parent.mkdir(exist_ok=True)
        staged_snapshot = staging / snapshot.name
        # Link the completed file when supported to avoid writing the same weights twice.
        try:
            os.link(temporary, staged_snapshot)
        except OSError:
            shutil.copyfile(temporary, staged_snapshot)
        temporary.replace(output)
        staged_snapshot.replace(snapshot)
        print(f"saved snapshot: {snapshot}", flush=True)
    finally:
        shutil.rmtree(staging)


def main():
    args = parse_args()
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as error:
        if error.name == "tensorboard":
            raise SystemExit("Install TensorBoard in this Python environment: python -m pip install tensorboard") from error
        raise

    device = resolve_device(args.device)
    state = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    fc_size = state["fc_size"] if state is not None else (args.fc_size or 4096)
    if args.fc_size is not None and args.fc_size != fc_size:
        raise SystemExit("--fc-size must match the checkpoint when resuming.")
    peak_lr = args.lr or (state["peak_lr"] if state is not None else 0.001)
    start_epoch = state["epoch"] if state is not None else 0
    saved_step = state.get("global_step") if state is not None else None
    saved_run = state.get("tensorboard_run") if state is not None else None
    if start_epoch >= args.epochs:
        raise SystemExit(f"--epochs must exceed the {start_epoch} completed epochs.")

    torch.manual_seed(0)
    model = YOLOv1(fc_size)
    if args.pretrained is not None:
        try:
            loaded = load_extraction_weights(model, args.pretrained)
        except (OSError, ValueError) as error:
            raise SystemExit(f"Cannot load {args.pretrained}: {error}") from error
        print(f"loaded {loaded} pretrained convolutional layers from {args.pretrained}", flush=True)
    model = model.to(device)
    criterion = YOLOv1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=peak_lr, momentum=0.9, weight_decay=0.0005)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scaler.is_enabled() and state["scaler"]:
            scaler.load_state_dict(state["scaler"])
        del state

    train_loader = make_loader(args, "train", device)
    val_loader = make_loader(args, "val", device)
    global_step = saved_step if saved_step is not None else start_epoch * len(train_loader)
    log_root = args.output.parent / "tensorboard"
    log_root.mkdir(parents=True, exist_ok=True)
    log_dir = None
    if saved_run and saved_run.startswith("run-") and Path(saved_run).name == saved_run:
        log_dir = log_root / saved_run
    reusing_log = log_dir is not None and log_dir.is_dir()
    if not reusing_log:
        log_dir = log_root / f"run-{uuid4().hex[:12]}"
        log_dir.mkdir()
    purge_step = global_step + 1 if reusing_log else None

    print(f"device={device} fc_size={fc_size} train={len(train_loader.dataset)} val={len(val_loader.dataset)}", flush=True)
    print(f"tensorboard: {log_dir}", flush=True)
    with SummaryWriter(str(log_dir), purge_step=purge_step, flush_secs=5) as writer:
        for epoch in range(start_epoch, args.epochs):
            torch.manual_seed(epoch + 1)
            random.seed(epoch + 1)
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device, epoch, peak_lr,
                writer, global_step,
            )
            global_step += len(train_loader)
            optimizer.zero_grad(set_to_none=True)
            val_loss, mean_ap = validate(model, val_loader, criterion, device, args.map)
            writer.add_scalar("loss/train_epoch", train_loss, global_step)
            writer.add_scalar("loss/val_epoch", val_loss, global_step)
            metrics = f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            if mean_ap is not None:
                writer.add_scalar("metrics/mAP50", mean_ap, global_step)
                metrics += f" mAP@0.5={mean_ap:.4f}"
            writer.flush()

            print(f"saving checkpoint: {args.output}", flush=True)
            save_checkpoint({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "fc_size": fc_size, "peak_lr": peak_lr,
                "epoch": epoch + 1, "global_step": global_step, "tensorboard_run": log_dir.name,
            }, args.output)
            print(f"epoch {epoch + 1}/{args.epochs} {metrics} saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
