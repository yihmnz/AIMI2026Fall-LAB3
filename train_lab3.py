"""Train multiclass CXR models, selecting checkpoint by validation macro-F1."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from lab3_common import (
    CLASSES, MODEL_CHOICES, CXRDataset, batch_to_device, create_model,
    csv_path, fix_seed, rad_norm_stats, read_labels, save_json,
)


def make_loss(mode, class_counts, device, gamma):
    counts = torch.tensor(class_counts, dtype=torch.float32, device=device)
    weights = (counts.sum() / counts).sqrt()
    weights /= weights.mean()
    if mode == "ce":
        return nn.CrossEntropyLoss()
    if mode == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights)
    if mode == "focal":
        def focal(logits, targets):
            logprobs = F.log_softmax(logits, dim=1)
            chosen = logprobs.gather(1, targets[:, None]).squeeze(1)
            pt = chosen.exp()
            return (weights[targets] * (1 - pt).pow(gamma) * (-chosen)).mean()
        return focal
    raise ValueError(mode)


BATCH_FIELDS = [
    "timestamp", "epoch", "phase", "batch", "total_batches", "samples_seen",
    "running_loss", "running_macro_f1", "running_covid_f1", "running_accuracy",
    "elapsed_seconds",
]


def setup_logger(output_dir: Path) -> logging.Logger:
    """Write both console and permanent log. File is overwritten per run."""
    logger = logging.getLogger("lab3")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for h in [logging.StreamHandler(sys.stdout),
              logging.FileHandler(output_dir / "train.log", mode="w", encoding="utf-8")]:
        h.setFormatter(formatter)
        logger.addHandler(h)
    return logger


def running_scores(cm: np.ndarray) -> tuple[float, float, float]:
    """Macro-F1 from cumulative confusion matrix, including absent classes as zero."""
    tp = np.diag(cm).astype(float)
    denominator = cm.sum(axis=1) + cm.sum(axis=0)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(4, dtype=float), where=denominator > 0)
    return float(f1.mean()), float(f1[3]), float(tp.sum() / max(cm.sum(), 1))


def write_status(path: Path, **values) -> None:
    """Small machine-readable snapshot, refreshed while the loop is running."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"updated_at": datetime.now().isoformat(timespec="seconds"),
                               **values}, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def score_epoch(
    model, loader, device, criterion, aux_weight, pos_weight, amp_enabled,
    *, epoch, total_epochs, phase, log_every, batch_csv, batch_file, status_path,
    show_progress=True, optimizer=None,
):
    train = optimizer is not None
    model.train(train)
    n, loss_sum = 0, 0.0
    ys, predictions, probability_chunks = [], [], []
    cm = np.zeros((4, 4), dtype=np.int64)
    started = time.perf_counter()
    total_batches = len(loader)
    write_status(status_path, state="running", epoch=epoch, total_epochs=total_epochs,
                 phase=phase, batch=0, total_batches=total_batches)

    iterator = tqdm(loader, desc=f"Epoch {epoch:03d}/{total_epochs:03d} {phase:5s}",
                    unit="batch", dynamic_ncols=True, leave=False,
                    mininterval=0.5, disable=not show_progress)
    for batch_idx, (data, targets) in enumerate(iterator, start=1):
        data = batch_to_device(data, device)
        targets = targets.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with (torch.enable_grad() if train else torch.inference_mode()):
            with (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp_enabled else nullcontext()):
                logits, covid_logits = model(data)
                loss = criterion(logits, targets)
                if covid_logits is not None:
                    binary_target = (targets == 3).to(dtype=logits.dtype)
                    aux_loss = F.binary_cross_entropy_with_logits(
                        covid_logits.float(), binary_target.float(), pos_weight=pos_weight
                    )
                    loss = loss + aux_weight * aux_loss
            if train:
                loss.backward()
                optimizer.step()
        probs = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
        t = targets.detach().cpu().numpy()
        pred = probs.argmax(axis=1)
        ys.extend(t.tolist())
        predictions.extend(pred.tolist())
        probability_chunks.append(probs)
        np.add.at(cm, (t, pred), 1)
        n += len(t)
        loss_sum += loss.detach().float().item() * len(t)

        if batch_idx % log_every == 0 or batch_idx == total_batches:
            macro_f1, covid_f1, accuracy = running_scores(cm)
            running_loss = loss_sum / max(n, 1)
            elapsed = time.perf_counter() - started
            batch_csv.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "epoch": epoch, "phase": phase, "batch": batch_idx,
                "total_batches": total_batches, "samples_seen": n,
                "running_loss": running_loss, "running_macro_f1": macro_f1,
                "running_covid_f1": covid_f1, "running_accuracy": accuracy,
                "elapsed_seconds": round(elapsed, 2),
            })
            # Flush allows reading batch_history.csv from another program mid-epoch.
            batch_file.flush()
            write_status(status_path, state="running", epoch=epoch,
                         total_epochs=total_epochs, phase=phase,
                         batch=batch_idx, total_batches=total_batches,
                         samples_seen=n, running_loss=running_loss,
                         running_macro_f1=macro_f1, running_covid_f1=covid_f1,
                         elapsed_seconds=round(elapsed, 1))
            if show_progress:
                iterator.set_postfix(loss=f"{running_loss:.4f}",
                                     macro_f1=f"{macro_f1:.4f}",
                                     covid_f1=f"{covid_f1:.4f}")
    y, pred = np.asarray(ys), np.asarray(predictions)
    return {
        "loss": loss_sum / max(n, 1),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "covid_f1": float(f1_score(y == 3, pred == 3, zero_division=0)),
        "accuracy": float(np.mean(y == pred)),
        "y": y,
        "pred": pred,
        "probs": np.concatenate(probability_chunks, axis=0),
    }


def save_curves(history: pd.DataFrame, out: Path):
    history.to_csv(out / "history.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, label in zip(axes, ["loss", "macro_f1", "covid_f1"], ["Loss", "Macro-F1", "COVID-19 F1"]):
        for prefix in ("train", "val"):
            ax.plot(history["epoch"], history[f"{prefix}_{metric}"], label=prefix)
        ax.set(title=label, xlabel="Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=180)
    plt.close(fig)


def save_best_validation(out, dataset, result):
    report = classification_report(
        result["y"], result["pred"], labels=[0, 1, 2, 3],
        target_names=CLASSES, output_dict=True, zero_division=0,
    )
    save_json(out / "best_val_report.json", report)
    cm = confusion_matrix(result["y"], result["pred"], labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")
    ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted", ylabel="True", title="Validation confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    probs = pd.DataFrame(result["probs"], columns=[f"p_{c}" for c in CLASSES])
    probs.insert(0, "pred_label", [CLASSES[i] for i in result["pred"]])
    probs.insert(0, "true_label", [CLASSES[i] for i in result["y"]])
    probs.insert(0, "new_filename", dataset.frame["new_filename"].tolist())
    probs.to_csv(out / "val_probabilities.csv", index=False)


def main():
    p = argparse.ArgumentParser(description="LAB3 CXR four-class training")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--model", choices=MODEL_CHOICES, default="xrv_densenet")
    p.add_argument("--output", type=Path, default=Path("outputs/baseline"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-3, help="Classifier / fusion head LR")
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="weighted_ce")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--sampler", choices=["natural", "balanced"], default="natural")
    p.add_argument("--aux-weight", type=float, default=0.0, help="COVID auxiliary binary loss weight")
    p.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder (default for RAD-DINO/fusion)")
    p.add_argument("--finetune-encoder", action="store_true", help="Fine tune encoder (default for DenseNet)")
    p.add_argument("--no-pretrained", action="store_true", help="Ablation: train from scratch")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0, help="Windows-friendly default")
    p.add_argument("--log-every", type=int, default=10,
                   help="Record running batch metrics and update status.json every N batches")
    p.add_argument("--no-progress", action="store_true",
                   help="Hide tqdm progress bars (epoch summaries will still print)")
    p.add_argument("--tensorboard", action="store_true",
                   help="Write TensorBoard scalars under output/tensorboard (pip install tensorboard)")
    args = p.parse_args()
    if args.freeze_encoder and args.finetune_encoder:
        p.error("Cannot use --freeze-encoder and --finetune-encoder together")
    if args.image_size % 14 != 0 and args.model in ("rad_dino", "fusion"):
        p.error("RAD-DINO requires input size divisible by 14 (e.g. 224, 336, 518)")
    if args.aux_weight < 0:
        p.error("--aux-weight cannot be negative")
    if args.log_every < 1:
        p.error("--log-every must be at least 1")
    fix_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output)
    started = time.perf_counter()
    status_path = args.output / "status.json"
    write_status(status_path, state="initializing", epoch=0, total_epochs=args.epochs,
                 phase="preparing", batch=0, total_batches=0)
    logger.info("Starting %s: %d epochs | batch_size=%d | output=%s",
                args.model, args.epochs, args.batch_size, args.output)
    logger.info("Reading CSVs, loading images/weights; pretrained downloads may take a while on the first run.")
    freeze_encoder = (args.model in ["rad_dino", "fusion"])
    if args.freeze_encoder:
        freeze_encoder = True
    if args.finetune_encoder:
        freeze_encoder = False
    if args.model == "fusion" and not freeze_encoder:
        p.error("Fusion currently supports frozen encoders only on 16GB GPU")
    config = dict(
        model=args.model, freeze_encoder=freeze_encoder,
        dropout=args.dropout, aux_weight=args.aux_weight,
        image_size=args.image_size,
    )
    train_frame = read_labels(csv_path(args.data, "train_data.csv"), require_labels=True)
    val_frame = read_labels(csv_path(args.data, "val_data.csv"), require_labels=True)
    assert not set(train_frame.new_filename) & set(val_frame.new_filename), "Train/val filenames overlap"
    rad_stats = rad_norm_stats() if args.model in {"rad_dino", "fusion"} else None
    train_set = CXRDataset(args.data / "train_images", train_frame, args.model,
                           args.image_size, augment=not args.no_augment, rad_stats=rad_stats)
    val_set = CXRDataset(args.data / "val_images", val_frame, args.model,
                         args.image_size, augment=False, rad_stats=rad_stats)
    counts = np.bincount(train_set.targets, minlength=4)
    sampler = None
    if args.sampler == "balanced":
        class_weights = 1.0 / np.sqrt(counts.astype(float))
        sample_weights = class_weights[train_set.targets]
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double),
                                        num_samples=len(train_set), replacement=True)
    if args.sampler == "balanced" and args.loss != "ce":
        logger.warning("Sampling + weighted loss may over-correct COVID; compare separate ablations.")
    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_set, shuffle=sampler is None, sampler=sampler, **kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model = create_model(config, pretrained=not args.no_pretrained).to(device)
    head_params = [p for name, p in model.named_parameters()
                   if p.requires_grad and not (name.startswith("encoder.") or name.startswith("xrv.") or name.startswith("rad."))]
    backbone_params = [p for name, p in model.named_parameters()
                       if p.requires_grad and (name.startswith("encoder.") or name.startswith("xrv.") or name.startswith("rad."))]
    groups = [{"params": head_params, "lr": args.lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = make_loss(args.loss, counts, device, args.focal_gamma)
    pos_weight = torch.tensor([min(12., math.sqrt((counts.sum() - counts[3]) / counts[3]))], device=device)
    logger.info("Device=%s | BF16=%s | samples=%s", device, amp_enabled,
                dict(zip(CLASSES, counts.tolist())))
    logger.info("Model=%s | freeze_encoder=%s | train batches=%d | val batches=%d",
                args.model, freeze_encoder, len(train_loader), len(val_loader))
    logger.info("Outputs: train.log, batch_history.csv, history.csv, curves.png, status.json")
    # Store config immediately, not just after training completes.
    config_for_json = {**config, **vars(args), "data": str(args.data),
                       "output": str(args.output), "class_counts": counts.tolist()}
    save_json(args.output / "run_config.json",
              {**config_for_json, "best_macro_f1": None})

    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise SystemExit("TensorBoard missing. Install with: pip install tensorboard") from exc
        writer = SummaryWriter(log_dir=str(args.output / "tensorboard"))
        logger.info("TensorBoard: tensorboard --logdir %s", args.output / "tensorboard")

    best_f1, best_epoch, stale, history = -1., 0, 0, []
    with (args.output / "batch_history.csv").open("w", newline="", encoding="utf-8") as batch_file:
        batch_csv = csv.DictWriter(batch_file, fieldnames=BATCH_FIELDS)
        batch_csv.writeheader()
        batch_file.flush()
        try:
            for epoch in range(1, args.epochs + 1):
                epoch_started = time.perf_counter()
                logger.info("---- Epoch %03d/%03d ----", epoch, args.epochs)
                learning_rates = [group["lr"] for group in optimizer.param_groups]
                tr = score_epoch(
                    model, train_loader, device, criterion, args.aux_weight, pos_weight,
                    amp_enabled, epoch=epoch, total_epochs=args.epochs, phase="train",
                    log_every=args.log_every, batch_csv=batch_csv, batch_file=batch_file,
                    status_path=status_path,
                    show_progress=not args.no_progress, optimizer=optimizer,
                )
                va = score_epoch(
                    model, val_loader, device, criterion, args.aux_weight, pos_weight,
                    amp_enabled, epoch=epoch, total_epochs=args.epochs, phase="val",
                    log_every=args.log_every, batch_csv=batch_csv, batch_file=batch_file,
                    status_path=status_path,
                    show_progress=not args.no_progress,
                )
                scheduler.step()
                improved = va["macro_f1"] > best_f1 + 1e-8
                if improved:
                    best_f1, best_epoch, stale = va["macro_f1"], epoch, 0
                    logger.info("New best validation Macro-F1: %.4f. Saving checkpoint...", best_f1)
                    torch.save({"state_dict": model.state_dict(), "config": config,
                                "best_epoch": epoch, "val_macro_f1": best_f1,
                                "classes": CLASSES}, args.output / "best.pt")
                    save_best_validation(args.output, val_set, va)
                else:
                    stale += 1

                elapsed_epoch = time.perf_counter() - epoch_started
                row = {"epoch": epoch,
                       **{f"train_{k}": tr[k] for k in ("loss", "macro_f1", "covid_f1", "accuracy")},
                       **{f"val_{k}": va[k] for k in ("loss", "macro_f1", "covid_f1", "accuracy")},
                       "lr_head": learning_rates[0],
                       "lr_backbone": learning_rates[1] if len(learning_rates) > 1 else 0,
                       "epoch_seconds": round(elapsed_epoch, 2),
                       "best_val_macro_f1": best_f1, "best_epoch": best_epoch}
                history.append(row)
                # Saved every epoch, even if the process ends early.
                save_curves(pd.DataFrame(history), args.output)
                logger.info("Train | Loss %.4f | Macro-F1 %.4f | COVID-F1 %.4f | Acc %.4f",
                            tr["loss"], tr["macro_f1"], tr["covid_f1"], tr["accuracy"])
                logger.info("Val   | Loss %.4f | Macro-F1 %.4f | COVID-F1 %.4f | Acc %.4f",
                            va["loss"], va["macro_f1"], va["covid_f1"], va["accuracy"])
                logger.info("Best  | Val Macro-F1 %.4f (epoch %d) | epoch %.1fs | no improvement %d/%d",
                            best_f1, best_epoch, elapsed_epoch, stale, args.patience)
                if writer is not None:
                    for split, results in (("Train", tr), ("Validation", va)):
                        for metric in ("loss", "macro_f1", "covid_f1", "accuracy"):
                            writer.add_scalar(f"{metric}/{split}", results[metric], epoch)
                    writer.add_scalar("best_val_macro_f1", best_f1, epoch)
                    writer.add_scalar("learning_rate/head", learning_rates[0], epoch)
                    if len(learning_rates) > 1:
                        writer.add_scalar("learning_rate/backbone", learning_rates[1], epoch)
                    writer.flush()
                write_status(status_path, state="running", epoch=epoch,
                             total_epochs=args.epochs, phase="epoch_complete",
                             batch=len(val_loader), total_batches=len(val_loader),
                             val_macro_f1=va["macro_f1"], best_val_macro_f1=best_f1,
                             best_epoch=best_epoch)
                if stale >= args.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, args.patience)
                    break
            duration = time.perf_counter() - started
            save_json(args.output / "run_config.json", {
                **config_for_json, "best_macro_f1": best_f1,
                "best_epoch": best_epoch, "finished_epochs": len(history),
                "total_seconds": round(duration, 2),
            })
            write_status(status_path, state="finished", epoch=len(history),
                         total_epochs=args.epochs, phase="done", batch=0, total_batches=0,
                         best_val_macro_f1=best_f1, best_epoch=best_epoch,
                         total_seconds=round(duration, 2))
            logger.info("Finished: best epoch %d, best Macro-F1 %.4f | outputs: %s",
                        best_epoch, best_f1, args.output)
        except KeyboardInterrupt:
            write_status(status_path, state="interrupted", epoch=len(history),
                         total_epochs=args.epochs, phase="stopped", batch=0, total_batches=0)
            logger.warning("Interrupted. Completed-epoch history/curves and the best checkpoint are preserved.")
            raise
        except Exception:
            write_status(status_path, state="error", epoch=len(history),
                         total_epochs=args.epochs, phase="failed", batch=0, total_batches=0)
            logger.exception("Training failed. See the traceback above and train.log.")
            raise
        finally:
            if writer is not None:
                writer.close()



if __name__ == "__main__":
    main()
