"""Convert LAB3 model predictions into Kaggle one-hot CSV, preserving template order."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lab3_common import (
    CLASSES, CXRDataset, batch_to_device, create_model, csv_path, rad_norm_stats, read_labels,
)


def main():
    p = argparse.ArgumentParser(description="LAB3 Kaggle submission creator")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("submission.csv"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config, pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()
    template = read_labels(csv_path(args.data, "test_data_sample.csv"), require_labels=False)
    stats = rad_norm_stats() if config["model"] in {"rad_dino", "fusion"} else None
    test_set = CXRDataset(args.data / "test_images", template, config["model"],
                          config["image_size"], augment=False, rad_stats=stats, labeled=False)
    loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, pin_memory=device.type == "cuda")
    probs = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch_to_device(batch, device)
            with (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext()):
                logits, _ = model(batch)
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    probabilities = np.concatenate(probs, axis=0)
    assert probabilities.shape == (len(template), 4)
    assert np.isfinite(probabilities).all()
    pred = probabilities.argmax(axis=1)
    # Build one-hot labels in a writable NumPy array. Pandas .values may be read-only.
    one_hot = np.zeros((len(template), len(CLASSES)), dtype=np.int64)
    one_hot[np.arange(len(template)), pred] = 1
    submission = pd.DataFrame(one_hot, columns=CLASSES)
    submission.insert(0, "new_filename", template["new_filename"].tolist())
    assert submission.columns.tolist() == template.columns.tolist()
    assert (submission[CLASSES].sum(axis=1) == 1).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    prob_df = pd.DataFrame(probabilities, columns=[f"p_{name}" for name in CLASSES])
    prob_df.insert(0, "new_filename", template["new_filename"].tolist())
    prob_path = args.output.with_name(args.output.stem + "_probabilities.csv")
    prob_df.to_csv(prob_path, index=False)
    print(f"Saved submission to {args.output} ({len(submission)} rows).")
    print(f"Saved probabilities to {prob_path}.")
    print("Predicted class counts:", dict(zip(CLASSES, np.bincount(pred, minlength=4).tolist())))


if __name__ == "__main__":
    main()
