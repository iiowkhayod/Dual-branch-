"""Paired lightweight ablation of minute labels versus DM-interpolated labels."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dbmf import (
    DBMF, Folsom, IRR, LEADS, MET, Frames, image_anchored_samples,
    prepare_year, read_observations, run_epoch,
)


def fit_and_predict(train, validation, test, seed, epochs, device):
    torch.manual_seed(seed)
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=64)
    model = DBMF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        run_epoch(model, train_loader, device, optimizer)
        validation_loss, _ = run_epoch(model, validation_loader, device)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    _, predictions = run_epoch(model, DataLoader(test, batch_size=64), device)
    return np.concatenate(predictions), best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    seeds = (42, 123, 2024)
    config = vars(args) | {"seeds": seeds, "torch": torch.__version__}
    (args.output / "config.json").write_text(json.dumps(config, default=str, indent=2))
    observations = read_observations(args.raw / "Folsom_irradiance.csv", IRR).join(
        read_observations(args.raw / "Folsom_weather.csv", MET))
    datasets = []
    for year in (2014, 2015, 2016):
        times, values, ghi, i0, windows = prepare_year(observations, year, 2, minute_targets=False)
        frames = Frames(args.cache / str(year), cached=True)
        rows, frame_ids, issue, targets, target_i0 = image_anchored_samples(frames, times, windows, ghi, 0)
        windows = windows[rows]
        history_end = windows[:, -1]
        # The minute control labels the frame on the first minute at or after DM.
        minute_targets = ghi[history_end[:, None] + LEADS + (issue.second.to_numpy() > 0)[:, None]]
        keep = (np.isfinite(minute_targets) & (minute_targets > 10)).all(axis=1)
        windows, frame_ids, history_end, minute_targets, targets, target_i0 = (
            array[keep] for array in (windows, frame_ids, history_end, minute_targets, targets, target_i0))
        issue = issue[keep]
        if year == 2014:
            observed = values[np.unique(windows)]
            std = observed.std(axis=0)
            std[std == 0] = 1
            scaler = (observed.mean(axis=0), std)
        issue_index = ghi[history_end] / i0[history_end]
        dataset = Folsom(values, windows, frames, frame_ids, scaler, targets, target_i0, issue_index)
        datasets.append(dataset)
        pd.DataFrame({"issue_utc": issue, "image": [frames.names[i] for i in frame_ids],
                      "history_end_utc": times[history_end]}).to_csv(
                          args.output / f"samples_{year}.csv", index=False)
        print(f"{year}: {len(dataset)} common samples", flush=True)
        if year == 2014:
            control = copy.copy(dataset)
            control.targets = minute_targets
        if year == 2016:
            test_times = issue
    train, validation, test = datasets
    metrics, forecasts = [], []
    for seed in seeds:
        for method, inputs in (("minute", control), ("interpolated", train)):
            prediction, epoch = fit_and_predict(inputs, validation, test, seed, args.epochs, device)
            errors = prediction.astype(np.float64) - test.targets
            print(f"{method}: seed={seed}, best_epoch={epoch}", flush=True)
            for index, lead in enumerate(LEADS):
                error = errors[:, index]
                metrics.append({"method": method, "seed": seed, "lead_min": lead, "n": len(test),
                                "best_epoch": epoch, "rmse": np.sqrt(np.mean(error**2)),
                                "mae": np.mean(np.abs(error))})
                forecasts.append(pd.DataFrame({"method": method, "seed": seed, "lead_min": lead,
                                               "issue_utc": test_times, "ghi": test.targets[:, index],
                                               "prediction": prediction[:, index]}))
    pd.DataFrame(metrics).to_csv(args.output / "metrics.csv", index=False)
    pd.concat(forecasts, ignore_index=True).to_csv(args.output / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
