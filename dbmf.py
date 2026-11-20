"""Independent reconstruction of the published DBMF training path."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pvlib import irradiance, solarposition
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


LEADS = np.array([10, 20, 30])
HISTORY = 30
IRR = ["ghi", "dni", "dhi"]
MET = ["air_temp", "relhum", "windsp", "winddir", "max_windsp"]


class DBMF(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for index, (incoming, outgoing) in enumerate(zip(
            [3, 32, 64, 128], [32, 64, 128, 128]
        )):
            pool = nn.MaxPool2d(2) if index < 3 else nn.AdaptiveAvgPool2d(1)
            blocks.extend([nn.Conv2d(incoming, outgoing, 3, padding=1),
                           nn.BatchNorm2d(outgoing), nn.ReLU(), pool])
        self.cnn = nn.Sequential(*blocks, nn.Flatten(), nn.Linear(128, 128))
        self.lstm = nn.LSTM(12, 128, num_layers=2, dropout=0.2, batch_first=True)
        self.norm = nn.LayerNorm(128)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(256, len(LEADS)))

    def forward(self, image, history):
        _, (hidden, _) = self.lstm(history)
        return self.head(torch.cat((self.cnn(image), self.norm(hidden[-1])), dim=1))


def read_observations(path, columns):
    table = pd.read_csv(path, usecols=["timeStamp", *columns], na_values=["NAN"])
    table.index = pd.DatetimeIndex(pd.to_datetime(table.pop("timeStamp"), utc=True))
    return table.loc[~table.index.duplicated(keep=False)].apply(pd.to_numeric, errors="coerce")


def prepare_year(observations, year):
    times = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01",
                          inclusive="left", freq="min", tz="UTC")
    table = observations.reindex(times)
    hour = 2 * np.pi * (times.hour + times.minute / 60) / 24
    year_days = len(times) / 1440
    doy = 2 * np.pi * (times.dayofyear - 1) / year_days
    values = np.column_stack((table[IRR + MET], np.sin(hour), np.cos(hour),
                              np.sin(doy), np.cos(doy))).astype(np.float32)
    ghi = values[:, 0].copy()
    zenith = solarposition.get_solarposition(times, 38.74, -121.19).zenith
    i0 = np.asarray(irradiance.get_extra_radiation(times) *
                    np.cos(np.deg2rad(zenith)), dtype=np.float32)
    daylight = np.isfinite(ghi) & (ghi > 10) & (i0 > 0)
    complete = np.isfinite(values).all(axis=1) & daylight
    history_ok = np.lib.stride_tricks.sliding_window_view(complete, HISTORY).all(axis=1)
    issues = np.flatnonzero(history_ok) + HISTORY - 1
    issues = issues[issues + LEADS[-1] < len(times)]
    issues = issues[daylight[issues[:, None] + LEADS].all(axis=1)]
    return times, values, ghi, i0, issues


def match_images(root, times, issues, clock, age, limit):
    image_paths = sorted(path for path in root.rglob("*")
                         if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"})
    if not image_paths:
        raise ValueError(f"No JPEGs in {root}")
    if clock == "mtime":
        seconds = np.array([int(path.stat().st_mtime) for path in image_paths])
    else:
        stamps = pd.to_datetime([path.stem for path in image_paths],
                                format="%Y%m%d_%H%M%S", utc=True)
        seconds = stamps.to_numpy(dtype="datetime64[s]").astype("int64")
    order = np.argsort(seconds, kind="stable")
    seconds = seconds[order]
    issue_seconds = times[issues].to_numpy(dtype="datetime64[s]").astype("int64")
    slots = np.searchsorted(seconds, issue_seconds, side="right") - 1
    valid = (slots >= 0) & (issue_seconds - seconds[np.maximum(slots, 0)] <= age)
    issues, slots = issues[valid], slots[valid]
    if not len(issues):
        raise ValueError(f"No causal image matches in {root} using {clock}")
    if limit and len(issues) > limit:
        keep = np.linspace(0, len(issues) - 1, limit, dtype=int)
        issues, slots = issues[keep], slots[keep]
    paths = [image_paths[index] for index in order[slots]]
    ages = times[issues].to_numpy(dtype="datetime64[s]").astype("int64") - seconds[slots]
    return issues, paths, ages


class Folsom(Dataset):
    def __init__(self, values, ghi, i0, issues, paths, scaler):
        self.values = (values - scaler[0]) / scaler[1]
        self.issues = issues
        self.paths = paths
        targets = issues[:, None] + LEADS
        self.targets = ghi[targets]
        self.i0 = i0[targets]
        self.baseline = ghi[issues, None] / i0[issues, None] * self.i0

    def __len__(self):
        return len(self.issues)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            rgb = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
            pixels = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255
        issue = self.issues[index]
        history = self.values[issue - HISTORY + 1:issue + 1]
        return pixels, history, self.targets[index], self.i0[index]


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    squared = 0.0
    predictions = []
    with torch.set_grad_enabled(optimizer is not None):
        for image, history, ghi, i0 in loader:
            image, history, ghi, i0 = (x.to(device) for x in (image, history, ghi, i0))
            prediction = model(image, history)
            loss = nn.functional.mse_loss(prediction, ghi / i0)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite clearness-index MSE")
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            else:
                predictions.append((prediction * i0).cpu().numpy())
            squared += loss.item() * len(image)
    return squared / len(loader.dataset), predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path, nargs=3,
                        metavar=("TRAIN_2014", "VAL_2015", "TEST_2016"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-clock", choices=["mtime", "filename"], default="mtime")
    parser.add_argument("--max-image-age", type=int, default=60)
    parser.add_argument("--limit-per-year", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or min(
        args.limit_per_year, args.max_image_age, args.workers
    ) < 0:
        parser.error("Invalid run size or worker count")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    paths = [args.raw / f"Folsom_{kind}.csv" for kind in ("irradiance", "weather")]
    obs = read_observations(paths[0], IRR).join(read_observations(paths[1], MET))
    config = vars(args) | {
        "versions": {"torch": torch.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__, "cuda": torch.version.cuda},
        "csv_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    (args.output / "config.json").write_text(json.dumps(config, default=str, indent=2))
    datasets, load, issue_times = [], [], []
    for year, root in zip((2014, 2015, 2016), args.images):
        times, values, ghi, i0, issues = prepare_year(obs, year)
        issues, frames, ages = match_images(root, times, issues, args.image_clock,
                                            args.max_image_age, args.limit_per_year)
        if year == 2014:
            windows = issues[:, None] + np.arange(1 - HISTORY, 1)
            training_rows = values[np.unique(windows)]
            mean = training_rows.mean(axis=0)
            std = training_rows.std(axis=0)
            std[std == 0] = 1
            scaler = (mean, std)
        pd.DataFrame({"issue_utc": times[issues], "image": [str(p.relative_to(root)) for p in frames],
                      "image_age_seconds": ages}).to_csv(args.output / f"samples_{year}.csv", index=False)
        dataset = Folsom(values, ghi, i0, issues, frames, scaler)
        datasets.append(dataset)
        issue_times.append(times[issues])
        load.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=year == 2014,
                               num_workers=args.workers))
        print(f"{year}: {len(dataset)} matched samples", flush=True)

    model = DBMF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    best, best_epoch = float("inf"), 0
    checkpoint = args.output / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, load[0], device, optimizer)
        val_loss, _ = run_epoch(model, load[1], device)
        schedule.step(val_loss)
        print(f"epoch={epoch} train_mse={train_loss:.8g} val_mse={val_loss:.8g}", flush=True)
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            temporary = checkpoint.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "mean": mean.tolist(), "std": std.tolist()}, temporary)
            os.replace(temporary, checkpoint)
        if epoch - best_epoch >= 8:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    _, batches = run_epoch(model, load[2], device)
    prediction = np.concatenate(batches)
    actual, baseline = datasets[2].targets, datasets[2].baseline
    metrics = []
    errors = prediction - actual
    for index, lead in enumerate(LEADS):
        error = errors[:, index].astype(np.float64)
        rmse = np.sqrt(np.mean(error**2))
        sp_rmse = np.sqrt(np.mean((baseline[:, index] - actual[:, index])**2))
        variance = np.var(actual[:, index].astype(np.float64))
        metrics.append({"lead_min": lead, "n": len(error), "rmse": rmse,
                        "mae": np.mean(np.abs(error)), "mbe": np.mean(error),
                        "r2": 1 - rmse**2 / variance if variance else np.nan,
                        "sp_rmse": sp_rmse, "skill_percent": 100 * (1 - rmse / sp_rmse)
                        if sp_rmse else np.nan})
    pd.DataFrame(metrics).to_csv(args.output / "metrics.csv", index=False)
    forecasts = pd.DataFrame({"issue_utc": issue_times[2]})
    for index, lead in enumerate(LEADS):
        forecasts[f"ghi_{lead}"] = actual[:, index]
        forecasts[f"dbmf_{lead}"] = prediction[:, index]
        forecasts[f"sp_{lead}"] = baseline[:, index]
    forecasts.to_csv(args.output / "predictions.csv", index=False)
    print(f"best_epoch={best_epoch} metrics={args.output / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
