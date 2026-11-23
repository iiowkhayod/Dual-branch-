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


def history_indices(complete, max_gap):
    valid = np.flatnonzero(complete)
    if len(valid) < HISTORY:
        return np.empty((0, HISTORY), dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(valid, HISTORY)
    return windows[(np.diff(windows, axis=1) <= max_gap).all(axis=1)]


def extraterrestrial_horizontal(times):
    zenith = solarposition.get_solarposition(times, 38.74, -121.19).zenith
    return np.asarray(irradiance.get_extra_radiation(times) *
                      np.cos(np.deg2rad(zenith)), dtype=np.float32)


def prepare_year(observations, year, max_gap, *, minute_targets):
    times = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01",
                          inclusive="left", freq="min", tz="UTC")
    table = observations.reindex(times)
    hour = 2 * np.pi * (times.hour + times.minute / 60) / 24
    year_days = len(times) / 1440
    doy = 2 * np.pi * (times.dayofyear - 1) / year_days
    values = np.column_stack((table[IRR + MET], np.sin(hour), np.cos(hour),
                              np.sin(doy), np.cos(doy))).astype(np.float32)
    ghi = values[:, 0].copy()
    i0 = extraterrestrial_horizontal(times)
    daylight = np.isfinite(ghi) & (ghi > 10) & (i0 > 0)
    complete = np.isfinite(values).all(axis=1) & daylight
    windows = history_indices(complete, max_gap)
    windows = windows[windows[:, -1] + LEADS[-1] < len(times)]
    if minute_targets:
        windows = windows[daylight[windows[:, -1, None] + LEADS].all(axis=1)]
    return times, values, ghi, i0, windows


class Frames:
    def __init__(self, root, cached):
        self.root = root
        self.rgb = None
        if cached:
            info = json.loads((root / "complete.json").read_text())
            manifest = pd.read_csv(root / "frames.csv")
            require_bytes = len(manifest) * 64 * 64 * 3
            if len(manifest) != info["frames"] or (root / "rgb.bin").stat().st_size != require_bytes:
                raise ValueError(f"Cache count or length mismatch: {root}")
            self.names = manifest.path.tolist()
            self.mtime = manifest.mtime.to_numpy(dtype=np.int64)
            self.rgb = np.memmap(root / "rgb.bin", dtype=np.uint8, mode="r",
                                 shape=(len(manifest), 64, 64, 3))
        else:
            paths = sorted(path for path in root.rglob("*")
                           if path.suffix.lower() in {".jpg", ".jpeg"} and path.is_file())
            self.names = [path.relative_to(root).as_posix() for path in paths]
            self.mtime = np.array([int(path.stat().st_mtime) for path in paths], dtype=np.int64)

    def read(self, index):
        if self.rgb is not None:
            return self.rgb[index]
        with Image.open(self.root / self.names[index]) as image:
            rgb = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
            return np.asarray(rgb)


def match_images(frames, times, issues, clock, age, limit):
    if not frames.names:
        raise ValueError(f"No JPEGs in {frames.root}")
    if clock == "mtime":
        seconds = frames.mtime
    else:
        stamps = pd.to_datetime([Path(name).stem for name in frames.names],
                                format="%Y%m%d_%H%M%S", utc=True)
        seconds = stamps.to_numpy(dtype="datetime64[s]").astype("int64")
    order = np.lexsort((frames.names, seconds))
    seconds = seconds[order]
    issue_seconds = times[issues].to_numpy(dtype="datetime64[s]").astype("int64")
    slots = np.searchsorted(seconds, issue_seconds, side="right") - 1
    valid = (slots >= 0) & (issue_seconds - seconds[np.maximum(slots, 0)] <= age)
    rows = np.flatnonzero(valid)
    if not len(rows):
        raise ValueError(f"No causal image matches in {frames.root} using {clock}")
    if limit and len(rows) > limit:
        rows = rows[np.linspace(0, len(rows) - 1, limit, dtype=int)]
    return rows, order[slots[rows]], issue_seconds[rows] - seconds[slots[rows]]


def interpolate_targets(ghi, minutes, seconds):
    fraction = seconds[:, None] / 60
    # Exact-minute labels need only the exact observation.
    later = minutes + (fraction > 0)
    return (ghi[minutes] * (1 - fraction) + ghi[later] * fraction).astype(np.float32)


def image_anchored_samples(frames, times, windows, ghi, limit):
    if not frames.names:
        raise ValueError(f"No JPEGs in {frames.root}")
    ordered = np.lexsort((frames.names, frames.mtime))
    unique_time = np.r_[True, np.diff(frames.mtime[ordered]) != 0]
    frame_ids = ordered[unique_time]
    year_start = times[0].value // 10**9
    minutes, seconds = np.divmod(frames.mtime[frame_ids] - year_start, 60)
    valid = (minutes >= 0) & (minutes + LEADS[-1] + (seconds > 0) < len(times))
    lookup = np.full(len(times), -1, dtype=np.int64)
    lookup[windows[:, -1]] = np.arange(len(windows))
    frame_ids, minutes, seconds = frame_ids[valid], minutes[valid], seconds[valid]
    rows = lookup[minutes]
    valid = rows >= 0
    frame_ids, minutes, seconds, rows = (x[valid] for x in (frame_ids, minutes, seconds, rows))
    targets = interpolate_targets(ghi, minutes[:, None] + LEADS, seconds)
    valid = (np.isfinite(targets) & (targets > 10)).all(axis=1)
    frame_ids, rows, targets = (x[valid] for x in (frame_ids, rows, targets))
    frame_times = pd.to_datetime(frames.mtime[frame_ids], unit="s", utc=True)
    target_seconds = frames.mtime[frame_ids, None] + LEADS * 60
    i0 = extraterrestrial_horizontal(pd.to_datetime(target_seconds.ravel(), unit="s", utc=True))
    i0 = i0.reshape(-1, len(LEADS))
    selected = np.flatnonzero((i0 > 0).all(axis=1))
    if not len(selected):
        raise ValueError(f"No valid image-anchored forecast samples in {frames.root}")
    if limit and len(selected) > limit:
        selected = selected[np.linspace(0, len(selected) - 1, limit, dtype=int)]
    return rows[selected], frame_ids[selected], frame_times[selected], targets[selected], i0[selected]


def write_clock_audit(frames, output):
    filenames = pd.to_datetime([Path(name).stem for name in frames.names],
                                format="%Y%m%d_%H%M%S", utc=True, errors="coerce")
    modified = pd.to_datetime(frames.mtime, unit="s", utc=True)
    offset = (filenames - modified).total_seconds()
    daily = pd.DataFrame({"dm_utc_day": modified.normalize(), "fn_minus_dm_s": offset})
    daily.groupby("dm_utc_day")["fn_minus_dm_s"].agg(
        frames="size", parsed_filenames="count", min_s="min", median_s="median", max_s="max"
    ).to_csv(output)


class Folsom(Dataset):
    def __init__(self, values, windows, frames, frame_ids, scaler, targets, target_i0, issue_index):
        self.values = (values - scaler[0]) / scaler[1]
        self.windows = windows
        self.frames = frames
        self.frame_ids = frame_ids
        self.targets = targets
        self.i0 = target_i0
        self.baseline = issue_index[:, None] * self.i0

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        rgb = self.frames.read(self.frame_ids[index])
        pixels = rgb.transpose(2, 0, 1).astype(np.float32) / 255
        history = self.values[self.windows[index]]
        return pixels, history, self.targets[index], self.i0[index]


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    squared = 0.0
    predictions = []
    with torch.set_grad_enabled(optimizer is not None):
        for batch, (image, history, ghi, i0) in enumerate(loader, 1):
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
            if optimizer is not None and batch % 250 == 0:
                print(f"train_batch={batch}/{len(loader)}", flush=True)
    return squared / len(loader.dataset), predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images", type=Path, nargs=3,
                        metavar=("TRAIN_2014", "VAL_2015", "TEST_2016"))
    inputs.add_argument("--image-cache", type=Path, nargs=3,
                        metavar=("CACHE_2014", "CACHE_2015", "CACHE_2016"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-clock", choices=["mtime", "filename"], default="mtime")
    parser.add_argument("--issue-clock", choices=["minute", "image"], default="minute")
    parser.add_argument("--max-image-age", type=int, default=60)
    parser.add_argument("--max-history-gap", type=int, choices=[1, 2], default=1)
    parser.add_argument("--limit-per-year", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Write configuration, sample manifests and clock audits without training.")
    args = parser.parse_args()
    if args.issue_clock == "image" and args.image_clock != "mtime":
        parser.error("Image-anchored forecasts use archive mtime")
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
    minute_targets = args.issue_clock == "minute"
    for year, root in zip((2014, 2015, 2016), args.images or args.image_cache):
        times, values, ghi, i0, windows = prepare_year(obs, year, args.max_history_gap,
                                                      minute_targets=minute_targets)
        frames = Frames(root, cached=args.image_cache is not None)
        if minute_targets:
            rows, frame_ids, ages = match_images(frames, times, windows[:, -1], args.image_clock,
                                                 args.max_image_age, args.limit_per_year)
            issues = windows[rows, -1]
            issue = times[issues]
            target_rows = issues[:, None] + LEADS
            targets, target_i0 = ghi[target_rows], i0[target_rows]
        else:
            rows, frame_ids, issue, targets, target_i0 = image_anchored_samples(
                frames, times, windows, ghi, args.limit_per_year)
            ages = np.zeros(len(rows), dtype=np.int64)
            write_clock_audit(frames, args.output / f"clock_audit_{year}.csv")
        windows = windows[rows]
        issues = windows[:, -1]
        if year == 2014:
            training_rows = values[np.unique(windows)]
            mean = training_rows.mean(axis=0)
            std = training_rows.std(axis=0)
            std[std == 0] = 1
            scaler = (mean, std)
        pd.DataFrame({"issue_utc": issue, "image": [frames.names[i] for i in frame_ids],
                      "image_age_seconds": ages, "history_end_utc": times[issues],
                      "history_start_utc": times[windows[:, 0]],
                      "history_2min_gaps": (np.diff(windows, axis=1) == 2).sum(axis=1)
                      }).to_csv(args.output / f"samples_{year}.csv", index=False)
        dataset = Folsom(values, windows, frames, frame_ids, scaler, targets, target_i0,
                         ghi[issues] / i0[issues])
        datasets.append(dataset)
        issue_times.append(issue)
        load.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=year == 2014,
                               num_workers=args.workers))
        print(f"{year}: {len(dataset)} matched samples", flush=True)

    if args.prepare_only:
        return
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
