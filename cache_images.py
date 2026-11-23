"""Stream a Folsom archive into a manifest and a compact, memory-mappable RGB cache."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from io import BytesIO
from itertools import islice
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

from PIL import Image


def jpeg_members(source):
    seen = set()
    for entry in source:
        if not entry.isfile() or PurePosixPath(entry.name).suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        if entry.name in seen:
            raise ValueError(f"Duplicate archive entry: {entry.name}")
        seen.add(entry.name)
        with source.extractfile(entry) as incoming:
            yield entry.name, int(entry.mtime), incoming.read()


def resize_jpeg(member):
    name, mtime, data = member
    try:
        with Image.open(BytesIO(data)) as image:
            resized = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    except OSError as error:
        return name, mtime, None, str(error)
    return name, mtime, resized.tobytes(), ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--decoder", default="bzip2")
    parser.add_argument("--decoder-option", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--prefix", type=Path, help="Reuse verified rows from a stopped strict cache.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    count = skipped = 0
    with subprocess.Popen([args.decoder, "-dc", *args.decoder_option, str(args.archive)],
                          stdout=subprocess.PIPE) as decoder:
        with (tarfile.open(fileobj=decoder.stdout, mode="r|") as source,
              (args.output / "rgb.bin").open("wb") as binary,
              (args.output / "frames.csv").open("w", newline="") as manifest,
              (args.output / "unreadable.csv").open("w", newline="") as rejected,
              ThreadPoolExecutor(max_workers=args.workers) as pool):
            rows = csv.writer(manifest)
            rows.writerow(["path", "mtime"])
            bad_rows = csv.writer(rejected)
            bad_rows.writerow(["path", "mtime", "error"])
            members = jpeg_members(source)
            if args.prefix:
                if (args.prefix / "unreadable.csv").exists():
                    raise ValueError("--prefix requires a strict cache with no skipped frames")
                with (args.prefix / "frames.csv").open(newline="") as prior:
                    for row in csv.DictReader(prior):
                        name, stamp, _ = next(members)
                        if (name, stamp) != (row["path"], int(row["mtime"])):
                            raise ValueError(f"Prefix mismatch: {name}")
                        rows.writerow([name, stamp])
                        count += 1
                with (args.prefix / "rgb.bin").open("rb") as previous:
                    if (args.prefix / "rgb.bin").stat().st_size != count * 64 * 64 * 3:
                        raise ValueError("Prefix byte length does not match its manifest")
                    shutil.copyfileobj(previous, binary, length=2**20)
                print(f"Reused {count} cached prefix frames", flush=True)
            while batch := list(islice(members, 64)):
                for name, mtime, rgb, error in pool.map(resize_jpeg, batch):
                    if rgb is None:
                        bad_rows.writerow([name, mtime, error])
                        skipped += 1
                        continue
                    binary.write(rgb)
                    rows.writerow([name, mtime])
                    count += 1
                    if count % 10000 == 0:
                        print(f"{args.archive.name}: {count} frames", flush=True)
            # tarfile stops at the end marker. Drain padding to verify the bzip2 stream.
            while decoder.stdout.read(2**20):
                pass
        if decoder.wait() or not count:
            raise RuntimeError(f"Incomplete cache for {args.archive}")
    info = {"archive": str(args.archive), "archive_bytes": args.archive.stat().st_size,
            "frames": count, "shape": [count, 64, 64, 3], "format": "RGB uint8 bilinear",
            "decoder": args.decoder, "decoder_options": args.decoder_option,
            "workers": args.workers, "prefix": str(args.prefix) if args.prefix else None,
            "unreadable_frames": skipped}
    marker = args.output / "complete.tmp"
    marker.write_text(json.dumps(info, indent=2))
    marker.replace(args.output / "complete.json")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
