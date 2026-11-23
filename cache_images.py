"""Stream a Folsom archive into a manifest and a compact, memory-mappable RGB cache."""
import argparse
import csv
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--decoder", default="bzip2")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    count = 0
    seen = set()
    with subprocess.Popen([args.decoder, "-dc", str(args.archive)],
                          stdout=subprocess.PIPE) as decoder:
        with (tarfile.open(fileobj=decoder.stdout, mode="r|") as source,
              (args.output / "rgb.bin").open("wb") as binary,
              (args.output / "frames.csv").open("w", newline="") as manifest):
            rows = csv.writer(manifest)
            rows.writerow(["path", "mtime"])
            for entry in source:
                if not entry.isfile() or PurePosixPath(entry.name).suffix.lower() not in {
                    ".jpg", ".jpeg"
                }:
                    continue
                if entry.name in seen:
                    raise ValueError(f"Duplicate archive entry: {entry.name}")
                seen.add(entry.name)
                with source.extractfile(entry) as incoming:
                    with Image.open(BytesIO(incoming.read())) as image:
                        resized = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
                binary.write(resized.tobytes())
                rows.writerow([entry.name, int(entry.mtime)])
                count += 1
                if count % 10000 == 0:
                    print(f"{args.archive.name}: {count} frames", flush=True)
            # tarfile stops at the end marker. Drain padding to verify the bzip2 stream.
            while decoder.stdout.read(2**20):
                pass
        if decoder.wait() or not count:
            raise RuntimeError(f"Incomplete cache for {args.archive}")
    info = {"archive": str(args.archive), "archive_bytes": args.archive.stat().st_size,
            "frames": count, "shape": [count, 64, 64, 3], "format": "RGB uint8 bilinear"}
    (args.output / "complete.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
