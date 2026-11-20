"""Extract JPEGs from a Folsom tar stream while preserving its UTC image clock."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after this many JPEGs; zero reads the entire archive.")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    count = 0
    with tarfile.open(args.archive, "r|*") as source:
        for entry in source:
            relative = PurePosixPath(entry.name)
            if not entry.isfile() or relative.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            if relative.is_absolute() or ".." in relative.parts or any(
                ":" in part or "\\" in part for part in relative.parts
            ):
                raise ValueError(f"Unsafe image path: {entry.name}")
            path = args.output.joinpath(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(entry) as incoming, path.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            os.utime(path, (entry.mtime, entry.mtime))
            count += 1
            if args.limit and count >= args.limit:
                break
    if not count:
        raise ValueError("No JPEGs in archive")
    info = {"archive": str(args.archive), "bytes": args.archive.stat().st_size,
            "jpeg_count": count, "limit": args.limit}
    (args.output / "extraction.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
