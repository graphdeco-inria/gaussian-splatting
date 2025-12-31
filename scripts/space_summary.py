#!/usr/bin/env python3
import argparse
import os
import sys
import time
from collections import defaultdict


def fmt_bytes(sz):
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    val = float(sz)
    for unit in units:
        if val < 1024:
            return f"{val:.2f}{unit}"
        val /= 1024.0
    return f"{val:.2f}EB"


def walk_files(paths, follow_symlinks=True, progress_interval=1.0):
    last_report = time.time()
    file_count = 0
    byte_count = 0
    last_path = ""

    for base in paths:
        if not os.path.exists(base):
            print(f"skip missing: {base}", file=sys.stderr)
            continue
        for dirpath, _, filenames in os.walk(base, followlinks=follow_symlinks):
            last_path = dirpath
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path, follow_symlinks=follow_symlinks)
                except FileNotFoundError:
                    continue
                if not os.path.isfile(path):
                    continue
                file_count += 1
                byte_count += st.st_size
                now = time.time()
                if now - last_report >= progress_interval:
                    rate = file_count / max(1e-9, now - last_report)
                    print(
                        f"scanned: {file_count} files, {fmt_bytes(byte_count)} "
                        f"(+{rate:.1f} files/s) in {last_path}",
                        file=sys.stderr,
                    )
                    last_report = now

                yield path, st.st_size

    print(
        f"done: {file_count} files, {fmt_bytes(byte_count)}",
        file=sys.stderr,
    )


def summarize(paths, follow_symlinks=True, progress_interval=1.0):
    total = 0
    by_ext = defaultdict(lambda: [0, 0])
    by_top = defaultdict(lambda: [0, 0])

    for path, sz in walk_files(paths, follow_symlinks, progress_interval):
        total += sz
        ext = os.path.splitext(path)[1].lower() or "<no_ext>"
        by_ext[ext][0] += sz
        by_ext[ext][1] += 1
        parts = path.split(os.sep)
        top = parts[0]
        if len(parts) > 1:
            top = os.path.join(parts[0], parts[1])
        by_top[top][0] += sz
        by_top[top][1] += 1

    return total, by_ext, by_top


def print_table(title, items, total, limit):
    print(title)
    print("name\tsize\tpercent\tavg\tcount")
    for name, (sz, cnt) in items[:limit]:
        pct = (sz / total * 100.0) if total else 0.0
        avg = (sz / cnt) if cnt else 0.0
        print(f"{name}\t{fmt_bytes(sz)}\t{pct:6.2f}%\t{fmt_bytes(avg)}\t{cnt}")
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize disk usage by extension and top-level path."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["data1", "data2"],
        help="Paths to scan (default: data1 data2)",
    )
    parser.add_argument(
        "--no-follow-symlinks",
        action="store_true",
        help="Do not follow symlinks when scanning",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of rows to show per table (default: 15)",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=1.0,
        help="Seconds between progress updates (default: 1.0)",
    )
    args = parser.parse_args()

    total, by_ext, by_top = summarize(
        args.paths,
        follow_symlinks=not args.no_follow_symlinks,
        progress_interval=args.progress_interval,
    )

    print(f"TOTAL\t{fmt_bytes(total)}\n")

    ext_items = sorted(by_ext.items(), key=lambda x: x[1][0], reverse=True)
    top_items = sorted(by_top.items(), key=lambda x: x[1][0], reverse=True)

    print_table("BY EXTENSION", ext_items, total, args.top)
    print_table("BY TOP PATH", top_items, total, args.top)


if __name__ == "__main__":
    main()
