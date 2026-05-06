"""Parse per-scene FPS logs under a dataset directory and write averages.

Reads <scene>/fps_results/fps_iteration_<iter>_<impl>.txt for each impl
(currently cuda and torch) and parses the "Test Set Average FPS" line.

Writes <dataset_dir>/fps_summary.txt with per-scene FPS columns and a final
average row, one column per implementation found across the scenes.

Usage:
    python tools/summarize_fps.py <dataset_dir> [--iter best]
    e.g. python tools/summarize_fps.py output/mcmc/dbs/7dgs_pbr
"""
import os
import re
import sys

IMPLS = ("cuda", "torch")
_TEST_AVG_RE = re.compile(r"Test Set Average FPS:\s*([0-9.]+)")


def parse_verbose_log(path):
    """Return float Test Set Average FPS from a verbose log, or None."""
    try:
        with open(path) as f:
            for line in f:
                m = _TEST_AVG_RE.search(line)
                if m:
                    return float(m.group(1))
    except FileNotFoundError:
        return None
    return None


def parse_scene(scene_dir, iter_name):
    """Return dict mapping impl -> fps, reading verbose logs for each impl."""
    out = {}
    fps_dir = os.path.join(scene_dir, "fps_results")
    for impl in IMPLS:
        path = os.path.join(fps_dir, f"fps_iteration_{iter_name}_{impl}.txt")
        v = parse_verbose_log(path)
        if v is not None:
            out[impl] = v
    return out


def main(dataset_dir, iter_name="best"):
    scenes = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
        and os.path.isdir(os.path.join(dataset_dir, d, "fps_results"))
    )
    if not scenes:
        print(f"No fps_results/ found under {dataset_dir}/*/")
        return

    rows = {scene: parse_scene(os.path.join(dataset_dir, scene), iter_name) for scene in scenes}
    scenes = [s for s in scenes if rows[s]]
    if not scenes:
        print(f"No fps_iteration_{iter_name}_*.txt files found under {dataset_dir}/*/fps_results/")
        return
    impls = sorted({impl for r in rows.values() for impl in r})

    name_w = max(len("Scene"), max(len(s) for s in scenes))
    header = f"{'Scene':<{name_w}}  " + "  ".join(f"{i:>10}" for i in impls)
    sep = "-" * len(header)
    out_lines = [
        f"{os.path.basename(os.path.normpath(dataset_dir))} - FPS summary",
        header,
        sep,
    ]
    sums = {i: 0.0 for i in impls}
    counts = {i: 0 for i in impls}
    for scene in scenes:
        cells = []
        for i in impls:
            v = rows[scene].get(i)
            if v is None:
                cells.append(f"{'---':>10}")
            else:
                cells.append(f"{v:>10.2f}")
                sums[i] += v
                counts[i] += 1
        out_lines.append(f"{scene:<{name_w}}  " + "  ".join(cells))
    out_lines.append(sep)
    avg_cells = []
    for i in impls:
        avg_cells.append(f"{sums[i] / counts[i]:>10.2f}" if counts[i] else f"{'---':>10}")
    out_lines.append(f"{'Average':<{name_w}}  " + "  ".join(avg_cells))
    out_lines.append("")

    text = "\n".join(out_lines)
    out_path = os.path.join(dataset_dir, "fps_summary.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(text)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    iter_kwargs = [a for a in sys.argv[1:] if a.startswith("--iter=")]
    if len(args) != 1:
        print(__doc__)
        sys.exit(1)
    iter_name = iter_kwargs[0].split("=", 1)[1] if iter_kwargs else "best"
    main(args[0], iter_name=iter_name)
