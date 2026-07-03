"""CUDA vecAdd + Nsight Compute block-size visualizer.

This script compiles a tiny CUDA vector-add kernel for multiple block sizes,
profiles each run with Nsight Compute, extracts a curated metric subset, and
shows polished matplotlib figures without writing persistent profiler artifacts.

"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

CUDA_SOURCE = r"""
#include <iostream>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);

    float *h_a = new float[n], *h_b = new float[n], *h_c = new float[n];
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize = BLOCK_SIZE;
    int gridSize = (n + blockSize - 1) / blockSize;
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    std::cout << "c[0] = " << h_c[0] << ", c[n-1] = " << h_c[n - 1] << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    return 0;
}
"""

METRICS = {
    "duration_us": "gpu__time_duration.sum",
    "mem_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "occ_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "regs_per_thread": "launch__registers_per_thread",
    "grid_size": "launch__grid_size",
    "block_size_reported": "launch__block_size",
}

VALID_BLOCK_SIZES = (32, 64, 128, 256, 512, 1024)


@dataclass(frozen=True)
class ProfileRow:
    block_size: int
    grid_size: float
    duration_us: float
    mem_pct: float
    dram_pct: float
    sm_pct: float
    occ_pct: float
    l2_hit_pct: float
    regs_per_thread: float
    block_size_reported: float


class CommandError(RuntimeError):
    pass


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required tool in PATH: {name}")


def validate_block_sizes(sizes: Iterable[int]) -> list[int]:
    sizes = list(sizes)
    invalid = [s for s in sizes if s not in VALID_BLOCK_SIZES]
    if invalid:
        raise SystemExit(
            f"Invalid block sizes: {invalid}. Use only {list(VALID_BLOCK_SIZES)}; CUDA blocks are limited to 1024 threads."
        )
    return sizes


def write_cuda_source(path: Path) -> None:
    path.write_text(CUDA_SOURCE)


def compile_binary(source_path: Path, output_path: Path, block_size: int) -> None:
    cmd = ["nvcc", "-O2", f"-DBLOCK_SIZE={block_size}", str(source_path), "-o", str(output_path)]
    res = run_command(cmd)
    if res.returncode != 0:
        raise CommandError(
            f"nvcc failed for block size {block_size}"
            f"CMD: {' '.join(cmd)},STDOUT:{res.stdout},STDERR:{res.stderr}"
        )


def parse_ncu_raw_csv(raw_text: str) -> pd.DataFrame:
    lines = raw_text.splitlines()

    header_index = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"') and '"Kernel Name"' in line:
            header_index = i
            break

    if header_index is None:
        raise CommandError(
            "Could not find a usable CSV header in Nsight Compute output.\n\n"
            f"RAW OUTPUT:\n{raw_text}"
        )

    df = pd.read_csv(io.StringIO("\n".join(lines[header_index:])))

    if "ID" not in df.columns:
        raise CommandError(f"CSV parsed, but no ID column was found. Columns: {list(df.columns)}")

    id_as_num = pd.to_numeric(df["ID"], errors="coerce")
    df = df[id_as_num.notna()].copy()
    df["ID"] = id_as_num[id_as_num.notna()].astype(int)

    if df.empty:
        raise CommandError(
            "CSV was parsed, but no numeric kernel rows remained after removing units/non-data rows.\n\n"
            f"PARSED COLUMNS: {list(df.columns)}\n\nRAW OUTPUT:\n{raw_text}"
        )

    return df.reset_index(drop=True)


def numeric(value):
    return pd.to_numeric(value, errors="coerce")


def create_report(binary_path: Path, report_base: Path) -> None:
    cmd = [
        "ncu",
        "--target-processes", "application-only",
        "--set", "full",
        "-o", str(report_base),
        str(binary_path),
    ]
    res = run_command(cmd)
    if res.returncode != 0:
        raise CommandError(f"ncu profile failed,CMD: {' '.join(cmd)},STDOUT:{res.stdout},STDERR:{res.stderr}"        )


def load_metrics_from_report(report_file: Path) -> pd.DataFrame:
    cmd = [
        "ncu",
        "--import", str(report_file),
        "--page", "raw",
        "--csv",
        "--print-kernel-base", "function",
        "--metrics", ",".join(METRICS.values()),
    ]
    res = run_command(cmd)
    if res.returncode != 0:
        raise CommandError(f"ncu import failed, CMD: {' '.join(cmd)},STDOUT:{res.stdout},STDERR:{res.stderr}"
        )
    return parse_ncu_raw_csv(res.stdout)


def profile_binary(binary_path: Path, block_size: int, temp_dir: Path) -> ProfileRow:
    report_base = temp_dir / f"vecadd_bs{block_size}"
    report_file = report_base.with_suffix(".ncu-rep")

    create_report(binary_path, report_base)

    if not report_file.exists():
        raise CommandError(
            f"Expected report was not created for block size {block_size}: {report_file}"
        )

    df = load_metrics_from_report(report_file)
    if df.empty:
        raise CommandError(f"No kernel rows found in the report for block size {block_size}.")

    row = df.iloc[0]
    return ProfileRow(
        block_size=block_size,
        grid_size=numeric(row.get(METRICS["grid_size"])),
        duration_us=numeric(row.get(METRICS["duration_us"])),
        mem_pct=numeric(row.get(METRICS["mem_pct"])),
        dram_pct=numeric(row.get(METRICS["dram_pct"])),
        sm_pct=numeric(row.get(METRICS["sm_pct"])),
        occ_pct=numeric(row.get(METRICS["occ_pct"])),
        l2_hit_pct=numeric(row.get(METRICS["l2_hit_pct"])),
        regs_per_thread=numeric(row.get(METRICS["regs_per_thread"])),
        block_size_reported=numeric(row.get(METRICS["block_size_reported"])),
    )


def collect_profiles(block_sizes: list[int]) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="ncu_tmp_") as tmp:
        temp_dir = Path(tmp)
        source_path = temp_dir / "vecadd.cu"
        write_cuda_source(source_path)

        rows: list[ProfileRow] = []
        for block_size in block_sizes:
            binary_path = temp_dir / f"vecadd_bs{block_size}"
            compile_binary(source_path, binary_path, block_size)
            rows.append(profile_binary(binary_path, block_size, temp_dir))

    return pd.DataFrame([row.__dict__ for row in rows]).sort_values("block_size").reset_index(drop=True)


def apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "#0f1117",
        "axes.facecolor": "#171a21",
        "axes.edgecolor": "#3d4450",
        "axes.labelcolor": "#e6edf3",
        "text.color": "#e6edf3",
        "xtick.color": "#c9d1d9",
        "ytick.color": "#c9d1d9",
        "grid.color": "#2b3240",
    })


def draw_runtime(ax, df: pd.DataFrame) -> None:
    ax.plot(df["block_size"], df["duration_us"], marker="o", linewidth=2.6, color="#58a6ff")
    ax.set_title("Runtime vs block size")
    ax.set_xlabel("Block size")
    ax.set_ylabel("Duration (us)")
    for x, y in zip(df["block_size"], df["duration_us"]):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)


def draw_throughput(ax, df: pd.DataFrame) -> None:
    ax.plot(df["block_size"], df["mem_pct"], marker="o", linewidth=2.2, label="Memory % peak", color="#7ee787")
    ax.plot(df["block_size"], df["dram_pct"], marker="o", linewidth=2.2, label="DRAM % peak", color="#ffa657")
    ax.plot(df["block_size"], df["sm_pct"], marker="o", linewidth=2.2, label="Compute % peak", color="#d2a8ff")
    ax.set_title("Throughput vs block size")
    ax.set_xlabel("Block size")
    ax.set_ylabel("Peak used (%)")
    ax.legend(frameon=True)


def draw_heatmap(ax, fig, df: pd.DataFrame) -> None:
    heat = df[["block_size", "duration_us", "mem_pct", "dram_pct", "sm_pct", "occ_pct", "l2_hit_pct"]].copy()
    heat = heat.set_index("block_size").T

    normalized = heat.copy()
    for metric_name in normalized.index:
        row = normalized.loc[metric_name]
        mn, mx = row.min(), row.max()
        normalized.loc[metric_name] = 0.5 if mx == mn else (row - mn) / (mx - mn)

    image = ax.imshow(normalized.values, aspect="auto", cmap="viridis")
    ax.set_title("Result matrix")
    ax.set_xlabel("Block size")
    ax.set_xticks(range(len(normalized.columns)))
    ax.set_xticklabels([str(c) for c in normalized.columns])
    ax.set_yticks(range(len(normalized.index)))
    ax.set_yticklabels(["duration_us", "mem_pct", "dram_pct", "sm_pct", "occ_pct", "l2_hit_pct"])

    for i, metric in enumerate(heat.index):
        for j, block_size in enumerate(heat.columns):
            ax.text(j, i, f"{heat.loc[metric, block_size]:.1f}", ha="center", va="center", color="white", fontsize=8)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative level")


def show_figures(df: pd.DataFrame) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    fig.suptitle("CUDA vecAdd block-size sweep", fontsize=16, fontweight="bold")

    draw_runtime(axes[0], df)
    draw_throughput(axes[1], df)
    draw_heatmap(axes[2], fig, df)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def print_summary(df: pd.DataFrame) -> None:
    print("Collected metrics:")
    print(df.to_string(index=False))
    best_runtime = df.loc[df["duration_us"].idxmin()]
    print(f"Fastest block size: {int(best_runtime['block_size'])} "
        f"with {best_runtime['duration_us']:.2f} us, "
        f"memory {best_runtime['mem_pct']:.2f}% peak, "
        f"compute {best_runtime['sm_pct']:.2f}% peak."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize CUDA vecAdd behavior across block sizes with Nsight Compute.")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512, 1024],
        help="Block sizes to test. Recommended: 64 128 256 512 1024",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_tool("nvcc")
    require_tool("ncu")

    block_sizes = validate_block_sizes(args.sizes)
    df = collect_profiles(block_sizes)
    print_summary(df)
    show_figures(df)


if __name__ == "__main__":
    main()
