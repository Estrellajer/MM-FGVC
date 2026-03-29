#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORDER = ["stv", "i2cl", "mimic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch method-specific failure panels into a single image.")
    parser.add_argument("--panel-manifest", required=True)
    parser.add_argument("--timestamp", default=datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--suite-name", default="write_failure_method_panels_stitched")
    parser.add_argument("--layout", choices=["vertical", "grid"], default="grid")
    parser.add_argument("--padding", type=int, default=36)
    parser.add_argument("--gap", type=int, default=28)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = max(int(round(image.height * width / image.width)), 1)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def stitch_vertical(images: list[Image.Image], *, padding: int, gap: int) -> Image.Image:
    target_width = max(image.width for image in images)
    resized = [resize_to_width(image, target_width) for image in images]
    total_height = padding * 2 + sum(image.height for image in resized) + gap * max(len(resized) - 1, 0)
    canvas = Image.new("RGB", (target_width + padding * 2, total_height), color="white")

    y = padding
    for image in resized:
        x = padding + (target_width - image.width) // 2
        canvas.paste(image, (x, y))
        y += image.height + gap
    return canvas


def stitch_grid(images: list[Image.Image], *, padding: int, gap: int) -> Image.Image:
    if len(images) != 3:
        return stitch_vertical(images, padding=padding, gap=gap)

    bottom_width = max(image.width for image in images)
    top_width = max((bottom_width - gap) // 2, 1)
    top_left = resize_to_width(images[0], top_width)
    top_right = resize_to_width(images[1], top_width)
    bottom = resize_to_width(images[2], bottom_width)

    top_row_width = top_left.width + gap + top_right.width
    content_width = max(top_row_width, bottom.width)
    top_row_height = max(top_left.height, top_right.height)
    total_height = padding * 2 + top_row_height + gap + bottom.height
    canvas = Image.new("RGB", (content_width + padding * 2, total_height), color="white")

    top_x = padding + (content_width - top_row_width) // 2
    bottom_x = padding + (content_width - bottom.width) // 2
    top_y = padding
    bottom_y = padding + top_row_height + gap

    canvas.paste(top_left, (top_x, top_y + (top_row_height - top_left.height) // 2))
    canvas.paste(top_right, (top_x + top_left.width + gap, top_y + (top_row_height - top_right.height) // 2))
    canvas.paste(bottom, (bottom_x, bottom_y))
    return canvas


def main() -> None:
    args = parse_args()
    panel_manifest_path = Path(args.panel_manifest).expanduser().resolve()
    manifest = load_json(panel_manifest_path)
    panels = manifest["panels"]
    ordered_paths = [Path(panels[name]).expanduser().resolve() for name in DEFAULT_ORDER if name in panels]
    images = [Image.open(path).convert("RGB") for path in ordered_paths]
    if not images:
        raise ValueError("No panel images found in manifest.")

    if args.layout == "vertical":
        stitched = stitch_vertical(images, padding=args.padding, gap=args.gap)
    elif args.layout == "grid":
        stitched = stitch_grid(images, padding=args.padding, gap=args.gap)
    else:
        raise ValueError(f"Unsupported layout: {args.layout}")

    figure_root = PROJECT_ROOT / "swap" / "paper" / "figures"
    output_root = PROJECT_ROOT / "swap" / "paper" / "outputs" / f"{args.timestamp}_{args.suite_name}"
    output_root.mkdir(parents=True, exist_ok=True)
    stitched_path = figure_root / f"{args.suite_name}_{args.timestamp}.png"
    stitched.save(stitched_path)

    out_manifest = {
        "panel_manifest": str(panel_manifest_path),
        "panel_order": DEFAULT_ORDER,
        "layout": args.layout,
        "stitched_image": str(stitched_path),
    }
    save_json(output_root / "stitched_method_panels_manifest.json", out_manifest)
    print(json.dumps(out_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
