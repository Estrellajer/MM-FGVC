#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


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
    parser.add_argument("--extra-panel", action="append", default=[], help="Additional panel image paths to append.")
    parser.add_argument("--panel-tag", action="append", default=[], help="Short tag like A, B, C, D for each block.")
    parser.add_argument("--panel-note", action="append", default=[], help="Short note shown under each block.")
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


def load_footer_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrapped_text(text: str, *, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def add_panel_footer(
    image: Image.Image,
    *,
    tag: str | None,
    note: str | None,
    horizontal_padding: int = 18,
    vertical_padding: int = 30,
) -> Image.Image:
    tag = (tag or "").strip()
    note = (note or "").strip()
    if not tag and not note:
        return image

    base = image.convert("RGB")
    font = load_footer_font(size=66)
    measure = ImageDraw.Draw(base)
    max_width = max(base.width - 2 * horizontal_padding, 40)

    lines: list[str] = []
    if tag and note:
        lines = wrapped_text(f"{tag}. {note}", draw=measure, font=font, max_width=max_width)
    elif tag:
        lines = [f"{tag}."]
    else:
        lines = wrapped_text(note, draw=measure, font=font, max_width=max_width)

    line_heights = []
    for line in lines:
        bbox = measure.textbbox((0, 0), line, font=font)
        line_heights.append(max(bbox[3] - bbox[1], 12))

    footer_height = vertical_padding * 2 + sum(line_heights) + 14 * max(len(lines) - 1, 0)
    canvas = Image.new("RGB", (base.width, base.height + footer_height), color="white")
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = base.height + vertical_padding
    for line, line_height in zip(lines, line_heights):
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        x = max((base.width - line_width) // 2, horizontal_padding)
        draw.text((x, y), line, fill="black", font=font)
        y += line_height + 14
    return canvas


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
    if len(images) == 4:
        cell_width = max(image.width for image in images)
        resized = [resize_to_width(image, cell_width) for image in images]
        top_height = max(resized[0].height, resized[1].height)
        bottom_height = max(resized[2].height, resized[3].height)
        content_width = cell_width * 2 + gap
        total_height = padding * 2 + top_height + gap + bottom_height
        canvas = Image.new("RGB", (content_width + padding * 2, total_height), color="white")

        positions = [
            (padding, padding + (top_height - resized[0].height) // 2),
            (padding + cell_width + gap, padding + (top_height - resized[1].height) // 2),
            (padding, padding + top_height + gap + (bottom_height - resized[2].height) // 2),
            (padding + cell_width + gap, padding + top_height + gap + (bottom_height - resized[3].height) // 2),
        ]
        for image, (x, y) in zip(resized, positions):
            canvas.paste(image, (x, y))
        return canvas

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
    order = list(manifest.get("panel_order") or DEFAULT_ORDER)
    ordered_paths = [Path(panels[name]).expanduser().resolve() for name in order if name in panels]
    ordered_paths.extend(Path(path).expanduser().resolve() for path in args.extra_panel)
    raw_images = [Image.open(path).convert("RGB") for path in ordered_paths]
    tags = list(args.panel_tag)
    notes = list(args.panel_note)
    images = []
    for idx, image in enumerate(raw_images):
        tag = tags[idx] if idx < len(tags) else None
        note = notes[idx] if idx < len(notes) else None
        images.append(add_panel_footer(image, tag=tag, note=note))
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
        "panel_order": [str(path) for path in ordered_paths],
        "layout": args.layout,
        "stitched_image": str(stitched_path),
    }
    save_json(output_root / "stitched_method_panels_manifest.json", out_manifest)
    print(json.dumps(out_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
