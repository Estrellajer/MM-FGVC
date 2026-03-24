import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Callable
import zipfile


Record = dict
SplitRecords = dict[str, list[Record]]


def write_json(path: Path, records: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def limit_reached(records: list[Record], max_items: int | None) -> bool:
    return max_items is not None and len(records) >= max_items


def trim(records: list[Record], n: int) -> list[Record]:
    return records[: min(len(records), n)]


def extract_zip_with_normalized_paths(zip_path: Path, output_root: Path) -> None:
    if not zip_path.exists():
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if not name or name.endswith("/"):
                continue
            target = output_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(target, "wb") as dst:
                dst.write(src.read())


def normalize_blink_answer(ans: str) -> str:
    text = normalize_spaces(str(ans)).upper()
    if text in {"A", "(A)"}:
        return "(A)"
    if text in {"B", "(B)"}:
        return "(B)"
    if text in {"C", "(C)"}:
        return "(C)"
    if text in {"D", "(D)"}:
        return "(D)"
    return text


def build_blink_question(question: str, choices: list[str], n_images: int) -> str:
    image_prefix = "\n".join(["<image>"] * n_images)
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    choice_text = " ".join([f"({letters[i]}) {normalize_spaces(str(choices[i]))}" for i in range(len(choices))])
    suffix = f"Select from the following choices. {choice_text}"
    return f"{image_prefix}\n{normalize_spaces(question)} {suffix}".strip()


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_flower_label(label: str) -> str:
    label = normalize_spaces(str(label))
    label = re.sub(r"\s+flower$", "", label, flags=re.IGNORECASE)
    return normalize_spaces(label)


def build_mcq_question(
    prompt: str,
    correct_label: str,
    label_pool: list[str],
    num_choices: int,
    rng: random.Random,
) -> tuple[str, str]:
    unique_pool = sorted(set(label_pool))
    distractors_pool = [x for x in unique_pool if x != correct_label]
    if len(distractors_pool) < num_choices - 1:
        raise ValueError(
            f"Not enough distractors. Need {num_choices - 1}, got {len(distractors_pool)}"
        )

    distractors = rng.sample(distractors_pool, num_choices - 1)
    options = distractors + [correct_label]
    rng.shuffle(options)

    letters = [chr(ord("A") + i) for i in range(num_choices)]
    answer = letters[options.index(correct_label)]
    option_text = " ".join([f"{letters[i]}. {options[i]}" for i in range(num_choices)])
    question = f"{prompt}\n{option_text}"
    return question, answer


def apply_mcq(
    records: list[Record],
    prompt: str,
    num_choices: int,
    seed: int,
    label_pool: list[str] | None = None,
) -> list[Record]:
    if label_pool is None:
        label_pool = [str(item["label"]) for item in records]
    rng = random.Random(seed)
    out: list[Record] = []
    for item in records:
        question, answer = build_mcq_question(
            prompt=prompt,
            correct_label=str(item["label"]),
            label_pool=label_pool,
            num_choices=num_choices,
            rng=rng,
        )
        cur = dict(item)
        cur["question"] = question
        cur["answer"] = answer
        out.append(cur)
    return out


def convert_naturalbench(data_root: Path, max_items: int | None) -> tuple[str, SplitRecords]:
    src = data_root / "NaturalBench" / "train_set.jsonl"
    nb_root = data_root / "NaturalBench"
    out: SplitRecords = {"train": []}

    def resolve_image(rel_path: str) -> Path:
        candidates = [
            nb_root / rel_path,
            nb_root / "NaturalBench-Retrieval" / rel_path,
            nb_root / "NaturalBench-Retrieval" / "images" / Path(rel_path).name,
            nb_root / Path(rel_path).name,
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return candidates[0].resolve()

    with open(src, "r", encoding="utf-8") as f:
        qid = 0
        for line in f:
            item = json.loads(line)
            messages = item.get("messages", [])
            if len(messages) < 3:
                continue
            question = str(messages[0].get("content", "")).strip()
            image_block = messages[1].get("content", [])
            if not image_block:
                continue
            img_rel = image_block[0].get("image_path", "")
            label = str(messages[2].get("content", "")).strip()
            out["train"].append(
                {
                    "image": str(resolve_image(img_rel)),
                    "question": question,
                    "question_id": qid,
                    "label": label,
                }
            )
            qid += 1
            if limit_reached(out["train"], max_items):
                break

    return "jsonl", out


def convert_vizwiz(data_root: Path, max_items: int | None) -> tuple[str, SplitRecords]:
    root = data_root / "VizWIz"
    out: SplitRecords = {"train": [], "val": []}

    for split in ["train", "val"]:
        ann_path = root / f"{split}.json"
        img_dir = root / split
        with open(ann_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        for idx, item in enumerate(items):
            label = "answerable" if int(item.get("answerable", 1)) != 0 else "unanswerable"

            image_name = item.get("image", "")
            out[split].append(
                {
                    "image": str((img_dir / image_name).resolve()),
                    "question": str(item.get("question", "")).strip(),
                    "question_id": idx,
                    "label": label,
                }
            )
            if limit_reached(out[split], max_items):
                break

    return "jsonl", out


def convert_eurosat(data_root: Path, max_items: int | None) -> tuple[str, SplitRecords]:
    root = data_root / "Eurosat"
    img_root = root / "2750"
    split_path = root / "split_zhou_EuroSAT.json"
    out: SplitRecords = {}

    with open(split_path, "r", encoding="utf-8") as f:
        split_obj = json.load(f)

    global_labels = []
    for split_name in ["train", "val", "test"]:
        if split_name in split_obj:
            global_labels.extend([normalize_spaces(str(row[2])) for row in split_obj[split_name]])

    for split in ["train", "val", "test"]:
        if split not in split_obj:
            continue
        out[split] = []
        for idx, row in enumerate(split_obj[split]):
            rel_path, class_id, class_name = row
            out[split].append(
                {
                    "image": str((img_root / rel_path).resolve()),
                    "question_id": idx,
                    "class_id": int(class_id),
                    "label": normalize_spaces(str(class_name)),
                }
            )
            if limit_reached(out[split], max_items):
                break

    for split in list(out.keys()):
        out[split] = apply_mcq(
            records=out[split],
            prompt="What type of remote sensing image does the given image belong to?",
            num_choices=4,
            seed=13,
            label_pool=global_labels,
        )

    return "json", out


def convert_pets(data_root: Path, max_items: int | None) -> tuple[str, SplitRecords]:
    root = data_root / "Pets"
    img_root = root / "images"
    ann_root = root / "annotations"
    out: SplitRecords = {"train": [], "test": []}

    def parse_line(line: str) -> tuple[str, int]:
        parts = line.strip().split()
        image_id = parts[0]
        class_id = int(parts[1])
        return image_id, class_id

    def label_from_image_id(image_id: str) -> str:
        return normalize_spaces(re.sub(r"_[0-9]+$", "", image_id).replace("_", " "))

    global_labels = []
    for file_name in ["trainval.txt", "test.txt"]:
        with open(ann_root / file_name, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image_id = line.split()[0]
                global_labels.append(label_from_image_id(image_id))

    for split, file_name in [("train", "trainval.txt"), ("test", "test.txt")]:
        with open(ann_root / file_name, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                image_id, class_id = parse_line(line)
                out[split].append(
                    {
                        "image": str((img_root / f"{image_id}.jpg").resolve()),
                        "question_id": idx,
                        "class_id": class_id,
                        "label": label_from_image_id(image_id),
                    }
                )
                if limit_reached(out[split], max_items):
                    break

    for split in list(out.keys()):
        out[split] = apply_mcq(
            records=out[split],
            prompt="What type of animal is in the image?",
            num_choices=4,
            seed=17,
            label_pool=global_labels,
        )

    return "json", out


def convert_cub(data_root: Path, max_items: int | None) -> tuple[str, SplitRecords]:
    root = data_root / "Cub" / "cub200"
    out: SplitRecords = {"train": [], "test": []}
    global_labels = []
    for split in ["train", "test"]:
        split_root = root / split
        global_labels.extend(
            [normalize_spaces(p.name.replace("_", " ")) for p in sorted(split_root.iterdir()) if p.is_dir()]
        )

    for split in ["train", "test"]:
        split_root = root / split
        idx = 0
        for class_dir in sorted([p for p in split_root.iterdir() if p.is_dir()]):
            label = normalize_spaces(class_dir.name.replace("_", " "))
            for img_path in sorted(class_dir.glob("*.jpg")):
                out[split].append(
                    {
                        "image": str(img_path.resolve()),
                        "question_id": idx,
                        "label": label,
                    }
                )
                idx += 1
                if limit_reached(out[split], max_items):
                    break
            if limit_reached(out[split], max_items):
                break

    for split in list(out.keys()):
        out[split] = apply_mcq(
            records=out[split],
            prompt="What bird species is shown in the image?",
            num_choices=4,
            seed=23,
            label_pool=global_labels,
        )

    return "json", out


def convert_tinyimage(data_root: Path, max_items: int | None) -> tuple[str, SplitRecords]:
    root = data_root / "TinyImage" / "tiny-imagenet-200"
    out: SplitRecords = {"train": [], "val": [], "test": []}

    wnids = [line.strip() for line in (root / "wnids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    wnid_set = set(wnids)

    wnid_to_name: dict[str, str] = {}
    with open(root / "words.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            wnid, names = line.split("\t", 1)
            if wnid in wnid_set and wnid not in wnid_to_name:
                wnid_to_name[wnid] = names.split(",")[0].strip()

    global_labels = [normalize_spaces(wnid_to_name.get(wnid, wnid).replace("_", " ")) for wnid in wnids]

    idx = 0
    for wnid in wnids:
        img_dir = root / "train" / wnid / "images"
        label = normalize_spaces(wnid_to_name.get(wnid, wnid).replace("_", " "))
        for img_path in sorted(img_dir.glob("*.JPEG")):
            out["train"].append(
                {
                    "image": str(img_path.resolve()),
                    "question_id": idx,
                    "wnid": wnid,
                    "label": label,
                }
            )
            idx += 1
            if limit_reached(out["train"], max_items):
                break
        if limit_reached(out["train"], max_items):
            break

    val_ann = root / "val" / "val_annotations.txt"
    idx = 0
    with open(val_ann, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            img_name, wnid = parts[0], parts[1]
            label = normalize_spaces(wnid_to_name.get(wnid, wnid).replace("_", " "))
            out["val"].append(
                {
                    "image": str((root / "val" / "images" / img_name).resolve()),
                    "question_id": idx,
                    "wnid": wnid,
                    "label": label,
                }
            )
            idx += 1
            if limit_reached(out["val"], max_items):
                break

    idx = 0
    for img_path in sorted((root / "test" / "images").glob("*.JPEG")):
        out["test"].append(
            {
                "image": str(img_path.resolve()),
                "question": "What object category is shown in the image?",
                "question_id": idx,
                "label": "unknown",
            }
        )
        idx += 1
        if limit_reached(out["test"], max_items):
            break

    out["train"] = apply_mcq(
        records=out["train"],
        prompt="What object category is shown in the image?",
        num_choices=16,
        seed=31,
        label_pool=global_labels,
    )
    out["val"] = apply_mcq(
        records=out["val"],
        prompt="What object category is shown in the image?",
        num_choices=16,
        seed=37,
        label_pool=global_labels,
    )

    return "json", out


def convert_coco2017(data_root: Path, max_items: int | None) -> tuple[str, SplitRecords]:
    root = data_root / "Coco2017" / "val2017"
    out: SplitRecords = {"val": []}

    idx = 0
    for img_path in sorted(root.glob("*.jpg")):
        out["val"].append(
            {
                "image": str(img_path.resolve()),
                "question": "Describe the image briefly.",
                "question_id": idx,
                "label": "unknown",
            }
        )
        idx += 1
        if limit_reached(out["val"], max_items):
            break

    return "jsonl", out


def convert_flowers(data_root: Path, max_items: int | None) -> tuple[str, SplitRecords]:
    root = data_root / "Flowers"
    img_root = root / "jpg"
    split_path = root / "split_zhou_OxfordFlowers.json"
    out: SplitRecords = {}

    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            split_obj = json.load(f)

        global_labels = []
        for split_name in ["train", "val", "test"]:
            if split_name in split_obj:
                global_labels.extend([normalize_flower_label(str(row[2])) for row in split_obj[split_name]])

        for split in ["train", "val", "test"]:
            if split not in split_obj:
                continue
            out[split] = []
            for idx, row in enumerate(split_obj[split]):
                image_name, class_id, class_name = row
                out[split].append(
                    {
                        "image": str((img_root / image_name).resolve()),
                        "question_id": idx,
                        "class_id": int(class_id),
                        "label": normalize_flower_label(str(class_name)),
                    }
                )
                if limit_reached(out[split], max_items):
                    break

        for split in list(out.keys()):
            out[split] = apply_mcq(
                records=out[split],
                prompt="What category does this flower image belong to?",
                num_choices=4,
                seed=41,
                label_pool=global_labels,
            )
        return "json", out

    cat_path = root / "cat_to_name.json"
    mat_path = root / "imagelabels.mat"
    out = {"all": []}

    cat_to_name: dict[str, str] = {}
    if cat_path.exists():
        with open(cat_path, "r", encoding="utf-8") as f:
            cat_to_name = json.load(f)

    labels: list[int] = []
    if mat_path.exists():
        try:
            from scipy.io import loadmat

            mat = loadmat(mat_path)
            labels = [int(x) for x in mat["labels"].squeeze().tolist()]
        except Exception:
            labels = []

    idx = 0
    for img_path in sorted(img_root.glob("*.jpg")):
        stem = img_path.stem
        match = re.search(r"(\d+)$", stem)
        image_index = int(match.group(1)) if match else idx + 1
        label_id = labels[image_index - 1] if 0 < image_index <= len(labels) else -1
        label_name = cat_to_name.get(str(label_id), "unknown")
        out["all"].append(
            {
                "image": str(img_path.resolve()),
                "question_id": idx,
                "class_id": label_id,
                "label": normalize_flower_label(label_name),
            }
        )
        idx += 1
        if limit_reached(out["all"], max_items):
            break

    out["all"] = apply_mcq(
        records=out["all"],
        prompt="What category does this flower image belong to?",
        num_choices=4,
        seed=43,
        label_pool=[str(item["label"]) for item in out["all"]],
    )

    return "json", out


def convert_sugarcrepe(data_root: Path, max_items: int | None) -> tuple[str, SplitRecords]:
    root = data_root / "Sugarcrepe"
    coco_root = data_root / "Coco2017" / "val2017"
    out: SplitRecords = {"sugarcrepe_all": []}

    for split_file in sorted(root.glob("*.json")):
        split_name = split_file.stem
        out[split_name] = []

        with open(split_file, "r", encoding="utf-8") as f:
            items = json.load(f)

        for question_id, item in items.items():
            image_path = (coco_root / item["filename"]).resolve()
            pos_caption = str(item.get("caption", "")).strip()
            neg_caption = str(item.get("negative_caption", "")).strip()

            pos_record = {
                "image": str(image_path),
                "question": f"Does this image show '{pos_caption}'?",
                "split": split_name,
                "question_id": str(question_id),
                "label": "Yes",
            }
            neg_record = {
                "image": str(image_path),
                "question": f"Does this image show '{neg_caption}'?",
                "split": split_name,
                "question_id": str(question_id),
                "label": "No",
            }

            out[split_name].append(pos_record)
            out["sugarcrepe_all"].append(pos_record)
            if limit_reached(out[split_name], max_items):
                break

            out[split_name].append(neg_record)
            out["sugarcrepe_all"].append(neg_record)
            if limit_reached(out[split_name], max_items):
                break

    return "jsonl", out


def convert_mhalubench(data_root: Path, max_items: int | None) -> tuple[str, SplitRecords]:
    root = data_root / "MHaluBench"
    image_to_text_dir = root / "data" / "image-to-text"
    text_to_image_dir = root / "data" / "text-to-image"
    test_text_to_image_dir = root / "test-data" / "text-to-image"

    out: SplitRecords = {"train": [], "val_v01": [], "val_v02": []}
    skipped_counts = {split: 0 for split in out}

    def resolve_image_path(original: str) -> Path:
        file_name = Path(original).name
        for candidate in [
            image_to_text_dir / file_name,
            text_to_image_dir / file_name,
            test_text_to_image_dir / file_name,
        ]:
            if candidate.exists():
                return candidate.resolve()
        return Path(original)

    train_items = json.loads((root / "MHaluBench_train.json").read_text(encoding="utf-8"))
    qid = 0
    for item in train_items:
        image_path = resolve_image_path(str(item.get("image_path", "")))
        if not image_path.exists():
            skipped_counts["train"] += len(list(zip(item.get("claim_list", []), item.get("ref_claim_label", []))))
            continue
        claims = item.get("claim_list", [])
        labels = item.get("ref_claim_label", [])
        for claim, label in zip(claims, labels):
            out["train"].append(
                {
                    "image_path": str(image_path),
                    "claim": normalize_spaces(str(claim)),
                    "question_id": qid,
                    "label": normalize_spaces(str(label)),
                }
            )
            qid += 1
            if limit_reached(out["train"], max_items):
                break
        if limit_reached(out["train"], max_items):
            break

    for split_name, file_name in [("val_v01", "MHaluBench_val-v0.1.json"), ("val_v02", "MHaluBench_val-v0.2.json")]:
        val_items = json.loads((root / file_name).read_text(encoding="utf-8"))
        for idx, item in enumerate(val_items):
            image_path = resolve_image_path(str(item.get("image_path", "")))
            if not image_path.exists():
                skipped_counts[split_name] += 1
                continue
            out[split_name].append(
                {
                    "image_path": str(image_path),
                    "claim": normalize_spaces(str(item.get("response", ""))),
                    "question_id": idx,
                    "label": normalize_spaces(str(item.get("label", ""))),
                }
            )
            if limit_reached(out[split_name], max_items):
                break

    skipped_info = ", ".join([f"{split}:{count}" for split, count in skipped_counts.items() if count > 0])
    if skipped_info:
        print(f"[mhalubench] skipped samples with missing images -> {skipped_info}")

    return "json", out


def convert_vlguard(data_root: Path, max_items: int | None) -> tuple[str, SplitRecords]:
    root = data_root / "VLGuard"
    if not (root / "VLGuard" / "train.json").exists() and (root / "VLGuard.zip").exists():
        extract_zip_with_normalized_paths(root / "VLGuard.zip", root)

    vl_root = root / "VLGuard"
    if not (vl_root / "train").exists() and (vl_root / "train.zip").exists():
        extract_zip_with_normalized_paths(vl_root / "train.zip", vl_root)
    if not (vl_root / "test").exists() and (vl_root / "test.zip").exists():
        extract_zip_with_normalized_paths(vl_root / "test.zip", vl_root)

    out: SplitRecords = {"train": [], "test": []}

    def convert_split(split: str, records: list[dict]) -> list[Record]:
        converted: list[Record] = []
        split_root = vl_root / split
        qid = 0
        for item in records:
            image_rel = str(item.get("image", ""))
            image_path = str((split_root / image_rel).resolve())
            safe_flag = bool(item.get("safe", False))
            instr_resp = item.get("instr-resp", [])

            if safe_flag:
                for pair in instr_resp:
                    safe_instruction = normalize_spaces(str(pair.get("safe_instruction", "")))
                    unsafe_instruction = normalize_spaces(str(pair.get("unsafe_instruction", "")))
                    if safe_instruction:
                        converted.append(
                            {
                                "image": image_path,
                                "instruction": safe_instruction,
                                "question_id": qid,
                                "label": "unharmful",
                            }
                        )
                        qid += 1
                        if limit_reached(converted, max_items):
                            return converted
                    if unsafe_instruction:
                        converted.append(
                            {
                                "image": image_path,
                                "instruction": unsafe_instruction,
                                "question_id": qid,
                                "label": "harmful",
                            }
                        )
                        qid += 1
                        if limit_reached(converted, max_items):
                            return converted
            else:
                for pair in instr_resp:
                    instruction = normalize_spaces(str(pair.get("instruction", "")))
                    if instruction:
                        converted.append(
                            {
                                "image": image_path,
                                "instruction": instruction,
                                "question_id": qid,
                                "label": "harmful",
                            }
                        )
                        qid += 1
                        if limit_reached(converted, max_items):
                            return converted
        return converted

    train_records = json.loads((vl_root / "train.json").read_text(encoding="utf-8"))
    test_records = json.loads((vl_root / "test.json").read_text(encoding="utf-8"))
    out["train"] = convert_split("train", train_records)
    out["test"] = convert_split("test", test_records)

    return "json", out


def convert_blink(data_root: Path, max_items: int | None) -> tuple[str, SplitRecords]:
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "BLINK conversion requires 'datasets' + parquet dependencies. "
            "Please run with 'uv run python scripts/convert_data_to_json.py ...'."
        ) from e

    root = data_root / "BLINK"
    image_export_root = root / "converted_images"
    out: SplitRecords = {}

    task_dirs = sorted(
        [
            p
            for p in root.iterdir()
            if p.is_dir() and (p / "val-00000-of-00001.parquet").exists() and (p / "test-00000-of-00001.parquet").exists()
        ],
        key=lambda p: p.name,
    )

    for task_dir in task_dirs:
        task_name = task_dir.name
        for split in ["val", "test"]:
            split_key = f"{task_name.lower()}_{split}"
            out[split_key] = []

            parquet_path = task_dir / f"{split}-00000-of-00001.parquet"
            dataset = load_dataset("parquet", data_files={split: str(parquet_path)})[split]

            export_dir = image_export_root / task_name / split
            export_dir.mkdir(parents=True, exist_ok=True)

            for idx, item in enumerate(dataset):
                image_paths: list[str | None] = []
                for image_idx in [1, 2, 3, 4]:
                    key = f"image_{image_idx}"
                    img = item.get(key)
                    if img is None:
                        image_paths.append(None)
                        continue
                    file_name = f"{item['idx']}_{image_idx}.png"
                    save_path = export_dir / file_name
                    if not save_path.exists():
                        img.save(save_path)
                    image_paths.append(str(save_path.resolve()))

                n_images = sum(1 for p in image_paths if p is not None)
                question = build_blink_question(
                    question=str(item.get("question", "")),
                    choices=[str(c) for c in item.get("choices", [])],
                    n_images=n_images,
                )

                out[split_key].append(
                    {
                        "question": question,
                        "label": normalize_blink_answer(str(item.get("answer", ""))),
                        "question_id": idx,
                        "sub_task": normalize_spaces(str(item.get("sub_task", ""))),
                        "image_1": image_paths[0],
                        "image_2": image_paths[1],
                        "image_3": image_paths[2],
                        "image_4": image_paths[3],
                    }
                )

                if limit_reached(out[split_key], max_items):
                    break

    return "json", out


CONVERTERS: dict[str, Callable[[Path, int | None], tuple[str, SplitRecords]]] = {
    "naturalbench": convert_naturalbench,
    "vizwiz": convert_vizwiz,
    "eurosat": convert_eurosat,
    "pets": convert_pets,
    "cub": convert_cub,
    "tinyimage": convert_tinyimage,
    "sugarcrepe": convert_sugarcrepe,
    "mhalubench": convert_mhalubench,
    "vlguard": convert_vlguard,
    "blink": convert_blink,
    "flowers": convert_flowers,
    "coco2017": convert_coco2017,
}

DEFAULT_ALL_DATASETS = [
    "naturalbench",
    "vizwiz",
    "eurosat",
    "pets",
    "cub",
    "tinyimage",
    "flowers",
    "sugarcrepe",
    "mhalubench",
    "vlguard",
    "blink",
]


def write_outputs(
    dataset_name: str,
    fmt: str,
    split_records: SplitRecords,
    output_root: Path,
    sample_root: Path,
    sample_size: int,
) -> None:
    for split, records in split_records.items():
        ext = "jsonl" if fmt == "jsonl" else "json"
        out_file = output_root / dataset_name / f"{split}.{ext}"
        sample_file = sample_root / dataset_name / f"{split}.sample.{ext}"

        if fmt == "jsonl":
            write_jsonl(out_file, records)
            write_jsonl(sample_file, trim(records, sample_size))
        else:
            write_json(out_file, records)
            write_json(sample_file, trim(records, sample_size))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert datasets under Data/ to unified json/jsonl format.")
    parser.add_argument("--data-root", type=str, default="Data")
    parser.add_argument("--output-root", type=str, default="dataset/converted_from_data")
    parser.add_argument("--sample-root", type=str, default="dataset/converted_samples")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--max-items-per-split", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    sample_root = Path(args.sample_root).resolve()
    max_items = None if args.max_items_per_split <= 0 else args.max_items_per_split

    if "all" in [d.lower() for d in args.datasets]:
        datasets = DEFAULT_ALL_DATASETS
    else:
        datasets = [d.lower() for d in args.datasets]

    for name in datasets:
        if name not in CONVERTERS:
            raise ValueError(f"Unsupported dataset: {name}. Supported: {list(CONVERTERS.keys())}")

    for name in datasets:
        fmt, split_records = CONVERTERS[name](data_root, max_items)
        write_outputs(
            dataset_name=name,
            fmt=fmt,
            split_records=split_records,
            output_root=output_root,
            sample_root=sample_root,
            sample_size=args.sample_size,
        )
        split_info = ", ".join([f"{k}:{len(v)}" for k, v in split_records.items()])
        print(f"[OK] {name} -> format={fmt} | {split_info}")

    print(f"[DONE] full outputs: {output_root}")
    print(f"[DONE] sample outputs: {sample_root}")


if __name__ == "__main__":
    main()
