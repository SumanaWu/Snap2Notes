"""对标注照片评估原图 OCR、完整流程或单独的公式转录。"""
import argparse
import json
import math
import os
import re
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import app
from recognition import bounds, formula_model_name, model, payload, plain_ocr, recognize_page


def edits(reference, actual):
    row = list(range(len(actual) + 1))
    for i, expected in enumerate(reference, 1):
        next_row = [i]
        for j, observed in enumerate(actual, 1):
            next_row.append(min(row[j] + 1, next_row[-1] + 1, row[j - 1] + (expected != observed)))
        row = next_row
    return row[-1]


def run_image(path, surface, pipeline):
    if pipeline == "raw":
        with Image.open(path) as photo:
            image = cv2.cvtColor(np.asarray(ImageOps.exif_transpose(photo).convert("RGB")), cv2.COLOR_RGB2BGR)
        return plain_ocr(image)
    blocks, warnings = recognize_page(path, surface)
    if warnings:
        raise RuntimeError(warnings)
    return blocks


def formula_text(text):
    return re.sub(r"\s+", "", text.strip().strip("$"))


def evaluate_formula_crops(manifest):
    """用标注框裁剪公式，单独测量转录错误。"""
    samples = json.loads(manifest.read_text(encoding="utf-8"))
    started = time.perf_counter()
    details = []
    for item in samples:
        formulas = item.get("formulas", [])
        if not formulas:
            continue
        with Image.open(manifest.parent / item["image"]) as photo:
            rgb = ImageOps.exif_transpose(photo).convert("RGB")
            patches = []
            for annotation in formulas:
                left, top, right, bottom = annotation["box"]
                if not (0 <= left < right <= rgb.width and 0 <= top < bottom <= rgb.height):
                    raise ValueError(f"公式框超出原图范围：{item['image']} {annotation['box']}")
                patch = np.asarray(rgb.crop((left, top, right, bottom)))
                if item["surface"] == "blackboard":
                    patch = cv2.bitwise_not(patch)
                patches.append(cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        predictions = model("formula").predict(patches, batch_size=2)
        for annotation, prediction in zip(formulas, predictions, strict=True):
            actual = str(payload(prediction)["rec_formula"])
            details.append({
                "image": item["image"], "box": annotation["box"],
                "reference": annotation["latex"], "prediction": actual,
                "exact": formula_text(actual) == formula_text(annotation["latex"]),
            })
    if not details:
        raise ValueError("公式裁剪评测至少需要一个带框的公式标注")
    device = os.getenv("SNAP_DEVICE", "cpu")
    return {"pipeline": "formula-crops", "device": device,
            "formula_model": formula_model_name(device),
            "formulas": len(details), "exact": sum(row["exact"] for row in details),
            "seconds": round(time.perf_counter() - started, 3), "details": details}


def box_iou(first, second):
    left, top, right, bottom = first
    other_left, other_top, other_right, other_bottom = second
    intersection = max(0, min(right, other_right) - max(left, other_left)) * max(
        0, min(bottom, other_bottom) - max(top, other_top)
    )
    union = (right - left) * (bottom - top) + (other_right - other_left) * (other_bottom - other_top) - intersection
    return intersection / union if union > 0 else 0.0


def match_regions(expected, predicted):
    # 标注与预测各最多匹配一次，避免重复候选抬高召回。
    pairs = sorted(
        ((box_iou(item["box"], bounds(block["polygon"])), i, j)
         for i, item in enumerate(expected) for j, block in enumerate(predicted)),
        reverse=True,
    )
    matched_expected = set()
    matched_predicted = set()
    for iou, i, j in pairs:
        if iou < 0.5:
            break
        if i in matched_expected or j in matched_predicted:
            continue
        matched_expected.add(i)
        matched_predicted.add(j)
        yield expected[i], predicted[j]


def score_formulas(expected, blocks):
    predicted = [block for block in blocks if block["kind"] == "formula"]
    matches = list(match_regions(expected, predicted))
    exact = sum(formula_text(annotation["latex"]) == formula_text(block["text"])
                for annotation, block in matches)
    return len(predicted), len(matches), exact


def evaluate(manifest, pipeline, repeats=1):
    samples = json.loads(manifest.read_text(encoding="utf-8"))
    if not samples or repeats < 1:
        raise ValueError("清单至少需要一张照片，重复次数至少为 1")
    totals = {"chars": 0, "char_errors": 0, "words": 0, "word_errors": 0}
    first = samples[0]
    started = time.perf_counter()
    run_image(manifest.parent / first["image"], first["surface"], pipeline)
    cold_seconds = time.perf_counter() - started
    # 部分模型只有在对应内容出现时才加载，先遍历一轮覆盖全部样例类型。
    for item in samples[1:]:
        run_image(manifest.parent / item["image"], item["surface"], pipeline)
    times = []
    expected_formulas = predicted_formulas = matched_formulas = exact_formulas = 0
    formula_pages = 0
    figure_pages = expected_figures = predicted_figures = matched_figures = 0
    for trial in range(repeats):
        for item in samples:
            started = time.perf_counter()
            blocks = run_image(manifest.parent / item["image"], item["surface"], pipeline)
            times.append(time.perf_counter() - started)
            if trial:
                continue
            if "formulas" in item:
                formula_pages += 1
                expected_formulas += len(item["formulas"])
                predicted, matched, exact = score_formulas(item["formulas"], blocks)
                predicted_formulas += predicted
                matched_formulas += matched
                exact_formulas += exact
            if "figures" in item:
                figure_pages += 1
                expected_figures += len(item["figures"])
                figures = [block for block in blocks if block["kind"] == "figure"]
                predicted_figures += len(figures)
                matched_figures += sum(1 for _ in match_regions(item["figures"], figures))
            if "text" not in item:
                continue
            reference = " ".join(item["text"].split())
            actual = " ".join(" ".join(b["text"] for b in blocks if b["kind"] in {"text", "mixed"}).split())
            totals["chars"] += len(reference)
            totals["char_errors"] += edits(reference, actual)
            totals["words"] += len(reference.split())
            totals["word_errors"] += edits(reference.split(), actual.split())
    warm = sorted(times)
    return {
        "pipeline": pipeline,
        "device": os.getenv("SNAP_DEVICE", "cpu"),
        "images": len(samples),
        "timed_runs": len(times),
        "cer": totals["char_errors"] / totals["chars"] if totals["chars"] else None,
        "wer": totals["word_errors"] / totals["words"] if totals["words"] else None,
        "formula_pages": formula_pages,
        "formula_region_precision": matched_formulas / predicted_formulas if predicted_formulas else None,
        "formula_region_recall": matched_formulas / expected_formulas if expected_formulas else None,
        "formula_exact_recall": exact_formulas / expected_formulas if expected_formulas else None,
        "figure_pages": figure_pages,
        "figure_region_precision": matched_figures / predicted_figures if predicted_figures else None,
        "figure_region_recall": matched_figures / expected_figures if expected_figures else None,
        "first_seconds": round(cold_seconds, 3),
        "warm_p50_seconds": round(statistics.median(warm), 3),
        "warm_p95_seconds": round(warm[math.ceil(0.95 * len(warm)) - 1], 3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--pipeline", choices=("raw", "full", "formula-crops"), required=True)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.pipeline == "formula-crops":
        if args.repeats != 1:
            parser.error("公式裁剪模式每个标注框运行一次")
        report = evaluate_formula_crops(args.manifest)
    else:
        report = evaluate(args.manifest, args.pipeline, args.repeats)
    print(json.dumps(report, indent=2, ensure_ascii=False))
