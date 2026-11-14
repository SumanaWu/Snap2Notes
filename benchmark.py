"""对标注照片比较原图 OCR 与完整流程，输出 CER、WER 和耗时。"""
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
from recognition import bounds, plain_ocr, recognize_page


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


def box_iou(first, second):
    left, top, right, bottom = first
    other_left, other_top, other_right, other_bottom = second
    intersection = max(0, min(right, other_right) - max(left, other_left)) * max(
        0, min(bottom, other_bottom) - max(top, other_top)
    )
    union = (right - left) * (bottom - top) + (other_right - other_left) * (other_bottom - other_top) - intersection
    return intersection / union if union > 0 else 0.0


def score_formulas(expected, blocks):
    # 标注与预测各最多匹配一次，避免重复候选抬高召回。
    predicted = [block for block in blocks if block["kind"] == "formula"]
    pairs = sorted(
        ((box_iou(item["box"], bounds(block["polygon"])), i, j)
         for i, item in enumerate(expected) for j, block in enumerate(predicted)),
        reverse=True,
    )
    matched_expected = set()
    matched_predicted = set()
    exact = 0
    for iou, i, j in pairs:
        if iou < 0.5:
            break
        if i in matched_expected or j in matched_predicted:
            continue
        matched_expected.add(i)
        matched_predicted.add(j)
        if re.sub(r"\s+", "", expected[i]["latex"]) == re.sub(r"\s+", "", predicted[j]["text"]):
            exact += 1
    return len(predicted), len(matched_expected), exact


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
        "first_seconds": round(cold_seconds, 3),
        "warm_p50_seconds": round(statistics.median(warm), 3),
        "warm_p95_seconds": round(warm[math.ceil(0.95 * len(warm)) - 1], 3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--pipeline", choices=("raw", "full"), required=True)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.manifest, args.pipeline, args.repeats), indent=2))
