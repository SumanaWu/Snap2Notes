"""对标注照片比较原图 OCR 与完整流程，输出 CER、WER 和耗时。"""
import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import app
from recognition import plain_ocr, recognize_page


def edits(reference, actual):
    row = list(range(len(actual) + 1))
    for i, expected in enumerate(reference, 1):
        next_row = [i]
        for j, observed in enumerate(actual, 1):
            next_row.append(min(row[j] + 1, next_row[-1] + 1, row[j - 1] + (expected != observed)))
        row = next_row
    return row[-1]


def evaluate(manifest, pipeline):
    totals = {"chars": 0, "char_errors": 0, "words": 0, "word_errors": 0}
    times = []
    for item in json.loads(manifest.read_text(encoding="utf-8")):
        path = manifest.parent / item["image"]
        started = time.perf_counter()
        if pipeline == "raw":
            with Image.open(path) as photo:
                image = cv2.cvtColor(np.asarray(ImageOps.exif_transpose(photo).convert("RGB")), cv2.COLOR_RGB2BGR)
            blocks = plain_ocr(image)
        else:
            blocks, warnings = recognize_page(path, item["surface"])
            if warnings:
                raise RuntimeError(warnings)
        times.append(time.perf_counter() - started)
        if "text" not in item:
            continue
        reference = " ".join(item["text"].split())
        actual = " ".join(" ".join(b["text"] for b in blocks if b["kind"] in {"text", "mixed"}).split())
        totals["chars"] += len(reference)
        totals["char_errors"] += edits(reference, actual)
        totals["words"] += len(reference.split())
        totals["word_errors"] += edits(reference.split(), actual.split())
    if not times:
        raise ValueError("清单至少需要一张照片")
    warm = sorted(times[1:])
    return {
        "pipeline": pipeline,
        "images": len(times),
        "cer": totals["char_errors"] / totals["chars"] if totals["chars"] else None,
        "wer": totals["word_errors"] / totals["words"] if totals["words"] else None,
        "first_seconds": round(times[0], 3),
        "warm_p50_seconds": round(statistics.median(warm), 3) if warm else None,
        "warm_p95_seconds": round(warm[math.ceil(0.95 * len(warm)) - 1], 3) if warm else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--pipeline", choices=("raw", "full"), required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.manifest, args.pipeline), indent=2))
