import os
import re
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageOps


@lru_cache
def model(kind):
    device = os.getenv("SNAP_DEVICE", "cpu")
    if kind == "layout":
        from paddlex import create_model
        return create_model("PP-DocLayoutV3", device=device)
    if kind == "formula":
        from paddleocr import FormulaRecognition
        default = "UniMERNet" if device.startswith("gpu") else "PP-FormulaNet_plus-S"
        return FormulaRecognition(model_name=os.getenv("SNAP_FORMULA_MODEL", default), device=device,
                                  enable_mkldnn=False)
    from paddleocr import PaddleOCR
    size = os.getenv("SNAP_OCR_SIZE", "medium" if device.startswith("gpu") else "small")
    if size not in {"small", "medium"}:
        raise ValueError("SNAP_OCR_SIZE 必须为 small 或 medium")
    return PaddleOCR(
        device=device, enable_mkldnn=False, text_detection_model_name=f"PP-OCRv6_{size}_det",
        text_recognition_model_name=f"PP-OCRv6_{size}_rec",
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False, text_rec_score_thresh=0.0,
        text_det_limit_side_len=2048, text_det_limit_type="max",
    )


def payload(result):
    data = result.json
    return data.get("res", data)


def rectangle(bounds):
    x, y, right, bottom = bounds
    return [[x, y], [right, y], [right, bottom], [x, bottom]]


def bounds(points):
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def overlap(a, b):
    x, y, r, d = a
    u, v, s, t = b
    intersection = max(0, min(r, s) - max(x, u)) * max(0, min(d, t) - max(y, v))
    return intersection / max(1, min((r - x) * (d - y), (s - u) * (t - v)))


def covered_fraction(inner, outer):
    x, y, right, bottom = inner
    left, top, other_right, other_bottom = outer
    intersection = max(0, min(right, other_right) - max(x, left)) * max(0, min(bottom, other_bottom) - max(y, top))
    return intersection / max(1, (right - x) * (bottom - y))


def ordered_quad(points):
    points = np.array(points, dtype=np.float32)
    center = points.mean(axis=0)
    delta = points - center
    points = points[np.argsort(np.arctan2(delta[:, 1], delta[:, 0]))]
    return np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)


def prepare(rgb, surface, corners=None):
    # 保留校正矩阵，识别区域最终映射回照片像素。
    height, width = rgb.shape[:2]
    scale = min(1, 2048 / max(height, width))
    preview = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))))
    transform = np.diag([preview.shape[1] / width, preview.shape[0] / height, 1.0])
    quad = ordered_quad(np.asarray(corners) * [transform[0, 0], transform[1, 1]]) if corners else None
    if quad is None:
        gray = cv2.cvtColor(preview, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 55, 160)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        area = preview.shape[0] * preview.shape[1]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            if not 0.45 * area < cv2.contourArea(contour) < 0.96 * area:
                continue
            polygon = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(polygon) == 4 and cv2.isContourConvex(polygon):
                quad = ordered_quad(polygon.reshape(4, 2))
                break
    if quad is not None:
        w = round(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
        h = round(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])))
        if w >= 32 and h >= 32 and cv2.isContourConvex(quad):
            warp = cv2.getPerspectiveTransform(quad, np.float32(rectangle((0, 0, w - 1, h - 1))))
            if abs(np.linalg.det(warp)) > 1e-9:
                preview = cv2.warpPerspective(preview, warp, (w, h))
                transform = warp @ transform
    bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    if surface == "blackboard":
        gray = cv2.bitwise_not(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        bgr = cv2.cvtColor(cv2.createCLAHE(2.0, (8, 8)).apply(gray), cv2.COLOR_GRAY2BGR)
    return bgr, transform


def project(points, transform):
    return cv2.perspectiveTransform(np.float32(points)[None], transform)[0].tolist()


def crop(image, points):
    x, y, r, d = bounds(points)
    h, w = image.shape[:2]
    return image[max(0, int(y)):min(h, int(np.ceil(d))), max(0, int(x)):min(w, int(np.ceil(r)))]


def plain_ocr(image):
    data = payload(next(iter(model("text").predict(image))))
    return [{"kind": "text", "text": str(text), "score": float(score),
             "polygon": np.asarray(poly, dtype=float).tolist()}
            for text, score, poly in zip(data["rec_texts"], data["rec_scores"], data["rec_polys"])]


def layout_regions(image, surface):
    regions = list(payload(next(iter(model("layout").predict(image))))["boxes"])
    if surface == "paper" or os.getenv("SNAP_LAYOUT_TILES") != "1":
        return regions
    height, width = image.shape[:2]
    # 对远距离板书局部放大检测，原图区域继续参与后续去重。
    tile_width, tile_height = int(width * 0.65), int(height * 0.65)
    for top in (0, height - tile_height):
        for left in (0, width - tile_width):
            tile = image[top:top + tile_height, left:left + tile_width]
            if min(tile.shape[:2]) < 64:
                continue
            boxes = payload(next(iter(model("layout").predict(tile))))["boxes"]
            for region in boxes:
                if region["label"].lower() not in {"inline_formula", "display_formula", "formula", "image", "figure", "chart"}:
                    continue
                if float(region["score"]) < 0.6:
                    continue
                x, y, right, bottom = region["coordinate"]
                coordinate = [x + left, y + top, right + left, bottom + top]
                if any(existing["label"] == region["label"] and
                       overlap(coordinate, existing["coordinate"]) > 0.65
                       for existing in regions):
                    continue
                regions.append({**region, "coordinate": coordinate, "order": None})
    return regions


def recognize_page(path, surface, corners=None):
    with Image.open(path) as original:
        rgb = np.asarray(ImageOps.exif_transpose(original).convert("RGB"))
    image, transform = prepare(rgb, surface, corners)
    inverse = np.linalg.inv(transform)
    warnings = []
    try:
        regions = layout_regions(image, surface)
    except Exception as error:
        warnings.append(f"版面模型：{error}")
        regions = []

    formulas = []
    figures = []
    for region in regions:
        label = region["label"].lower()
        polygon = rectangle(region["coordinate"])
        block = {"polygon": polygon, "score": float(region["score"]), "text": ""}
        if label in {"inline_formula", "display_formula", "formula"}:
            block.update(kind="formula", inline=label == "inline_formula")
            formulas.append(block)
        elif label in {"image", "figure", "chart"}:
            block.update(kind="figure")
            figures.append(block)

    # 公式从校正图裁剪；独立公式与文本行重叠时去掉重复转录。
    if formulas:
        try:
            patches = [crop(image, block["polygon"]) for block in formulas]
            valid = [(block, patch) for block, patch in zip(formulas, patches) if patch.size]
            if valid:
                results = model("formula").predict([patch for _, patch in valid], batch_size=2)
                for (block, _), result in zip(valid, results, strict=True):
                    block["text"] = str(payload(result)["rec_formula"]).strip().strip("$")
        except Exception as error:
            warnings.append(f"公式模型：{error}")

    lines = plain_ocr(image)
    inserted = set()
    for line in lines[:]:
        hit = [block for block in formulas if block["text"] and id(block) not in inserted
               and overlap(bounds(line["polygon"]), bounds(block["polygon"])) > 0.55]
        if not hit:
            continue
        if any(not block["inline"] and covered_fraction(bounds(line["polygon"]), bounds(block["polygon"])) > 0.85
               for block in hit):
            lines.remove(line)
            continue
        hit = [block for block in hit if block["inline"]]
        if not hit:
            continue
        x, y, right, bottom = bounds(line["polygon"])
        parts = []
        for block in sorted(hit, key=lambda item: bounds(item["polygon"])[0]):
            left, _, end, _ = bounds(block["polygon"])
            if left - x > 8:
                patch = crop(image, rectangle((x, y, left, bottom)))
                parts.extend(item["text"] for item in plain_ocr(patch))
            parts.append(f"${block['text']}$")
            x = max(x, end)
            inserted.add(id(block))
        if right - x > 8:
            parts.extend(item["text"] for item in plain_ocr(crop(image, rectangle((x, y, right, bottom)))))
        line.update(kind="mixed", text=" ".join(parts))

    for figure in figures:
        x, y, right, bottom = bounds(figure["polygon"])
        labels = []
        for line in lines:
            lx, ly, lr, lb = bounds(line["polygon"])
            near_bottom = bottom <= ly <= bottom + (bottom - y) * 0.15
            if (x <= lx and lr <= right) and (y <= ly and lb <= bottom or near_bottom):
                labels.append(line["text"])
        figure["text"] = " · ".join(labels)

    blocks = lines + [block for block in formulas if id(block) not in inserted] + figures
    def reading_order(block):
        box = bounds(block["polygon"])
        containers = [
            region for region in regions
            if region.get("order") is not None and overlap(box, region["coordinate"]) >= 0.8
        ]
        order = min((region["order"] for region in containers), default=10000)
        return order, round(box[1] / 12), box[0]

    blocks.sort(key=reading_order)
    for block in blocks:
        block["polygon"] = [
            [min(max(x, 0), rgb.shape[1]), min(max(y, 0), rgb.shape[0])]
            for x, y in project(block["polygon"], inverse)
        ]
    return [block for block in blocks
            if (box := bounds(block["polygon"]))[2] - box[0] >= 2 and box[3] - box[1] >= 2], warnings


def alignment(previous, current):
    # 只对用户指定的同一板面求配准，特征不足时保留两张图。
    detector = cv2.ORB_create(2500)
    def features(path):
        with Image.open(path) as photo:
            photo.thumbnail((1600, 1600))
            return detector.detectAndCompute(np.array(photo.convert("L")), None), photo.size
    (first, desc_first), size_first = features(previous)
    (second, desc_second), size_second = features(current)
    if desc_first is None or desc_second is None:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_second, desc_first, k=2)
    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
    if len(matches) < 18:
        return None
    source = np.float32([second[m.queryIdx].pt for m in matches])
    target = np.float32([first[m.trainIdx].pt for m in matches])
    matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if matrix is None or inliers.sum() < 15 or inliers.mean() < 0.65:
        return None
    mask = inliers.ravel().astype(bool)
    for points, size in ((source, size_second), (target, size_first)):
        if cv2.contourArea(cv2.convexHull(points[mask])) < 0.035 * np.prod(size):
            return None
    with Image.open(previous) as first_image, Image.open(current) as second_image:
        to_preview = np.diag([size_second[0] / second_image.width, size_second[1] / second_image.height, 1])
        from_preview = np.diag([first_image.width / size_first[0], first_image.height / size_first[1], 1])
    return from_preview @ matrix @ to_preview


def merge_pages(path, pages):
    for page in pages:
        for block in page["blocks"]:
            block.pop("duplicate_of", None)
            block.pop("possible_revision_of", None)
    for previous, current in zip(pages, pages[1:]):
        transform = alignment(path / previous["file"], path / current["file"])
        if transform is None:
            continue
        for block in current["blocks"]:
            if block.get("kind", "text") not in {"text", "formula", "mixed"} or not block.get("id") or not block["text"]:
                continue
            projected = bounds(project(block["polygon"], transform))
            matches = [candidate for candidate in previous["blocks"] if candidate.get("kind", "text") == block["kind"]
                       and candidate.get("id")
                       and overlap(projected, bounds(candidate["polygon"])) > 0.8]
            if len(matches) != 1:
                continue
            candidate = matches[0]
            # 公式保留大小写和符号，避免把变量改写判为重复。
            key = lambda item: re.sub(r"\s+", " ", item["text"]).strip()
            area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
            candidate_area = area(bounds(candidate["polygon"]))
            area_ratio = min(area(projected), candidate_area) / max(1, area(projected), candidate_area)
            if area_ratio >= 0.70 and key(candidate) == key(block):
                block["duplicate_of"] = candidate.get("duplicate_of", candidate["id"])
            else:
                block["possible_revision_of"] = candidate["id"]
