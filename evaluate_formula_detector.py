"""在独立环境比较 Pix2Text 公式检测，输出可核对的区域。"""
import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageOps
from pix2text.formula_detector import MathFormulaDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="MFD 1.5 的 ONNX 权重路径")
    parser.add_argument("image", type=Path, help="待检测的照片")
    parser.add_argument("--surface", choices=("paper", "whiteboard", "blackboard"), default="paper")
    parser.add_argument("--tiles", action="store_true", help="检测四个相互重叠的区域")
    parser.add_argument("--output", type=Path, required=True, help="输出 JSON 路径")
    args = parser.parse_args()

    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if args.surface == "blackboard":
        image = ImageOps.invert(image)

    detector = MathFormulaDetector(model_path=args.model, device="cpu")
    width, height = image.size
    if args.tiles:
        tile_width, tile_height = int(width * 0.65), int(height * 0.65)
        windows = [(left, top, left + tile_width, top + tile_height)
                   for top in (0, height - tile_height)
                   for left in (0, width - tile_width)]
    else:
        windows = [(0, 0, width, height)]

    started = time.perf_counter()
    detections = []
    for left, top, right, bottom in windows:
        for result in detector(image.crop((left, top, right, bottom)), resized_shape=1024):
            polygon = result["box"].tolist()
            detections.append({
                "type": result["type"],
                "score": float(result["score"]),
                "polygon": [[float(x) + left, float(y) + top] for x, y in polygon],
            })
    args.output.write_text(json.dumps({
        "image": str(args.image), "surface": args.surface, "tiles": args.tiles,
        "seconds": round(time.perf_counter() - started, 3), "detections": detections,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 {len(detections)} 个候选区域到 {args.output}")


if __name__ == "__main__":
    main()
