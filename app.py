import base64
import binascii
import io
import json
import os
import shutil
import subprocess
import threading
import tempfile
import time
import uuid
import zipfile
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageOps, UnidentifiedImageError
from recognition import recognize_page, merge_pages, model, plain_ocr, payload, ordered_quad
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(DATA / "paddlex"))
app = FastAPI(title="Snap2Notes")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
lock = threading.RLock()
worker = ThreadPoolExecutor(max_workers=1)
pending = set()


class ImageInput(BaseModel):
    name: str
    data: str = Field(max_length=27962028)
    surface: Literal["paper", "whiteboard", "blackboard"] = "paper"


class Upload(BaseModel):
    images: list[ImageInput] = Field(min_length=1, max_length=12)
    same_board: bool = False


class Note(BaseModel):
    markdown: str = Field(max_length=1000000)
    confirmed: bool = False


class Region(BaseModel):
    kind: Literal["text", "formula", "figure"]
    box: tuple[int, int, int, int]


class Retry(BaseModel):
    corners: list[tuple[int, int]] | None = Field(default=None, min_length=4, max_length=4)


def folder(note_id: str) -> Path:
    try:
        uuid.UUID(note_id)
    except ValueError:
        raise HTTPException(404, "笔记不存在")
    path = DATA / note_id
    if not path.is_dir():
        raise HTTPException(404, "笔记不存在")
    return path


def save(path: Path, result: dict):
    temp = path / "note.tmp"
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path / "note.json")


def load(path: Path) -> dict:
    return json.loads((path / "note.json").read_text(encoding="utf-8"))


def block_markdown(block):
    if block.get("duplicate_of"):
        return ""
    text = block["text"]
    kind = block.get("kind", "text")
    if kind == "figure" or (kind == "formula" and not text):
        label = "Figure" if kind == "figure" else "Formula"
        caption = text.replace("[", "(").replace("]", ")") if kind == "figure" else ""
        return f"![{caption or label}](assets/{block['id']}.jpg)"
    if kind == "formula":
        if block.get("inline"):
            return " ".join(part for part in (block.get("text_before"), f"${text}$", block.get("text_after")) if part)
        return f"$$\n{text}\n$$"
    return text


def add_assets(path, page, blocks):
    assets = path / "assets"
    assets.mkdir(exist_ok=True)
    with Image.open(path / page["file"]) as image:
        for block in blocks:
            block["id"] = uuid.uuid4().hex
            if block["kind"] in {"figure", "formula"}:
                xs, ys = zip(*block["polygon"])
                box = (max(0, int(min(xs))), max(0, int(min(ys))),
                       min(image.width, int(max(xs))), min(image.height, int(max(ys))))
                if box[0] < box[2] and box[1] < box[3]:
                    image.crop(box).save(assets / f"{block['id']}.jpg", quality=94)


def recognize(note_id: str, only=None, corners=None):
    path = folder(note_id)
    try:
        with lock:
            note = load(path)
            note["status"] = "processing"
            save(path, note)
        indexes = range(len(note["pages"])) if only is None else [only]
        for index in indexes:
            page = note["pages"][index]
            started = time.perf_counter()
            try:
                blocks, warnings = recognize_page(path / page["file"], page.get("surface", "paper"), corners)
                add_assets(path, page, blocks)
                blocks.extend(block for block in page["blocks"] if block.get("manual"))
                page.update(blocks=blocks, warnings=warnings)
                page["error"] = None
            except Exception as exc:
                page["error"] = str(exc)
            page["seconds"] = round(time.perf_counter() - started, 3)
            with lock:
                latest = load(path)
                latest["pages"][index] = page
                save(path, latest)
        if note.get("same_board"):
            try:
                merge_pages(path, note["pages"])
            except (cv2.error, ValueError):
                note["warning"] = "多图配准失败，已保留各页内容。"
        with lock:
            latest = load(path)
            latest.update(pages=note["pages"], warning=note.get("warning"))
            latest["draft"] = "\n\n".join(filter(None, (
                block_markdown(block) for page in note["pages"] for block in page["blocks"]
            )))
            if not latest.get("edited"):
                latest["markdown"] = latest["draft"]
            latest["status"] = "error" if any(page.get("error") for page in note["pages"]) else "complete"
            save(path, latest)
    except Exception as error:
        with lock:
            latest = load(path)
            latest.update(status="error", warning=str(error))
            save(path, latest)
    finally:
        with lock:
            pending.discard(note_id)


def enqueue(note_id, page=None, corners=None):
    with lock:
        if note_id in pending:
            raise HTTPException(409, "这份笔记正在识别")
        pending.add(note_id)
        worker.submit(recognize, note_id, page, corners)


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/api/notes")
def create_note(upload: Upload):
    pages = []
    for item in upload.images:
        try:
            data = base64.b64decode(item.data, validate=True)
            if len(data) > 20 * 1024 * 1024:
                raise ValueError("图片超过 20 MB")
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                pass
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG", "HEIF"}:
                    raise ValueError("仅支持 JPEG、PNG 和 HEIC")
                if image.width * image.height > 40000000:
                    raise ValueError("图片超过 4000 万像素")
                image = ImageOps.exif_transpose(image).convert("RGB")
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=95)
                pages.append((item.name, output.getvalue(), image.size, item.surface))
        except (ValueError, binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(400, f"图片 {item.name} 无法读取：{exc}")
    note_id = str(uuid.uuid4())
    path = DATA / note_id
    path.mkdir()
    records = []
    for index, (name, content, size, surface) in enumerate(pages):
        filename = f"page-{index + 1}.jpg"
        (path / filename).write_bytes(content)
        records.append({"name": name, "file": filename, "width": size[0], "height": size[1], "blocks": [], "surface": surface})
    save(path, {"id": note_id, "status": "queued", "pages": records, "markdown": "", "same_board": upload.same_board})
    enqueue(note_id)
    return {"id": note_id}


@app.get("/api/notes/{note_id}")
def get_note(note_id: str):
    with lock:
        note = load(folder(note_id))
        if note["status"] in {"queued", "processing"} and note_id not in pending:
            note["status"] = "interrupted"
        return note


@app.get("/api/notes")
def history():
    files = sorted(DATA.glob("*/note.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [{"id": note["id"], "name": note["pages"][0]["name"]}
            for file in files if (note := load(file.parent)).get("pages")]


@app.put("/api/notes/{note_id}")
def update_note(note_id: str, update: Note):
    path = folder(note_id)
    with lock:
        note = load(path)
        note["markdown"] = update.markdown
        note["confirmed"] = update.confirmed
        note["edited"] = True
        save(path, note)
    return {"ok": True}


@app.get("/api/notes/{note_id}/pages/{page_number}")
def get_page(note_id: str, page_number: int):
    path = folder(note_id)
    pages = load(path)["pages"]
    if page_number < 0 or page_number >= len(pages):
        raise HTTPException(404, "图片不存在")
    return FileResponse(path / pages[page_number]["file"])


@app.get("/api/notes/{note_id}/assets/{filename}")
def get_asset(note_id: str, filename: str):
    if not re.fullmatch(r"[a-f0-9]{32}\.jpg", filename):
        raise HTTPException(404, "图片不存在")
    asset = folder(note_id) / "assets" / filename
    if not asset.is_file():
        raise HTTPException(404, "图片不存在")
    return FileResponse(asset, media_type="image/jpeg")


@app.post("/api/notes/{note_id}/pages/{page_number}/retry")
def retry_page(note_id: str, page_number: int, request: Retry):
    path = folder(note_id)
    with lock:
        note = load(path)
        if not 0 <= page_number < len(note["pages"]):
            raise HTTPException(404, "图片不存在")
        page = note["pages"][page_number]
        if request.corners and any(not 0 <= x < page["width"] or not 0 <= y < page["height"]
                                   for x, y in request.corners):
            raise HTTPException(422, "角点超出照片")
        if request.corners:
            polygon = ordered_quad(request.corners)
            if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) < 1024:
                raise HTTPException(422, "请标记四个不同的角，围成足够大的区域")
        enqueue(note_id, page_number, request.corners)
        note["status"] = "queued"
        save(path, note)
    return {"ok": True}


@app.post("/api/notes/{note_id}/pages/{page_number}/region")
def recognize_region(note_id: str, page_number: int, request: Region):
    path = folder(note_id)
    note = get_note(note_id)
    if not 0 <= page_number < len(note["pages"]):
        raise HTTPException(404, "图片不存在")
    page = note["pages"][page_number]
    x, y, right, bottom = request.box
    if not (0 <= x < right <= page["width"] and 0 <= y < bottom <= page["height"]):
        raise HTTPException(422, "区域超出照片")
    polygon = [[x, y], [right, y], [right, bottom], [x, bottom]]
    block = {"kind": request.kind, "text": "", "polygon": polygon, "manual": True}
    try:
        with Image.open(path / page["file"]) as original:
            patch = cv2.cvtColor(np.asarray(original.crop(request.box).convert("RGB")), cv2.COLOR_RGB2BGR)
        if page.get("surface") == "blackboard":
            patch = cv2.bitwise_not(patch)
        if request.kind == "formula":
            result = worker.submit(lambda: payload(next(iter(model("formula").predict(patch))))).result()
            block["text"] = result["rec_formula"].strip().strip("$")
        elif request.kind == "text":
            block["text"] = " ".join(item["text"] for item in worker.submit(plain_ocr, patch).result())
    except Exception as error:
        raise HTTPException(503, str(error)) from error
    with lock:
        latest = load(path)
        if note_id in pending:
            raise HTTPException(409, "请在整页识别完成后补圈")
        add_assets(path, page, [block])
        latest["pages"][page_number]["blocks"].append(block)
        save(path, latest)
    return {**block, "markdown": block_markdown(block)}


@app.get("/api/notes/{note_id}/pages/{page_number}/formula-candidates")
def formula_candidates(note_id: str, page_number: int):
    path = folder(note_id)
    note = get_note(note_id)
    if not 0 <= page_number < len(note["pages"]):
        raise HTTPException(404, "图片不存在")
    if note_id in pending:
        raise HTTPException(409, "请等待整页识别完成")
    python = Path(os.getenv("SNAP_MFD_PYTHON", str(DATA / "mfd-env" / "Scripts" / "python.exe")))
    weights = Path(os.getenv("SNAP_MFD_MODEL", str(DATA / "mfd" / "pix2text-mfd-1.5.onnx")))
    if not python.is_file() or not weights.is_file():
        raise HTTPException(503, "请按照 README 安装可选的公式候选检测环境")
    page = note["pages"][page_number]
    with tempfile.TemporaryDirectory(dir=path) as temporary:
        output = Path(temporary) / "candidates.json"
        args = [str(python), str(ROOT / "evaluate_formula_detector.py"), str(weights),
                str(path / page["file"]), "--surface", page.get("surface", "paper"),
                "--tiles", "--output", str(output)]
        environment = os.environ.copy()
        environment["YOLO_CONFIG_DIR"] = str(DATA / "mfd-config")
        try:
            result = subprocess.run(args, capture_output=True, encoding="utf-8", errors="replace",
                                    timeout=120, cwd=ROOT, env=environment)
        except subprocess.TimeoutExpired as error:
            raise HTTPException(504, "公式候选检测超时") from error
        if result.returncode:
            raise HTTPException(503, f"公式候选检测失败：{result.stderr[-800:]}")
        detections = json.loads(output.read_text(encoding="utf-8"))["detections"]
    return detections


@app.get("/api/notes/{note_id}/summary")
def summary(note_id: str):
    note = get_note(note_id)
    if not note.get("confirmed"):
        raise HTTPException(409, "请先确认并保存笔记")
    excerpt = []
    for page_number, page in enumerate(note["pages"]):
        for block in page["blocks"]:
            text = block["text"].strip()
            if block.get("kind", "text") == "text" and text and text in note["markdown"] and not block.get("duplicate_of"):
                excerpt.append({"text": text, "page": page_number, "polygon": block["polygon"]})
    return excerpt[:8]


@app.get("/api/notes/{note_id}/export/{format}")
def export(note_id: str, format: str):
    path = folder(note_id)
    markdown = load(path)["markdown"]
    if format not in ("md", "tex", "docx", "pdf"):
        raise HTTPException(400, "不支持此格式")
    assets = sorted(set(re.findall(r"\]\(assets/([a-f0-9]{32}\.jpg)\)", markdown)))
    missing = [name for name in assets if not (path / "assets" / name).is_file()]
    if missing:
        raise HTTPException(422, "笔记中引用的图片不存在：" + ", ".join(missing))
    if format == "md":
        return download(markdown.encode("utf-8"), format, path, assets)
    pandoc = shutil.which("pandoc")
    if not pandoc:
        try:
            from pypandoc import get_pandoc_path
            pandoc = get_pandoc_path()
        except (ImportError, OSError):
            pass
    if not pandoc:
        raise HTTPException(503, "请安装 Pandoc 并将其加入 PATH")
    if format == "pdf":
        try:
            import typst
        except ImportError as error:
            raise HTTPException(503, "PDF 导出需要安装 Typst Python 包") from error
    # 导出在独立目录运行，仅复制应用生成且被引用的图片。
    with tempfile.TemporaryDirectory(dir=path) as directory:
        root = Path(directory)
        output = root / ("notes.typ" if format == "pdf" else f"notes.{format}")
        (root / "assets").mkdir()
        for name in assets:
            source = path / "assets" / name
            if source.is_file():
                shutil.copyfile(source, root / "assets" / name)
        filter_path = root / "assets.lua"
        filter_path.write_text(
            'function Image(img)\n'
            '  if not img.src:match("^assets/%x+%.jpg$") then return img.caption end\n'
            '  return img\nend\n', encoding="utf-8"
        )
        args = [pandoc, "--from=markdown-raw_tex-raw_html", "--standalone",
                "--lua-filter", str(filter_path), "--fail-if-warnings", "-o", str(output)]
        if re.search(r"\\(?:input|include|write|openin|openout|read|csname|catcode|def|newcommand|usepackage|special)\b", markdown):
            raise HTTPException(422, "请移除文件访问或宏定义类 LaTeX 命令")
        try:
            process = subprocess.run(args, input=markdown, encoding="utf-8", errors="replace",
                                     capture_output=True, cwd=root, timeout=90)
        except subprocess.TimeoutExpired as error:
            raise HTTPException(504, "文档转换超时") from error
        if process.returncode:
            raise HTTPException(422, f"导出失败，请检查公式或图片：{process.stderr[-1500:]}")
        if format == "pdf":
            try:
                return download(typst.compile(str(output), root=str(root)), format, path, assets)
            except (RuntimeError, ValueError) as error:
                raise HTTPException(422, f"PDF 排版失败：{error}") from error
        return download(output.read_bytes(), format, path, assets)


def download(content, format, path, assets):
    media = {"md": "text/markdown", "tex": "application/x-tex", "pdf": "application/pdf",
             "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}[format]
    filename = f"notes.{format}"
    if format in {"md", "tex"} and assets:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(filename, content)
            for name in assets:
                source = path / "assets" / name
                if source.is_file():
                    archive.write(source, f"assets/{name}")
        content, filename, media = buffer.getvalue(), f"notes-{format}.zip", "application/zip"
    return Response(content, media_type=media, headers={"Content-Disposition": f'attachment; filename="{filename}"'})
