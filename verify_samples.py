"""用公开样例跑真实识别与导出，输出逐张耗时和块数量。"""
import argparse
import base64
import io
import json
import os
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import app


def verify_exports(note_id):
    for extension in ("md", "tex", "docx", "pdf"):
        data = app.export(note_id, extension).body
        if extension in {"md", "tex", "docx"}:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if extension == "docx":
                    assert b"<m:oMath" in archive.read("word/document.xml")
                else:
                    assert f"notes.{extension}" in archive.namelist()
                    assert any(name.startswith("assets/") for name in archive.namelist())
        else:
            assert data.startswith(b"%PDF")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path, help="包含四张样图的目录")
    args = parser.parse_args()
    inputs = [("blackboard.jpg", "blackboard"), ("whiteboard.jpg", "whiteboard"),
              ("paper.jpg", "paper"), ("formula.png", "paper")]
    images = [app.ImageInput(name=name, surface=surface,
                             data=base64.b64encode((args.samples / name).read_bytes()).decode())
              for name, surface in inputs]
    with TemporaryDirectory(dir=app.DATA) as temporary:
        app.DATA = Path(temporary)
        start = time.perf_counter()
        note_id = app.create_note(app.Upload(images=images))["id"]
        deadline = start + 300
        while time.perf_counter() < deadline:
            note = app.get_note(note_id)
            if note["status"] not in {"queued", "processing"}:
                break
            time.sleep(0.25)
        else:
            raise TimeoutError("识别超过五分钟")
        assert note["status"] == "complete", note
        for page in note["pages"]:
            assert page["blocks"] and not page["warnings"], page
            print(json.dumps({
                "file": page["name"],
                "blocks": len(page["blocks"]),
                "seconds": page.get("seconds"),
                "formula_blocks": sum(block["kind"] == "formula" for block in page["blocks"]),
            }))
        assert any(block["kind"] == "formula" and block["text"] for block in note["pages"][-1]["blocks"])
        region = app.recognize_region(note_id, 0, app.Region(kind="figure", box=(10, 10, 180, 150)))
        app.update_note(note_id, app.Note(markdown=region["markdown"] + "\n\n$$x^2+1$$", confirmed=True))
        verify_exports(note_id)
        print("识别和四种导出检查通过")


if __name__ == "__main__":
    main()
