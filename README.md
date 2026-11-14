# Snap2Notes

上传黑板、白板或纸质手写笔记照片，在本机识别英文、数学公式和图形。对照原图修改 Markdown 后，可以下载 Markdown、LaTeX、Word 或 PDF。

## Windows 启动

需要 64 位 Python 3.12。在项目目录运行 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PADDLE_PDX_MODEL_SOURCE = 'bos'
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

打开 [本地网页](http://127.0.0.1:8000)。首次识别下载官方模型到 `data/paddlex`，之后可使用本地缓存。照片、笔记和裁剪图也存放在 `data/`。识别任务中断时，可以从历史笔记重试单页。

GPU 使用单独的虚拟环境，先按官方指南安装 CUDA 12.6 对应的 PaddlePaddle 3.3.0 及其运行库，再安装项目依赖，并设置 `$env:SNAP_DEVICE = 'gpu:0'`。可用 `SNAP_OCR_SIZE` 选择 `small` 或 `medium`，通过 `SNAP_FORMULA_MODEL` 选择公式模型。为复现下方 RTX 4090 对照，使用 `small` 和 `PP-FormulaNet_plus-S`，与 CPU 使用相同权重。 [PaddlePaddle Windows 安装](https://github.com/PaddlePaddle/docs/blob/develop/docs/install/pip/windows-pip.md)

## 使用

1. 选择最多 12 张照片，逐张选纸张、白板或黑板，需要时上移调整顺序。拍自同一板面时勾选多图合并。
2. 在原图上点识别框核对文字。识别漏掉的区域可以补圈。斜拍照片可以标记四角后重试。重试保留已保存的正文，新结果可从识别稿插入。
3. 在编辑器修改文字、公式和顺序。笔记中引用的裁剪图会显示在编辑区下方，丢失的图片会在导出时提示。疑似改写保留在识别稿，由用户编辑正文决定保留哪些内容。勾选已确认后，可以查看有原图出处的原文摘录。
4. 下载所需格式。含裁剪图的 Markdown 和 LaTeX 为带图片的 ZIP。Word 由 Pandoc 写入可编辑的 OMML 公式。PDF 由 Pandoc 转换为 Typst，再在本机排版。

## 第三方实现与研究

| 用途 | 项目与版本 | 来源 |
| --- | --- | --- |
| 透视、增强和同板配准 | OpenCV，Apache-2.0。采用四点单应性、CLAHE、ORB 和 RANSAC | [几何变换](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html) · [配准教程](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html) |
| 文字与版面 | PaddleOCR 3.7.0 / PaddleX 3.7.2，Apache-2.0。PP-OCRv6 和 PP-DocLayoutV3 从官方模型源加载 | [OCR 接口源码](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.7.0/paddleocr/_pipelines/ocr.py) · [PP-DocLayoutV3 模型卡](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) · [RT-DocLayout 论文](https://arxiv.org/abs/2606.23344) |
| 公式 | PaddleOCR 官方 PP-FormulaNet_plus-S（CPU）和 UniMERNet（GPU 配置）。区域混排参考 Pix2Text | [公式接口和模型列表](https://www.paddleocr.ai/latest/en/version3.x/module_usage/formula_recognition.html) · [UniMERNet 论文](https://arxiv.org/abs/2404.15254) · [Pix2Text 源码](https://github.com/breezedeus/Pix2Text/blob/main/pix2text/text_formula_ocr.py) |
| 公式预览与导出 | KaTeX 0.16.27（MIT）、Pandoc（GPL-2.0-or-later，独立进程）、Typst（Apache-2.0） | [KaTeX 许可](https://github.com/KaTeX/KaTeX/blob/v0.16.27/LICENSE) · [Pandoc 手册](https://pandoc.org/MANUAL.html) · [Typst Python 包](https://pypi.org/project/typst/) |

评测候选和实验范围见 [中文 proposal](docs/proposal.md)。

## 连接 GitHub

先在 PowerShell 运行 `git --version`。如果找不到 `git`，用下面的命令安装 Git for Windows。安装完成后，重新打开 PowerShell。若使用 VS Code 集成终端，请重启 VS Code，再打开终端。 [Git for Windows 安装文档](https://git-scm.com/install/windows)

```powershell
winget install --id Git.Git -e --source winget
```

在新终端确认 `git --version` 成功后，进入项目目录。如果 GitHub 仓库为空，而且本地尚未初始化，依次执行下面的命令。提交前可用 `git status --short` 检查将要提交的文件。首次提交若提示需要身份信息，用 `git config user.name "你的显示名称"` 和 `git config user.email "你的 GitHub 提交邮箱"` 设置当前仓库的身份，再重试提交。 [GitHub 提交邮箱设置](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address)

```powershell
git init -b main
git add .gitignore app.py recognition.py verify_samples.py benchmark.py evaluate_formula_detector.py requirements.txt README.md docs static
git status --short
git commit -m "Implement Snap2Notes local workflow"
git remote add origin https://github.com/SumanaWu/Snap2Notes.git
git push -u origin main
```

如果本地已经有提交并设置了 `origin`，用 `git remote -v` 确认目标仓库，再运行 `git push -u origin main`。首次 HTTPS 推送可能打开浏览器，请按 Git Credential Manager 提示登录 GitHub。 [GitHub 凭据管理文档](https://docs.github.com/en/get-started/git-basics/caching-your-github-credentials-in-git)

如果远端已经有提交，先克隆到另一个目录，再把上述源码和文档复制过去提交并推送。原始课程 Word 文件可按课程提交要求另行添加。`data/` 与 `.venv/` 已排除。

## 复现公开样本验证

在 `data/samples/` 放入以下样本，然后执行 `.\.venv\Scripts\python.exe -X utf8 verify_samples.py data/samples`。脚本调用真实模型，输出各页耗时与块数量，并检查带裁剪图的四种导出和 Word 的 OMML 公式。

| 保存为 | 来源 |
| --- | --- |
| `blackboard.jpg` | David Malone，CC BY 4.0。[黑板实拍](https://commons.wikimedia.org/wiki/File:Mathematics_(cryptography)_on_a_blackboard.jpg) |
| `whiteboard.jpg` | Skullovitch，CC BY-SA 4.0。[白板实拍](https://commons.wikimedia.org/wiki/File:White_board_with_math.jpg) |
| `paper.jpg` | SucreRouge，CC BY-SA 3.0。[纸质笔记实拍](https://commons.wikimedia.org/wiki/File:Mathnote894462.jpg) |
| `formula.png` | PaddleOCR 官方公式模块示例。[示例文档](https://www.paddleocr.ai/latest/en/version3.x/module_usage/formula_recognition.html) · [原图](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_formula_rec_001.png) |

版面模型可以定位并转录官方公式样例。版面模型漏掉白板上的 `1+1=3-1`，PP-OCRv6 small 直接读对了整行，PP-FormulaNet_plus-S 对 OCR 行框的裁剪则读成了 `I+I=3-I`。因此主流程会将置信度至少为 0.95、整行符合数字等式语法且包含算术运算的 OCR 结果转成公式块，沿用该行的原图坐标。复杂公式仍由版面和公式模型处理，漏检区域可在原图手动补圈或通过 Pix2Text 候选校订。 [PaddleOCR 公式模型文档](https://www.paddleocr.ai/latest/en/version3.x/module_usage/formula_recognition.html) · [Pix2Text 混合识别源码](https://github.com/breezedeus/Pix2Text/blob/main/pix2text/text_formula_ocr.py)



标注后可用 `benchmark.py` 比较原图 OCR 和完整流程。清单为 JSON 数组，每项填写相对于清单的 `image`、`surface` 和人工转录的 `text`，例如 `{"image":"blackboard.jpg","surface":"blackboard","text":"人工转录"}`。固定同一清单和图片顺序，分别在独立进程运行 `python benchmark.py manifest.json --pipeline raw` 与 `python benchmark.py manifest.json --pipeline full`。两次运行均使用同一个虚拟环境和设备配置。

下载上表四张照片后，可用 `python benchmark.py docs/sample-benchmark.json --pipeline full --repeats 3` 复现小样本基准。脚本先加载首张图并记录耗时，再把其余图片跑一遍，最后重复计时。清单包含黑板、白板和独立公式图的 10 个公式标注。定位按原图矩形 IoU ≥ 0.5 一对一匹配，整式转录按去空白后的 LaTeX 字符串比较。这个指标会把一些等价写法计为不同。纸质照片参与计时，尚未标注公式。CER 和 WER 需要补充人工转录后才能计算。

图形标注使用 `"figures": [{"box": [左, 上, 右, 下]}]`，坐标对应 EXIF 方向校正后的原图。黑板上的消息箭头作为一个流程图标注，白板和独立公式图是已核对的无图样例。空数组表示已核对且无图，省略字段表示该页暂不参加图形评分。图形与公式定位使用同一 IoU 阈值。RTX 4090 上用 small 跑完整流程，三个已标注页面没有自动输出图形框，图形召回为 0/1。

以下完整流程均使用 PP-DocLayoutV3。除 UniMERNet 对照外，表中公式转录使用 PP-FormulaNet_plus-S。RTX 4090 每次在独立进程加载模型，四张图各重复三次。Paddle 提示编译时 cuDNN 为 9.9，本机运行库为 9.5，样例推理成功。

| 流程 | 首张图（秒） | 预热 P50（秒） | 预热 P95（秒） | 公式定位匹配 | 整式转录匹配 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原图文字 OCR，CPU，历史结果 | 10.836 | 2.910 | 6.571 | 0/10 | 0/10 |
| 完整流程，CPU，历史结果 | 15.433 | 4.615 | 9.614 | 1/10 | 0/10 |
| 完整流程，CPU，small，改动后 | 11.661 | 4.122 | 7.104 | 2/10 | 1/10 |
| 完整流程，RTX 4090，small，改动前 | 6.852 | 0.191 | 0.280 | 1/10 | 0/10 |
| 完整流程，RTX 4090，small，改动后 | 7.662 | 0.191 | 0.280 | 2/10 | 1/10 |
| 完整流程，RTX 4090，medium，改动后 | 9.140 | 0.214 | 0.337 | 2/10 | 1/10 |
| 完整流程，RTX 4090，small + UniMERNet，改动后 | 6.665 | 0.195 | 1.200 | 2/10 | 1/10 |

改动后两种 OCR 在三个有标注的页面均只输出两个公式框，分别匹配白板和官方公式图。白板整式匹配，官方公式图的 LaTeX 字符串与参考仍有差异，八处黑板标注仍漏检。白板公式参与规则开发，结果用于回归验证。手写识别模型比较仍需按照片来源划分的独立留出集。 [UniMERNet 论文与实景公式评测](https://openaccess.thecvf.com/content/CVPR2026/html/Gu_UniMERNet_A_Universal_Network_for_Real-World_Mathematical_Expression_Recognition_CVPR_2026_paper.html)

Pix2Text 在同样三张有公式标注的图上共返回 7 个候选，其中 3 个与标注匹配。黑板与白板使用分块，黑板额外反色，独立公式图使用整图。候选匹配数为 3/7，标注召回为 3/10，适合继续作为人工校订入口。这批图用于功能验证与初步对照，完整留出集和其他模型对照仍待补充。

### 单独测量公式转录

```powershell
$env:SNAP_FORMULA_MODEL = 'PP-FormulaNet_plus-S'
python benchmark.py docs/sample-benchmark.json --pipeline formula-crops
$env:SNAP_FORMULA_MODEL = 'UniMERNet'
python benchmark.py docs/sample-benchmark.json --pipeline formula-crops
```

这个模式按人工标注框从方向校正后的原图裁剪，黑板裁剪反色。输出逐框参考、预测和整式匹配结果，耗时包含模型加载。RTX 4090 上 PP-FormulaNet_plus-S 在 8.165 秒内得到 0/10 整式字符串匹配，UniMERNet 在 11.918 秒内得到 1/10。UniMERNet 读对了白板等式，两种模型的八处黑板转录都与参考不一致。官方印刷样例的输出与参考有 LaTeX 分组和间距差异。UniMERNet 测试权重来自 PaddlePaddle 发布的 `795faf5` 版本。 [PaddleOCR 公式模块](https://www.paddleocr.ai/latest/en/version3.x/module_usage/formula_recognition.html) · [UniMERNet 官方权重](https://huggingface.co/PaddlePaddle/UniMERNet/tree/795faf56325b38c845bbc31535d661e4c72e5fe0) · [CDM 公式评测论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Image_Over_Text_Transforming_Formula_Recognition_Evaluation_with_Character_Detection_CVPR_2025_paper.pdf)

## Pix2Text 公式检测对照

在单独的 Windows CPU 测试环境中，使用 Pix2Text 1.1.4、CnSTD 1.2.8、RapidOCR 3.9.2、ONNX Runtime 1.19.2 和 MFD 1.5 权重。对四张前述样例直接检测时，三个实拍样例均无候选，官方公式图有一个候选。黑板反色并对三张实拍各做四个 65% 重叠分块后，黑板得到 5 个候选，白板和纸笔各得到 1 个候选。人工查看裁剪图发现黑板候选中包含公式，也包含英文文字与较大混合区域，白板候选包含数字和图形。测试样本较少，尚无法计算召回率和误检率。检测结果用于人工对照，主流程继续使用已配置的版面模型与手动补圈。 [MFD 1.5 模型卡](https://huggingface.co/breezedeus/pix2text-mfd-1.5) · [Pix2Text 检测器源码](https://github.com/breezedeus/Pix2Text/blob/main/pix2text/formula_detector.py)

使用独立环境安装上述版本并下载模型卡中的 `pix2text-mfd-1.5.onnx` 后，可以运行以下命令。输出坐标对应经 EXIF 方向校正后的原图。重叠分块可能产生重复候选，输出保留每个分块的检测结果，便于人工复核。

```powershell
python evaluate_formula_detector.py 模型路径 照片路径 --surface blackboard --tiles --output 候选.json
```

大幅板书可以通过设置 `$env:SNAP_LAYOUT_TILES='1'` 试用重叠分块版面检测。在现有黑板和白板样例上，CPU 测试分别耗时约 26.6 秒和 14.7 秒，未检出额外公式或图形。该选项默认关闭，效果仍需在更多标注照片上验证。

### 在网页中查找公式候选

公式候选使用 Pix2Text 的独立环境。下面的命令在项目目录创建该环境并下载官方 MIT 许可的权重。需要安装系统 Python 3.12，首次安装会下载 PyTorch 等依赖。

```powershell
python -m venv data/mfd-env
.\data\mfd-env\Scripts\python.exe -m pip install pix2text==1.1.4 cnstd==1.2.8 rapidocr==3.9.2 onnxruntime==1.19.2
New-Item -ItemType Directory -Force data/mfd
Invoke-WebRequest 'https://huggingface.co/breezedeus/pix2text-mfd-1.5/resolve/main/pix2text-mfd-1.5.onnx' -OutFile data/mfd/pix2text-mfd-1.5.onnx
```

重启网页服务，打开照片后点击“查找公式候选”。橙色框是待核对的区域。点击一个框，再点击“识别选中候选”，公式识别结果会插入编辑器的光标位置。检测可能误选文字或图形，请对照原图确认。可以通过 `SNAP_MFD_PYTHON` 和 `SNAP_MFD_MODEL` 指定独立环境与模型路径。
