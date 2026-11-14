# Snap2Notes 项目方案

## 目标

将课堂黑板、白板和纸质手写笔记照片转换为可校订的数字笔记，优先识别英文手写和数学公式，保留简单图形及图注。面向学生、教师和研究人员，输出 Markdown、LaTeX、Word 和 PDF。用户可以追溯识别结果到原始照片并修正错误。

## 处理流程

1. 按用户顺序接收多张照片，校正拍摄方向，尝试识别板面并透视矫正。
2. 利用 PP-OCRv6 定位和识别英文手写，利用 PP-DocLayoutV3 检测布局、公式和图形。公式转录支持 PP-FormulaNet_plus-S 和 UniMERNet。对版面未检出的高置信度数字等式，使用整行 OCR 结果生成 LaTeX。保留原图坐标、置信度与阅读顺序。
3. 同一块板的照片通过 OpenCV 特征匹配与 RANSAC 配准。对齐可靠且位置和识别内容一致时去重，疑似改写交由用户确认。
4. 网页中对照原图修改文字、公式和图形，调整阅读顺序。对已确认内容提供可追溯的原文摘录。
5. 从统一的有序笔记数据生成各格式，并保存所引用的图片。

## 开发阶段

本地工作流包含照片预处理、英文 OCR、公式与图形检测、手动补框、同板配准、原图校订、历史笔记、原文摘录及四种导出。PDF 采用 Pandoc 与 Typst。已在浏览器验证上传黑板照片、选择公式候选、校订和下载 Markdown。Windows CPU 和 RTX 4090 已完成四张公开样例的小规模对照。新增的公式裁剪评测可用人工标注框单独测转录，端到端识别和裁剪基准的复现步骤见 README。扩展留出集和其他模型对照仍待完成。

## 实验与指标

收集有明确授权的黑板、白板和纸质笔记公开图片并标注文字、公式、图形和顺序。按源图片划分开发集和留出集。比较原图与预处理、PP-OCRv6 与 TrOCR、UniMERNet 与 PP-FormulaNet、PP-StructureV3 与 PaddleOCR-VL。记录英文 CER、WER、检测召回，公式区域召回与整式准确率、图形召回、多图误合并率、GPU 显存与冷启动和预热耗时。MathWriting 手写公式数据采用其自身的非商业许可，使用时记录来源。参考 [MathWriting 论文](https://arxiv.org/abs/2404.10690)、[OmniDocBench 论文](https://openaccess.thecvf.com/content/CVPR2025/html/Ouyang_OmniDocBench_Benchmarking_Diverse_PDF_Document_Parsing_with_Comprehensive_Annotations_CVPR_2025_paper.html) 和 [CDM 论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Image_Over_Text_Transforming_Formula_Recognition_Evaluation_with_Character_Detection_CVPR_2025_paper.pdf)。

## 主要技术依据

版面和顺序参考 [RT-DocLayout](https://arxiv.org/abs/2606.23344)。公式检测与混排参考 [Pix2Text](https://github.com/breezedeus/Pix2Text)，公式识别参考 [UniMERNet](https://github.com/opendatalab/UniMERNet)。整页方法参考 [PaddleOCR-VL-1.6](https://arxiv.org/abs/2606.03264)。来源追踪和文档表示参考 [Docling](https://docling-project.github.io/docling/reference/docling_document/) 与 [MinerU](https://opendatalab.github.io/MinerU/reference/output_files/)。每个实际集成的模型或代码版本、许可证和权重来源在实现时登记于 README。
