# HyperLPR3 临时车牌识别

基于 HyperLPR3 + RapidOCR 的离线临时车牌（纸质临时号牌）识别工具。

## 功能

- **单图识别** — 输入图片路径，返回车牌号、置信度与耗时
- **批量测试** — 无参数运行时自动识别 `test/` 目录下所有 PNG 图片并汇总结果
- **USB 摄像头实时识别** — 后台异步识别，实时叠加结果到预览画面
- **视频文件识别** — 以 5 FPS 采样，输出每帧识别结果

## 识别流程（三级递进）

| 层级 | 方法 | 置信度阈值 | 说明 |
|------|------|------------|------|
| Tier 1 | HyperLPR3 检测区域 + RapidOCR 原图 | ≥ 0.9 | 最快，高置信度立即返回 |
| Tier 2 | OTSU 二值化预处理 | ≥ 0.85 | 中等速度 |
| Tier 3 | 自适应阈值、膨胀等多种预处理 | ≥ 0.7 | 最慢，兜底策略 |

所有识别完全离线，不依赖网络。

## 安装

```bash
pip install -r requirements.txt
```

依赖：`hyperlpr3` `rapidocr-onnxruntime` `opencv-python` `onnxruntime` `Pillow`

## 使用方法

### 批量测试

```bash
python temp_plate_recog.py
```

自动识别 `test/` 下所有图片，输出每张结果与汇总统计。

### 单张图片

```bash
python temp_plate_recog.py path/to/image.png
```

### USB 摄像头

```bash
python temp_plate_recog.py --camera [摄像头索引]
# 默认索引 0
```

按 `ESC` 或 `Q` 退出。

### 视频文件

```bash
python temp_plate_recog.py --video path/to/video.mp4
```

每 5 FPS 采样一帧识别，输出识别到的不同车牌。

## API 调用

```python
from temp_plate_recog import recognize_plate

result = recognize_plate("test/26.png")
# result: {"plate": str|None, "confidence": float, "time_ms": float, "bbox": tuple|None}
print(result)
```

也支持传入 OpenCV 图像 (numpy.ndarray)。
