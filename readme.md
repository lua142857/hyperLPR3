# 车牌识别

基于 RapidOCR 的离线车牌识别工具，支持标准金属号牌与纸质临时号牌。

## 功能

- **单图识别** — 输入图片路径，返回车牌号、置信度与耗时
- **批量测试** — 无参数运行时自动识别 `test/` 目录下所有 PNG 图片并汇总结果
- **USB 摄像头实时识别** — 后台异步识别，实时叠加结果到预览画面，新车牌打印到控制台
- **视频文件识别** — 以 5 FPS 采样，输出识别到的不同车牌

## 识别流程

四级递进识别策略，低层级仅在高层级未达阈值时执行：

| 层级 | 处理方法 | 置信度阈值 |
|------|----------|------------|
| 1 | RapidOCR 原图 | ≥ 0.9 |
| 2 | OTSU 二值化 | ≥ 0.85 |
| 3 | 自适应阈值 / OTSU+膨胀 / OTSU+扩边 | ≥ 0.7 |
| 4 | 大核自适应阈值+膨胀等（最慢） | 无阈值兜底 |

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
# 批量测试 test/ 目录下所有图片
python temp_plate_recog.py

# 单张图片
python temp_plate_recog.py path/to/image.png

# USB 摄像头
python temp_plate_recog.py --camera [索引]

# 视频文件
python temp_plate_recog.py --video path/to/video.mp4
```

摄像头模式下按 `ESC` 或 `Q` 退出，点击关闭窗口也可退出。

## API

```python
from temp_plate_recog import recognize_plate

result = recognize_plate("test/26.png")
# {"plate": str|None, "confidence": float, "time_ms": float}
```

也支持传入 OpenCV 图像 (numpy.ndarray)。
