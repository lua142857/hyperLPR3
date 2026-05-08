import os
import re
import glob
import time
import threading

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rapidocr_onnxruntime import RapidOCR

PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
PLATE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"

_ocr_engine = None


def draw_chinese_text(img, text, pos, font_size=20, color=(0, 255, 0)):
    """Draw Chinese text on OpenCV image using PIL."""
    # Convert OpenCV BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Load font with Chinese support
    font = None
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except:
            continue
    if font is None:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(pil_img)
    # Convert BGR color to RGB
    rgb_color = color[::-1]
    draw.text(pos, text, font=font, fill=rgb_color)

    # Convert back to BGR
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _ocr(image):
    results, _ = _get_ocr()(image)
    if not results:
        return []
    return [{"bbox": item[0], "text": item[1], "confidence": item[2]} for item in results]


def _extract_plate(text):
    cleaned = re.sub(r"[·\.\-\s\*]", "", text)
    for ch in PROVINCES:
        idx = cleaned.find(ch)
        if idx == -1:
            continue
        sub = cleaned[idx:]
        for length in (8, 7, 6, 5):
            if len(sub) < length:
                continue
            candidate = sub[:length]
            if candidate[1] in PLATE_CHARS and all(c in PLATE_CHARS for c in candidate[2:]):
                return candidate
    return None


def _merge_overlap(a, b):
    max_overlap = min(len(a), len(b))
    for i in range(max_overlap, 0, -1):
        if a[-i:] == b[:i]:
            return a + b[i:]
    return a + b


def _merge_texts(texts):
    fragments = []
    for item in texts:
        if "bbox" not in item or item["confidence"] < 0.5:
            t = re.sub(r"[·\.\-\s\*]", "", item["text"])
            if t:
                fragments.append({"text": t, "bbox": None, "conf": item["confidence"]})
            continue
        t = re.sub(r"[·\.\-\s\*]", "", item["text"])
        if not t:
            continue
        bbox = item["bbox"]
        fragments.append({
            "text": t,
            "x1": bbox[0][0], "x2": bbox[1][0],
            "y1": bbox[0][1], "y2": bbox[2][1],
            "conf": item["confidence"]
        })
    fragments.sort(key=lambda f: (f.get("y1", 0) if f.get("x1") is not None else 0, f.get("x1", 0) if f.get("x1") is not None else 0))
    rows = []
    current_row = []
    prev_y2 = -100
    for f in fragments:
        if f.get("x1") is None:
            current_row.append(f)
            continue
        if not current_row or (f["y1"] - prev_y2) < 20:
            current_row.append(f)
        else:
            rows.append(current_row)
            current_row = [f]
        prev_y2 = f["y2"]
    if current_row:
        rows.append(current_row)
    merged = []
    for row in rows:
        row.sort(key=lambda f: f.get("x1", 0) if f.get("x1") is not None else 0)
        combined = ""
        min_conf = 1.0
        for f in row:
            combined = _merge_overlap(combined, f["text"])
            min_conf = min(min_conf, f["conf"])
        merged.append({"text": combined, "confidence": min_conf})
    return merged


def _try_extract(texts):
    merged = _merge_texts(texts)
    all_candidates = texts + merged

    best = None
    best_score = 0
    province_parts = []
    other_parts = []
    for item in all_candidates:
        plate = _extract_plate(item["text"])
        if plate:
            score = item["confidence"] * (1.0 if len(plate) == 7 else 0.7)
            if score > best_score:
                best = plate
                best_score = score
        t = re.sub(r"[·\.\-\s\*]", "", item["text"])
        if t and t[0] in PROVINCES:
            province_parts.append(t)
        else:
            other_parts.append(t)
    if best and best_score >= 0.7:
        return best, best_score
    for op in other_parts:
        for pp in province_parts:
            plate = _extract_plate(pp + op)
            if plate:
                score = 0.5 * (1.0 if len(plate) == 7 else 0.7)
                if score > best_score:
                    best = plate
                    best_score = score
    return best, best_score


def _preprocess(image, method):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "otsu":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    if method == "otsu_pad":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        padded = cv2.copyMakeBorder(bw, 20, 20, 40, 40, cv2.BORDER_CONSTANT, value=255)
        return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    if method == "otsu_dil":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(bw, kern, iterations=1)
        return cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    if method == "adpt":
        adpt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        return cv2.cvtColor(adpt, cv2.COLOR_GRAY2BGR)
    if method == "adpt51_dil":
        adpt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 15)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        dilated = cv2.dilate(adpt, kern, iterations=1)
        return cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    if method == "adpt61_dil":
        adpt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 20)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        dilated = cv2.dilate(adpt, kern, iterations=1)
        return cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    return image


_FAST_METHODS = ["otsu"]
_MID_METHODS = ["adpt", "otsu_dil", "otsu_pad"]
_SLOW_METHODS = ["adpt51_dil", "adpt61_dil"]


def _recognize_frame(image):
    """Core recognition with tiered early-exit strategy.
    
    Returns:
        tuple: (plate_text, confidence, None)
    """
    all_texts = []

    texts = _ocr(image)
    all_texts.extend(texts)
    plate, score = _try_extract(all_texts)
    if plate and score >= 0.9:
        return plate, score, None

    for method in _FAST_METHODS:
        preprocessed = _preprocess(image, method)
        texts = _ocr(preprocessed)
        all_texts.extend(texts)
    plate, score = _try_extract(all_texts)
    if plate and score >= 0.85:
        return plate, score, None

    for method in _MID_METHODS:
        preprocessed = _preprocess(image, method)
        texts = _ocr(preprocessed)
        all_texts.extend(texts)
    plate, score = _try_extract(all_texts)
    if plate and score >= 0.7:
        return plate, score, None

    for method in _SLOW_METHODS:
        preprocessed = _preprocess(image, method)
        texts = _ocr(preprocessed)
        all_texts.extend(texts)
    plate, score = _try_extract(all_texts)
    if plate:
        return plate, score, None

    return None, 0.0, None


def recognize_plate(image_input):
    """识别车牌号。

    Args:
        image_input: 图片文件路径(str) 或 OpenCV 图像 (numpy.ndarray)。

    Returns:
        dict: {"plate": str|None, "confidence": float, "time_ms": float}
    """
    t0 = time.perf_counter()

    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            return {"plate": None, "confidence": 0.0, "time_ms": (time.perf_counter() - t0) * 1000}
    else:
        image = image_input

    h, w = image.shape[:2]
    max_dim = 1280
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    plate, score, _ = _recognize_frame(image)
    elapsed = (time.perf_counter() - t0) * 1000

    return {"plate": plate, "confidence": round(score, 2), "time_ms": round(elapsed, 1)}


def run_camera(camera_index=0):
    """启动 USB 摄像头实时识别。

    Args:
        camera_index: 摄像头设备索引，默认 0。
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"无法打开摄像头 (index={camera_index})")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_plate = None
    last_conf = 0.0
    last_ms = 0.0
    printed_plate = None
    frame_count = 0
    PROCESS_INTERVAL = 3
    processing = False
    result_lock = threading.Lock()
    shared_result = {"plate": None, "confidence": 0.0, "time_ms": 0.0}

    def _process_async(frame):
        nonlocal processing
        try:
            result = recognize_plate(frame)
            with result_lock:
                shared_result["plate"] = result["plate"]
                shared_result["confidence"] = result["confidence"]
                shared_result["time_ms"] = result["time_ms"]
        finally:
            processing = False

    print("摄像头已启动，按 ESC 或 Q 退出...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        frame_count += 1

        if frame_count % PROCESS_INTERVAL == 0 and not processing:
            processing = True
            t = threading.Thread(target=_process_async, args=(frame.copy(),), daemon=True)
            t.start()

        with result_lock:
            last_plate = shared_result["plate"]
            last_conf = shared_result["confidence"]
            last_ms = shared_result["time_ms"]
            if last_plate and last_plate != printed_plate and last_conf >= 0.8:
                printed_plate = last_plate
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {last_plate}  conf:{last_conf:.2f}  {last_ms:.0f}ms")

        # Draw text with plate info
        if last_plate:
            label = f"{last_plate}  conf:{last_conf:.2f}  {last_ms:.0f}ms"
            display = draw_chinese_text(display, label, (10, 20), font_size=20, color=(0, 255, 0))
        else:
            display = draw_chinese_text(display, "未识别", (10, 20), font_size=20, color=(0, 0, 255))

        cv2.imshow("Temporary Plate Recognition - ESC to quit", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        if cv2.getWindowProperty("Temporary Plate Recognition - ESC to quit", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


def run_video(video_path):
    """识别视频文件中的临时车牌。

    Args:
        video_path: 视频文件路径。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(fps / 5))
    frame_count = 0
    results = {}

    print(f"视频: {video_path}  FPS: {fps:.1f}  每 {interval} 帧识别一次\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % interval != 0:
            continue

        result = recognize_plate(frame)
        plate = result["plate"]
        if plate and plate not in results:
            results[plate] = result
            ts = frame_count / fps
            print(f"  [{ts:6.1f}s] 帧#{frame_count:<6d}  {plate:<10s}  置信度: {result['confidence']:.2f}  耗时: {result['time_ms']:.0f}ms")

    cap.release()

    print(f"\n共处理 {frame_count} 帧，识别到 {len(results)} 个不同车牌:")
    for plate, result in results.items():
        print(f"  {plate}  置信度: {result['confidence']:.2f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--camera":
            idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            run_camera(idx)
        elif arg == "--video":
            if len(sys.argv) < 3:
                print("用法: python main.py --video <视频文件路径>")
                sys.exit(1)
            run_video(sys.argv[2])
        else:
            result = recognize_plate(arg)
            plate = result["plate"]
            tag = plate if plate else "未识别"
            print(f"{tag}  置信度: {result['confidence']:.2f}  耗时: {result['time_ms']:.0f}ms")
    else:
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")
        paths = sorted(
            glob.glob(os.path.join(test_dir, "*.png")),
            key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)),
        )
        if not paths:
            print(f"测试目录中没有图片: {test_dir}")
            sys.exit(1)

        print(f"找到 {len(paths)} 张图片，开始识别...\n")
        ok = 0
        total_ms = 0.0
        for p in paths:
            result = recognize_plate(p)
            plate = result["plate"]
            conf = result["confidence"]
            ms = result["time_ms"]
            total_ms += ms
            tag = f"{plate}" if plate else "未识别"
            print(f"{os.path.basename(p):>8s}  →  {tag:<10s}  置信度: {conf:.2f}  耗时: {ms:.0f}ms")
            if plate:
                ok += 1
        print(f"\n汇总: {ok}/{len(paths)} 张成功识别  总耗时: {total_ms:.0f}ms  平均: {total_ms / len(paths):.0f}ms/张")
