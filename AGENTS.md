# AGENTS.md

## Environment

- **Conda env**: `conda activate hyperlpr3` (Python 3.10)
- **Run**: `D:\Programs\anaconda3\envs\hyperlpr3\python.exe -u -X utf8 temp_plate_recog.py`
  - Windows `conda run` hits GBK encoding errors; invoke env Python directly with `-X utf8`

## Structure

- `temp_plate_recog.py` — entrypoint with `recognize_plate()` API and CLI
- `plate.py` — thin wrapper importing `recognize_plate` for quick tests
- `test/` — 26 PNG images (` 1.png`–` 25.png`, `26.png`) of temporary Chinese license plates

## CLI Usage

```bash
# Batch test all images in test/
python temp_plate_recog.py

# Single image
python temp_plate_recog.py path/to/image.png

# USB camera (async recognition in background thread)
python temp_plate_recog.py --camera [index]

# Video file (samples at 5 FPS)
python temp_plate_recog.py --video path/to/video.mp4
```

## Recognition Pipeline

Tiered early-exit (all offline, no network):

1. **HyperLPR3** — plate region detection (limited support for temporary paper plates)
2. **RapidOCR** — full image, then preprocessed variants (OTSU, adaptive threshold, dilation)
3. **Text merging** — merges province prefix + fragmented digit sequences
4. **Plate extraction** — regex: province char + letter + 3-6 alphanumeric chars

Confidence thresholds: Tier 1 >= 0.9, Tier 2 >= 0.7, Tier 3 (slowest) runs last.

## Dependencies

See `requirements.txt`: `hyperlpr3`, `rapidocr-onnxruntime`, `opencv-python`, `onnxruntime`, `Pillow`
