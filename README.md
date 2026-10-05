# vision-stack

Day-1 CPU perception pipeline for a 90-day public Optimus software sprint. It reads a webcam, a video file, or a still image, runs a lightweight detector, writes an annotated result, and logs per-frame latency to CSV.

The same process runs on a MacBook Air tonight and on a GPU tower later. `device` is a setting (`cpu`, `cuda`, or `mps`) passed through to Ultralytics. The default is `cpu`.

## Quickstart (Mac, CPU)

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py --source data/sample/synthetic_demo.mp4 --device cpu
```

That writes:

- `outputs/annotated.mp4` — frames with boxes and a timing line
- `outputs/latency.csv` — one row per frame

The first YOLO run downloads `yolov8n.pt` (about 6 MB) into the working directory. The file is gitignored.

Python 3.10 or newer. `pip` installs PyTorch via Ultralytics and picks the wheel for the machine you are on.

## Sample commands

Webcam, short capture (Ctrl+C also stops and keeps the CSV):

```bash
python scripts/run_pipeline.py --source 0 --device cpu --max-frames 120
```

Faster CPU pass on a file you already have:

```bash
python scripts/run_pipeline.py --source path/to/clip.mp4 --device cpu --imgsz 320 --max-frames 60
```

Still image (writes `outputs/annotated.jpg` plus one CSV row):

```bash
python scripts/run_pipeline.py --source path/to/photo.jpg --device cpu
```

GPU tower, same code path:

```bash
python scripts/run_pipeline.py --source 0 --device cuda
```

Apple GPU on a machine where PyTorch reports MPS:

```bash
python scripts/run_pipeline.py --source 0 --device mps
```

Regenerate the checked-in sample clip:

```bash
python scripts/make_sample_video.py
```

`data/sample/synthetic_demo.mp4` is a generated moving-shape clip, so the repo does not ship third-party footage. The moving circle is often scored as a `sports ball`, which is enough to see a box and a non-zero `num_detections`. Point `--source` at a webcam or a photograph for a real scene.

`outputs/annotated.mp4` is MPEG-4 (`mp4v`), which is what the OpenCV wheel writes. VLC and IINA open it if Preview does not.

## Configuration

Precedence is CLI flag, then environment variable, then `configs/default.yaml`.

| Knob | CLI | Environment | Default |
| --- | --- | --- | --- |
| device | `--device` | `VISION_STACK_DEVICE` | `cpu` |
| model | `--model` | `VISION_STACK_MODEL` | `yolov8n.pt` |
| source | `--source` | `VISION_STACK_SOURCE` | `data/sample/synthetic_demo.mp4` |
| backend | `--backend` | `VISION_STACK_BACKEND` | `yolo` |
| confidence | `--confidence` | | `0.25` |
| imgsz | `--imgsz` | | `640` |
| max frames | `--max-frames` | | `0` (whole file, or until Ctrl+C) |
| output dir | `--out` | | `outputs/` |

`--backend yolo` loads Ultralytics. `--backend mobilenet_ssd` loads OpenCV DNN and leaves `device` for the YOLO path (see below).

## Latency CSV

`outputs/latency.csv` columns:

| Column | Meaning |
| --- | --- |
| `frame_idx` | 0-based index in this run |
| `timestamp_ms` | Capture time in milliseconds. File position when OpenCV reports it, otherwise `frame_idx / fps` |
| `infer_ms` | Time inside the detector call: preprocess, forward, and postprocess |
| `total_ms` | `infer_ms` plus draw and write for that frame |
| `num_detections` | Boxes kept at the configured confidence |

Frame 0 is usually a warmup outlier. Keep it in the file and use the median from the printed summary when you compare runs.

Example:

```csv
frame_idx,timestamp_ms,infer_ms,total_ms,num_detections
0,0.000,312.450,318.102,1
1,100.000,28.113,30.441,1
```

## Architecture

```
source → FrameSource → detector.predict → draw → annotated.mp4 / annotated.jpg
                            │                              │
                            └── infer_ms ──────────────────┴── total_ms → latency.csv
```

`src/vision_stack/capture.py` only opens a camera or a file. `detect.py` chooses the backend. `YoloDetector` calls `model.predict(..., device=settings.device)`. Swapping the laptop for the tower is `--device cuda` (or `device: cuda` in the YAML). CPU stays the default, so a fresh checkout runs with the quickstart above.

## Lighter fallback (OpenCV DNN)

If the Ultralytics install is too heavy for the machine you are on, install the small stack and use MobileNet-SSD. It writes the same annotated output and the same CSV columns. Inference in this backend uses OpenCV's CPU DNN runtime.

```bash
pip install numpy==2.4.6 opencv-python==4.14.0.94 PyYAML==6.0.3
python scripts/download_mobilenet_ssd.py
python scripts/run_pipeline.py --source data/sample/synthetic_demo.mp4 --device cpu --backend mobilenet_ssd
```

The download script fetches the public Caffe MobileNet-SSD prototxt and caffemodel (VOC labels, including `person`) into `models/`. Those weights are gitignored.

## Layout

```
configs/default.yaml          device, model, confidence, max_frames
src/vision_stack/capture.py   webcam, video, still
src/vision_stack/detect.py    backend selection
src/vision_stack/pipeline.py  annotate + write
src/vision_stack/latency.py   CSV log
scripts/run_pipeline.py       CLI
data/sample/                  synthetic demo clip
outputs/                      annotated media and latency.csv
```

## Week 1

1. Run the quickstart, then a 120-frame webcam capture, and keep both `latency.csv` files.
2. Record median `infer_ms` at `--imgsz 640` and `--imgsz 320` on the laptop CPU. Treat frame 0 as warmup.
3. Film one short handheld clip and set `confidence` in `configs/default.yaml` once person boxes stay stable.
4. Re-run that same clip with `--backend mobilenet_ssd` and compare `infer_ms` and `num_detections`.
5. On the GPU machine, change the command to `--device cuda` and diff the CSV against the CPU run.

## Tests

Unit tests cover capture, the CSV schema, and the pipeline with a stand-in detector (no weights):

```bash
python -m unittest discover -s tests
```
