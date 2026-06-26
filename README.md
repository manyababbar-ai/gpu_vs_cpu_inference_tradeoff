# GPU vs CPU Inference Tradeoffs — CV12

## Overview
This project compares GPU and CPU performance for running 
computer vision model inference using YOLOv8n.


## What This Project Does
- Runs YOLOv8n object detection on a test image using CPU
- Measures inference time in milliseconds
- Outputs structured JSON with hardware info and recommendation

## Output Example
{
    "inference_time_ms": 939.72,
    "hardware_used": "cpu",
    "recommendation": "use GPU for production"
}

## Files
| File | Description |
|------|-------------|
| inference_task.py | Main script — runs YOLOv8n and measures inference time |
| test.jpg | Sample image used for inference |

## How to Run
1. Install uv
2. Run: uv add ultralytics
3. Run: uv run inference_task.py

## Key Findings
- CPU inference: 80.4ms per image
- GPU inference: 0.99ms per image
- GPU is 81x faster — recommended for real-time production use

## Tech Stack
- Python
- YOLOv8n (Ultralytics)
- uv (package manager)