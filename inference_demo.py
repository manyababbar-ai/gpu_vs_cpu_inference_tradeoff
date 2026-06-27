import time
import json
from ultralytics import YOLO
model=YOLO("yolov8n.pt")
# CPU inference
start=time.perf_counter()
results=model.predict("test.jpg",device='cpu')
end=time.perf_counter()
inference_time=(end-start)*1000
output={
    "inference_time_ms": inference_time,
    "hardware_used": 'cpu',  
    "recommendation": 'use GPU for production'
}
print(json.dumps(output,indent=4))
# GPU inference
try:
    start=time.perf_counter()

    results=model.predict("test.jpg",device=0)

    end=time.perf_counter()
    inference_time=(end-start)*1000
    output={
        "inference_time_ms": inference_time,
        "hardware_used": 'gpu',  
        "recommendation": 'keep using GPU'
    }
    print(json.dumps(output,indent=4))
except Exception as e:
    print(f"Error occurred while running GPU inference: {e}")
    results=None