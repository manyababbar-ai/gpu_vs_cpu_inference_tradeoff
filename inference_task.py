import time
import json
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
image = "test.jpg"
start = time.perf_counter()
results = model.predict(image)
end = time.perf_counter()
inference_time = (end - start) * 1000
output= {
    "inference_time_ms": inference_time,
    "hardware_used": 'cpu' ,
    "recommendation": 'use GPU for production' 
}
print(json.dumps(output,indent=4))