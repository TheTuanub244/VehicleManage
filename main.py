from flask import Flask, request, jsonify
import time
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
import base64
from paddleocr import PaddleOCR
from ultralytics import YOLO

import requests
import cv2
app = FastAPI()
SERVER_URL = "https://vehicle-manage.vercel.app/api/access-logs"  # thay bằng server thực tế
API_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI2OGQ3NGUzZjViNTQ1YzExMmI2MzQ0YTUiLCJ1c2VybmFtZSI6Im1haXRoZXR1YW4iLCJyb2xlIjoiYWRtaW4iLCJpYXQiOjE3NTg5ODY3MTgsImV4cCI6MTc1OTA3MzExOH0.Z-iyHMeP0WMa8CvsfNMtjj7qi5_PLxEtINyI35VUvyI"
track_plate = YOLO("Yolo11SegPlate.pt")
track_vehicle = YOLO("yolo11n.pt")
ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            device='cpu',
        )
def point_position(x, y, x1, y1, x2, y2):
    cross = (x2-x1)*(y-y1) - (y2-y1)*(x-x1)
    if cross > 0:
        return 1   
    elif cross < 0:
        return -1  
    else:
        return 0 

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    lines = [0, 300, 1200, 600]
    pos_prev = {}
    barrier_open = False
    last_seen_time = 0
    vehicle_data = {}
    HOLD_TIME = 10
    cap = cv2.VideoCapture('cut_video.mp4')  
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_proc = time.time()
        vehicle_detect = False
        output = frame.copy()
        results = track_vehicle.track(frame, imgsz=1280, verbose=False, persist=True)
        r = results[0]
        if results and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                track_id = int(box.id[0]) if box.id is not None else None
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if track_id is None:
                    continue
                if conf < 0.75:
                    continue
        
                if r.names[cls_id] in ["car", "truck", "motorcycle"]:
                    plate_text = "29A12345"
                    roi_car = frame[y1:y2, x1:x2]
                    plates = track_plate(roi_car, verbose=False)
                    pos = point_position(cx, cy, lines[0], lines[1], lines[2], lines[3])
                    width, height = x2 - x1, y2 - y1
                    if track_id not in vehicle_data:
                        recognition_dict = {
                            "licensePlate": plate_text,
                            "action": "entry",           # hoặc "exit" tùy logic
                            "gateId": "GATE_001",
                            "gateName": "Cổng chính",
                            "recognitionData": {
                                "confidence": round(conf, 2),
                                "processedImage": f"frames/{track_id}.jpg",  
                                "processingTime": round((time.time() - start_proc) * 1000),  # ms
                                "boundingBox": {
                                    "x": int(x1),
                                    "y": int(y1),
                                    "width": int(width),
                                    "height": int(height)
                                }
                            },
                            "positions": [pos]
                        }

                        vehicle_data[track_id] = recognition_dict
                    else:
                        vehicle_data[track_id]["licensePlate"] = plate_text
                        vehicle_data[track_id]["recognitionData"]["confidence"] = round(conf, 2)
                        vehicle_data[track_id]["recognitionData"]["processingTime"] = round((time.time() - start_proc) * 1000)
                        vehicle_data[track_id]["recognitionData"]["boundingBox"] = {
                            "x": int(x1),
                            "y": int(y1),
                            "width": int(width),
                            "height": int(height)
                        }
                        vehicle_data[track_id]["positions"].append(pos)
                    if len(vehicle_data[track_id]["positions"]) >= 2:
                        if vehicle_data[track_id]["positions"][-1] != vehicle_data[track_id]["positions"][-2]:
                            vehicle_detect = True
                            barrier_open = True
                            print(f"Xe {track_id} ({plate_text}) vừa cắt qua đường!")
                            _, buffer = cv2.imencode('.jpg', frame)
                            jpg_as_text = base64.b64encode(buffer).decode("utf-8")

                            await websocket.send_json({
                                "image": jpg_as_text,
                                "plate": vehicle_data[track_id]["licensePlate"]
                            })

                            await asyncio.sleep(0.05)  # ~20 FPS
                    pos_prev[track_id] = pos
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(output, f"ID {track_id}: {r.names[cls_id]}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        if vehicle_detect:
            last_seen_time = time.time()
        else:
            if time.time() - last_seen_time > HOLD_TIME:
                barrier_open = False
        if barrier_open:
            cv2.putText(output, "Barrier: OPEN", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        else:
            cv2.putText(output, "Barrier: CLOSED", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        cv2.line(output, (lines[0], lines[1]), (lines[2], lines[3]), (0,0,255), 2)

        
    cap.release()
@app.get("/")
async def get():
    return HTMLResponse("""
    <!DOCTYPE html>
<html>
<body>
    <h2>Vehicle Plate Detection</h2>
    <img id="video" width="640" height="400"/>   
    <p id="plate"></p>
    <script>
        var ws = new WebSocket("ws://localhost:8000/ws");
        ws.onmessage = function(event) {
            var data = JSON.parse(event.data);
            document.getElementById("video").src = "data:image/jpeg;base64," + data.image;
            if (data.plate) {
                document.getElementById("plate").innerText = "Plate: " + data.plate;
            }
        };
    </script>
</body>
</html>
    """)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
