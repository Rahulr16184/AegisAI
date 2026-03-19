from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import os
import io

app = FastAPI()

# ✅ CORS (IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔧 CONFIG
model = None
CONF_THRESHOLD = 0.5

# ✅ Load model once (FASTER)
def get_model():
    global model
    if model is None:
        model = YOLO("best.pt")
        model.fuse()  # 🔥 speed boost
    return model

# ✅ Health check
@app.get("/")
def home():
    return {"message": "Weapon Detection API Running"}

# ✅ JSON RESPONSE
@app.post("/detect")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        model = get_model()

        # 🔥 FAST inference
        results = model.predict(img, imgsz=224, stream=False)

        detections = []
        names = model.names

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])

                if conf > CONF_THRESHOLD:
                    detections.append({
                        "class": names[int(box.cls[0])],
                        "confidence": conf,
                        "bbox": box.xyxy[0].tolist()
                    })

        print("Detections:", detections)  # 🔥 debug log

        return {
            "status": "success",
            "count": len(detections),
            "detections": detections
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# ✅ IMAGE RESPONSE
@app.post("/detect-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        model = get_model()
        results = model.predict(img, imgsz=224)

        names = model.names

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])

                if conf > CONF_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{names[int(box.cls[0])]} {conf:.2f}"

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    cv2.putText(
                        img,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )

        _, buffer = cv2.imencode(".jpg", img)

        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )

    except Exception as e:
        return {"status": "error", "message": str(e)}

# ✅ Render start
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
