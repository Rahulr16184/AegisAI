from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# Load your trained YOLO model (replace 'best.pt' with your model path)
model = YOLO("best.pt")   # <-- your trained weapon detection model

@app.route('/')
def home():
    return "API is running 🚀"

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Convert file to OpenCV image
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Run detection
    results = model.predict(img)

    # Extract labels
    detections = []
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            detections.append(label)

    return jsonify({
        "status": "success",
        "message": "Detection complete",
        "detections": detections
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
