from flask import Flask, request, jsonify
from flask_cors import CORS   # <--- import CORS

app = Flask(__name__)
CORS(app)   # <--- enable CORS for all routes

@app.route('/')
def home():
    return "API is running 🚀"

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files.get('image')

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # TEMP: dummy response
    return jsonify({
        "status": "success",
        "message": "Image received",
        "detections": ["gun"]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
