@app.route('/detect', methods=['POST'])
def detect():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        results = model.predict(img)

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
    except Exception as e:
        # Return the actual Python error
        return jsonify({"status": "error", "message": str(e)}), 500
