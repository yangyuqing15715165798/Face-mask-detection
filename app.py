from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# Load the trained YOLOv8 model
model = YOLO('trained_weights/yolov8n_trained.onnx')  # Make sure the path to your model file is correct

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))

    # Perform inference
    results = model.predict(img)

    # Extract predictions
    predictions = results[0].boxes

    # Convert predictions to a list of dictionaries
    predictions_list = []
    for box in predictions:
        predictions_list.append({
            'class': int(box.cls),
            'confidence': float(box.conf),
            'x1': float(box.xyxy[0]),
            'y1': float(box.xyxy[1]),
            'x2': float(box.xyxy[2]),
            'y2': float(box.xyxy[3])
        })

    return jsonify(predictions_list)

if __name__ == '__main__':
    app.run(debug=True)
