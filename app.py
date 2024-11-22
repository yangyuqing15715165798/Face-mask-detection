from flask import Flask, request, render_template, redirect, url_for, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the trained YOLOv8 model
model = YOLO('trained_weights/exp1_yolov8n_trained.pt')  # Make sure the path to your trained model file is correct

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            return redirect(url_for('predict', filename=file.filename))
    return render_template('index.html')

@app.route('/predict/<filename>')
def predict(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image = cv2.imread(filepath)
    
    # Get predictions from the model
    results = model.predict(source=filepath, save=False)
    
    # Draw bounding boxes and labels on the image
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Convert the image to a format that can be sent as a response
    _, buffer = cv2.imencode('.jpg', image)
    io_buf = BytesIO(buffer)
    return send_file(io_buf, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
