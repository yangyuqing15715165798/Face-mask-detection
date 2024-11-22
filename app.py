from flask import Flask, Response, render_template, redirect, url_for, send_file, request, flash
from PIL import Image, ImageDraw, ImageFont, ImageGrab
import numpy as np
import os
from io import BytesIO
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load the ONNX model
model = YOLO('trained_weights/exp1_yolov8n_trained.onnx', task='detect') 
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            return redirect(url_for('predict', filename=file.filename))
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)
    return render_template('index.html')


@app.route('/predict/<filename>')
def predict(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image = Image.open(filepath)
    
    # Ensure the image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize the image to 640x640 before inference
    original_size = image.size
    image_resized = image.resize((640, 640))
    image_np = np.array(image_resized)

    # Preprocess the image
    input_image = image_np.astype(np.float32) / 255.0  # Normalize pixel values
    input_image = np.transpose(input_image, (2, 0, 1))  # Convert to CHW format
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

    # Run inference
    results = model.predict(image_resized)

    # Process the results
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls.item())  # Convert tensor to int
            conf = float(box.conf.item())  # Convert tensor to float
            label = f"{model.names[cls]}, prob:{conf:.2f}"  # Format label as requested
            
            # Adjust coordinates back to original image size
            x1 = int(x1 * original_size[0] / 640)
            y1 = int(y1 * original_size[1] / 640)
            x2 = int(x2 * original_size[0] / 640)
            y2 = int(y2 * original_size[1] / 640)

            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, y1 - 10), label, fill="green")
    
    # Convert the image to a format that can be sent as a response
    io_buf = BytesIO()
    image.save(io_buf, format='JPEG')
    io_buf.seek(0)
    return send_file(io_buf, mimetype='image/jpeg')

@app.route('/live')
def live():
    return render_template('live.html')

if __name__ == '__main__':
    app.run(debug=True)