from flask import Flask, Response, render_template, redirect, url_for, send_file, request, flash
from PIL import Image, ImageDraw, ImageFont, ImageGrab
import numpy as np
import onnxruntime as ort
import os
from io import BytesIO

app = Flask(__name__)

# Load the ONNX model
ort_session = ort.InferenceSession('trained_weights/exp1_yolov8n_trained.onnx')

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
    
    # Resize the image to 640x640 before inference
    original_size = image.size
    image = image.resize((640, 640))
    image_np = np.array(image)

    # Preprocess the image
    input_image = image_np.astype(np.float32)
    input_image = np.transpose(input_image, (2, 0, 1))  # Convert to CHW format
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_image}
    ort_outs = ort_session.run(None, ort_inputs)

    # Process the results
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for result in ort_outs[0]:
        x1, y1, x2, y2, conf, cls = result[:6]
        label = f'{int(cls)} {float(conf):.2f}' 
        # Adjust coordinates back to original image size
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(lambda x: int(x.item()), [x1, y1, x2, y2])

        # Ensure coordinates are in the correct order
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Scale coordinates back to original image size
        x1 = int(x1 * original_size[0] / 640)
        y1 = int(y1 * original_size[1] / 640)
        x2 = int(x2 * original_size[0] / 640)
        y2 = int(y2 * original_size[1] / 640)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        draw.text((x1, y1 - 10), label, fill="green", font=font)

    # Convert the image to a format that can be sent as a response
    io_buf = BytesIO()
    image.save(io_buf, format='JPEG')
    io_buf.seek(0)
    return send_file(io_buf, mimetype='image/jpeg')



def generate_frames():
    while True:
        # Capture frame-by-frame using ImageGrab
        frame = ImageGrab.grab()
        frame = frame.resize((640, 640))
        frame_np = np.array(frame)

        # Preprocess the image
        input_image = frame_np.astype(np.float32)
        input_image = np.transpose(input_image, (2, 0, 1))  # Convert to CHW format
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

        # Run inference
        ort_inputs = {ort_session.get_inputs()[0].name: input_image}
        ort_outs = ort_session.run(None, ort_inputs)

        # Process the results
        draw = ImageDraw.Draw(frame)
        font = ImageFont.load_default()
        for result in ort_outs[0]:
            x1, y1, x2, y2, conf, cls = result[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Ensure coordinates are integers
            label = f'{int(cls[0])} {float(conf[0]):.2f}'  # Extract single values correctly
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, y1 - 10), label, fill="green", font=font)

        # Convert the image to bytes
        io_buf = BytesIO()
        frame.save(io_buf, format='JPEG')
        frame_bytes = io_buf.getvalue()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live')
def live():
    return render_template('live.html')

if __name__ == '__main__':
    app.run(debug=True)
