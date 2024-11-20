# Import libraries
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
import cv2
from ultralytics import YOLO

# Load trained YOLOv8 model
model = YOLO('trained_weights/exp1_yolov8n_trained.onnx', task='detect')  # Make sure the path to your model file is correct

# To initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# To capture image from webcam
def capture_image():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return None
    # Convert image to frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return img

# To perform inference on the webcam image and draw bounding boxes
def perform_inference(img):
    # run inference
    results = model.predict(img)

    # Draw bounding boxes on image
    draw = ImageDraw.Draw(img)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cls = int(box.cls.item())  
            conf = float(box.conf.item()) 
            label = f"{model.names[cls]}, prob:{conf:.2f}" 
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, y1 - 10), label, fill="green")

    return img

# Update GUI with latest frame
def update_frame():
    img = capture_image()
    if img is not None:
        img_with_boxes = perform_inference(img)
        img_tk = ImageTk.PhotoImage(img_with_boxes)
        label.config(image=img_tk)
        label.image = img_tk
    root.after(10, update_frame)

# To create the GUI window using tkteach
root = tk.Tk()
root.title("Live Prediction")

# Give label to the image
label = tk.Label(root)
label.pack()

# Start the prediction
update_frame()

# Run the GUI
root.mainloop()

# To stop webcam
cap.release()
