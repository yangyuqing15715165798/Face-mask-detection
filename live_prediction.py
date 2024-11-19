from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('exp1_yolov8n_trained.pt')  # Make sure the path to your model file is correct


# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to capture image from webcam
def capture_image():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return None
    # Convert the frame to a PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return img

# Function to perform inference on an image and draw bounding boxes
def perform_inference(img):
    # Perform inference
    results = model.predict(img)

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(img)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy)
            label = f"{box.cls} {box.conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, y1 - 10), label, fill="green")

    return img

# Function to update the GUI with the latest frame
def update_frame():
    img = capture_image()
    if img is not None:
        img_with_boxes = perform_inference(img)
        img_tk = ImageTk.PhotoImage(img_with_boxes)
        label.config(image=img_tk)
        label.image = img_tk
    root.after(10, update_frame)

# Create the GUI window
root = tk.Tk()
root.title("Live Prediction")

# Create a label to display the image
label = tk.Label(root)
label.pack()

# Start the update loop
update_frame()

# Run the GUI loop
root.mainloop()

# Release the webcam
cap.release()
