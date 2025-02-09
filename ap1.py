import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
import time

# Cache model to prevent reloading
@st.cache_resource()
def load_model():
    return YOLO('yolov8s.pt')

yolo = load_model()

# Function to get class colors
def get_colours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [
        base_colors[color_index][i] +
        increments[color_index][i] * (cls_num // len(base_colors)) % 256
        for i in range(3)
    ]
    return tuple(color)

st.title("🚀 YOLO for contextual object perception ")

# Sidebar settings
st.sidebar.title("⚙️ Settings")
source_option = st.sidebar.radio("Select Input Source:", ("Webcam", "Upload Image"))

# Webcam detection
if source_option == "Webcam":
    if st.sidebar.button("Start Detection"):
        st.write("Starting Webcam... Press 'Stop' to end detection.")
        videoCap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Reduce latency on Windows
        videoCap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        videoCap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        videoCap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS limit
        
        FRAME_WINDOW = st.empty()
        stop_button = st.sidebar.button("Stop")

        prev_time = 0
        FRAME_RATE = 10  # Adjust this to control processing speed

        while not stop_button:
            ret, frame = videoCap.read()
            if not ret:
                st.error("Error accessing webcam.")
                break

            # Limit frame rate to improve performance
            curr_time = time.time()
            if curr_time - prev_time < 1 / FRAME_RATE:
                continue  # Skip frame processing if time threshold not met
            prev_time = curr_time

            frame = cv2.resize(frame, (640, 480))  # Resize for performance boost
            results = yolo.track(frame, stream=True)

            for result in results:
                classes_names = result.names
                for box in result.boxes:
                    if box.conf[0] > 0.4:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        class_name = classes_names[cls]
                        colour = get_colours(cls)

                        # Draw bounding boxes and labels
                        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                        cv2.putText(
                            frame,
                            f'{class_name} {box.conf[0]:.2f}',
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            colour,
                            2,
                        )

            # Convert frame to RGB and update UI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb, channels="RGB")

        videoCap.release()

# Image Upload Detection
if source_option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Run YOLO detection
        results = yolo.track(img, stream=True)

        for result in results:
            classes_names = result.names
            for box in result.boxes:
                if box.conf[0] > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    class_name = classes_names[cls]
                    colour = get_colours(cls)

                    # Draw bounding boxes and labels
                    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(
                        img,
                        f'{class_name} {box.conf[0]:.2f}',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        colour,
                        2,
                    )

        # Convert frame to RGB and display results
        result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(result_img, caption="Detection Results", use_column_width=True)
