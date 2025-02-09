YOLO Object Detection with Streamlit and OpenCV

This project showcases real-time object detection using YOLOv8 from Ultralytics, combined with OpenCV for image processing and Streamlit for an interactive web-based interface. Users can detect objects via webcam or by uploading images.

Features:

Real-time detection using your webcam.

Upload images to perform object detection.

Custom color schemes for bounding boxes based on object classes.

Simple and interactive Streamlit interface.





Requirements

Python 3.7+

OpenCV

Ultralytics YOLO

Streamlit

NumPy


Installation

1. Clone the Repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


2. Install Dependencies:


pip install opencv-python ultralytics streamlit numpy


3. Download YOLOv8 Model: The script automatically downloads yolov8s.pt if it's not present in your directory.



Usage

1. Run the Streamlit App:

streamlit run detect.py


2. Select Input Source:

Webcam: Choose Webcam in the sidebar and click Start Detection.

Upload Image: Select Upload Image and upload an image (.jpg, .jpeg, .png) to detect objects.



3. Stop Webcam Detection: Click the Stop button in the sidebar to end webcam detection.



Project Structure

your-repo-name/
│
├── detect.py           # Streamlit app with YOLO detection
├── models/             # YOLO models (if any custom ones are added)
├── demo.gif            # Optional demo of the app in action
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

Customization

Change Detection Confidence:
Modify the confidence threshold in the code (if box.conf[0] > 0.4) to increase or decrease detection sensitivity.

Add Custom YOLO Models:
Replace yolov8s.pt with your own trained YOLO model for specific object detection.


Known Issues

Webcam Latency on Some Systems:
If you're facing high latency on Windows, cv2.CAP_DSHOW is used to reduce lag.

Performance Adjustments:
You can modify FRAME_RATE and cv2.resize() to optimize performance on lower-end systems.


Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request with improvements or fixes.

License

This project is licensed under the MIT License.


