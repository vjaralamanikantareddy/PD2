# app.py

from flask import Flask, Response, render_template, request
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Flag to indicate if pose detection is active
pose_detection_active = False

@app.route('/')
def index():
    return render_template('index.html')

# Function to capture video frames
def generate_frames():
    camera = cv2.VideoCapture(0)  # Access the first camera (index 0)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")
    
    while True:
        success, frame = camera.read()
        if not success:
            raise RuntimeError("Failed to read frame from camera.")
        
        # Perform pose detection if active
        if pose_detection_active:
            frame = detect_pose(frame)
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Function to detect poses
def detect_pose(frame):
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and get the pose landmarks
    results = pose.process(rgb_frame)
    
    # Draw the landmarks on the frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
    return frame

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global pose_detection_active
    pose_detection_active = True
    return "Pose detection started."

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global pose_detection_active
    pose_detection_active = False
    return "Pose detection stopped."

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
