from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Function to capture video frames
def generate_frames():
    camera = cv2.VideoCapture(0) # Access the first camera (index 0)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")
    
    while True:
        success, frame = camera.read()
        if not success:
            raise RuntimeError("Failed to read frame from camera.")
        
        # Perform pose detection
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
