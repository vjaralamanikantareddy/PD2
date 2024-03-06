from flask import Flask, Response, render_template, request
import cv2
import numpy as np
import mediapipe as mp
import base64

app = Flask(__name__)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Flag to indicate if pose detection is active
pose_detection_active = False

@app.route('/')
def index():
    return render_template('index.html')

# Route to receive webcam frames from frontend
@app.route('/send_frame', methods=['POST'])
def receive_frame():
    # Decode base64 string to image
    frame_data = request.form['frame']
    frame_bytes = base64.b64decode(frame_data)
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform pose detection if active
    if pose_detection_active:
        frame = detect_pose(frame)

    # Encode frame to base64 string
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')

    return frame_base64

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

if __name__ == '__main__':
    app.run(debug=True)
