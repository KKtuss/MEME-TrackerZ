"""
Full MEME Tracker Web Application with all features
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
import os
import base64
import numpy as np
import time
from typing import Dict, Optional

# Setup logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import OpenCV, fallback to mock detection if not available
try:
    import cv2
    OPENCV_AVAILABLE = True
    logger.info("OpenCV imported successfully")
except ImportError as e:
    OPENCV_AVAILABLE = False
    logger.warning(f"OpenCV not available: {e}. Using mock detection.")

# Try to import MediaPipe for hand tracking
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe imported successfully")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    logger.warning(f"MediaPipe not available: {e}. Hand tracking disabled.")

app = FastAPI(title="MEME Tracker Web", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# User sessions storage
user_sessions: Dict[str, Dict] = {}

@app.get("/")
async def read_root():
    """Serve the main web interface"""
    return HTMLResponse(content=get_html_content())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "MEME Tracker is running!"}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time detection"""
    await websocket.accept()
    logger.info(f"Client {client_id} connected")
    
    # Initialize user session
    user_sessions[client_id] = {
        "images": {},
        "current_expression": None,
        "last_valid_expression": None,
        "auto_trigger": True
    }
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            # Process frame and return results
            result = await process_frame(frame_data, client_id)
            await websocket.send_text(json.dumps(result))
            
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
        if client_id in user_sessions:
            del user_sessions[client_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

async def process_frame(frame_data: dict, client_id: str) -> dict:
    """Process a single frame and return detection results"""
    try:
        if OPENCV_AVAILABLE:
            return await process_frame_full(frame_data, client_id)
        else:
            return await process_frame_mock(frame_data, client_id)
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return {"error": str(e)}

async def process_frame_full(frame_data: dict, client_id: str) -> dict:
    """Process frame with full detection capabilities"""
    # Decode base64 frame
    frame_bytes = base64.b64decode(frame_data["frame"])
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "Invalid frame data"}
    
    # Get user session
    session = user_sessions.get(client_id, {"images": {}, "current_expression": None, "last_valid_expression": None})
    
    # Initialize cascades if not already done
    if not hasattr(process_frame_full, 'face_cascade'):
        process_frame_full.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if not hasattr(process_frame_full, 'eye_cascade'):
        process_frame_full.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    if not hasattr(process_frame_full, 'smile_cascade'):
        process_frame_full.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    # Initialize MediaPipe if available
    if MEDIAPIPE_AVAILABLE and not hasattr(process_frame_full, 'mp_hands'):
        process_frame_full.mp_hands = mp.solutions.hands
        process_frame_full.mp_drawing = mp.solutions.drawing_utils
        process_frame_full.hands = process_frame_full.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Face detection
    faces = process_frame_full.face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Initialize variables
    expression = None
    face_ratio = 0
    faces_detected = len(faces)
    is_smiling = False
    is_mouth_open = False
    eyes_closed = False
    gaze_direction = "center"
    hand_gestures = []
    
    if len(faces) > 0:
        # Get the largest face
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = largest_face
        
        # Calculate face size ratio
        face_area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        face_ratio = face_area / frame_area
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Eye detection
        eyes = process_frame_full.eye_cascade.detectMultiScale(face_roi, 1.1, 5)
        eyes_closed = len(eyes) == 0
        
        # Gaze direction (simplified)
        if len(eyes) >= 2:
            # Sort eyes by x position
            eyes_sorted = sorted(eyes, key=lambda eye: eye[0])
            left_eye = eyes_sorted[0]
            right_eye = eyes_sorted[1]
            
            # Calculate relative positions for gaze direction
            left_center = left_eye[0] + left_eye[2]//2
            right_center = right_eye[0] + right_eye[2]//2
            
            if left_center < w//3:
                gaze_direction = "left"
            elif right_center > 2*w//3:
                gaze_direction = "right"
            else:
                gaze_direction = "center"
        
        # Smile detection
        smiles = process_frame_full.smile_cascade.detectMultiScale(face_roi, 1.8, 20)
        is_smiling = len(smiles) > 0
        
        # Mouth opening detection (simplified)
        mouth_region = face_roi[int(h*0.6):int(h*0.9), int(w*0.2):int(w*0.8)]
        if mouth_region.size > 0:
            mouth_edges = cv2.Canny(mouth_region, 50, 150)
            mouth_ratio = np.sum(mouth_edges > 0) / mouth_region.size
            is_mouth_open = mouth_ratio > 0.02
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 1)
        
        # Draw smiles
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(frame, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 0, 255), 1)
    
    # Hand detection (if MediaPipe available)
    if MEDIAPIPE_AVAILABLE:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = process_frame_full.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                process_frame_full.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, process_frame_full.mp_hands.HAND_CONNECTIONS)
                
                # Simple gesture detection
                landmarks = hand_landmarks.landmark
                
                # Thumbs up detection
                if (landmarks[4].y < landmarks[3].y and  # Thumb tip above thumb joint
                    landmarks[8].y > landmarks[6].y and  # Index finger down
                    landmarks[12].y > landmarks[10].y and  # Middle finger down
                    landmarks[16].y > landmarks[14].y and  # Ring finger down
                    landmarks[20].y > landmarks[18].y):   # Pinky down
                    hand_gestures.append("thumbs_up")
                
                # Open hand detection
                fingers_up = 0
                for finger_tip, finger_pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
                    if landmarks[finger_tip].y < landmarks[finger_pip].y:
                        fingers_up += 1
                
                if fingers_up >= 4:
                    hand_gestures.append("open_hand")
                elif fingers_up <= 1:
                    hand_gestures.append("fist")
    
    # Determine expression based on all detections
    expression = determine_expression(
        eyes_closed, gaze_direction, is_smiling, is_mouth_open, 
        hand_gestures, face_ratio, session["images"]
    )
    
    # Update session
    if expression != session["current_expression"]:
        if expression is not None and session["images"].get(expression) is not None:
            session["current_expression"] = expression
            session["last_valid_expression"] = expression
    
    # Encode result frame
    _, buffer = cv2.imencode('.jpg', frame)
    result_frame = base64.b64encode(buffer).decode()
    
    return {
        "success": True,
        "expression": session["current_expression"],
        "frame": result_frame,
        "debug": {
            "face_size": face_ratio,
            "faces_detected": faces_detected,
            "smiling": is_smiling,
            "mouth_open": is_mouth_open,
            "eyes_closed": eyes_closed,
            "gaze_direction": gaze_direction,
            "hand_gestures": hand_gestures,
            "mode": "full_detection"
        }
    }

async def process_frame_mock(frame_data: dict, client_id: str) -> dict:
    """Process frame with mock detection (fallback)"""
    import time
    current_time = time.time()
    
    # Mock detection data
    mock_face_ratio = (current_time % 100) / 100
    mock_smiling = int(current_time) % 2 == 0
    mock_eyes_closed = int(current_time * 2) % 3 == 0
    mock_gaze = ["left", "center", "right"][int(current_time) % 3]
    mock_hands = ["thumbs_up", "open_hand", "fist"][int(current_time * 1.5) % 3]
    
    # Mock expression
    expressions = ["smiling", "looking_center", "closeup", "eyes_closed", "thumbs_up"]
    expression = expressions[int(current_time) % len(expressions)]
    
    return {
        "success": True,
        "expression": expression,
        "frame": frame_data.get("frame", ""),
        "debug": {
            "face_size": mock_face_ratio,
            "faces_detected": 1 if mock_face_ratio > 0.3 else 0,
            "smiling": mock_smiling,
            "mouth_open": False,
            "eyes_closed": mock_eyes_closed,
            "gaze_direction": mock_gaze,
            "hand_gestures": [mock_hands],
            "mode": "mock_detection"
        }
    }

def determine_expression(eyes_closed, gaze_direction, is_smiling, is_mouth_open, hand_gestures, face_ratio, images):
    """Determine expression based on all detection results"""
    # Check for closeup first
    if face_ratio > 0.3 and images.get("closeup"):
        return "closeup"
    
    # Check hand gestures
    for gesture in hand_gestures:
        if images.get(gesture):
            return gesture
    
    # Check facial expressions
    if eyes_closed:
        if is_smiling and images.get("eyes_closed_smiling"):
            return "eyes_closed_smiling"
        elif images.get("eyes_closed_neutral"):
            return "eyes_closed_neutral"
    elif is_smiling:
        if gaze_direction == "left" and images.get("looking_left_smiling"):
            return "looking_left_smiling"
        elif gaze_direction == "right" and images.get("looking_right_smiling"):
            return "looking_right_smiling"
        elif gaze_direction == "center" and images.get("looking_center_smiling"):
            return "looking_center_smiling"
        elif images.get("smiling"):
            return "smiling"
    elif is_mouth_open and images.get("shocked"):
        return "shocked"
    else:
        if gaze_direction == "left" and images.get("looking_left"):
            return "looking_left"
        elif gaze_direction == "right" and images.get("looking_right"):
            return "looking_right"
        elif gaze_direction == "center" and images.get("looking_center"):
            return "looking_center"
    
    return None

def get_html_content():
    """Return the HTML content for the web interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MEME Tracker Web - Full Version</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #333;
            margin: 0;
            font-size: 2.5em;
        }
        .main-content {
            display: flex;
            gap: 30px;
            margin-bottom: 30px;
        }
        .video-section {
            flex: 1;
        }
        .controls-section {
            flex: 1;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            max-height: 600px;
            overflow-y: auto;
        }
        #video {
            width: 100%;
            max-width: 640px;
            border-radius: 10px;
            border: 3px solid #007bff;
        }
        #canvas {
            display: none;
        }
        .status {
            margin: 15px 0;
            padding: 15px;
            background: #e9ecef;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }
        .expression-display {
            margin: 20px 0;
            padding: 25px;
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            box-shadow: 0 5px 15px rgba(0,123,255,0.3);
        }
        .controls {
            margin: 20px 0;
        }
        .controls button {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            margin: 5px;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .controls button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(40,167,69,0.3);
        }
        .controls button:disabled {
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .image-uploads {
            margin: 20px 0;
        }
        .image-uploads h4 {
            margin: 10px 0 5px 0;
            color: #495057;
        }
        .image-uploads input[type="file"] {
            margin: 5px 0;
            width: 100%;
        }
        .debug-info {
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            border: 1px solid #dee2e6;
        }
        .debug-info h4 {
            margin-top: 0;
            color: #495057;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-connected { background: #28a745; }
        .status-disconnected { background: #dc3545; }
        .status-connecting { background: #ffc107; }
        .feature-notice {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎭 MEME Tracker Web - Full Version</h1>
            <p>Complete facial expression and hand gesture detection with image selection</p>
        </div>
        
        <div class="feature-notice">
            <strong>🚀 Full Feature Set:</strong> Face detection, gaze tracking, smile detection, mouth opening, hand gestures, and custom image selection!
        </div>
        
        <div class="main-content">
            <div class="video-section">
                <video id="video" autoplay muted></video>
                <canvas id="canvas"></canvas>
                
                <div class="status">
                    <div id="connection-status">
                        <span class="status-indicator status-connecting"></span>
                        Connecting...
                    </div>
                    <div id="detection-status">Ready to start detection</div>
                </div>
                
                <div class="expression-display">
                    <div id="current-expression">No expression detected</div>
                </div>
            </div>
            
            <div class="controls-section">
                <h3>🎮 Controls</h3>
                
                <div class="controls">
                    <button id="start-btn" onclick="startDetection()">🚀 Start Detection</button>
                    <button id="stop-btn" onclick="stopDetection()" disabled>⏹️ Stop Detection</button>
                </div>
                
                <div class="image-uploads">
                    <h4>📸 Image Uploads</h4>
                    
                    <h4>Facial Expressions:</h4>
                    <input type="file" id="smiling-file" accept="image/*" onchange="uploadImage('smiling', this)">
                    <input type="file" id="shocked-file" accept="image/*" onchange="uploadImage('shocked', this)">
                    <input type="file" id="looking_left-file" accept="image/*" onchange="uploadImage('looking_left', this)">
                    <input type="file" id="looking_right-file" accept="image/*" onchange="uploadImage('looking_right', this)">
                    <input type="file" id="looking_center-file" accept="image/*" onchange="uploadImage('looking_center', this)">
                    
                    <h4>Combined Expressions:</h4>
                    <input type="file" id="eyes_closed_smiling-file" accept="image/*" onchange="uploadImage('eyes_closed_smiling', this)">
                    <input type="file" id="looking_center_smiling-file" accept="image/*" onchange="uploadImage('looking_center_smiling', this)">
                    <input type="file" id="eyes_closed_neutral-file" accept="image/*" onchange="uploadImage('eyes_closed_neutral', this)">
                    
                    <h4>Hand Gestures:</h4>
                    <input type="file" id="thumbs_up-file" accept="image/*" onchange="uploadImage('thumbs_up', this)">
                    <input type="file" id="open_hand-file" accept="image/*" onchange="uploadImage('open_hand', this)">
                    <input type="file" id="fist-file" accept="image/*" onchange="uploadImage('fist', this)">
                    
                    <h4>Special:</h4>
                    <input type="file" id="closeup-file" accept="image/*" onchange="uploadImage('closeup', this)">
                </div>
                
                <div class="debug-info" id="debug-info">
                    <h4>🔍 Debug Information</h4>
                    <div id="debug-content">Waiting for detection to start...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let video = document.getElementById('video');
        let canvas = document.getElementById('canvas');
        let ctx = canvas.getContext('2d');
        let ws = null;
        let clientId = 'client_' + Math.random().toString(36).substr(2, 9);
        let isDetecting = false;
        let stream = null;

        // Initialize video stream
        async function initVideo() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: 640, 
                        height: 480 
                    } 
                });
                video.srcObject = stream;
                updateConnectionStatus('connected', 'Camera connected ✅');
            } catch (err) {
                updateConnectionStatus('disconnected', 'Camera error: ' + err.message);
                console.error('Camera error:', err);
            }
        }

        // Update connection status
        function updateConnectionStatus(status, message) {
            const statusElement = document.getElementById('connection-status');
            const indicator = statusElement.querySelector('.status-indicator');
            
            // Remove all status classes
            indicator.classList.remove('status-connected', 'status-disconnected', 'status-connecting');
            
            // Add current status class
            indicator.classList.add('status-' + status);
            
            statusElement.innerHTML = `<span class="status-indicator status-${status}"></span>${message}`;
        }

        // Start detection
        function startDetection() {
            isDetecting = true;
            document.getElementById('start-btn').disabled = true;
            document.getElementById('stop-btn').disabled = false;
            document.getElementById('detection-status').textContent = 'Detection running...';

            // Connect to WebSocket
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/${clientId}`);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
                updateConnectionStatus('connected', 'WebSocket connected ✅');
                sendFrames();
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.success) {
                    updateDisplay(data);
                } else if (data.error) {
                    console.error('Detection error:', data.error);
                    updateConnectionStatus('disconnected', 'Detection error: ' + data.error);
                }
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected');
                isDetecting = false;
                document.getElementById('start-btn').disabled = false;
                document.getElementById('stop-btn').disabled = true;
                document.getElementById('detection-status').textContent = 'Detection stopped';
                updateConnectionStatus('disconnected', 'WebSocket disconnected');
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateConnectionStatus('disconnected', 'Connection error');
            };
        }

        // Stop detection
        function stopDetection() {
            isDetecting = false;
            if (ws) {
                ws.close();
            }
        }

        // Send frames to server
        function sendFrames() {
            if (!isDetecting) return;

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            
            const frameData = canvas.toDataURL('image/jpeg', 0.7);
            const base64Data = frameData.split(',')[1];
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    frame: base64Data,
                    timestamp: Date.now()
                }));
            }
            
            setTimeout(sendFrames, 200); // 5 FPS for real detection
        }

        // Update display with detection results
        function updateDisplay(data) {
            // Update current expression
            const expressionDiv = document.getElementById('current-expression');
            if (data.expression) {
                expressionDiv.textContent = `🎯 Current Expression: ${data.expression}`;
            } else {
                expressionDiv.textContent = '👤 No expression detected';
            }

            // Update debug info
            const debugContent = document.getElementById('debug-content');
            if (data.debug) {
                debugContent.innerHTML = `
                    <strong>📊 Detection Stats:</strong><br>
                    • Face Size: ${(data.debug.face_size * 100).toFixed(1)}%<br>
                    • Faces: ${data.debug.faces_detected}<br>
                    • Smiling: ${data.debug.smiling}<br>
                    • Mouth Open: ${data.debug.mouth_open}<br>
                    • Eyes Closed: ${data.debug.eyes_closed}<br>
                    • Gaze: ${data.debug.gaze_direction}<br>
                    • Hands: ${data.debug.hand_gestures.join(', ') || 'None'}<br>
                    • Mode: ${data.debug.mode}<br>
                    • Time: ${new Date().toLocaleTimeString()}
                `;
            }
        }

        // Upload image for expression
        async function uploadImage(expression, input) {
            const file = input.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch(`/upload-image/${expression}`, {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                
                if (result.success) {
                    console.log(`Image uploaded for ${expression}:`, result.file_path);
                    // Update session images
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'update_image',
                            expression: expression,
                            image_path: result.file_path
                        }));
                    }
                } else {
                    console.error('Upload failed:', result.error);
                }
            } catch (err) {
                console.error('Upload error:', err);
            }
        }

        // Initialize on page load
        window.onload = function() {
            initVideo();
        };
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    try:
        port = int(os.environ.get("PORT", "8000"))
        logger.info(f"Starting MEME Tracker on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start app: {e}")
        raise
