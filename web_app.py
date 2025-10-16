"""
Web version of the MEME Tracker application using FastAPI
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import base64
import cv2
import numpy as np
from PIL import Image
import io
import os
import time
from typing import Dict, List, Optional
import logging

# Import our detection modules
from facial_landmarks import FacialLandmarksDetector
from gaze_tracker import GazeTracker
from hand_tracker import HandTracker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MEME Tracker Web", version="1.0.0")

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detection instances
facial_detector = None
gaze_tracker = None
hand_tracker = None
is_initialized = False

# User sessions storage
user_sessions: Dict[str, Dict] = {}

async def initialize_detectors():
    """Initialize detection modules"""
    global facial_detector, gaze_tracker, hand_tracker, is_initialized
    try:
        facial_detector = FacialLandmarksDetector()
        gaze_tracker = GazeTracker()
        hand_tracker = HandTracker()
        is_initialized = True
        logger.info("Detection modules initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize detectors: {e}")
        is_initialized = False

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    await initialize_detectors()

@app.get("/")
async def read_root():
    """Serve the main web interface"""
    return HTMLResponse(content=get_html_content())

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
            
            if not is_initialized:
                await websocket.send_text(json.dumps({
                    "error": "Detection modules not initialized"
                }))
                continue
            
            # Process frame and return results
            result = await process_frame(frame_data, client_id)
            await websocket.send_text(json.dumps(result))
            
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
        if client_id in user_sessions:
            del user_sessions[client_id]

async def process_frame(frame_data: dict, client_id: str) -> dict:
    """Process a single frame and return detection results"""
    try:
        # Decode base64 frame
        frame_bytes = base64.b64decode(frame_data["frame"])
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Invalid frame data"}
        
        # Get user session
        session = user_sessions.get(client_id, {})
        
        # Process detections
        landmark_data = facial_detector.get_landmark_data(frame)
        
        if landmark_data["faces_detected"] > 0:
            landmark = landmark_data["landmarks"][0]
            face_coords = landmark["face"]
            eyes = landmark["eyes"]
            eye_analysis = landmark["eye_analysis"]
            
            # Analyze gaze direction
            gaze_result = gaze_tracker.analyze_gaze_direction(frame, eyes, face_coords)
            
            # Analyze smile
            smile_result = facial_detector.detect_smile_simple(frame, face_coords)
            
            # Analyze mouth opening
            mouth_result = facial_detector.detect_mouth_opening(frame, face_coords)
            
            # Analyze hand gestures
            hand_result = hand_tracker.get_hand_gestures(frame, face_coords)
            
            # Determine expression
            expression = determine_expression(
                eye_analysis, gaze_result, smile_result, 
                mouth_result, hand_result, face_coords, 
                frame.shape, session["images"]
            )
            
            # Update session
            if expression != session["current_expression"] and session["auto_trigger"]:
                if expression is not None:
                    session["current_expression"] = expression
                    session["last_valid_expression"] = expression
                elif session["last_valid_expression"] is not None:
                    session["current_expression"] = session["last_valid_expression"]
            
            # Draw landmarks for visualization
            frame_with_landmarks = facial_detector.draw_landmarks(frame, landmark_data)
            frame_with_hands = hand_tracker.draw_hands(frame_with_landmarks, hand_tracker.detect_hands(frame))
            
            # Encode result frame
            _, buffer = cv2.imencode('.jpg', frame_with_hands)
            result_frame = base64.b64encode(buffer).decode()
            
            return {
                "success": True,
                "expression": session["current_expression"],
                "frame": result_frame,
                "debug": {
                    "face_size": get_face_size_ratio(face_coords, frame.shape),
                    "smile": smile_result.get("is_smiling", False),
                    "mouth_open": mouth_result.get("is_mouth_open", False),
                    "gaze": gaze_result.get("direction", "center"),
                    "hands": len(hand_tracker.detect_hands(frame))
                }
            }
        else:
            # No face detected
            session["current_expression"] = None
            return {
                "success": True,
                "expression": None,
                "frame": frame_data["frame"],  # Return original frame
                "debug": {"face_size": 0}
            }
            
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return {"error": str(e)}

def determine_expression(eye_analysis, gaze_result, smile_result, mouth_result, hand_result, face_coords, frame_shape, images):
    """Determine current expression (same logic as desktop version)"""
    # Check for closeup detection
    if face_coords is not None and frame_shape is not None and images.get("closeup") is not None:
        fx, fy, fw, fh = face_coords
        frame_height, frame_width = frame_shape[:2]
        face_area = fw * fh
        frame_area = frame_width * frame_height
        face_ratio = face_area / frame_area
        
        if face_ratio > 0.3:
            return "closeup"
    
    # Get hand gestures
    left_hand_gesture = hand_result.get("left_hand")
    right_hand_gesture = hand_result.get("right_hand")
    special_gesture = hand_result.get("special_gesture")
    
    # Prioritize special gestures
    if special_gesture and special_gesture != "None" and images.get(special_gesture) is not None:
        return special_gesture
    
    # Check individual hand gestures
    if left_hand_gesture and left_hand_gesture != "unknown" and left_hand_gesture != "None" and images.get(left_hand_gesture) is not None:
        return left_hand_gesture
    elif right_hand_gesture and right_hand_gesture != "unknown" and right_hand_gesture != "None" and images.get(right_hand_gesture) is not None:
        return right_hand_gesture
    
    # Check facial expressions
    is_smiling = smile_result.get("is_smiling", False)
    is_mouth_open = mouth_result.get("is_mouth_open", False)
    eyes_closed = gaze_result.get("is_eyes_closed", False)
    gaze_direction = gaze_result.get("direction", "center")
    
    if eyes_closed:
        if is_smiling and images.get("eyes_closed_smiling") is not None:
            return "eyes_closed_smiling"
        elif images.get("eyes_closed_neutral") is not None:
            return "eyes_closed_neutral"
    elif is_smiling:
        if gaze_direction == "left" and images.get("looking_left_smiling") is not None:
            return "looking_left_smiling"
        elif gaze_direction == "right" and images.get("looking_right_smiling") is not None:
            return "looking_right_smiling"
        elif gaze_direction == "center" and images.get("looking_center_smiling") is not None:
            return "looking_center_smiling"
        elif images.get("eyes_open_smiling") is not None:
            return "eyes_open_smiling"
        elif images.get("smiling") is not None:
            return "smiling"
    elif is_mouth_open and not is_smiling:
        if images.get("shocked") is not None:
            return "shocked"
    else:
        if gaze_direction == "left" and images.get("looking_left") is not None:
            return "looking_left"
        elif gaze_direction == "right" and images.get("looking_right") is not None:
            return "looking_right"
        elif gaze_direction == "center" and images.get("looking_center") is not None:
            return "looking_center"
        elif images.get("eyes_open") is not None:
            return "eyes_open"
    
    return None

def get_face_size_ratio(face_coords, frame_shape):
    """Calculate face size ratio"""
    if face_coords is None:
        return 0
    fx, fy, fw, fh = face_coords
    frame_height, frame_width = frame_shape[:2]
    face_area = fw * fh
    frame_area = frame_width * frame_height
    return face_area / frame_area

@app.post("/upload-image/{expression}")
async def upload_image(expression: str, file: UploadFile = File(...)):
    """Upload image for an expression"""
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Save uploaded file
        file_path = f"uploads/{expression}_{int(time.time())}.jpg"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {"success": True, "file_path": file_path}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_html_content():
    """Return the HTML content for the web interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MEME Tracker Web</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .video-container {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .video-section {
            flex: 1;
        }
        .controls-section {
            flex: 1;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        #video {
            width: 100%;
            max-width: 640px;
            border-radius: 5px;
        }
        #canvas {
            display: none;
        }
        .status {
            margin: 10px 0;
            padding: 10px;
            background: #e9ecef;
            border-radius: 5px;
        }
        .expression-display {
            margin: 20px 0;
            padding: 20px;
            background: #007bff;
            color: white;
            border-radius: 5px;
            text-align: center;
            font-size: 18px;
        }
        .controls {
            margin: 10px 0;
        }
        .controls label {
            display: block;
            margin: 5px 0;
        }
        .controls input[type="file"] {
            margin: 5px 0;
        }
        .controls button {
            background: #28a745;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin: 2px;
        }
        .controls button:hover {
            background: #218838;
        }
        .controls button:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        .debug-info {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
            font-family: monospace;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 MEME Tracker Web</h1>
        
        <div class="video-container">
            <div class="video-section">
                <video id="video" autoplay muted></video>
                <canvas id="canvas"></canvas>
                
                <div class="status">
                    <div id="connection-status">Connecting...</div>
                    <div id="detection-status">Starting detection...</div>
                </div>
                
                <div class="expression-display">
                    <div id="current-expression">No expression detected</div>
                </div>
            </div>
            
            <div class="controls-section">
                <h3>🎮 Controls</h3>
                
                <div class="controls">
                    <button id="start-btn" onclick="startDetection()">Start Detection</button>
                    <button id="stop-btn" onclick="stopDetection()" disabled>Stop Detection</button>
                </div>
                
                <h4>📸 Image Uploads</h4>
                <div class="controls">
                    <label>Smiling:</label>
                    <input type="file" id="smiling-file" accept="image/*" onchange="uploadImage('smiling', this)">
                    
                    <label>Eyes Closed:</label>
                    <input type="file" id="eyes_closed-file" accept="image/*" onchange="uploadImage('eyes_closed', this)">
                    
                    <label>Looking Left:</label>
                    <input type="file" id="looking_left-file" accept="image/*" onchange="uploadImage('looking_left', this)">
                    
                    <label>Looking Right:</label>
                    <input type="file" id="looking_right-file" accept="image/*" onchange="uploadImage('looking_right', this)">
                    
                    <label>Looking Center:</label>
                    <input type="file" id="looking_center-file" accept="image/*" onchange="uploadImage('looking_center', this)">
                    
                    <label>Shocked (Mouth Open):</label>
                    <input type="file" id="shocked-file" accept="image/*" onchange="uploadImage('shocked', this)">
                    
                    <label>Closeup:</label>
                    <input type="file" id="closeup-file" accept="image/*" onchange="uploadImage('closeup', this)">
                </div>
                
                <div class="debug-info" id="debug-info">
                    <h4>🔍 Debug Info</h4>
                    <div id="debug-content">Waiting for detection...</div>
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
                document.getElementById('connection-status').textContent = 'Camera connected ✅';
            } catch (err) {
                document.getElementById('connection-status').textContent = 'Camera error: ' + err.message;
                console.error('Camera error:', err);
            }
        }

        // Start detection
        function startDetection() {
            if (!stream) {
                alert('Please allow camera access first');
                return;
            }

            isDetecting = true;
            document.getElementById('start-btn').disabled = true;
            document.getElementById('stop-btn').disabled = false;
            document.getElementById('detection-status').textContent = 'Detection running...';

            // Connect to WebSocket
            ws = new WebSocket(`ws://localhost:8000/ws/${clientId}`);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
                sendFrames();
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.success) {
                    updateDisplay(data);
                } else if (data.error) {
                    console.error('Detection error:', data.error);
                }
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected');
                isDetecting = false;
                document.getElementById('start-btn').disabled = false;
                document.getElementById('stop-btn').disabled = true;
                document.getElementById('detection-status').textContent = 'Detection stopped';
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
            
            setTimeout(sendFrames, 100); // ~10 FPS
        }

        // Update display with detection results
        function updateDisplay(data) {
            // Update current expression
            const expressionDiv = document.getElementById('current-expression');
            if (data.expression) {
                expressionDiv.textContent = `Current Expression: ${data.expression}`;
            } else {
                expressionDiv.textContent = 'No expression detected';
            }

            // Update debug info
            const debugContent = document.getElementById('debug-content');
            if (data.debug) {
                debugContent.innerHTML = `
                    Face Size: ${(data.debug.face_size * 100).toFixed(1)}%
                    <br>Smile: ${data.debug.smile}
                    <br>Mouth Open: ${data.debug.mouth_open}
                    <br>Gaze: ${data.debug.gaze}
                    <br>Hands: ${data.debug.hands}
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
