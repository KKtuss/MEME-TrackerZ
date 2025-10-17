"""
MEME Tracker Web Application - Main Entry Point
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
import os
import base64
import numpy as np

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

app = FastAPI(title="MEME Tracker Web", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

async def process_frame(frame_data: dict, client_id: str) -> dict:
    """Process a single frame and return detection results"""
    try:
        if OPENCV_AVAILABLE:
            return await process_frame_opencv(frame_data, client_id)
        else:
            return await process_frame_mock(frame_data, client_id)
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return {"error": str(e)}

async def process_frame_opencv(frame_data: dict, client_id: str) -> dict:
    """Process frame with OpenCV detection"""
    # Decode base64 frame
    frame_bytes = base64.b64decode(frame_data["frame"])
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "Invalid frame data"}
    
    # Initialize cascades if not already done
    if not hasattr(process_frame_opencv, 'face_cascade'):
        process_frame_opencv.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if not hasattr(process_frame_opencv, 'smile_cascade'):
        process_frame_opencv.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = process_frame_opencv.face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Initialize variables
    expression = None
    face_ratio = 0
    faces_detected = len(faces)
    is_smiling = False
    
    if len(faces) > 0:
        # Get the largest face
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = largest_face
        
        # Calculate face size ratio
        face_area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        face_ratio = face_area / frame_area
        
        # Extract face region for smile detection
        face_roi = gray[y:y+h, x:x+w]
        
        # Detect smiles in the face region
        smiles = process_frame_opencv.smile_cascade.detectMultiScale(face_roi, 1.8, 20)
        
        if len(smiles) > 0:
            is_smiling = True
            # Draw smile rectangles
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(frame, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 255, 0), 2)
        
        # Determine expression based on face size and smile
        if face_ratio > 0.3:
            expression = "closeup_smiling" if is_smiling else "closeup"
        else:
            expression = "looking_center_smiling" if is_smiling else "looking_center"
        
        # Draw rectangle around face
        color = (0, 255, 0) if is_smiling else (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Add text overlay
        cv2.putText(frame, f"Face: {face_ratio:.2f}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Smile: {is_smiling}", (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Encode result frame
    _, buffer = cv2.imencode('.jpg', frame)
    result_frame = base64.b64encode(buffer).decode()
    
    return {
        "success": True,
        "expression": expression,
        "frame": result_frame,
        "debug": {
            "face_size": face_ratio,
            "faces_detected": faces_detected,
            "smiling": is_smiling,
            "mode": "real_detection"
        }
    }

async def process_frame_mock(frame_data: dict, client_id: str) -> dict:
    """Process frame with mock detection (fallback)"""
    import time
    current_time = time.time()
    
    # Mock expression detection based on time
    expressions = ["smiling", "looking_center", "closeup", "eyes_closed"]
    mock_expression = expressions[int(current_time) % len(expressions)]
    
    # Mock face detection
    mock_face_ratio = (current_time % 100) / 100  # 0 to 1
    mock_smiling = int(current_time) % 2 == 0
    
    # Combine mock data
    if mock_face_ratio > 0.3:
        expression = f"closeup_smiling" if mock_smiling else "closeup"
    else:
        expression = f"looking_center_smiling" if mock_smiling else "looking_center"
    
    return {
        "success": True,
        "expression": expression,
        "frame": frame_data.get("frame", ""),  # Echo back the frame
        "debug": {
            "face_size": mock_face_ratio,
            "faces_detected": 1 if mock_face_ratio > 0.3 else 0,
            "smiling": mock_smiling,
            "mode": "mock_detection"
        }
    }

def get_html_content():
    """Return the HTML content for the web interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MEME Tracker Web - Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
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
        .header p {
            color: #666;
            margin: 10px 0 0 0;
        }
        .video-container {
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
        .demo-notice {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎭 MEME Tracker Web</h1>
            <p>Real-time facial expression detection powered by AI</p>
        </div>
        
        <div class="demo-notice">
            <strong>🎯 Smart Detection Active:</strong> This app automatically detects your environment and uses the best available detection method. 
            Move closer to the camera for "closeup" mode, and smile for "smiling" expressions!
        </div>
        
        <div class="video-container">
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
                    • Faces Detected: ${data.debug.faces_detected}<br>
                    • Smiling: ${data.debug.smiling}<br>
                    • Mode: ${data.debug.mode}<br>
                    • Timestamp: ${new Date().toLocaleTimeString()}
                `;
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