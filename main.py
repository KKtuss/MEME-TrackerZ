"""
MEME Tracker Web Application - Main Entry Point
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import base64
import cv2
import numpy as np
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # Decode base64 frame
        frame_bytes = base64.b64decode(frame_data["frame"])
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Invalid frame data"}
        
        # Simple face detection for demo
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Encode result frame
        _, buffer = cv2.imencode('.jpg', frame)
        result_frame = base64.b64encode(buffer).decode()
        
        if len(faces) > 0:
            # Face detected
            face_area = faces[0][2] * faces[0][3]
            frame_area = frame.shape[0] * frame.shape[1]
            face_ratio = face_area / frame_area
            
            expression = "closeup" if face_ratio > 0.3 else "looking_center"
            
            return {
                "success": True,
                "expression": expression,
                "frame": result_frame,
                "debug": {
                    "face_size": face_ratio,
                    "faces_detected": len(faces)
                }
            }
        else:
            # No face detected
            return {
                "success": True,
                "expression": None,
                "frame": result_frame,
                "debug": {"face_size": 0, "faces_detected": 0}
            }
            
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return {"error": str(e)}

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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎭 MEME Tracker Web</h1>
            <p>Real-time facial expression detection powered by AI</p>
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
            if (!stream) {
                alert('Please allow camera access first');
                return;
            }

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
            
            setTimeout(sendFrames, 200); // ~5 FPS for better performance
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
