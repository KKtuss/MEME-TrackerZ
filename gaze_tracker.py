import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

class GazeTracker:
    """
    Eye gaze direction detection using OpenCV.
    Analyzes eye positions and movements to determine gaze direction.
    """
    
    def __init__(self):
        # Eye detection parameters
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Gaze tracking parameters
        self.gaze_threshold = 0.15  # Threshold for left/right gaze detection
        self.vertical_threshold = 0.1  # Threshold for up/down gaze detection
        
        # Eye openness detection
        self.eye_open_threshold = 0.25  # Threshold for eye openness
        
        # Gaze direction emojis
        self.gaze_emojis = {
            'left': '👀⬅️',
            'right': '👀➡️',
            'up': '👀⬆️',
            'down': '👀⬇️',
            'center': '👀',
            'closed': '😴'
        }
        
        # History for smoothing
        self.gaze_history = []
        self.max_history = 5
        
    def detect_eyes(self, frame: np.ndarray, face_roi: np.ndarray, face_coords: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        """Detect eyes within a face region"""
        x, y, w, h = face_coords
        
        # Adaptive eye region and parameters based on face size
        face_size_factor = min(w, h) / 100.0  # Normalize by face size
        
        if face_size_factor < 0.8:  # Distant face - larger eye region
            eye_region = face_roi[0:int(h*0.7), 0:w]
            scale_factor = 1.05
            min_neighbors = 3
            min_size = (15, 15)
        elif face_size_factor > 1.5:  # Very close face - more precise eye region
            eye_region = face_roi[0:int(h*0.5), 0:w]
            scale_factor = 1.15
            min_neighbors = 7
            min_size = (25, 25)
        else:  # Normal distance
            eye_region = face_roi[0:int(h*0.6), 0:w]
            scale_factor = 1.1
            min_neighbors = 5
            min_size = (20, 20)
        
        eyes = self.eye_cascade.detectMultiScale(
            eye_region,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        # Adjust coordinates to full frame
        adjusted_eyes = []
        for (ex, ey, ew, eh) in eyes:
            adjusted_eyes.append((x + ex, y + ey, ew, eh))
        
        return adjusted_eyes
    
    def analyze_eye_openness(self, frame: np.ndarray, eye_coords: Tuple[int, int, int, int]) -> Dict:
        """Analyze if an eye is open or closed using simple brightness analysis"""
        x, y, w, h = eye_coords
        eye_roi = frame[y:y+h, x:x+w]
        
        if eye_roi.size == 0:
            return {"is_open": False, "openness_ratio": 0.0}
        
        # Convert to grayscale if needed
        if len(eye_roi.shape) == 3:
            gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_eye = eye_roi
        
        # Resize for consistent analysis
        height, width = gray_eye.shape
        if height < 15 or width < 15:
            return {"is_open": False, "openness_ratio": 0.0}
        
        # Simple method: analyze the center region of the eye
        center_y = height // 2
        center_region = gray_eye[center_y-3:center_y+3, :]
        
        if center_region.size == 0:
            return {"is_open": False, "openness_ratio": 0.0}
        
        # Calculate brightness statistics
        mean_brightness = np.mean(gray_eye)
        std_brightness = np.std(gray_eye)
        center_brightness = np.mean(center_region)
        
        # Simple heuristic: if the eye is open, there should be:
        # 1. Reasonable brightness variation (not too uniform)
        # 2. Center should not be significantly darker than edges
        brightness_variation = std_brightness
        
        # Adaptive thresholds based on eye size for distance robustness
        eye_size_factor = min(gray_eye.shape[0], gray_eye.shape[1]) / 20.0  # Normalize by eye size
        
        # More sensitive thresholds for eye closing detection
        if eye_size_factor < 1.0:  # Small eyes (distant face) - very sensitive
            min_std_threshold = max(1.5, 3 * eye_size_factor)  # More sensitive std threshold
            min_brightness_diff = max(3, 8 * eye_size_factor)  # More sensitive brightness difference
        else:  # Normal/large eyes (close face) - more sensitive
            min_std_threshold = max(2, 4 * eye_size_factor)  # More sensitive thresholds
            min_brightness_diff = max(5, 10 * eye_size_factor)  # More sensitive thresholds
        
        # Eye is considered open if:
        # - There's sufficient brightness variation (adaptive to eye size)
        # - The center isn't much darker than the overall mean (adaptive threshold)
        is_open = (brightness_variation > min_std_threshold) and (center_brightness > mean_brightness - min_brightness_diff)
        
        # Calculate openness score (0-1)
        openness_score = min(1.0, brightness_variation / 20.0)
        
        return {
            "is_open": is_open,
            "openness_ratio": openness_score
        }
    
    def detect_pupil_position(self, eye_roi: np.ndarray) -> Tuple[int, int]:
        """Simple and reliable pupil detection within an eye region"""
        if eye_roi.size == 0:
            return (50, 50)  # Default fallback
            
        # Convert to grayscale
        if len(eye_roi.shape) == 3:
            gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_roi
        
        height, width = gray.shape
        
        # Simple approach: find the darkest point (likely pupil)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
        
        # Use the darkest point as pupil center
        pupil_x, pupil_y = min_loc
        
        # Ensure coordinates are within bounds
        pupil_x = max(0, min(width-1, pupil_x))
        pupil_y = max(0, min(height-1, pupil_y))
        
        return (pupil_x, pupil_y)
    
    def analyze_gaze_direction(self, frame: np.ndarray, eyes: List[Tuple[int, int, int, int]], face_coords: Tuple[int, int, int, int]) -> Dict:
        """Analyze gaze direction based on eye positions and pupil locations"""
        if len(eyes) < 2:
            return {
                "direction": "closed",
                "confidence": 1.0,
                "is_eyes_closed": True,
                "emoji": self.gaze_emojis['closed']
            }
        
        # Filter out eyes that are too close together (likely false detections)
        filtered_eyes = []
        for eye in eyes:
            ex, ey, ew, eh = eye
            
            # Additional filtering for eye validity
            # Filter 1: Size constraints
            if ew < 10 or eh < 8 or ew > 100 or eh > 60:
                continue
            
            # Filter 2: Aspect ratio (eyes are wider than they are tall)
            aspect_ratio = ew / eh
            if aspect_ratio < 1.0 or aspect_ratio > 5.0:
                continue
            
            # Filter 3: Position within face
            face_center_y = face_coords[1] + face_coords[3] // 2
            if ey > face_center_y - 10:  # Too low in face
                continue
            
            # Filter 4: Distance from other eyes
            is_valid = True
            for other_eye in filtered_eyes:
                # Check if eyes are too close (less than 40 pixels apart)
                distance = abs(eye[0] - other_eye[0])
                if distance < 40:
                    is_valid = False
                    break
            if is_valid:
                filtered_eyes.append(eye)
        
        # If we don't have at least 2 valid eyes, consider eyes closed
        if len(filtered_eyes) < 2:
            return {
                "direction": "closed",
                "confidence": 1.0,
                "is_eyes_closed": True,
                "emoji": self.gaze_emojis['closed']
            }
        
        # Sort eyes by x position (left to right)
        sorted_eyes = sorted(filtered_eyes, key=lambda e: e[0])
        left_eye_coords, right_eye_coords = sorted_eyes[0], sorted_eyes[1]
        
        # Check if eyes are open
        left_eye_analysis = self.analyze_eye_openness(frame, left_eye_coords)
        right_eye_analysis = self.analyze_eye_openness(frame, right_eye_coords)
        
        # If eyes are closed
        if not left_eye_analysis["is_open"] and not right_eye_analysis["is_open"]:
            return {
                "direction": "closed",
                "confidence": 1.0,
                "is_eyes_closed": True,
                "emoji": self.gaze_emojis['closed']
            }
        
        # Extract eye regions
        lx, ly, lw, lh = left_eye_coords
        rx, ry, rw, rh = right_eye_coords
        
        left_eye_roi = frame[ly:ly+lh, lx:lx+lw]
        right_eye_roi = frame[ry:ry+rh, rx:rx+rw]
        
        # Detect pupil positions
        left_pupil = self.detect_pupil_position(left_eye_roi)
        right_pupil = self.detect_pupil_position(right_eye_roi)
        
        # Calculate relative pupil positions
        left_pupil_rel_x = left_pupil[0] / lw
        right_pupil_rel_x = right_pupil[0] / rw
        
        # Average horizontal gaze
        avg_horizontal_gaze = (left_pupil_rel_x + right_pupil_rel_x) / 2
        
        # Calculate vertical gaze (average of both eyes)
        left_pupil_rel_y = left_pupil[1] / lh
        right_pupil_rel_y = right_pupil[1] / rh
        avg_vertical_gaze = (left_pupil_rel_y + right_pupil_rel_y) / 2
        
        # Determine gaze direction
        direction = "center"
        confidence = 0.0
        
        # Horizontal gaze
        if avg_horizontal_gaze < 0.5 - self.gaze_threshold:
            direction = "left"
            confidence = abs(avg_horizontal_gaze - 0.5) * 2
        elif avg_horizontal_gaze > 0.5 + self.gaze_threshold:
            direction = "right"
            confidence = abs(avg_horizontal_gaze - 0.5) * 2
        
        # Vertical gaze (if significant)
        if abs(avg_vertical_gaze - 0.5) > self.vertical_threshold:
            if avg_vertical_gaze < 0.5:
                direction = "up" if direction == "center" else direction
            else:
                direction = "down" if direction == "center" else direction
        
        # Smooth gaze direction using history
        self.gaze_history.append(direction)
        if len(self.gaze_history) > self.max_history:
            self.gaze_history.pop(0)
        
        # Use most common direction in recent history
        most_common_direction = max(set(self.gaze_history), key=self.gaze_history.count)
        
        return {
            "direction": most_common_direction,
            "confidence": min(confidence, 1.0),
            "is_eyes_closed": False,
            "emoji": self.gaze_emojis.get(most_common_direction, '👀'),
            "raw_gaze": {
                "horizontal": avg_horizontal_gaze,
                "vertical": avg_vertical_gaze,
                "left_pupil": left_pupil_rel_x,
                "right_pupil": right_pupil_rel_x
            }
        }
    
    def get_gaze_text(self, gaze_result: Dict) -> str:
        """Get formatted gaze direction text"""
        direction = gaze_result["direction"]
        emoji = gaze_result["emoji"]
        confidence = gaze_result["confidence"]
        
        if direction == "closed":
            return f"{emoji} Eyes Closed"
        elif direction == "center":
            return f"{emoji} Looking Center"
        else:
            return f"{emoji} Looking {direction.title()} ({confidence:.1%})"
    
    def draw_gaze_overlay(self, frame: np.ndarray, gaze_result: Dict, eyes: List[Tuple[int, int, int, int]], position: Tuple[int, int] = (10, 100)) -> np.ndarray:
        """Draw gaze information on the frame"""
        x, y = position
        
        # Draw gaze direction text
        gaze_text = self.get_gaze_text(gaze_result)
        cv2.putText(frame, gaze_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw eye rectangles with pupil positions
        for eye_coords in eyes:
            ex, ey, ew, eh = eye_coords
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)
            
            # Draw pupil position
            eye_roi = frame[ey:ey+eh, ex:ex+ew]
            pupil_pos = self.detect_pupil_position(eye_roi)
            pupil_x = ex + pupil_pos[0]
            pupil_y = ey + pupil_pos[1]
            cv2.circle(frame, (pupil_x, pupil_y), 3, (0, 255, 255), -1)
        
        # Draw raw gaze data
        if "raw_gaze" in gaze_result:
            raw = gaze_result["raw_gaze"]
            y_offset = y + 30
            cv2.putText(frame, f"H: {raw['horizontal']:.2f}, V: {raw['vertical']:.2f}", 
                       (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def track_gaze(self, frame: np.ndarray, face_roi: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Dict:
        """Complete gaze tracking analysis"""
        eyes = self.detect_eyes(frame, face_roi, face_coords)
        gaze_result = self.analyze_gaze_direction(frame, eyes, face_coords)
        
        return {
            "gaze": gaze_result,
            "eyes": eyes,
            "eyes_detected": len(eyes)
        }

if __name__ == "__main__":
    # Test the gaze tracker
    cap = cv2.VideoCapture(0)
    tracker = GazeTracker()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("Testing Gaze Tracking...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            gaze_data = tracker.track_gaze(frame, face_roi, (x, y, w, h))
            
            # Draw gaze overlay
            frame = tracker.draw_gaze_overlay(frame, gaze_data["gaze"], gaze_data["eyes"])
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow('Gaze Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
