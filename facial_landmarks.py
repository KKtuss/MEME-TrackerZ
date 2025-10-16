import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

class FacialLandmarks:
    """
    Facial landmark detection using OpenCV Haar cascades.
    Detects eyes, mouth, and basic facial features for expression analysis.
    """
    
    def __init__(self):
        # Initialize OpenCV classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Smile detection debouncing
        self.smile_history = []
        self.smile_history_size = 5  # Keep last 5 detections
        
        # Mouth opening detection debouncing
        self.mouth_history = []
        self.mouth_history_size = 5  # Keep last 5 detections
        
        # Eye detection parameters (balanced to detect eyes but filter nose)
        self.eye_params = {
            'scaleFactor': 1.03,  # Very sensitive scaling for distant faces
            'minNeighbors': 3,    # Lower threshold for better distant detection
            'minSize': (8, 6),    # Even smaller minimum size for distant faces
            'maxSize': (150, 100) # Larger maximum size for close faces
        }
        
        # Face detection parameters
        self.face_params = {
            'scaleFactor': 1.1,
            'minNeighbors': 5,
            'minSize': (100, 100)
        }
        
        # Smile detection parameters - balanced sensitivity
        self.smile_params = {
            'scaleFactor': 1.1,   # Balanced scaling
            'minNeighbors': 12,   # Moderate threshold
            'minSize': (25, 25),  # Balanced minimum size
            'maxSize': (180, 110) # Balanced maximum size
        }
    
    def detect_face(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, **self.face_params)
        return faces
    
    def detect_eyes(self, frame: np.ndarray, face_roi: np.ndarray, face_coords: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        """Detect eyes within a face region with improved filtering"""
        x, y, w, h = face_coords
        
        # Adaptive eye region based on face size
        face_size_factor = min(w, h) / 100.0  # Normalize by face size
        
        if face_size_factor < 0.8:  # Distant face - larger eye region
            eye_region = face_roi[0:int(h*0.7), 0:w]
        elif face_size_factor > 1.5:  # Very close face - more precise eye region
            eye_region = face_roi[0:int(h*0.5), 0:w]
        else:  # Normal distance
            eye_region = face_roi[0:int(h*0.6), 0:w]
        
        # Adaptive eye detection parameters based on face size
        adaptive_eye_params = self.eye_params.copy()
        
        if face_size_factor < 0.8:  # Distant face - more sensitive
            adaptive_eye_params['minNeighbors'] = max(2, int(self.eye_params['minNeighbors'] * 0.7))
            adaptive_eye_params['minSize'] = (max(6, int(8 * face_size_factor)), max(4, int(6 * face_size_factor)))
        elif face_size_factor > 1.5:  # Very close face - less sensitive
            adaptive_eye_params['minNeighbors'] = int(self.eye_params['minNeighbors'] * 1.3)
            adaptive_eye_params['minSize'] = (int(8 * 1.2), int(6 * 1.2))
        else:  # Normal distance
            adaptive_eye_params = self.eye_params.copy()
        
        eyes = self.eye_cascade.detectMultiScale(eye_region, **adaptive_eye_params)
        
        # Adjust coordinates to full frame and filter
        adjusted_eyes = []
        for (ex, ey, ew, eh) in eyes:
            adjusted_eyes.append((x + ex, y + ey, ew, eh))
        
        # Enhanced filtering for false detections
        filtered_eyes = []
        for eye in adjusted_eyes:
            ex, ey, ew, eh = eye
            
            # Filter 1: Size constraints (eyes should be reasonably sized)
            if ew < 10 or eh < 8 or ew > 100 or eh > 60:
                continue
            
            # Filter 2: Aspect ratio (eyes are wider than they are tall)
            aspect_ratio = ew / eh
            if aspect_ratio < 1.0 or aspect_ratio > 5.0:
                continue
            
            # Filter 3: Position within face (eyes should be in upper portion)
            face_center_y = y + h // 2
            if ey > face_center_y - 10:  # Too low in face
                continue
            
            # Filter 4: Distance from other eyes
            is_valid = True
            for other_eye in filtered_eyes:
                # Check if eyes are too close (less than 40 pixels apart)
                distance = abs(eye[0] - other_eye[0])
                if distance < 40:
                    # Keep the larger eye
                    if eye[2] * eye[3] > other_eye[2] * other_eye[3]:
                        filtered_eyes.remove(other_eye)
                    else:
                        is_valid = False
                        break
            
            if is_valid:
                filtered_eyes.append(eye)
        
        # Sort by size and take only the largest 2 eyes
        filtered_eyes.sort(key=lambda e: e[2] * e[3], reverse=True)
        return filtered_eyes[:2]
    
    def detect_smile(self, frame: np.ndarray, face_roi: np.ndarray, face_coords: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        """Detect smiles within a face region"""
        x, y, w, h = face_coords
        
        # Focus on lower half of face for smile detection
        smile_region = face_roi[int(h*0.5):h, 0:w]
        
        smiles = self.smile_cascade.detectMultiScale(smile_region, **self.smile_params)
        
        # Adjust coordinates to full frame
        adjusted_smiles = []
        for (sx, sy, sw, sh) in smiles:
            adjusted_smiles.append((x + sx, y + int(h*0.5) + sy, sw, sh))
        
        return adjusted_smiles
    
    def detect_smile_simple(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Dict:
        """Conservative smile detection to avoid false positives"""
        x, y, w, h = face_coords
        
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return {"is_smiling": False, "confidence": 0.0, "smile_count": 0}
        
        # Adaptive mouth region based on face size
        face_size_factor = min(w, h) / 100.0  # Normalize by face size
        
        if face_size_factor < 0.8:  # Distant face - larger mouth region
            mouth_region = face_roi[int(h*0.5):int(h*0.9), int(w*0.15):int(w*0.85)]
        elif face_size_factor > 1.5:  # Very close face - more precise mouth region
            mouth_region = face_roi[int(h*0.6):int(h*0.8), int(w*0.25):int(w*0.75)]
        else:  # Normal distance
            mouth_region = face_roi[int(h*0.55):int(h*0.85), int(w*0.2):int(w*0.8)]
        
        # Convert to grayscale for better detection
        if len(mouth_region.shape) == 3:
            gray_mouth_region = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_mouth_region = mouth_region
        
        # Adaptive smile detection based on face size (balanced for all distances)
        adaptive_smile_params = self.smile_params.copy()
        
        # Adjust parameters based on face size for distance robustness
        if face_size_factor < 0.8:  # Distant face - more sensitive
            adaptive_smile_params['minNeighbors'] = max(6, int(self.smile_params['minNeighbors'] * 0.6))
            adaptive_smile_params['minSize'] = (max(15, int(25 * face_size_factor)), max(15, int(25 * face_size_factor)))
        elif face_size_factor > 1.5:  # Very close face - less sensitive
            adaptive_smile_params['minNeighbors'] = int(self.smile_params['minNeighbors'] * 1.5)
            adaptive_smile_params['minSize'] = (int(25 * 1.2), int(25 * 1.2))
        else:  # Normal distance
            adaptive_smile_params['minNeighbors'] = self.smile_params['minNeighbors']
            adaptive_smile_params['minSize'] = self.smile_params['minSize']
        
        smiles = self.smile_cascade.detectMultiScale(gray_mouth_region, **adaptive_smile_params)
        
        # Ensure only ONE smile is detected (logical constraint - one face = one smile max)
        if len(smiles) > 1:
            # Sort by area (width * height) and keep only the largest
            areas = [smile[2] * smile[3] for smile in smiles]
            best_smile_idx = areas.index(max(areas))
            smiles = [smiles[best_smile_idx]]
        elif len(smiles) == 0:
            # No smiles detected
            pass
        else:
            # Exactly one smile - validate it's reasonable size
            smile = smiles[0]
            smile_area = smile[2] * smile[3]
            face_area = w * h
            smile_ratio = smile_area / face_area
            
            # If smile is too small or too large relative to face, reject it
            if smile_ratio < 0.02 or smile_ratio > 0.25:  # 2% to 25% of face area
                smiles = []
        
        # Add current detection to history (only if exactly one valid smile)
        current_detection = len(smiles) == 1
        self.smile_history.append(current_detection)
        
        # Keep only recent history
        if len(self.smile_history) > self.smile_history_size:
            self.smile_history.pop(0)
        
        # Balanced debouncing - not too strict, not too loose
        if len(self.smile_history) >= 3:
            smile_votes = sum(self.smile_history)
            is_smiling = smile_votes >= 2  # Need at least 2 out of 5 recent detections
        else:
            is_smiling = current_detection  # Use current detection if not enough history
        
        confidence = 0.8 if is_smiling else 0.0
        
        return {
            "is_smiling": is_smiling,
            "confidence": confidence,
            "smile_count": 1 if is_smiling else 0  # Only 0 or 1, never more
        }
    
    def detect_mouth_opening(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Dict:
        """Detect if mouth is open (without smiling) using contour analysis"""
        x, y, w, h = face_coords
        
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return {"is_mouth_open": False, "confidence": 0.0, "mouth_ratio": 0.0}
        
        # Adaptive mouth region based on face size
        face_size_factor = min(w, h) / 100.0  # Normalize by face size
        
        if face_size_factor < 0.8:  # Distant face - larger mouth region
            mouth_region = face_roi[int(h*0.45):int(h*0.95), int(w*0.1):int(w*0.9)]
        elif face_size_factor > 1.5:  # Very close face - more precise mouth region
            mouth_region = face_roi[int(h*0.55):int(h*0.85), int(w*0.2):int(w*0.8)]
        else:  # Normal distance
            mouth_region = face_roi[int(h*0.5):int(h*0.95), int(w*0.15):int(w*0.85)]
        
        # Convert to grayscale
        if len(mouth_region.shape) == 3:
            gray_mouth_region = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_mouth_region = mouth_region
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_mouth_region, (5, 5), 0)
        
        # Apply adaptive threshold to create binary image
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"is_mouth_open": False, "confidence": 0.0, "mouth_ratio": 0.0}
        
        # Find the largest contour (likely the mouth)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Calculate mouth opening ratio relative to face size
        face_area = w * h
        mouth_ratio = area / face_area
        
        # Mouth is considered open if:
        # 1. The contour area is significant relative to face
        # 2. The contour has a reasonable aspect ratio (mouth-like shape)
        is_mouth_open = False
        confidence = 0.0
        
        # Adaptive threshold based on face size
        if face_size_factor < 0.8:  # Distant face - more sensitive
            threshold = 0.010  # 1.0% of face area
            aspect_min, aspect_max = 1.2, 5.0
            confidence_scale = 30
        elif face_size_factor > 1.5:  # Very close face - less sensitive
            threshold = 0.020  # 2.0% of face area
            aspect_min, aspect_max = 1.8, 4.0
            confidence_scale = 20
        else:  # Normal distance
            threshold = 0.015  # 1.5% of face area
            aspect_min, aspect_max = 1.5, 4.5
            confidence_scale = 25
        
        if mouth_ratio > threshold:
            # Check aspect ratio - mouths are typically wider than tall
            x_contour, y_contour, w_contour, h_contour = cv2.boundingRect(largest_contour)
            aspect_ratio = w_contour / h_contour if h_contour > 0 else 0
            
            if aspect_min <= aspect_ratio <= aspect_max:
                is_mouth_open = True
                confidence = min(0.9, mouth_ratio * confidence_scale)
        
        # Add current detection to history for debouncing
        current_detection = is_mouth_open
        self.mouth_history.append(current_detection)
        
        # Keep only recent history
        if len(self.mouth_history) > self.mouth_history_size:
            self.mouth_history.pop(0)
        
        # Use majority voting from recent history to reduce sensitivity
        if len(self.mouth_history) >= 3:
            mouth_votes = sum(self.mouth_history)
            is_mouth_open = mouth_votes >= 3  # Need at least 3 out of 5 recent detections (more strict)
        else:
            is_mouth_open = current_detection  # Use current detection if not enough history
        
        return {
            "is_mouth_open": is_mouth_open,
            "confidence": confidence,
            "mouth_ratio": mouth_ratio
        }
    
    def analyze_eye_openness(self, frame: np.ndarray, eyes: List[Tuple[int, int, int, int]]) -> Dict[str, bool]:
        """Analyze if eyes are open or closed"""
        if len(eyes) < 2:
            return {"left_eye_open": False, "right_eye_open": False, "both_eyes_open": False}
        
        # Sort eyes by x position (left to right)
        sorted_eyes = sorted(eyes, key=lambda e: e[0])
        left_eye, right_eye = sorted_eyes[0], sorted_eyes[1]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Analyze left eye
        lx, ly, lw, lh = left_eye
        left_eye_roi = gray[ly:ly+lh, lx:lx+lw]
        left_eye_open = self._is_eye_open(left_eye_roi)
        
        # Analyze right eye
        rx, ry, rw, rh = right_eye
        right_eye_roi = gray[ry:ry+rh, rx:rx+rw]
        right_eye_open = self._is_eye_open(right_eye_roi)
        
        return {
            "left_eye_open": left_eye_open,
            "right_eye_open": right_eye_open,
            "both_eyes_open": left_eye_open and right_eye_open
        }
    
    def _is_eye_open(self, eye_roi: np.ndarray) -> bool:
        """Determine if an eye is open using the same logic as gaze_tracker"""
        if eye_roi.size == 0:
            return False
        
        # Resize for consistent analysis
        height, width = eye_roi.shape
        if height < 15 or width < 15:
            return False
        
        # Simple method: analyze the center region of the eye
        center_y = height // 2
        center_region = eye_roi[center_y-3:center_y+3, :]
        
        if center_region.size == 0:
            return False
        
        # Calculate brightness statistics
        mean_brightness = np.mean(eye_roi)
        std_brightness = np.std(eye_roi)
        center_brightness = np.mean(center_region)
        
        # Adaptive thresholds based on eye size for distance robustness
        eye_size_factor = min(height, width) / 20.0  # Normalize by eye size
        
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
        is_open = (std_brightness > min_std_threshold) and (center_brightness > mean_brightness - min_brightness_diff)
        
        return is_open
    
    def analyze_gaze_direction(self, frame: np.ndarray, eyes: List[Tuple[int, int, int, int]], face_coords: Tuple[int, int, int, int]) -> str:
        """Analyze gaze direction based on eye positions"""
        if len(eyes) < 2:
            return "center"
        
        fx, fy, fw, fh = face_coords
        face_center_x = fx + fw // 2
        
        # Calculate average eye position relative to face center
        eye_centers = []
        for (ex, ey, ew, eh) in eyes:
            eye_center_x = ex + ew // 2
            eye_centers.append(eye_center_x)
        
        avg_eye_center = sum(eye_centers) / len(eye_centers)
        gaze_offset = (avg_eye_center - face_center_x) / (fw / 2)
        
        # Determine gaze direction based on offset
        if gaze_offset < -0.2:
            return "left"
        elif gaze_offset > 0.2:
            return "right"
        else:
            return "center"
    
    def get_landmark_data(self, frame: np.ndarray) -> Dict:
        """Get comprehensive landmark data from frame"""
        faces = self.detect_face(frame)
        
        if len(faces) == 0:
            return {
                "faces_detected": 0,
                "landmarks": []
            }
        
        # Get the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = largest_face
        face_roi = frame[fy:fy+fh, fx:fx+fw]
        
        # Detect features
        eyes = self.detect_eyes(frame, face_roi, largest_face)
        smiles = self.detect_smile(frame, face_roi, largest_face)
        
        # Analyze features
        eye_analysis = self.analyze_eye_openness(frame, eyes)
        gaze_direction = self.analyze_gaze_direction(frame, eyes, largest_face)
        
        return {
            "faces_detected": len(faces),
            "landmarks": [{
                "face": largest_face,
                "eyes": eyes,
                "smiles": smiles,
                "eye_analysis": eye_analysis,
                "gaze_direction": gaze_direction,
                "face_roi": face_roi
            }]
        }
    
    def draw_landmarks(self, frame: np.ndarray, landmark_data: Dict) -> np.ndarray:
        """Draw landmarks on the frame"""
        if landmark_data["faces_detected"] == 0:
            return frame
        
        for landmark in landmark_data["landmarks"]:
            face = landmark["face"]
            eyes = landmark["eyes"]
            smiles = landmark["smiles"]
            eye_analysis = landmark["eye_analysis"]
            gaze_direction = landmark["gaze_direction"]
            
            # Draw face rectangle
            fx, fy, fw, fh = face
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Draw eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                cv2.putText(frame, "Eye", (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw smiles
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
                cv2.putText(frame, "Smile", (sx, sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Draw analysis results
            y_offset = fy + fh + 20
            cv2.putText(frame, f"Gaze: {gaze_direction}", (fx, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset += 25
            cv2.putText(frame, f"Eyes: {'Open' if eye_analysis['both_eyes_open'] else 'Closed'}", 
                       (fx, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset += 25
            cv2.putText(frame, f"Smiles: {len(smiles)}", (fx, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

if __name__ == "__main__":
    # Test the facial landmarks detector
    cap = cv2.VideoCapture(0)
    detector = FacialLandmarks()
    
    print("Testing Facial Landmarks Detection...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror effect
        
        # Get landmark data
        landmark_data = detector.get_landmark_data(frame)
        
        # Draw landmarks
        frame = detector.draw_landmarks(frame, landmark_data)
        
        cv2.imshow('Facial Landmarks Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
