import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Dict, List, Tuple, Optional

class HandTracker:
    def __init__(self):
        """Initialize MediaPipe hand tracking"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hands model
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Track both hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Hand gesture recognition thresholds
        self.gesture_thresholds = {
            'thumb_up': 0.7,
            'peace_sign': 0.6,
            'ok_sign': 0.5,
            'fist': 0.6,
            'open_hand': 0.5
        }
    
    def detect_hands(self, frame: np.ndarray) -> List[Dict]:
        """Detect hands and return landmark data"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        hands_data = []
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand label (Left/Right)
                hand_label = results.multi_handedness[idx].classification[0].label
                
                # Convert landmarks to numpy array
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                
                # Detect gesture
                gesture = self._detect_gesture(landmarks, hand_label)
                
                hands_data.append({
                    'landmarks': landmarks,
                    'label': hand_label,
                    'gesture': gesture,
                    'confidence': results.multi_handedness[idx].classification[0].score
                })
        
        return hands_data
    
    def _detect_gesture(self, landmarks: np.ndarray, hand_label: str) -> str:
        """Detect hand gesture based on landmark positions"""
        # Get key landmark indices
        THUMB_TIP = 4
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20
        
        THUMB_MCP = 3
        INDEX_MCP = 5
        MIDDLE_MCP = 9
        RING_MCP = 13
        PINKY_MCP = 17
        WRIST = 0
        
        # Check if fingers are extended
        thumb_up = landmarks[THUMB_TIP][1] < landmarks[THUMB_MCP][1]
        index_up = landmarks[INDEX_TIP][1] < landmarks[INDEX_MCP][1]
        middle_up = landmarks[MIDDLE_TIP][1] < landmarks[MIDDLE_MCP][1]
        ring_up = landmarks[RING_TIP][1] < landmarks[RING_MCP][1]
        pinky_up = landmarks[PINKY_TIP][1] < landmarks[PINKY_MCP][1]
        
        # Count extended fingers
        fingers_up = [thumb_up, index_up, middle_up, ring_up, pinky_up]
        num_fingers = sum(fingers_up)
        
        # Get wrist position (base of hand)
        wrist_y = landmarks[WRIST][1]
        
        # Check if hand is raised (wrist is higher than typical head position)
        # Assuming head is roughly in the upper third of the frame
        hand_raised = wrist_y < 0.4  # Adjust threshold as needed
        
        # Check if hand is touching head area
        # This is a simple approximation - in a real implementation you'd need face detection
        hand_touching_head = wrist_y < 0.3 and wrist_y > 0.1
        
        # Gesture recognition
        if num_fingers == 0:
            return "fist"
        elif num_fingers == 1 and thumb_up:
            return "thumbs_up"
        elif num_fingers == 5:
            return "open_hand"
        elif num_fingers == 1 and index_up:
            return "pointing"
        else:
            return "unknown"
    
    def _calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _is_hand_touching_head(self, hand_landmarks: np.ndarray, face_coords: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> bool:
        """Check if hand landmarks are touching the head area"""
        if face_coords is None:
            return False
        
        fx, fy, fw, fh = face_coords
        frame_h, frame_w = frame_shape[:2]
        
        # Convert face coordinates to normalized coordinates (0-1)
        face_left = fx / frame_w
        face_right = (fx + fw) / frame_w
        face_top = fy / frame_h
        face_bottom = (fy + fh) / frame_h
        
        # Expand the head area slightly for better detection
        head_expansion = 0.1  # 10% expansion
        face_left = max(0, face_left - head_expansion)
        face_right = min(1, face_right + head_expansion)
        face_top = max(0, face_top - head_expansion)
        face_bottom = min(1, face_bottom + head_expansion)
        
        # Check if any hand landmarks are within the head area
        for landmark in hand_landmarks:
            x, y = landmark[0], landmark[1]  # x, y are already normalized (0-1)
            
            # Check if landmark is within head area
            if face_left <= x <= face_right and face_top <= y <= face_bottom:
                return True
        
        return False
    
    def _is_hand_raised_above_head(self, hand_landmarks: np.ndarray, face_coords: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> bool:
        """Check if any part of the hand is above the middle of the head"""
        if face_coords is None:
            return False
        
        fx, fy, fw, fh = face_coords
        frame_h = frame_shape[0]
        
        # Calculate the middle Y position of the head in normalized coordinates
        head_middle_y = (fy + fh/2) / frame_h
        
        # Check if any hand landmark is above the middle of the head
        for landmark in hand_landmarks:
            y = landmark[1]  # y is already normalized (0-1)
            
            # If any landmark is above the middle of the head, hand is raised
            if y < head_middle_y:
                return True
        
        return False
    
    def get_hand_gestures(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int] = None) -> Dict:
        """Get hand gestures from frame - compatible with existing interface"""
        hands_data = self.detect_hands(frame)
        
        result = {
            'left_hand': 'None',
            'right_hand': 'None',
            'special_gesture': 'None'
        }
        
        left_gesture = 'None'
        right_gesture = 'None'
        left_raised = False
        right_raised = False
        left_touching_head = False
        right_touching_head = False
        
        for hand in hands_data:
            landmarks = hand['landmarks']
            wrist_y = landmarks[0][1]  # WRIST = 0
            
            # Check if hand is raised (any part above middle of head)
            hand_raised = False
            hand_touching_head = False
            if face_coords is not None:
                hand_raised = self._is_hand_raised_above_head(landmarks, face_coords, frame.shape)
                hand_touching_head = self._is_hand_touching_head(landmarks, face_coords, frame.shape)
            else:
                # Fallback to old method if no face coordinates
                hand_raised = wrist_y < 0.5
            
            if hand['label'] == 'Left':
                left_gesture = hand['gesture']
                left_raised = hand_raised
                left_touching_head = hand_touching_head
            elif hand['label'] == 'Right':
                right_gesture = hand['gesture']
                right_raised = hand_raised
                right_touching_head = hand_touching_head
        
        # Determine special gestures (prioritize hand touching head)
        if left_touching_head or right_touching_head:
            result['special_gesture'] = 'hand_touching_head'
        elif left_raised and right_raised:
            result['special_gesture'] = 'both_hands_raised'
        elif left_raised or right_raised:
            result['special_gesture'] = 'one_hand_raised'
        
        result['left_hand'] = left_gesture
        result['right_hand'] = right_gesture
        
        return result
    
    def draw_hands(self, frame: np.ndarray, hands_data: List[Dict]) -> np.ndarray:
        """Draw hand landmarks and connections on frame"""
        for hand in hands_data:
            landmarks = hand['landmarks']
            gesture = hand['gesture']
            
            # Convert landmarks to MediaPipe format for drawing
            hand_landmarks = self.mp_hands.HandLandmark
            mp_landmarks = []
            
            for i in range(21):  # MediaPipe has 21 landmarks per hand
                x = int(landmarks[i][0] * frame.shape[1])
                y = int(landmarks[i][1] * frame.shape[0])
                mp_landmarks.append([x, y])
            
            # Draw landmarks
            for landmark in mp_landmarks:
                cv2.circle(frame, tuple(landmark), 5, (0, 255, 0), -1)
            
            # Draw gesture text
            if hand['label'] == 'Left':
                cv2.putText(frame, f"L: {gesture}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(frame, f"R: {gesture}", (frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'hands'):
            self.hands.close()
