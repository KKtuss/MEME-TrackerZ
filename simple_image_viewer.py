import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import json
import os

from facial_landmarks import FacialLandmarks
from gaze_tracker import GazeTracker
# from emotion_detector import EmotionDetector  # Disabled due to protobuf conflicts
from hand_tracker import HandTracker

class SimpleImageViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Facial Expression Image Viewer")
        self.root.geometry("1200x900")
        
        # Initialize detectors
        self.landmarks_detector = FacialLandmarks()
        self.gaze_tracker = GazeTracker()
        self.hand_tracker = HandTracker()
        # self.emotion_detector = EmotionDetector()  # Disabled due to protobuf conflicts
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Image storage - expanded with combinations
        self.images = {
            'eyes_open': None,
            'eyes_closed': None,
            'looking_left': None,
            'looking_right': None,
            'looking_center': None,
            'smiling': None,
            'shocked': None,  # Mouth open without smiling
            # Combined expressions
            'eyes_closed_smiling': None,
            'eyes_open_smiling': None,
            'looking_left_smiling': None,
            'looking_right_smiling': None,
            'looking_center_smiling': None,
            'eyes_closed_neutral': None,
            # Hand gestures
            'thumbs_up': None,
            'thumbs_down': None,
            'open_hand': None,
            'fist': None,
            'pointing': None,
            # Special hand positions
            'one_hand_raised': None,
            'both_hands_raised': None,
            'hand_touching_head': None,
            # Face size detection
            'closeup': None
        }
        
        # Current state
        self.current_expression = None  # Start with no expression
        self.last_valid_expression = None  # Keep track of last valid expression
        self.is_running = False
        
        # Preset management
        self.presets_dir = "presets"
        if not os.path.exists(self.presets_dir):
            os.makedirs(self.presets_dir)
        
        self.setup_ui()
        # self.load_default_images()  # Don't load default images - only show user-selected ones
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top section - Camera and Image side by side
        top_frame = tk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left side - Camera feed (smaller)
        camera_panel = tk.Frame(top_frame)
        camera_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Camera feed label
        tk.Label(camera_panel, text="Camera Feed", font=('Arial', 12, 'bold')).pack(pady=5)
        
        # Camera display (smaller, no fixed size to prevent cropping)
        self.camera_display = tk.Label(camera_panel, bg='black')
        self.camera_display.pack(pady=5)
        
        # Control buttons
        control_frame = tk.Frame(camera_panel)
        control_frame.pack(pady=5, fill=tk.X)
        
        tk.Button(control_frame, text="Start Detection", command=self.start_detection, 
                 bg='green', fg='white').pack(side=tk.LEFT, padx=2)
        
        self.stop_button = tk.Button(control_frame, text="Stop Detection", command=self.stop_detection, 
                                   bg='red', fg='white', state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=2)
        
        # Status display
        self.status_label = tk.Label(camera_panel, text="Status: Stopped", font=('Arial', 10))
        self.status_label.pack(pady=2, fill=tk.X)
        
        # Expression info
        self.expression_label = tk.Label(camera_panel, text="Current Expression: None", font=('Arial', 10))
        self.expression_label.pack(pady=2, fill=tk.X)
        
        # Right side - Image display
        image_panel = tk.Frame(top_frame)
        image_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image display
        tk.Label(image_panel, text="Expression Image", font=('Arial', 12, 'bold')).pack(pady=5)
        
        self.image_display = tk.Label(image_panel, bg='lightgray', width=50, height=20)
        self.image_display.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
        # Bottom section - Image controls
        controls_frame = tk.LabelFrame(main_frame, text="Image Controls", padx=10, pady=10)
        controls_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create image selection buttons
        expressions = [
            ('eyes_open', 'Eyes Open'),
            ('eyes_closed', 'Eyes Closed'),
            ('looking_left', 'Looking Left'),
            ('looking_right', 'Looking Right'),
            ('looking_center', 'Looking Center'),
            ('smiling', 'Smiling'),
            ('shocked', 'Shocked (Mouth Open)'),
            # Combined expressions
            ('eyes_closed_smiling', 'Eyes Closed + Smiling'),
            ('eyes_open_smiling', 'Eyes Open + Smiling'),
            ('looking_left_smiling', 'Looking Left + Smiling'),
            ('looking_right_smiling', 'Looking Right + Smiling'),
            ('looking_center_smiling', 'Looking Center + Smiling'),
            ('eyes_closed_neutral', 'Eyes Closed + Neutral'),
            # Hand gestures
            ('thumbs_up', 'Thumbs Up 👍'),
            ('thumbs_down', 'Thumbs Down 👎'),
            ('open_hand', 'Open Hand ✋'),
            ('fist', 'Fist ✊'),
            ('pointing', 'Pointing 👉'),
            # Special hand positions
            ('one_hand_raised', 'One Hand Raised 🙋‍♂️'),
            ('both_hands_raised', 'Both Hands Raised 🙌'),
            ('hand_touching_head', 'Hand Touching Head 🤔'),
            # Face size detection
            ('closeup', 'Closeup 📸')
        ]
        
        for i, (key, label) in enumerate(expressions):
            row = i // 3
            col = i % 3
            
            btn_frame = tk.Frame(controls_frame)
            btn_frame.grid(row=row, column=col, padx=5, pady=2, sticky=tk.W)
            
            tk.Button(btn_frame, text=f"Select {label}", 
                     command=lambda k=key: self.select_image(k),
                     width=15).pack(side=tk.LEFT)
            
            status_label = tk.Label(btn_frame, text="Not set", fg='red')
            status_label.pack(side=tk.LEFT, padx=5)
            
            # Store reference for updating
            setattr(self, f"{key}_status", status_label)
        
        # Calculate the next available row after all expression buttons
        total_expressions = len(expressions)
        last_expression_row = (total_expressions - 1) // 2  # Last row used by expressions
        next_row = last_expression_row + 1
        
        # Clear all button
        tk.Button(controls_frame, text="Clear All Images", command=self.clear_all_images,
                 bg='orange', fg='white').grid(row=next_row, column=0, columnspan=2, pady=10)
        
        # Preset management section
        preset_frame = tk.LabelFrame(controls_frame, text="Preset Management", padx=5, pady=5)
        preset_frame.grid(row=next_row + 1, column=0, columnspan=2, pady=10, sticky=tk.W+tk.E)
        
        tk.Button(preset_frame, text="Save Preset", command=self.save_preset,
                 bg='blue', fg='white').pack(side=tk.LEFT, padx=2)
        
        tk.Button(preset_frame, text="Load Preset", command=self.load_preset,
                 bg='purple', fg='white').pack(side=tk.LEFT, padx=2)
        
        tk.Button(preset_frame, text="Delete Preset", command=self.delete_preset,
                 bg='darkred', fg='white').pack(side=tk.LEFT, padx=2)
        
        # Auto-trigger checkbox
        self.auto_trigger_var = tk.BooleanVar(value=True)
        tk.Checkbutton(controls_frame, text="Auto-trigger images", 
                      variable=self.auto_trigger_var).grid(row=next_row + 2, column=0, columnspan=2, pady=5)
        
    def load_default_images(self):
        """Load default placeholder images"""
        # Create a simple colored rectangle for each expression type
        default_images = {
            'eyes_open': self.create_default_image("Eyes Open!", "green"),
            'eyes_closed': self.create_default_image("Eyes Closed", "blue"),
            'looking_left': self.create_default_image("Looking Left", "orange"),
            'looking_right': self.create_default_image("Looking Right", "purple"),
            'looking_center': self.create_default_image("Looking Center", "yellow"),
            'neutral': self.create_default_image("Neutral", "gray")
        }
        
        for key, img in default_images.items():
            self.images[key] = img
            
    def create_default_image(self, text, color):
        """Create a default colored image with text"""
        img = Image.new('RGB', (400, 300), color)
        return img
        
    def select_image(self, expression_type):
        """Select an image for a specific expression"""
        file_path = filedialog.askopenfilename(
            title=f"Select image for {expression_type}",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        
        if file_path:
            try:
                img = Image.open(file_path)
                # Resize to fit display
                img = img.resize((400, 300), Image.Resampling.LANCZOS)
                self.images[expression_type] = img
                
                # Update status
                status_label = getattr(self, f"{expression_type}_status")
                status_label.config(text="Set", fg='green')
                
                print(f"Loaded image for {expression_type}: {file_path}")
            except Exception as e:
                print(f"Error loading image: {e}")
                
    def clear_all_images(self):
        """Clear all loaded images"""
        # Clear all images (don't load defaults anymore)
        for key in self.images.keys():
            self.images[key] = None
            
        # Reset all status labels (only if they exist)
        for key in self.images.keys():
            try:
                status_label = getattr(self, f"{key}_status")
                status_label.config(text="Not set", fg='red')
            except AttributeError:
                # Status label doesn't exist (e.g., for removed expressions like "neutral")
                pass
            
        print("All images cleared")
        
    def start_detection(self):
        """Start the facial expression detection"""
        self.is_running = True
        # Update button states
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Frame):
                        for btn in child.winfo_children():
                            if isinstance(btn, tk.Button) and "Start" in btn.cget('text'):
                                btn.config(state=tk.DISABLED)
                            elif isinstance(btn, tk.Button) and "Stop" in btn.cget('text'):
                                btn.config(state=tk.NORMAL)
        
        self.status_label.config(text="Status: Running")
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
    def stop_detection(self):
        """Stop the facial expression detection"""
        self.is_running = False
        # Update button states
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Frame):
                        for btn in child.winfo_children():
                            if isinstance(btn, tk.Button) and "Start" in btn.cget('text'):
                                btn.config(state=tk.NORMAL)
                            elif isinstance(btn, tk.Button) and "Stop" in btn.cget('text'):
                                btn.config(state=tk.DISABLED)
        
        self.status_label.config(text="Status: Stopped")
        
    def detection_loop(self):
        """Main detection loop"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)  # Mirror effect
            
            # Get landmark data
            landmark_data = self.landmarks_detector.get_landmark_data(frame)
            
            if landmark_data["faces_detected"] > 0:
                landmark = landmark_data["landmarks"][0]
                face_coords = landmark["face"]
                eyes = landmark["eyes"]
                eye_analysis = landmark["eye_analysis"]
                
                # Analyze gaze direction
                gaze_result = self.gaze_tracker.analyze_gaze_direction(frame, eyes, face_coords)
                
                # Analyze smile using facial landmarks (more reliable)
                smile_result = self.landmarks_detector.detect_smile_simple(frame, face_coords)
                
                # Analyze mouth opening (for shocked expression)
                mouth_result = self.landmarks_detector.detect_mouth_opening(frame, face_coords)
                
                # Analyze hand gestures (pass face coordinates for hand touching head detection)
                hand_result = self.hand_tracker.get_hand_gestures(frame, face_coords)
                # Get raw hand data for visualization
                hands_data = self.hand_tracker.detect_hands(frame)
                
                # Determine current expression
                new_expression = self.determine_expression(eye_analysis, gaze_result, smile_result, mouth_result, hand_result, face_coords, frame.shape)
                
                # Update display if expression changed
                if new_expression != self.current_expression and self.auto_trigger_var.get():
                    # If new_expression is None (no image set), keep the last valid expression
                    if new_expression is not None:
                        self.current_expression = new_expression
                        self.last_valid_expression = new_expression
                        self.root.after(0, self.update_expression_display)
                    # If new_expression is None, keep showing the last valid expression
                    elif self.last_valid_expression is not None:
                        self.current_expression = self.last_valid_expression
                        # Don't call update_expression_display since we're keeping the same image
                
                # Draw landmarks on frame
                frame = self.landmarks_detector.draw_landmarks(frame, landmark_data)
                
                # Draw hand tracking data
                frame = self.hand_tracker.draw_hands(frame, hands_data)
                
                # Add expression text to frame
                display_expression = self.current_expression if self.current_expression else "None (no image set)"
                cv2.putText(frame, f"Expression: {display_expression}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Smile: {smile_result.get('is_smiling', False)} (count: {smile_result.get('smile_count', 0)})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Mouth Open: {mouth_result.get('is_mouth_open', False)} (ratio: {mouth_result.get('mouth_ratio', 0):.3f})", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # Add closeup detection info
                if face_coords is not None:
                    fx, fy, fw, fh = face_coords
                    frame_height, frame_width = frame.shape[:2]
                    face_area = fw * fh
                    frame_area = frame_width * frame_height
                    face_ratio = face_area / frame_area
                    cv2.putText(frame, f"Face Size: {face_ratio:.2f} ({'CLOSEUP!' if face_ratio > 0.3 else 'Normal'})", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Face Size: No face detected", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                
                # Add gaze debugging info
                raw_gaze = gaze_result.get('raw_gaze', {})
                cv2.putText(frame, f"Gaze: {raw_gaze.get('horizontal', 0.5):.2f}", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Add detailed hand tracking info
                left_hand = hand_result.get('left_hand', 'None')
                right_hand = hand_result.get('right_hand', 'None')
                special_gesture = hand_result.get('special_gesture', 'None')
                
                # Count detected hands
                hand_count = len(hands_data)
                cv2.putText(frame, f"Hands Detected: {hand_count}", 
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Show individual hand data
                for i, hand in enumerate(hands_data):
                    landmarks = hand['landmarks']
                    wrist_y = landmarks[0][1]  # WRIST = 0, get Y coordinate
                    hand_label = hand['label']
                    gesture = hand['gesture']
                    
                    cv2.putText(frame, f"{hand_label}: {gesture} (Y: {wrist_y:.3f})", 
                               (10, 210 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                
                # Show special gesture info
                cv2.putText(frame, f"Special Gesture: {special_gesture}", 
                           (10, 270 + len(hands_data)*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 255), 2)
                
                # Update status
                display_text = self.current_expression if self.current_expression else "None (no image set)"
                self.root.after(0, lambda: self.expression_label.config(
                    text=f"Current Expression: {display_text}"))
            else:
                # No faces detected - set face_coords to None
                face_coords = None
                eye_analysis = {"both_eyes_open": False}
                gaze_result = {"is_eyes_closed": False, "direction": "center", "raw_gaze": {}}
                smile_result = {"is_smiling": False, "smile_count": 0}
                mouth_result = {"is_mouth_open": False, "mouth_ratio": 0.0}
                
                self.root.after(0, lambda: self.expression_label.config(
                    text="Current Expression: No face detected"))
            
            # Update camera display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            # Resize to fit the display area better
            frame_pil = frame_pil.resize((400, 300), Image.Resampling.LANCZOS)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            self.root.after(0, lambda: self.camera_display.config(image=frame_tk))
            self.root.after(0, lambda: setattr(self.camera_display, 'image', frame_tk))
            
            time.sleep(0.03)  # ~30 FPS
            
    def determine_expression(self, eye_analysis, gaze_result, smile_result, mouth_result, hand_result, face_coords=None, frame_shape=None):
        """Determine the current facial expression based on detection results"""
        # Get smile status
        is_smiling = smile_result.get("is_smiling", False)
        
        # Get mouth opening status
        is_mouth_open = mouth_result.get("is_mouth_open", False)
        
        # Check if eyes are closed
        eyes_closed = gaze_result.get("is_eyes_closed", False)
        
        # Check gaze direction
        gaze_direction = gaze_result.get("direction", "center")
        
        # Check if eyes are open
        eyes_open = eye_analysis.get("both_eyes_open", False)
        
        # Get hand gestures (prioritize hand gestures over facial expressions)
        left_hand_gesture = hand_result.get("left_hand")
        right_hand_gesture = hand_result.get("right_hand")
        special_gesture = hand_result.get("special_gesture")
        
        # Check for closeup detection (face size > 50% of camera frame)
        is_closeup = False
        if face_coords is not None and frame_shape is not None and self.images.get("closeup") is not None:
            fx, fy, fw, fh = face_coords
            frame_height, frame_width = frame_shape[:2]
            
            # Calculate face area vs frame area
            face_area = fw * fh
            frame_area = frame_width * frame_height
            face_ratio = face_area / frame_area
            
            # Trigger closeup if face takes up more than 30% of frame
            if face_ratio > 0.3:
                is_closeup = True
        
        # Prioritize closeup detection (highest priority)
        if is_closeup:
            return "closeup"
        
        # Then prioritize special gestures (only if image is set)
        if special_gesture and special_gesture != "None" and self.images.get(special_gesture) is not None:
            return special_gesture
        
        # Then check individual hand gestures (only if image is set)
        if left_hand_gesture and left_hand_gesture != "unknown" and left_hand_gesture != "None" and self.images.get(left_hand_gesture) is not None:
            return left_hand_gesture
        elif right_hand_gesture and right_hand_gesture != "unknown" and right_hand_gesture != "None" and self.images.get(right_hand_gesture) is not None:
            return right_hand_gesture
        
        # Create combined expressions (facial expressions only if no hand gestures and image is set)
        if eyes_closed:
            if is_smiling and self.images.get("eyes_closed_smiling") is not None:
                return "eyes_closed_smiling"
            elif self.images.get("eyes_closed_neutral") is not None:
                return "eyes_closed_neutral"
        elif is_smiling:
            if gaze_direction == "left" and self.images.get("looking_left_smiling") is not None:
                return "looking_left_smiling"
            elif gaze_direction == "right" and self.images.get("looking_right_smiling") is not None:
                return "looking_right_smiling"
            elif gaze_direction == "center" and self.images.get("looking_center_smiling") is not None:
                return "looking_center_smiling"
            elif eyes_open and self.images.get("eyes_open_smiling") is not None:
                return "eyes_open_smiling"
            elif self.images.get("smiling") is not None:
                return "smiling"
        elif is_mouth_open and not is_smiling:
            # Mouth open without smiling = shocked
            if self.images.get("shocked") is not None:
                return "shocked"
        elif eyes_open:
            if gaze_direction == "left" and self.images.get("looking_left") is not None:
                return "looking_left"
            elif gaze_direction == "right" and self.images.get("looking_right") is not None:
                return "looking_right"
            elif gaze_direction == "center" and self.images.get("looking_center") is not None:
                return "looking_center"
        
        # If no image is set for any detected expression, return None to show no image
        return None
        
    def update_expression_display(self):
        """Update the image display based on current expression"""
        if self.current_expression and self.current_expression in self.images and self.images[self.current_expression] is not None:
            # Convert PIL image to PhotoImage
            img_tk = ImageTk.PhotoImage(self.images[self.current_expression])
            
            # Update the display
            self.image_display.config(image=img_tk)
            self.image_display.image = img_tk  # Keep a reference
            
            print(f"Displaying image for: {self.current_expression}")
        else:
            # Clear display when no image is set for the detected expression
            self.image_display.config(image='')
            self.image_display.image = None
            if self.current_expression is None:
                print("No image set for detected expression - clearing display")
    
    def save_preset(self):
        """Save current image configuration as a preset"""
        # Get preset name from user
        preset_name = tk.simpledialog.askstring("Save Preset", "Enter preset name:")
        if not preset_name:
            return
        
        # Validate preset name (no special characters)
        if not preset_name.replace("_", "").replace("-", "").isalnum():
            messagebox.showerror("Invalid Name", "Preset name can only contain letters, numbers, underscores, and hyphens.")
            return
        
        preset_path = os.path.join(self.presets_dir, f"{preset_name}.json")
        
        # Check if preset already exists
        if os.path.exists(preset_path):
            if not messagebox.askyesno("Overwrite Preset", f"Preset '{preset_name}' already exists. Overwrite?"):
                return
        
        # Collect current image paths
        preset_data = {}
        for key, image_data in self.images.items():
            # Check if it's a file path (string) or PIL Image object
            if image_data and isinstance(image_data, str) and os.path.exists(image_data):
                preset_data[key] = image_data
        
        # Count actual images (before adding metadata)
        image_count = len(preset_data)
        
        # Add metadata
        preset_data["_metadata"] = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_images": image_count
        }
        
        try:
            with open(preset_path, 'w') as f:
                json.dump(preset_data, f, indent=2)
            messagebox.showinfo("Success", f"Preset '{preset_name}' saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save preset: {str(e)}")
    
    def load_preset(self):
        """Load a preset configuration"""
        # Get list of available presets
        preset_files = [f for f in os.listdir(self.presets_dir) if f.endswith('.json')]
        if not preset_files:
            messagebox.showinfo("No Presets", "No presets found. Save some image configurations first!")
            return
        
        # Create preset selection dialog
        preset_window = tk.Toplevel(self.root)
        preset_window.title("Load Preset")
        preset_window.geometry("400x300")
        preset_window.transient(self.root)
        preset_window.grab_set()
        
        # Center the window
        preset_window.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        tk.Label(preset_window, text="Select a preset to load:", font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Preset listbox
        listbox_frame = tk.Frame(preset_window)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        listbox = tk.Listbox(listbox_frame)
        scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate listbox with preset info
        for preset_file in sorted(preset_files):
            preset_name = preset_file[:-5]  # Remove .json extension
            preset_path = os.path.join(self.presets_dir, preset_file)
            
            try:
                with open(preset_path, 'r') as f:
                    preset_data = json.load(f)
                metadata = preset_data.get("_metadata", {})
                created = metadata.get("created", "Unknown")
                image_count = metadata.get("total_images", 0)
                listbox.insert(tk.END, f"{preset_name} ({image_count} images) - {created}")
            except:
                listbox.insert(tk.END, f"{preset_name} (Error loading info)")
        
        # Buttons
        button_frame = tk.Frame(preset_window)
        button_frame.pack(pady=10)
        
        def load_selected():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a preset to load.")
                return
            
            selected_file = sorted(preset_files)[selection[0]]
            preset_path = os.path.join(self.presets_dir, selected_file)
            
            try:
                with open(preset_path, 'r') as f:
                    preset_data = json.load(f)
                
                # Clear current images
                self.clear_all_images()
                
                # Load preset images
                loaded_count = 0
                for key, image_path in preset_data.items():
                    if key.startswith("_"):  # Skip metadata
                        continue
                    if os.path.exists(image_path):
                        self.images[key] = image_path
                        # Update status label if it exists
                        try:
                            status_label = getattr(self, f"{key}_status")
                            status_label.config(text="Set", fg='green')
                        except AttributeError:
                            # Status label doesn't exist (e.g., for removed expressions)
                            pass
                        loaded_count += 1
                
                preset_window.destroy()
                messagebox.showinfo("Success", f"Loaded preset with {loaded_count} images!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load preset: {str(e)}")
        
        tk.Button(button_frame, text="Load", command=load_selected, bg='green', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=preset_window.destroy).pack(side=tk.LEFT, padx=5)
    
    def delete_preset(self):
        """Delete a preset"""
        # Get list of available presets
        preset_files = [f for f in os.listdir(self.presets_dir) if f.endswith('.json')]
        if not preset_files:
            messagebox.showinfo("No Presets", "No presets found to delete.")
            return
        
        # Create preset selection dialog (similar to load_preset)
        preset_window = tk.Toplevel(self.root)
        preset_window.title("Delete Preset")
        preset_window.geometry("400x300")
        preset_window.transient(self.root)
        preset_window.grab_set()
        
        # Center the window
        preset_window.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        tk.Label(preset_window, text="Select a preset to delete:", font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Preset listbox
        listbox_frame = tk.Frame(preset_window)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        listbox = tk.Listbox(listbox_frame)
        scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate listbox
        for preset_file in sorted(preset_files):
            preset_name = preset_file[:-5]  # Remove .json extension
            listbox.insert(tk.END, preset_name)
        
        # Buttons
        button_frame = tk.Frame(preset_window)
        button_frame.pack(pady=10)
        
        def delete_selected():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a preset to delete.")
                return
            
            selected_file = sorted(preset_files)[selection[0]]
            preset_name = selected_file[:-5]
            preset_path = os.path.join(self.presets_dir, selected_file)
            
            if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete preset '{preset_name}'?"):
                try:
                    os.remove(preset_path)
                    preset_window.destroy()
                    messagebox.showinfo("Success", f"Preset '{preset_name}' deleted successfully!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to delete preset: {str(e)}")
        
        tk.Button(button_frame, text="Delete", command=delete_selected, bg='red', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=preset_window.destroy).pack(side=tk.LEFT, padx=5)
            
    def on_closing(self):
        """Handle application closing"""
        self.stop_detection()
        if self.cap:
            self.cap.release()
        # Clean up hand tracker
        if hasattr(self, 'hand_tracker'):
            self.hand_tracker.cleanup()
        cv2.destroyAllWindows()
        self.root.destroy()
        
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

if __name__ == "__main__":
    app = SimpleImageViewer()
    app.run()
