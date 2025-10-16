# 🎭 MEME Tracker

A real-time facial expression and hand gesture detection application that displays custom images based on detected expressions and gestures.

## 🌟 Features

### **Facial Expression Detection**
- 😊 **Smile Detection**: Detects both open-mouth and closed-mouth smiles
- 👀 **Eye Tracking**: Monitors eye openness and gaze direction (left, right, center)
- 😮 **Mouth Opening**: Detects when mouth is open (shocked expression)
- 📸 **Closeup Detection**: Triggers when face takes up >30% of camera frame

### **Hand Gesture Recognition**
- ✋ **Hand Raised**: One or both hands raised above head
- 🤚 **Hand Touching Head**: Hand landmarks touching face area
- 🤏 **Individual Gestures**: Thumbs up, pointing, open hand, fist

### **Image Management**
- 🖼️ **Custom Images**: Upload your own images for each expression
- 💾 **Preset System**: Save and load image configurations
- 🎨 **Real-time Display**: Instant image switching based on detection

## 🚀 Quick Start

### **Desktop Version**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python simple_image_viewer.py
```

### **Web Version**
```bash
# Install web dependencies
pip install -r web_requirements.txt

# Run web server
python web_app.py

# Open http://localhost:8000 in browser
```

## 📋 Requirements

### **Desktop App**
- Python 3.12
- OpenCV 4.8+
- MediaPipe 0.10+
- NumPy 1.24+
- Pillow 10.0+

### **Web App**
- All desktop requirements plus:
- FastAPI 0.104+
- Uvicorn 0.24+
- WebSockets 12.0+

## 🎮 How to Use

1. **Launch the App**: Run either desktop or web version
2. **Allow Camera Access**: Grant permissions when prompted
3. **Upload Images**: Click buttons to upload custom images for expressions
4. **Start Detection**: Click "Start Detection" button
5. **Enjoy**: Watch as images change based on your expressions!

## 🎯 Supported Expressions

| Expression | Description | Trigger |
|------------|-------------|---------|
| `smiling` | General smile detection | Any smile |
| `eyes_closed_smiling` | Smiling with eyes closed | Smile + eyes closed |
| `eyes_closed_neutral` | Eyes closed, not smiling | Eyes closed, no smile |
| `looking_left` | Gaze to the left | Looking left |
| `looking_right` | Gaze to the right | Looking right |
| `looking_center` | Gaze straight ahead | Looking center |
| `looking_center_smiling` | Smiling while looking center | Smile + center gaze |
| `shocked` | Mouth open, not smiling | Mouth open, no smile |
| `closeup` | Face very close to camera | Face >30% of frame |
| `one_hand_raised` | One hand above head | Single raised hand |
| `both_hands_raised` | Both hands above head | Both hands raised |
| `hand_touching_head` | Hand touching face area | Hand-face collision |

## 🛠️ Technical Details

### **Detection Pipeline**
1. **Face Detection**: OpenCV Haar cascades for face localization
2. **Landmark Analysis**: MediaPipe for precise facial landmarks
3. **Expression Classification**: Custom algorithms for smile, gaze, eye openness
4. **Hand Tracking**: MediaPipe hand landmarks for gesture recognition
5. **Image Display**: Tkinter (desktop) or WebSocket (web) for real-time updates

### **Adaptive Detection**
- **Face Size Adaptation**: All thresholds adjust based on face distance
- **Debouncing**: Prevents flickering between expressions
- **Multi-frame Validation**: Ensures stable detection results

## 🌐 Web Deployment

### **Heroku (Recommended)**
```bash
# Create Heroku app
heroku create your-meme-tracker-app

# Deploy
git push heroku main
```

### **Railway**
1. Connect GitHub repository
2. Auto-deploy from repository
3. Access via provided URL

See `deploy_instructions.md` for detailed deployment guide.

## 📁 Project Structure

```
MEMETRACKER/
├── simple_image_viewer.py      # Desktop GUI application
├── web_app.py                  # Web application (FastAPI)
├── facial_landmarks.py         # Face and expression detection
├── gaze_tracker.py             # Eye tracking and gaze detection
├── hand_tracker.py             # Hand gesture recognition
├── create_test_images.py       # Test image generator
├── requirements.txt            # Desktop dependencies
├── web_requirements.txt        # Web dependencies
├── deploy_instructions.md      # Deployment guide
└── README.md                   # This file
```

## 🎨 Customization

### **Adding New Expressions**
1. Add expression to `self.images` dictionary
2. Create UI button in `setup_ui()`
3. Add detection logic in `determine_expression()`
4. Update `create_test_images.py` for test images

### **Adjusting Detection Sensitivity**
- **Smile**: Modify `smile_params` in `facial_landmarks.py`
- **Eye Detection**: Adjust `eye_params` in detection modules
- **Hand Gestures**: Modify thresholds in `hand_tracker.py`

## 🐛 Troubleshooting

### **Common Issues**
- **Camera not working**: Check camera permissions and close other apps using camera
- **Detection not accurate**: Adjust lighting and ensure good face visibility
- **Performance issues**: Reduce camera resolution or detection frequency

### **Debug Mode**
The application includes debug overlays showing:
- Face detection confidence
- Smile count and confidence
- Eye openness status
- Gaze direction
- Hand detection count

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **OpenCV**: Computer vision library
- **MediaPipe**: Hand and facial landmark detection
- **FastAPI**: Modern web framework
- **Tkinter**: Desktop GUI framework

---

**Made with ❤️ for fun facial expression tracking!**