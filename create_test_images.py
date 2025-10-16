from PIL import Image, ImageDraw, ImageFont
import os

def create_test_images():
    """Create test images for different facial expressions"""
    
    # Create sample_images directory if it doesn't exist
    os.makedirs('sample_images', exist_ok=True)
    
    # Define test images
    test_images = {
        'eyes_open': {
            'color': (0, 255, 0),  # Green
            'text': 'Eyes Open!',
            'emoji': '👀'
        },
        'eyes_closed': {
            'color': (0, 0, 255),  # Blue
            'text': 'Eyes Closed',
            'emoji': '😴'
        },
        'looking_left': {
            'color': (255, 165, 0),  # Orange
            'text': 'Looking Left',
            'emoji': '👀⬅️'
        },
        'looking_right': {
            'color': (128, 0, 128),  # Purple
            'text': 'Looking Right',
            'emoji': '👀➡️'
        },
        'looking_center': {
            'color': (255, 255, 0),  # Yellow
            'text': 'Looking Center',
            'emoji': '👀⬆️'
        },
        'smiling': {
            'color': (255, 0, 0),  # Red
            'text': 'Smiling!',
            'emoji': '😊'
        },
    'shocked': {
        'color': (255, 0, 255),  # Magenta
        'text': 'Shocked!',
        'emoji': '😮'
    },
    # Hand gestures
    'thumbs_up': {
        'color': (0, 255, 0),  # Green
        'text': 'Thumbs Up!',
        'emoji': '👍'
    },
    'thumbs_down': {
        'color': (255, 0, 0),  # Red
        'text': 'Thumbs Down!',
        'emoji': '👎'
    },
    'open_hand': {
        'color': (128, 0, 128),  # Purple
        'text': 'Open Hand!',
        'emoji': '✋'
    },
    'fist': {
        'color': (64, 64, 64),  # Dark Gray
        'text': 'Fist!',
        'emoji': '✊'
    },
    'pointing': {
        'color': (255, 192, 203),  # Pink
        'text': 'Pointing!',
        'emoji': '👉'
    },
    # Special hand positions
    'one_hand_raised': {
        'color': (255, 165, 0),  # Orange
        'text': 'One Hand Raised!',
        'emoji': '🙋‍♂️'
    },
    'both_hands_raised': {
        'color': (0, 255, 255),  # Cyan
        'text': 'Both Hands Raised!',
        'emoji': '🙌'
    },
    'hand_touching_head': {
        'color': (255, 255, 0),  # Yellow
        'text': 'Hand Touching Head!',
        'emoji': '🤔'
    },
    'closeup': {
        'color': (255, 0, 128),  # Hot Pink
        'text': 'CLOSEUP DETECTED!',
        'emoji': '📸'
    },
        # Combined expressions
        'eyes_closed_smiling': {
            'color': (255, 128, 0),  # Orange
            'text': 'Eyes Closed + Smiling',
            'emoji': '😴😊'
        },
        'eyes_open_smiling': {
            'color': (128, 255, 0),  # Lime
            'text': 'Eyes Open + Smiling',
            'emoji': '👀😊'
        },
        'looking_left_smiling': {
            'color': (128, 0, 255),  # Purple
            'text': 'Looking Left + Smiling',
            'emoji': '👀⬅️😊'
        },
        'looking_right_smiling': {
            'color': (255, 128, 128),  # Pink
            'text': 'Looking Right + Smiling',
            'emoji': '👀➡️😊'
        },
        'looking_center_smiling': {
            'color': (128, 255, 128),  # Light Green
            'text': 'Looking Center + Smiling',
            'emoji': '👀⬆️😊'
        },
        'eyes_closed_neutral': {
            'color': (192, 192, 192),  # Light Gray
            'text': 'Eyes Closed + Neutral',
            'emoji': '😴😐'
        }
    }
    
    # Create each test image
    for expression, data in test_images.items():
        # Create image
        img = Image.new('RGB', (400, 300), data['color'])
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback if not available
        try:
            font = ImageFont.truetype("arial.ttf", 24)
            emoji_font = ImageFont.truetype("arial.ttf", 48)
        except:
            font = ImageFont.load_default()
            emoji_font = ImageFont.load_default()
        
        # Draw text
        text_bbox = draw.textbbox((0, 0), data['text'], font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_x = (400 - text_width) // 2
        text_y = 100
        
        draw.text((text_x, text_y), data['text'], fill='white', font=font)
        
        # Draw emoji
        emoji_bbox = draw.textbbox((0, 0), data['emoji'], font=emoji_font)
        emoji_width = emoji_bbox[2] - emoji_bbox[0]
        emoji_x = (400 - emoji_width) // 2
        emoji_y = 150
        
        draw.text((emoji_x, emoji_y), data['emoji'], fill='white', font=emoji_font)
        
        # Save image
        filename = f'sample_images/{expression}.png'
        img.save(filename)
        print(f"Created: {filename}")
    
    print("\nTest images created successfully!")
    print("You can now run the facial expression image app and load these images.")

if __name__ == "__main__":
    create_test_images()
