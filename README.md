# Spades - Neural Style Transfer

Spades is a Neural Style Transfer (NST) project that applies artistic styles to images using deep learning. The project leverages a pre-trained VGG19 model to extract content and style features, blending them to create visually stunning artwork.

## Features
- Uses **VGG19** for feature extraction
- Transfers artistic styles from one image to another
- Supports high-resolution image processing
- Can be extended into a web-based application using Flask

## Tech Stack
- **Python** (Primary language)
- **TensorFlow / PyTorch** (Deep learning framework)
- **OpenCV** (Image processing)
- **Matplotlib** (Visualization)
- **Flask** (Optional - for a web UI)

## Installation
Ensure you have Python installed, then install dependencies:
```bash
pip install tensorflow torch torchvision opencv-python numpy matplotlib flask
```

## Project Structure
```
Spades/
│── main.py                # Runs the NST process
│── style_transfer.py       # Core NST logic
│── utils.py                # Helper functions
│── models/                 # Pre-trained VGG19
│── images/                 # Input and output images
│── static/                 # (For Flask) Stores processed images
│── templates/              # (For Flask) HTML files for UI
```

## Usage
Run the script to apply style transfer:
```bash
python main.py
```

## Next Steps
- Implement optimization for style blending
- Add a Flask web interface
- Allow real-time NST with video processing

## License
This project is open-source under the MIT License.

