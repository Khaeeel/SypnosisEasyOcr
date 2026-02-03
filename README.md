# SypnosisEasyOcr

## Requirements

### System Requirements
- Python 3.7 or higher
- Windows OS (paths use Windows format)
- 2GB minimum RAM (4GB+ recommended for OCR processing)

### Python Dependencies
- **easyocr** - Optical Character Recognition library
- **opencv-python (cv2)** - Computer vision and image processing
- **numpy** - Numerical computing
- **Pillow** - Image processing
- **regex** - Regular expression operations
- **difflib** - Built-in Python library for sequence matching

### Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install easyocr opencv-python numpy pillow
   ```

### Optional
- **GPU Support**: If you have an NVIDIA GPU, install CUDA and cuDNN for faster OCR processing
  ```bash
  pip install easyocr[gpu]
  ```

## Project Structure
- `test_ocr.py` - Main OCR processing script for Viber screenshots
- `bubble_check.py` - Bubble detection and text extraction from chat messages
- `viber_fixed_files.json` - Output file containing extracted messages with OCR results
- `sample image/` - Directory containing sample Viber screenshots for processing
- `experiment output/` - Directory for output results