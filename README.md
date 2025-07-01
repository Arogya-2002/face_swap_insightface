
```markdown
# Face Swapping with InsightFace

This project enables face swapping between two images using the InsightFace library, leveraging state-of-the-art face analysis and swapping capabilities. It detects faces in a target image and replaces them with a face from a source image, supporting both CPU and GPU acceleration.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Clone the Repository](#clone-the-repository)
  - [Set Up Virtual Environment](#set-up-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Download Weights File](#download-weights-file)
- [Usage](#usage)
  - [Prepare Images](#prepare-images)
  - [Run the Script](#run-the-script)
  - [Command-Line Arguments](#command-line-arguments)
- [Code Structure](#code-structure)
- [Customization](#customization)
  - [Adjust Detection Thresholds](#adjust-detection-thresholds)
  - [Load the Weights File](#load-the-weights-file)
  - [Save Output Programmatically](#save-output-programmatically)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

## Features
- Accurate face detection in images.
- Face swapping from a source image onto multiple faces in a target image.
- Support for GPU and CPU computation via configurable `ctx_id`.
- Modular codebase for easy extension and customization.

## Prerequisites
- Python 3.8 or higher
- Git
- Compatible operating system (Windows, Linux, or macOS)
- For GPU acceleration: NVIDIA GPU with CUDA support and appropriate drivers
- Images in supported formats (e.g., JPG, PNG)

## Installation

### Clone the Repository
Clone the project repository to your local machine:
```bash
git clone https://github.com/yourusername/face-swapping.git
cd face-swapping
```

### Set Up Virtual Environment
Create and activate a virtual environment to isolate dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
.\venv\Scripts\activate   # On Windows
```

### Install Dependencies
Install the required Python packages:
```bash
pip install insightface onnxruntime opencv-python matplotlib
```
For GPU acceleration, install the GPU-specific package:
```bash
pip install insightface onnxruntime-gpu opencv-python matplotlib
```

### Download Weights File
Download the `inswapper_128.onnx` weights file:
- Access the weights file from [this Google Drive link](https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view?usp=sharing).
- Place the `inswapper_128.onnx` file in the project directory or a `models` subdirectory.
- Update the script to reference the correct path to `inswapper_128.onnx` (see [Load the Weights File](#load-the-weights-file)).

## Usage

### Prepare Images
Place the following images in the project directory:
- `target_image.jpg`: Image containing one or more faces to be swapped.
- `source_image.jpg`: Image containing a single face to apply to the target.

### Run the Script
Execute the main script to perform face swapping:
```bash
python face_swap.py
```

### Command-Line Arguments
The script supports optional command-line arguments for custom input and output paths. Example usage:
```bash
python face_swap.py --target my_group_photo.jpg --source my_face.jpg --output result.jpg
```

To enable command-line argument parsing, include the following in `face_swap.py`:
```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="Face swapping using InsightFace")
    parser.add_argument("--target", type=str, default="target_image.jpg", help="Path to target image with multiple faces")
    parser.add_argument("--source", type=str, default="source_image.jpg", help="Path to source image with single face")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save the output image")
    args = parser.parse_args()

    # Load images
    img_multi_faces = cv2.imread(args.target)
    img_single_face = cv2.imread(args.source)
```

## Code Structure
| Component                | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `face_swap.py`          | Main script containing modular functions for face detection and swapping.   |
| `initialize_face_analysis()` | Initializes the `buffalo_l` model for face detection and analysis.       |
| `perform_face_swapping()`   | Performs the face-swapping operation using `inswapper_128.onnx`.         |
| `main()`                | Orchestrates the workflow: loads images, detects faces, swaps faces, and saves/displays the result. |

## Customization

### Adjust Detection Thresholds
Modify the detection parameters in `initialize_face_analysis()` to adjust sensitivity:
```python
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)  # Lower det_thresh to detect more faces
```

### Load the Weights File
Ensure the `inswapper_128.onnx` weights file is loaded correctly in `face_swap.py`:
```python
from insightface.app import FaceAnalysis
from insightface.utils import face_swapper

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = face_swapper.get_swapper('path/to/inswapper_128.onnx')  # Update path as needed
```

### Save Output Programmatically
To save the output image, add the following to `main()`:
```python
cv2.imwrite(args.output, cv2.cvtColor(swapped_img, cv2.COLOR_RGB2BGR))
```

## Troubleshooting
| Issue                          | Solution                                                                 |
|--------------------------------|--------------------------------------------------------------------------|
| `No module named 'onnxruntime'` | Install the missing package: `pip install onnxruntime` or `onnxruntime-gpu`. |
| `No faces detected`            | Ensure images are clear and well-lit. Lower `det_thresh` or use higher-resolution images. |
| `CUDA out of memory`           | Reduce `det_size` (e.g., `(320, 320)`) or switch to CPU mode (`ctx_id=-1`). |
| `inswapper_128.onnx not found` | Verify the weights file is in the correct path and update the script accordingly. |

## License
This project is licensed under the MIT License, permitting both personal and commercial use.

## Contact
For questions or issues, open an issue on the GitHub repository or contact `your.email@example.com`.

---

### Notes for Updating This README
1. Replace `example_output.png` with a screenshot of your project's output.
2. Update the **Installation** and **Usage** sections if new dependencies or features are added.
3. Expand the **Customization** section to document any additional configuration options implemented.
```

