Below is the updated `README.md` for your face-swapping project, with the video face-swapping section removed and code blocks retained only where necessary for clarity. The `inswapper_128.onnx` weights file link is included, and the content is formatted for easy copy-and-paste, maintaining a formal tone and structured layout.

```markdown
# Face Swapping with InsightFace

This project enables face swapping between two images using the InsightFace library, which provides state-of-the-art face analysis and swapping capabilities. The application detects faces in a target image and swaps them with a face from a source image, supporting both CPU and GPU acceleration.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

## Features
- Detect faces in images with high accuracy.
- Swap faces from a source image onto multiple faces in a target image.
- Support for both GPU and CPU computation via configurable `ctx_id`.
- Modular codebase for easy extension and customization.

## Prerequisites
- Python 3.8 or higher
- Git
- A compatible operating system (Windows, Linux, or macOS)
- For GPU acceleration: NVIDIA GPU with CUDA support and appropriate drivers
- Images in a supported format (e.g., JPG, PNG)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-swapping.git
   cd face-swapping
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   .\venv\Scripts\activate   # On Windows
   ```

3. Install required dependencies:
   ```bash
   pip install insightface onnxruntime opencv-python matplotlib
   ```
   For GPU acceleration, install `onnxruntime-gpu` instead:
   ```bash
   pip install insightface onnxruntime-gpu opencv-python matplotlib
   ```

4. Download the `inswapper_128.onnx` weights file:
   - Access the weights file from [this Google Drive link](https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view?usp=sharing).
   - Place the `inswapper_128.onnx` file in the project directory or a designated `models` folder.
   - Update the script to reference the correct path to `inswapper_128.onnx` (see [Customization](#customization)).

## Usage
### 1. Prepare Images
Place the following images in the project directory:
- `target_image.jpg`: The image containing one or more faces to be swapped.
- `source_image.jpg`: The image containing a single face to apply to the target.

### 2. Run the Script
Execute the main script to perform face swapping:
   ```bash
   python face_swap.py
   ```

### 3. Command-Line Arguments
The script supports optional command-line arguments for custom input and output paths. Example usage:
   ```bash
   python face_swap.py --target my_group_photo.jpg --source my_face.jpg --output result.jpg
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
Add the following to `main()` to save the output image:
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
This project is licensed under the MIT License. It is free for both personal and commercial use.

## Contact
For questions or issues, please open an issue on the GitHub repository or contact `your.email@example.com`.

---

### Notes for Updating This README
1. Replace `example_output.png` with a screenshot of your project's output.
2. Update the **Installation** and **Usage** sections if new dependencies or features are added.
3. Expand the **Customization** section to document any additional configuration options you implement.
```

### Notes
- The video face-swapping section has been removed as requested.
- Code blocks are included only for essential parts: installation commands, command-line argument parsing, and customization snippets (detection thresholds, weights file loading, and output saving).
- The `inswapper_128.onnx` weights file link is retained in the **Installation** section, with instructions for placement and script configuration.
- The content is formatted with triple backticks (```) for code blocks, ensuring easy copy-and-paste.
- The structure remains consistent, professional, and concise.

You can copy the entire content above and paste it into your `README.md` file. Let me know if you need further adjustments or additional details!