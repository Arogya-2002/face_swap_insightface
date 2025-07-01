import cv2
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis

def initialize_face_analysis(ctx_id=0, det_size=(640, 640)):
    """Initialize the FaceAnalysis model for face detection."""
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app

def perform_face_swapping(app, swapper, img_multi_faces, img_single_face):
    """Perform face swapping between two images."""
    # Convert images to RGB
    img_multi_faces_rgb = cv2.cvtColor(img_multi_faces, cv2.COLOR_BGR2RGB)
    img_single_face_rgb = cv2.cvtColor(img_single_face, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces_multi = app.get(img_multi_faces_rgb)
    faces_single = app.get(img_single_face_rgb)

    # Check if faces are detected
    if len(faces_multi) == 0:
        raise ValueError("No faces detected in the multi-face image!")
    if len(faces_single) == 0:
        raise ValueError("No faces detected in the single-face image!")

    # Select the first face from the single-face image as the source
    source_face = faces_single[0]

    # Perform face swapping
    result_image = img_multi_faces_rgb.copy()
    for face in faces_multi:
        result_image = swapper.get(result_image, face, source_face, paste_back=True)

    return img_multi_faces_rgb, img_single_face_rgb, result_image

def main():
    """Main function to execute face swapping."""
    # Initialize models
    app = initialize_face_analysis(ctx_id=0)  # Use ctx_id=0 for CPU, ctx_id=0+ for GPU
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False)

    # Load images
    img_multi_faces = cv2.imread("example-1.jpeg")  # Image with multiple faces
    img_single_face = cv2.imread("example-2.jpeg")  # Image with single face (source)

    # Perform face swapping
    original_img, source_img, swapped_img = perform_face_swapping(app, swapper, img_multi_faces, img_single_face)

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original Image (Multiple Faces)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(source_img)
    plt.title("Source Face (Single Face)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(swapped_img)
    plt.title("Result After Face Swapping")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()