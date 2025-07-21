from paddleocr import PaddleOCR
import cv2
import os
import re
import numpy as np
from typing import List, Tuple, Optional


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by specified angle while maintaining full content visibility."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounding dimensions
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    new_width = int((height * sin_val) + (width * cos_val))
    new_height = int((height * cos_val) + (width * sin_val))

    # Adjust rotation matrix for translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Perform rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Apply preprocessing to improve OCR accuracy."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply adaptive thresholding to handle varying lighting
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Apply morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    # Convert back to BGR for consistency
    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    return processed


def detect_and_correct_orientation(image: np.ndarray, ocr: PaddleOCR) -> Tuple[np.ndarray, float]:
    """Detect image orientation and return corrected image with rotation angle."""
    # Try different angles to find the best orientation
    test_angles = [0, 90, 180, 270]
    best_score = 0
    best_angle = 0
    best_result = None

    for angle in test_angles:
        # Rotate image
        if angle == 0:
            test_image = image.copy()
        else:
            test_image = rotate_image(image, angle)

        # Run OCR on rotated image
        try:
            result = ocr.predict(test_image)
            if result and result[0]:
                # Calculate average confidence score
                scores = []
                for item in result[0]:
                    try:
                        if isinstance(item[1], tuple) and len(item[1]) >= 2:
                            score = item[1][1]
                            if isinstance(score, float) and score > 0.5:
                                scores.append(score)
                    except Exception as e:
                        print(f"⚠️ Skipping malformed OCR item: {e}")
                        continue
                if scores:
                    avg_score = np.mean(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_angle = angle
                        best_result = result
        except Exception as e:
            print(f"Error processing angle {angle}: {e}")
            continue

    # Return best oriented image
    if best_angle == 0:
        return image, 0
    else:
        return rotate_image(image, best_angle), best_angle


def run_ocr_and_draw_boxes(image_path: str, output_dir: str,
                           test_rotations: bool = True,
                           preprocess: bool = True,
                           confidence_threshold: float = 0.6):
    """
    Run OCR on image with multi-angle processing capabilities.

    Args:
        image_path: Path to input image
        output_dir: Directory to save output
        test_rotations: Whether to test different rotations for best results
        preprocess: Whether to apply image preprocessing
        confidence_threshold: Minimum confidence score to accept OCR results
    """

    # Create PaddleOCR object with orientation classification enabled
    ocr = PaddleOCR(
        use_doc_orientation_classify=True,  # Enable document orientation detection
        use_doc_unwarping=True,  # Enable document unwarping
        use_textline_orientation=True,  # Enable text line orientation
        lang='en',
    )

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return

    original_image = image.copy()

    # Apply preprocessing if requested
    if preprocess:
        image = preprocess_image(image)

    # Detect and correct orientation if requested
    rotation_angle = 0
    if test_rotations:
        print("🔄 Testing different orientations...")
        image, rotation_angle = detect_and_correct_orientation(image, ocr)
        if rotation_angle != 0:
            print(f"📐 Best orientation found at {rotation_angle}° rotation")

    # Run OCR on the (possibly corrected) image
    print("🔍 Running OCR...")
    try:
        result = ocr.predict(image)

        if not result or not result[0]:
            print("❌ No text detected in image")
            return

        result = result[0]  # Extract the main result

    except Exception as e:
        print(f"❌ OCR processing failed: {e}")
        return

    # Process results - handle both old and new PaddleOCR formats
    detected_texts = []

    for item in result:
        try:
            # New format: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)]
            if len(item) == 2 and len(item[0]) == 4:
                box = np.array(item[0]).astype(int)
                text = item[1][0]
                score = item[1][1]
            else:
                # Fallback for other formats
                continue

            # Filter by confidence threshold
            if score >= confidence_threshold:
                detected_texts.append((box, text, score))

        except Exception as e:
            print(f"Warning: Could not process OCR result item: {e}")
            continue

    print(f"📝 Found {len(detected_texts)} text regions above confidence threshold ({confidence_threshold})")

    # Draw boxes and text on the original image
    display_image = original_image.copy()

    # If we rotated the image, we need to adjust coordinates back to original orientation
    if rotation_angle != 0:
        # For visualization, let's use the rotated image instead
        display_image = image.copy()

    valid_detections = 0
    for box, text, score in detected_texts:
        # Filter for 1-4 digits pattern (adjust regex as needed)
        if re.fullmatch(r'\d{1,4}', text.strip()):
            valid_detections += 1

            # Draw bounding box
            cv2.polylines(display_image, [box.reshape((-1, 1, 2))],
                          isClosed=True, color=(0, 0, 255), thickness=2)

            # Draw text and confidence score
            text_label = f"{text} ({score:.2f})"
            cv2.putText(
                display_image,
                text_label,
                tuple(box[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

            print(f"  📋 Detected: '{text}' (confidence: {score:.3f})")

    # Save result
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create descriptive filename
    rotation_suffix = f"_rot{int(rotation_angle)}" if rotation_angle != 0 else ""
    preprocess_suffix = "_processed" if preprocess else ""
    out_filename = f"annotated_{base_name}{rotation_suffix}{preprocess_suffix}.png"
    out_path = os.path.join(output_dir, out_filename)

    cv2.imwrite(out_path, display_image)

    print(f"✅ Processing complete!")
    print(f"   📁 Saved to: {out_path}")
    print(f"   🎯 Valid detections (1-4 digits): {valid_detections}")
    if rotation_angle != 0:
        print(f"   📐 Applied rotation: {rotation_angle}°")


def process_multiple_images(image_dir: str, output_dir: str,
                            image_extensions: List[str] = None):
    """Process multiple images from a directory."""
    if image_extensions is None:
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']

    if not os.path.exists(image_dir):
        print(f"❌ Input directory does not exist: {image_dir}")
        return

    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(image_dir)
                            if f.lower().endswith(ext.lower())])

    if not image_files:
        print(f"❌ No image files found in {image_dir}")
        return

    print(f"📁 Found {len(image_files)} image(s) to process")

    for i, filename in enumerate(image_files, 1):
        print(f"\n🔄 Processing {i}/{len(image_files)}: {filename}")
        image_path = os.path.join(image_dir, filename)
        run_ocr_and_draw_boxes(image_path, output_dir)


if __name__ == "__main__":
    # Single image processing
    image_path = r"C:\Users\Anushka Verma\OneDrive\Pictures\Screenshots\Screenshot 2025-07-21 104026.png"
    output_dir = r"E:\PyCharm\PycharmProjects\Caption\output"

    print("🚀 Starting multi-angle OCR processing...")
    run_ocr_and_draw_boxes(
        image_path,
        output_dir,
        test_rotations=True,  # Test different rotations
        preprocess=True,  # Apply image preprocessing
        confidence_threshold=0.2  # Minimum confidence threshold
    )

    # Uncomment below to process multiple images from a directory
    # image_directory = r"C:\path\to\your\images"
    # process_multiple_images(image_directory, output_dir)