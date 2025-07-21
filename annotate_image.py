#annotate_image

import cv2
import numpy as np
import os
import re
from paddleocr import PaddleOCR

# Mapping: extend as needed
# number_to_object = {
#     "100": "flexible main body",
#     "120": "outer shell",
#     "150": "hinge",
#     "200": "circuit board",
#     "250": "battery",
#     "300": "antenna",
#     "600": "sensor module",
#     "610": "power port",
#     "650": "wire harness"
# }

def annotate_image_with_labels(image_path, output_dir, pad=100, number_to_object=None):
    if number_to_object is None:
        number_to_object = {}  # Fallback in case nothing passed
    # Initialize PaddleOCR
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='en'
    )

    # Read image and add white padding
    original = cv2.imread(image_path)
    padded = cv2.copyMakeBorder(original, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Save temporary padded image for OCR
    temp_path = "temp_padded_image.png"
    cv2.imwrite(temp_path, padded)

    # Run OCR
    result = ocr.ocr(temp_path)[0]

    # Annotate
    for box, text, score in zip(result['rec_polys'], result['rec_texts'], result['rec_scores']):
        if re.fullmatch(r'\d{1,4}', text.strip()):
            object_name = number_to_object.get(text.strip())
            if object_name:
                box = np.array(box).astype(int)

                # Draw bounding box
                cv2.polylines(padded, [box.reshape((-1, 1, 2))], isClosed=True, color=(0, 0, 255), thickness=1)

                # Draw label above the box
                x, y = int(box[0][0]), int(box[0][1]) - 5
                cv2.putText(
                    padded,
                    object_name,
                    (x, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.5,
                    color=(255, 0, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, padded)
    print(f"✅ Annotated image saved to: {output_path}")

def annotate_folder(images_folder, output_folder, number_to_object, pad=100):
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(images_folder):
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(images_folder, filename)
                annotate_image_with_labels(image_path, output_folder, pad=pad, number_to_object=number_to_object)


if __name__ == "__main__":
    image_path = r"C:\Users\Anushka Verma\OneDrive\Pictures\Screenshots\Screenshot 2025-07-15 180343.png"
    output_dir = r"E:\PyCharm\PycharmProjects\Caption\output"
    annotate_image_with_labels(image_path, output_dir)
