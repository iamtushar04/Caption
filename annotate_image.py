import os
import re
import cv2
import numpy as np
from paddleocr import PaddleOCR


def preprocess_image(image_path: str, pad: int = 100, max_side: int = 3840) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.equalizeHist(gray)
    upscaled = cv2.resize(hist, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    color = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

    padded = cv2.copyMakeBorder(color, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Resize if any side exceeds limit
    h, w = padded.shape[:2]
    scale = min(max_side / h, max_side / w, 1.0)
    if scale < 1.0:
        padded = cv2.resize(padded, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return padded


def annotate_image_with_labels(image_path, output_dir, pad=100, number_to_object=None, ocr=None):
    if number_to_object is None:
        number_to_object = {}
    if ocr is None:
        ocr = PaddleOCR(use_angle_cls=False, use_doc_unwarping=False, lang='en')

    padded_img = preprocess_image(image_path, pad)

    # Run OCR (use .predict(), not .ocr())
    result = ocr.predict(padded_img)[0]  # dict with 'rec_texts', 'rec_scores', 'rec_polys'

    for box, text, score in zip(result['rec_polys'], result['rec_texts'], result['rec_scores']):
        if re.fullmatch(r'\d{1,4}', text.strip()):
            object_name = number_to_object.get(text.strip())
            if object_name:
                box = np.array(box).astype(int)
                cv2.polylines(padded_img, [box.reshape((-1, 1, 2))], isClosed=True, color=(0, 0, 255), thickness=1)
                x, y = box[0]
                cv2.putText(padded_img, object_name, (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 1, lineType=cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, padded_img)
    print(f"✅ Annotated image saved to: {output_path}")


def annotate_folder(images_folder, output_folder, number_to_object, pad=100):
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    os.makedirs(output_folder, exist_ok=True)

    ocr = PaddleOCR(use_angle_cls=False, use_doc_unwarping=False, lang='en')

    filenames = sorted(os.listdir(images_folder))
    for i, filename in enumerate(filenames):
        if filename.lower().endswith(image_extensions) and i % 2 == 1:
            image_path = os.path.join(images_folder, filename)
            try:
                annotate_image_with_labels(image_path, output_folder, pad=pad, number_to_object=number_to_object, ocr=ocr)
            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")
