from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
import re

def run_ocr_and_draw_boxes(image_path, output_dir):
    ocr = PaddleOCR(use_angle_cls=False, lang='en')  # SIMPLER MODE
    results = ocr.ocr(image_path)[0]  # Use simpler ocr()

    image = cv2.imread(image_path)

    for line in results:
        box, (text, score) = line
        if re.fullmatch(r'\d{1,4}', text.strip()):
            box = np.array(box).astype(int)
            cv2.polylines(image, [box.reshape((-1, 1, 2))], isClosed=True, color=(0, 0, 255), thickness=1)
            cv2.putText(image, text, tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, image)
    print(f"✅ Saved to: {out_path}")

if __name__ == "__main__":
    image_path = r"C:\Users\Anushka Verma\OneDrive\Pictures\Screenshots\Screenshot 2025-07-15 165922.png"
    output_dir = r"E:\PyCharm\PycharmProjects\Caption\output"
    run_ocr_and_draw_boxes(image_path, output_dir)
