from paddleocr import PaddleOCR
import cv2
import os
import re
import numpy as np

def run_ocr_and_draw_boxes(image_path, output_dir):
    # Create PaddleOCR object; update parameters as needed
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='en'
    )

    # Run OCR
    result = ocr.ocr(image_path)[0]  # The output is a dict in latest PaddleOCR

    # Read the image
    image = cv2.imread(image_path)

    # Unpack the detection results
    rec_texts  = result['rec_texts']   # list of recognized texts
    rec_scores = result['rec_scores']  # list of scores
    rec_polys  = result['rec_polys']   # list of polygons (boxes)

    for box, text, score in zip(rec_polys, rec_texts, rec_scores):
        # (optional) Only draw if text matches a pattern; here: 1-4 digits
        if re.fullmatch(r'\d{1,4}', text.strip()):
            box = np.array(box).astype(int)
            cv2.polylines(image, [box.reshape((-1, 1, 2))], isClosed=True, color=(0, 0, 255), thickness=1)
            cv2.putText(
                image,
                text,
                tuple(box[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1
            )

    # Save result
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, image)
    print(f"✅ Saved to: {out_path}")

if __name__ == "__main__":
    image_path = r"image_path"
    output_dir = r"output_dir_path"
    run_ocr_and_draw_boxes(image_path, output_dir)
