from paddleocr import PaddleOCR
import cv2
import os
import re
import numpy as np

# Number to label mapping (add as many as you need)
number_to_object = {
    "100": "flexible main body",
    "110": "outer surface",
    # ... add all your mappings here
}

def run_ocr_and_draw_boxes_with_padding_and_labels(image_path, output_dir, pad=100):
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='en'
    )

    # 1. Load and pad image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    padded_img = cv2.copyMakeBorder(
        image,
        pad, pad, pad, pad,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )

    # 2. Run OCR on the padded image
    # Save temp file or use numpy array directly if PaddleOCR supports it
    temp_path = "temp_padded_image.png"
    cv2.imwrite(temp_path, padded_img)
    result = ocr.ocr(temp_path)[0]
    os.remove(temp_path)

    rec_texts  = result['rec_texts']
    rec_scores = result['rec_scores']
    rec_polys  = result['rec_polys']

    for box, text, score in zip(rec_polys, rec_texts, rec_scores):
        clean_text = text.strip()
        if re.fullmatch(r'\d{1,4}', clean_text):
            label = number_to_object.get(clean_text, "(unknown)")
            display_text = f"{clean_text}: {label}"

            box_arr = np.array(box).astype(int)
            cv2.polylines(padded_img, [box_arr.reshape((-1, 1, 2))], isClosed=True, color=(0, 0, 255), thickness=1)
            # Draw text with "safe" padding if near edge
            text_x, text_y = box_arr[0]
            text_x = max(text_x, 5)
            text_y = max(text_y, 15)  # leave room above
            cv2.putText(
                padded_img,
                display_text,
                (text_x, text_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, padded_img)
    print(f"✅ Saved to: {out_path}")

if __name__ == "__main__":
    image_path = r"C:\Users\Anushka Verma\OneDrive\Pictures\Screenshots\Screenshot 2025-07-15 180343.png"
    output_dir = r"E:\PyCharm\PycharmProjects\Caption\output"
    run_ocr_and_draw_boxes_with_padding_and_labels(image_path, output_dir, pad=100)
