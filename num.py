import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
from PIL import Image, ImageDraw, ImageFont
import os
import json


class NumberDetector:
    def __init__(self, use_angle_cls=True, lang='en', det_db_thresh=0.3, det_db_box_thresh=0.5):
        """
        Initialize PaddleOCR with optimized settings for number detection
        """
        self.ocr = PaddleOCR(
            lang=lang,
            use_textline_orientation=use_angle_cls,  # Replaces deprecated use_angle_cls
            text_det_thresh=det_db_thresh,
            text_det_box_thresh=det_db_box_thresh
        )

    def preprocess_image(self, image):
        """
        Preprocess image to improve OCR accuracy for numbers
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        blurred = cv2.GaussianBlur(gray, (1, 1), 0)

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)

        adaptive_thresh = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def is_number(self, text):
        cleaned_text = re.sub(r'[^\w.]', '', text)

        try:
            float(cleaned_text)
            return True
        except ValueError:
            pass

        digit_count = sum(c.isdigit() for c in cleaned_text)
        total_chars = len(cleaned_text)

        if total_chars == 0:
            return False

        return digit_count / total_chars >= 0.7

    def correct_common_ocr_errors(self, text):
        corrections = {
            'b': '6', 'B': '6', 'o': '0', 'O': '0', 'l': '1', 'I': '1',
            'S': '5', 's': '5', 'Z': '2', 'z': '2', 'g': '9', 'G': '9',
            'q': '9', 'Q': '9', 'D': '0', 'i': '1', 'T': '7', 't': '7'
        }

        corrected = text
        for wrong, right in corrections.items():
            corrected = corrected.replace(wrong, right)

        corrected = re.sub(r'[^\d.]', '', corrected)
        return corrected

    def filter_valid_numbers(self, results):
        valid_numbers = []
        for line in results:
            if 'bbox' not in line or 'text' not in line:
                continue

            bbox = line['bbox']
            text, confidence = line['text']

            if confidence < 0.6:
                continue

            corrected_text = self.correct_common_ocr_errors(text)

            if self.is_number(corrected_text) and len(corrected_text) > 0:
                try:
                    number_value = float(corrected_text)
                    valid_numbers.append({
                        'bbox': bbox,
                        'original_text': text,
                        'corrected_text': corrected_text,
                        'number_value': number_value,
                        'confidence': confidence
                    })
                except ValueError:
                    continue

        return valid_numbers

    def detect_numbers(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")

        results_all = []

        try:
            results = self.ocr.predict(image)
            if results:
                results_all.extend(results)
        except Exception as e:
            print(f"Error processing original image: {e}")

        try:
            preprocessed = self.preprocess_image(image)
            results_preprocessed = self.ocr.predict(preprocessed)
            if results_preprocessed:
                results_all.extend(results_preprocessed)
        except Exception as e:
            print(f"Error processing preprocessed image: {e}")

        valid_numbers = self.filter_valid_numbers(results_all)
        filtered_numbers = self.remove_duplicate_detections(valid_numbers)

        return filtered_numbers, image

    def remove_duplicate_detections(self, numbers, overlap_threshold=0.5):
        if not numbers:
            return numbers

        def calculate_overlap(bbox1, bbox2):
            def bbox_to_coords(bbox):
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

            coords1 = bbox_to_coords(bbox1)
            coords2 = bbox_to_coords(bbox2)

            x1 = max(coords1[0], coords2[0])
            y1 = max(coords1[1], coords2[1])
            x2 = min(coords1[2], coords2[2])
            y2 = min(coords1[3], coords2[3])

            if x2 <= x1 or y2 <= y1:
                return 0

            intersection = (x2 - x1) * (y2 - y1)
            area1 = (coords1[2] - coords1[0]) * (coords1[3] - coords1[1])
            area2 = (coords2[2] - coords2[0]) * (coords2[3] - coords2[1])
            union = area1 + area2 - intersection
            return intersection / union if union > 0 else 0

        numbers.sort(key=lambda x: x['confidence'], reverse=True)
        filtered = []
        for i, num1 in enumerate(numbers):
            is_duplicate = any(
                calculate_overlap(num1['bbox'], num2['bbox']) > overlap_threshold
                for num2 in filtered
            )
            if not is_duplicate:
                filtered.append(num1)

        return filtered

    def draw_results(self, image, numbers, output_path):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

        for i, number in enumerate(numbers):
            bbox = number['bbox']
            text = number['corrected_text']
            confidence = number['confidence']

            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            color = colors[i % len(colors)]
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

            label = f"{text} ({confidence:.2f})"
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = x_min
            text_y = max(0, y_min - text_height - 5)

            draw.rectangle([text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2],
                           fill='white', outline=color, width=1)
            draw.text((text_x, text_y), label, fill=color, font=font)

        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_image)

        return result_image


def main():
    detector = NumberDetector(det_db_thresh=0.2, det_db_box_thresh=0.4)

    image_path = r"C:\Users\Anushka Verma\OneDrive\Pictures\Screenshots\Screenshot 2025-07-15 165922.png"
    output_path = "detected_numbers.jpg"

    try:
        numbers, original_image = detector.detect_numbers(image_path)

        print(f"Detected {len(numbers)} numbers:")
        for i, num in enumerate(numbers):
            print(f"{i + 1}. Text: '{num['original_text']}' → Corrected: '{num['corrected_text']}' "
                  f"(Value: {num['number_value']}, Confidence: {num['confidence']:.3f})")

        detector.draw_results(original_image, numbers, output_path)
        print(f"Results saved to {output_path}")

    except Exception as e:
        print(f"Error: {e}")


def process_multiple_images(image_folder, output_folder):
    detector = NumberDetector(det_db_thresh=0.2, det_db_box_thresh=0.4)

    os.makedirs(output_folder, exist_ok=True)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    for filename in os.listdir(image_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(image_folder, filename)
            output_path = os.path.join(output_folder, f"detected_{filename}")

            try:
                numbers, original_image = detector.detect_numbers(image_path)
                detector.draw_results(original_image, numbers, output_path)
                print(f"Processed {filename}: {len(numbers)} numbers detected")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    main()
    # Uncomment to batch process
    # process_multiple_images("input_images", "output_images")
