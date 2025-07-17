import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
from PIL import Image, ImageDraw, ImageFont
import os


class ImageAnnotator:
    def __init__(self, lang='en'):
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)

    def detect_numbers(self, image_path):
        result = self.ocr.ocr(image_path)
        number_detections = []

        if result and result[0]:
            for detection in result[0]:
                bbox = detection[0]
                text = detection[1][0]
                confidence = detection[1][1]

                numbers = re.findall(r'\d+', text)
                for number in numbers:
                    number_detections.append({
                        'number': number,
                        'bbox': bbox,
                        'confidence': confidence,
                        'original_text': text
                    })
        return number_detections

    def find_annotation_position(self, bbox, image, annotation_size, margin=5):
        img_height, img_width = image.shape[:2]
        ann_width, ann_height = annotation_size
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        x_coords = [int(p[0]) for p in bbox]
        y_coords = [int(p[1]) for p in bbox]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        positions = [
            (max_x + margin, min_y),  # right
            (min_x - ann_width - margin, min_y),  # left
            (min_x, min_y - ann_height - margin),  # top
            (min_x, max_y + margin),  # bottom
            (max_x + margin, max_y + margin),  # bottom-right
            (min_x - ann_width - margin, max_y + margin),  # bottom-left
        ]

        best_pos = (min_x, max_y + margin)
        max_brightness = -1

        for x, y in positions:
            if (x >= 0 and y >= 0 and x + ann_width <= img_width and y + ann_height <= img_height):
                region = gray[y:y + ann_height, x:x + ann_width]
                if region.size == 0:
                    continue
                brightness = region.mean()
                if brightness > max_brightness:
                    max_brightness = brightness
                    best_pos = (x, y)

        return best_pos

    def annotate_image(self, image_path, number_descriptions, output_path=None,
                       font_size=20, box_color=(0, 255, 0), text_color=(0, 0, 255),
                       confidence_threshold=0.5):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                font = ImageFont.load_default()

        detections = self.detect_numbers(image_path)
        annotated_numbers = []

        for detection in detections:
            if detection['confidence'] < confidence_threshold:
                continue

            number = detection['number']
            bbox = detection['bbox']

            if number in number_descriptions:
                description = number_descriptions[number]
                annotation_text = f"{number}: {description}"
                bbox_points = [(int(p[0]), int(p[1])) for p in bbox]
                draw.polygon(bbox_points, outline=box_color, width=2)

                text_bbox = draw.textbbox((0, 0), annotation_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                ann_x, ann_y = self.find_annotation_position(
                    bbox, image, (text_width, text_height)
                )

                draw.text((ann_x, ann_y), annotation_text, fill=text_color, font=font)

                annotated_numbers.append({
                    'number': number,
                    'description': description,
                    'bbox': bbox,
                    'annotation_position': (ann_x, ann_y)
                })

        annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved to: {output_path}")
        else:
            cv2.imshow('Annotated Image', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return annotated_image, annotated_numbers


def main():
    annotator = ImageAnnotator()

    number_descriptions = {
        "1": "fig",
        "2": "fig",
        "3": "fig",
        "4": "fig",
        "5": "fig",
        "6": "fig",
        "7": "fig",
        "8": "fig",
        "9": "fig",
        "100": "flexible main body",
        "110": "outer surface",
        "112": "upper edge",
        "113": "lower edge",
        "120": "front flap",
        "130": "rear flap",
        "150": "main compartment",
        "151": "inner surface",
        "160": "laptop pocket",
        "161": "plurality pen pocket",
        "162": "bottle pocket",
        "163": "phone pocket",
        "164": "credit card slip",
        "165": "large pocket",
        "166": "small pocket",
        "167": "key clip",
        "200": "rigid compartment",
        "201": "upper surface",
        "202": "lower surface",
        "203": "lateral surface",
        "204": "medial surface",
        "206": "posterior surface",
        "250": "insulated compartment",
        "300": "rigid non - insulated compartment",
        "301": "non - insulated upper surface",
        "302": "non - insulated lower surface",
        "304": "non - insulated medial surface",
        "305": "non - insulated anterior surface",
        "350": "non - insulated compartment",
        "351": "non - insulated upper edge",
        "352": "non - insulated lower edge",
        "400": "front pocket",
        "500": "base",
        "501": "front edge",
        "502": "edge",
        "503": "side",
        "504": "side",
        "600": "handle",
        "610": "detachable shoulder strap",
        "611": "shoulder strap ring",
        "620": "zipper",
        "640": "metal foot",
        "650": "logo tag",
        "660": "embodiment plurality shoulder strap",
    }

    image_path = r"C:\Users\Anushka Verma\OneDrive\Pictures\Screenshots\Screenshot 2025-07-15 165951.png"
    output_path = "annotated_image.jpg"

    try:
        annotated_img, detections = annotator.annotate_image(
            image_path=image_path,
            number_descriptions=number_descriptions,
            output_path=output_path,
            font_size=16,
            confidence_threshold=0.7
        )

        print("\nDetected and annotated numbers:")
        for detection in detections:
            print(f"Number {detection['number']}: {detection['description']}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Installed required packages: pip install paddleocr opencv-python pillow")
        print("2. Provided a valid image path")


if __name__ == "__main__":
    main()
