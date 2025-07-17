import re
import spacy
import inflect
import requests
from bs4 import BeautifulSoup
import time
import os
from urllib.parse import urljoin
import json
from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import Image
import io

# Load spaCy model (fall back if large model unavailable)
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    nlp = spacy.load("en_core_web_sm")

p = inflect.engine()


def fetch_patent_data(patent_number):
    """Fetch patent text and high-quality images from multiple sources"""
    try:
        # Clean patent number (remove spaces, hyphens)
        clean_patent = re.sub(r'[^\w]', '', patent_number.upper())

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        patent_text = ""
        image_urls = []

        # Try Google Patents first for text
        google_urls = [
            f"https://patents.google.com/patent/US{clean_patent}A1",
            f"https://patents.google.com/patent/US{clean_patent}B2",
            f"https://patents.google.com/patent/US{clean_patent}",
            f"https://patents.google.com/patent/{clean_patent}"
        ]

        for url in google_urls:
            try:
                print(f"Fetching patent text from: {url}")
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Extract patent text
                    text_selectors = [
                        'div.description',
                        'div[itemprop="description"]',
                        'section.description',
                        'div.patent-text',
                        'div.detailed-description'
                    ]

                    for selector in text_selectors:
                        description_div = soup.select_one(selector)
                        if description_div:
                            patent_text = description_div.get_text()
                            break

                    if not patent_text:
                        patent_text = soup.get_text()

                    # Extract HIGH-QUALITY image URLs from Google Patents
                    img_tags = soup.find_all('img')
                    for img in img_tags:
                        src = img.get('src')
                        if src and ('figure' in src.lower() or 'fig' in src.lower()):
                            # Try to get higher quality version
                            if 'w=' in src:
                                # Remove width restrictions for higher quality
                                src = re.sub(r'w=\d+', 'w=1200', src)
                            if 'h=' in src:
                                # Remove height restrictions
                                src = re.sub(r'h=\d+', 'h=1200', src)

                            # Convert relative URLs to absolute
                            if src.startswith('//'):
                                src = 'https:' + src
                            elif src.startswith('/'):
                                src = 'https://patents.google.com' + src
                            image_urls.append(src)

                    if patent_text:
                        break

            except Exception as e:
                print(f"Error with Google Patents URL {url}: {e}")
                continue

        # Try USPTO for higher quality images
        try:
            print(f"Fetching high-quality images from USPTO...")
            uspto_url = f"https://ppubs.uspto.gov/dirsearch-public/print/downloadPdf/{clean_patent}"

            # Also try PatentScope for images
            patentscope_search = f"https://patentscope.wipo.int/search/en/detail.jsf?docId=US{clean_patent}"

            # Try FreePatentsOnline
            fpo_url = f"https://www.freepatentsonline.com/{clean_patent}.html"

            try:
                response = requests.get(fpo_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    fpo_imgs = soup.find_all('img')
                    for img in fpo_imgs:
                        src = img.get('src')
                        if src and ('drawing' in src.lower() or 'patent' in src.lower() or 'figure' in src.lower()):
                            if src.startswith('//'):
                                src = 'https:' + src
                            elif src.startswith('/'):
                                src = 'https://www.freepatentsonline.com' + src
                            image_urls.append(src)

            except Exception as e:
                print(f"Error fetching from FreePatentsOnline: {e}")

        except Exception as e:
            print(f"Error fetching from alternative sources: {e}")

        # Remove duplicates and filter out very small images
        unique_urls = []
        for url in set(image_urls):
            # Skip very small thumbnail images
            if not any(term in url.lower() for term in ['thumb', 'small', 'w=50', 'w=100', 'h=50', 'h=100']):
                unique_urls.append(url)

        print(f"Found {len(unique_urls)} high-quality image URLs")
        return patent_text, unique_urls

    except Exception as e:
        print(f"Error fetching patent: {e}")
        return None, []


def enhance_image_quality(image_path):
    """Enhance image quality for better OCR results"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return image_path

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Apply sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # Scale up image for better OCR (2x)
        height, width = sharpened.shape
        scaled = cv2.resize(sharpened, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

        # Save enhanced image
        enhanced_path = image_path.replace('.jpg', '_enhanced.jpg').replace('.png', '_enhanced.png')
        cv2.imwrite(enhanced_path, scaled)

        print(f"✅ Enhanced image saved: {enhanced_path}")
        return enhanced_path

    except Exception as e:
        print(f"Error enhancing image {image_path}: {e}")
        return image_path


def download_image(url, save_path):
    """Download a high-quality image from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        response = requests.get(url, headers=headers, timeout=15, stream=True)
        if response.status_code == 200:
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'image' in content_type:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Check if image was downloaded successfully
                if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:  # At least 1KB
                    print(f"✅ Downloaded image: {save_path} ({os.path.getsize(save_path)} bytes)")
                    return True
                else:
                    print(f"❌ Downloaded image too small: {save_path}")
                    return False
            else:
                print(f"❌ URL did not return image content: {url}")
                return False
        else:
            print(f"❌ Failed to download image. Status: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error downloading image from {url}: {e}")
        return False


def run_ocr_and_draw_boxes(image_path, output_dir):
    """Run OCR on image, extract numbers, and draw bounding boxes with annotations"""
    try:
        ocr = PaddleOCR(use_angle_cls=False, lang='en')  # SIMPLER MODE
        results = ocr.ocr(image_path)[0]  # Use simpler ocr()

        if not results:
            print(f"No OCR results for {image_path}")
            return []

        image = cv2.imread(image_path)
        extracted_numbers = []

        for line in results:
            box, (text, score) = line
            # Extract numbers (1-4 digits) - matching your original regex
            if re.fullmatch(r'\d{1,4}', text.strip()):
                extracted_numbers.append(text.strip())

                # Draw bounding box and annotation (matching your original style)
                box = np.array(box).astype(int)
                cv2.polylines(image, [box.reshape((-1, 1, 2))], isClosed=True, color=(0, 0, 255), thickness=1)
                cv2.putText(image, text, tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Save annotated image
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
        cv2.imwrite(out_path, image)
        print(f"✅ Saved annotated image to: {out_path}")

        return extracted_numbers

    except Exception as e:
        print(f"Error running OCR on {image_path}: {e}")
        return []


def normalize_phrase(phrase):
    """Extract clean, concise descriptions from phrases"""
    # Remove common stopwords and noise
    stopwords_re = re.compile(
        r'\b(?:wherein|each|the|and|a|an|when|all|of|may be|is|are|with|such as|general(?:ly)?|indicated|identified|'
        r'numeral|no|shown|that defines|controlled be|may be made of|or includes|i\.e\.|e\.g\.|as by|in use|'
        r'considering again|be it|some other|one embodiment|roughly|such that|whether by|to the extent that|'
        r'as suggested by|mounted|attached|respectively|similarly|or|this|that|these|those|some|any|all|every|'
        r'each|either|neither|both|few|many|much|more|most|other|such|what|however|with|within|without|'
        r'comprises?|comprising|includes?|including|having|being|noting|shows?|showing)\b',
        re.IGNORECASE
    )

    # Clean the phrase
    phrase = stopwords_re.sub('', phrase.lower()).strip()
    phrase = re.sub(r'\bfigs?\.?\s*\d+\w*\b', '', phrase, flags=re.IGNORECASE).strip()
    phrase = re.sub(r'\s+', ' ', phrase).strip(" ,.-:;()")

    if not phrase:
        return ""

    # Use spaCy to extract meaningful noun phrases
    doc = nlp(phrase)

    # Look for the main noun chunk (preferably the last meaningful one)
    main_noun = ""
    for chunk in reversed(list(doc.noun_chunks)):
        if chunk.root.pos_ in ("NOUN", "PROPN") and len(chunk.text) > 1:
            # Skip generic words
            if chunk.root.text.lower() not in {
                "it", "access", "extent", "width", "ends", "structure", "point",
                "form", "define", "has", "portion", "side", "area", "view", "figure"
            }:
                main_noun = chunk.text
                break

    # If no good noun chunk, try individual tokens
    if not main_noun:
        for token in reversed(doc):
            if token.pos_ in ("NOUN", "PROPN") and len(token.text) > 1:
                if token.text.lower() not in {
                    "it", "access", "extent", "width", "ends", "structure", "point",
                    "form", "define", "has", "portion", "view", "figure"
                }:
                    main_noun = token.text
                    break

    if not main_noun:
        return ""

    # Process the main noun to get clean description
    doc2 = nlp(main_noun.lower())
    words = []

    for token in doc2:
        if token.pos_ in ("NOUN", "PROPN"):
            # Convert to singular form
            singular = p.singular_noun(token.text)
            words.append(singular if singular else token.text)
        elif token.pos_ == "ADJ":
            words.append(token.text)

    # Join words and clean up
    label = ' '.join(words)
    label = re.sub(r'\b(\w+)\b(?: \1\b)+', r'\1', label)  # Remove duplicates

    return label.strip() if len(label) > 1 else ""


def extract_descriptions_from_text(patent_text, detected_numbers):
    """Extract descriptions for detected numbers from patent text"""
    # Enhanced regex to capture number-description pairs
    patterns = [
        # Pattern 1: "description number" or "description numbers"
        r'([\w\s\-\.,;:\(\)]+?)\s*(\d{1,4}(?:,\s*\d{1,4})*)',
        # Pattern 2: More specific patterns for patents
        r'(?:[\w\s\-,;:\(\)]*?\s(?:indicated\s+(?:generally\s+)?as|identified\s+as|as|no\.?|reference\s+numeral|shown\s+as)\s+)?([\w\s\-\.,;:\(\)]+?)\s*(\d{1,4}(?:,\s*\d{1,4})*)'
    ]

    candidate_labels = {}

    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)

        for match in regex.finditer(patent_text):
            phrase, nums = match.group(1), match.group(2)

            # Skip if phrase contains figure references
            if re.search(r'\bfigs?\.?\s*\d+\w*\b', phrase, re.IGNORECASE):
                continue

            # Clean and normalize the phrase
            label = normalize_phrase(phrase)
            if not label:
                continue

            # Extract individual numbers
            for num in nums.split(','):
                num = num.strip()
                if num.isdigit() and len(num) <= 4:
                    candidate_labels.setdefault(num, []).append(label)

    # Select best label for each number, prioritizing detected numbers
    final_labels = {}

    # First, process detected numbers from OCR
    for num in detected_numbers:
        if num in candidate_labels:
            unique_labels = list(set(candidate_labels[num]))
            best_label = sorted(unique_labels, key=len)[0]
            final_labels[num] = best_label

    # Then add any other numbers found in text
    for num, labels in candidate_labels.items():
        if num not in final_labels and labels:
            unique_labels = list(set(labels))
            best_label = sorted(unique_labels, key=len)[0]
            final_labels[num] = best_label

    return final_labels


def process_patent_images_and_text(patent_number, output_dir="patent_output"):
    """Main function to process patent images and extract descriptions"""

    print(f"Processing patent {patent_number}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Fetch patent data
    patent_text, image_urls = fetch_patent_data(patent_number)

    if not patent_text:
        print("Could not fetch patent text. Please provide patent text manually.")
        return {}

    if not image_urls:
        print("No images found in patent. Processing text only...")
        # Process text without images
        descriptions = extract_descriptions_from_text(patent_text, [])
        return descriptions

    print(f"Found {len(image_urls)} images in patent")

    # Download and process images
    all_detected_numbers = []
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    for i, url in enumerate(image_urls):
        try:
            image_filename = f"patent_figure_{i + 1}.jpg"
            image_path = os.path.join(image_dir, image_filename)

            print(f"Downloading image {i + 1}/{len(image_urls)}: {url}")
            if download_image(url, image_path):
                # Enhance image quality before OCR
                print(f"Enhancing image quality for better OCR...")
                enhanced_path = enhance_image_quality(image_path)

                print(f"Running OCR on enhanced image {i + 1}...")
                numbers = run_ocr_and_draw_boxes(enhanced_path, image_dir)
                all_detected_numbers.extend(numbers)
                print(f"Detected numbers: {numbers}")
            else:
                print(f"Failed to download image {i + 1}")

        except Exception as e:
            print(f"Error processing image {i + 1}: {e}")

    # Remove duplicates from detected numbers
    unique_detected_numbers = list(set(all_detected_numbers))
    print(f"Total unique numbers detected from images: {unique_detected_numbers}")

    # Extract descriptions from patent text
    descriptions = extract_descriptions_from_text(patent_text, unique_detected_numbers)

    return descriptions


def main():
    """Main execution function"""
    patent_number = input("Enter patent number (e.g., 10123456, US10123456A1): ").strip()

    if not patent_number:
        print("No patent number provided.")
        return

    output_dir = input("Enter output directory (default: patent_output): ").strip() or "patent_output"

    descriptions = process_patent_images_and_text(patent_number, output_dir)

    if not descriptions:
        print("No descriptions extracted.")
        return

    # Output in the required dictionary format
    print("\n" + "=" * 50)
    print("EXTRACTED DESCRIPTIONS:")
    print("=" * 50)
    print("number_descriptions = {")
    for num in sorted(descriptions.keys(), key=int):
        label = descriptions[num].replace('"', "'")  # Avoid breaking quotes
        print(f'    "{num}": "{label}",')
    print("}")

    print(f"\nExtracted {len(descriptions)} descriptions.")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()