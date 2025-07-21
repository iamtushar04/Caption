import os
import shutil
from extract import PatentExtractor
from extract_labels import TextAnalyzer
from annotate_image import annotate_image_with_labels

def main(patent_number):
    print(f"\n🔍 Starting extraction for patent: {patent_number}")

    # Initialize extractor and analyzer
    extractor = PatentExtractor()
    analyzer = TextAnalyzer()

    # Step 1: Extract patent text and images
    extract_result = extractor.extract_patent_data(patent_number)
    if not extract_result:
        print("❌ Extraction failed. Exiting.")
        return

    folders = extract_result["folders"]
    text_content = extract_result.get("text_content", "")

    if not text_content or len(text_content.strip()) < 100:
        print("⚠️ Extracted text is too short or empty. Skipping analysis.")
        return

    # Step 2: Extract number-to-object mappings from text
    print("📄 Analyzing patent text for number-to-object mappings...")
    number_to_object = analyzer.analyze_patent_text_content(text_content)

    if not number_to_object:
        print("⚠️ No number descriptions found in patent text.")
        return

    # Step 3: Save number-to-object mapping to Python file
    mapping_path = os.path.join(folders["main"], "number_descriptions.py")
    if analyzer.save_number_descriptions(number_to_object, mapping_path):
        print(f"🧠 Saved number mappings to: {mapping_path}")
    else:
        print("⚠️ Failed to save number mappings. Exiting.")
        return

    # Step 4: Dynamically load saved mapping
    try:
        local_vars = {}
        with open(mapping_path, "r", encoding="utf-8") as f:
            exec(f.read(), {}, local_vars)
        number_to_object = local_vars.get("number_descriptions", {})
    except Exception as e:
        print(f"❌ Error loading number mappings: {e}")
        return

    # Step 5: Annotate only even‑indexed images (2nd, 4th, ...)
    images_folder = folders.get("images")
    annotated_folder = folders.get("annotated")

    if not os.path.exists(images_folder) or not os.listdir(images_folder):
        print(f"⚠️ No images found to annotate in: {images_folder}")
        return

    print(f"🖼️ Annotating even‑indexed images (2nd, 4th, …) from: {images_folder}")
    os.makedirs(annotated_folder, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(images_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ])

    for idx, filename in enumerate(image_files, start=1):
        if idx % 2 != 0:
            continue  # Skip odd-indexed files (1-based index)
        image_path = os.path.join(images_folder, filename)
        annotate_image_with_labels(
            image_path,
            annotated_folder,
            pad=100,
            number_to_object=number_to_object
        )

    # Step 6: Delete original images after annotation
    try:
        shutil.rmtree(images_folder)
        print(f"🗑️ Deleted original images folder: {images_folder}")
    except Exception as e:
        print(f"⚠️ Could not delete images folder: {e}")

    print(f"\n✅ Done! Annotated data saved in: {folders['main']}")


if __name__ == "__main__":
    patent_number = input("🔢 Enter the patent number (e.g., US10602821B2 or EP3744866A1): ").strip()
    if patent_number:
        main(patent_number)
    else:
        print("⚠️ No patent number entered. Exiting.")
