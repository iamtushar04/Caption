from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import os
import shutil

from extract import PatentExtractor
from extract_labels import TextAnalyzer
from annotate_image import annotate_image_with_labels

app = FastAPI()

def run_patent_processing(patent_number: str) -> str:
    """Run your existing main logic but return status message instead of print."""
    extractor = PatentExtractor()
    analyzer = TextAnalyzer()

    extract_result = extractor.extract_patent_data(patent_number)
    if not extract_result:
        return f"❌ Extraction failed for patent {patent_number}."

    folders = extract_result["folders"]
    text_content = extract_result.get("text_content", "")

    if not text_content or len(text_content.strip()) < 100:
        return f"⚠️ Extracted text too short or empty for patent {patent_number}."

    number_to_object = analyzer.analyze_patent_text_content(text_content)
    if not number_to_object:
        return f"⚠️ No number descriptions found in patent text."

    mapping_path = os.path.join(folders["main"], "number_descriptions.py")
    if not analyzer.save_number_descriptions(number_to_object, mapping_path):
        return f"⚠️ Failed to save number mappings."

    try:
        local_vars = {}
        with open(mapping_path, "r", encoding="utf-8") as f:
            exec(f.read(), {}, local_vars)
        number_to_object = local_vars.get("number_descriptions", {})
    except Exception as e:
        return f"❌ Error loading number mappings: {str(e)}"

    images_folder = folders.get("images")
    annotated_folder = folders.get("annotated")

    if not os.path.exists(images_folder) or not os.listdir(images_folder):
        return f"⚠️ No images found to annotate in: {images_folder}"

    os.makedirs(annotated_folder, exist_ok=True)
    image_files = sorted([
        f for f in os.listdir(images_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ])

    for idx, filename in enumerate(image_files, start=1):
        if idx % 2 != 0:
            continue
        image_path = os.path.join(images_folder, filename)
        annotate_image_with_labels(
            image_path,
            annotated_folder,
            pad=100,
            number_to_object=number_to_object
        )

    try:
        shutil.rmtree(images_folder)
    except Exception as e:
        return f"⚠️ Could not delete images folder: {str(e)}"

    return f"✅ Patent {patent_number} processed successfully! Output saved at: {folders['main']}"

@app.get("/", response_class=HTMLResponse)
async def form():
    return """
    <html>
    <head>
        <title>Captions</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f7f9fc;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
            }
            h1 {
                font-size: 72px;
                margin: 0;
                font-weight: bold;
            }
            h2 {
                font-size: 36px;
                margin: 0 0 40px 0;
                color: #555;
                font-weight: normal;
            }
            input[type=text] {
                padding: 12px;
                font-size: 20px;
                width: 320px;
                border-radius: 6px;
                border: 1px solid #ccc;
                margin-bottom: 20px;
            }
            button {
                padding: 12px 30px;
                font-size: 20px;
                background-color: #0077cc;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
            }
            button:hover {
                background-color: #005fa3;
            }
            form {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
        </style>
    </head>
    <body>
        <h1>Captions</h1>
        <h2>Image Annotator</h2>
        <form method="post" action="/process">
            <input type="text" name="patent_number" placeholder="Enter patent number" required />
            <button type="submit">Process</button>
        </form>
    </body>
    </html>
    """

@app.post("/process", response_class=HTMLResponse)
async def process(patent_number: str = Form(...)):
    try:
        result = run_patent_processing(patent_number)
        return f"""
        <html>
        <head><title>Processing Result</title></head>
        <body style="font-family: Arial, sans-serif; text-align: center; margin-top: 100px;">
            <h1>Captions</h1>
            <h2>Image Annotator</h2>
            <p style="font-size: 18px;">{result}</p>
            <a href="/">🔙 Go back</a>
        </body>
        </html>
        """
    except Exception as e:
        return f"<h3>❌ Internal error: {str(e)}</h3>"
