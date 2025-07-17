import requests
import os
import json
import re
import time
from datetime import datetime
from bs4 import BeautifulSoup
import sys


class PatentExtractor:
    def __init__(self, base_output_dir="patents"):
        self.base_output_dir = base_output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                           '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        })

    # Removed normalization to keep full patent number intact
    # Only use if you want to extract pure numeric part for image folder or other logic

    def create_patent_folder(self, patent_number):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patent_folder = os.path.join(self.base_output_dir, f"{patent_number}_{timestamp}")

        folders = {
            'main': patent_folder,
            'images': os.path.join(patent_folder, 'images'),
            'annotated': os.path.join(patent_folder, 'annotated'),
            'text': os.path.join(patent_folder, 'text')
        }

        for folder in folders.values():
            os.makedirs(folder, exist_ok=True)

        return folders

    def extract_patent_text(self, patent_number):
        try:
            print(f"📄 Extracting text for patent {patent_number}...")

            url = f"https://patents.google.com/patent/{patent_number}/en"
            response = self.session.get(url, timeout=15)

            if response.status_code != 200:
                print(f"⚠️ Could not retrieve page for {patent_number}")
                return ""

            soup = BeautifulSoup(response.text, 'html.parser')

            abstract_text = ""
            description_text = ""
            claims_text = ""

            # Attempt 1: Extract JSON-LD
            ld_json_tags = soup.find_all("script", type="application/ld+json")
            for tag in ld_json_tags:
                try:
                    json_data = json.loads(tag.string)
                    if json_data.get('@type', '').lower() == "patent":
                        abstract_text = json_data.get("abstract", "")
                        description_text = json_data.get("description", "")
                        break
                except Exception:
                    continue

            # Attempt 2: __INITIAL_STATE__ JS variable extraction
            if not description_text or not abstract_text:
                script = soup.find("script", string=re.compile(r"window\.__INITIAL_STATE__ = "))
                if script and script.string:
                    try:
                        match = re.search(r"window\.__INITIAL_STATE__ = ({.*});", script.string, re.DOTALL)
                        if match:
                            initial_state = json.loads(match.group(1))
                            patent_doc = initial_state.get('patent', {}).get('patent', {})
                            if patent_doc:
                                if not abstract_text:
                                    abstract_text = patent_doc.get('abstract', '')
                                if not description_text:
                                    description_text = patent_doc.get('description', '')
                                claims_sections = initial_state.get('claims', {})
                                claims_list = []
                                if isinstance(claims_sections, dict):
                                    for v in claims_sections.values():
                                        if isinstance(v, str):
                                            claims_list.append(v.strip())
                                        elif isinstance(v, dict) and 'text' in v:
                                            claims_list.append(v['text'].strip())
                                elif isinstance(claims_sections, list):
                                    for claim in claims_sections:
                                        claims_list.append(str(claim).strip())
                                claims_text = "\n\n".join(claims_list)
                    except Exception:
                        pass

            # Attempt 3: Fallback raw scraping from HTML elements used on page
            if not abstract_text:
                abs_div = soup.find('div', {'itemprop': 'abstract'})
                if abs_div:
                    abstract_text = abs_div.get_text(separator="\n", strip=True)

            if not description_text:
                describ_div = soup.find('section', {'itemprop': 'description'})
                if describ_div:
                    paragraphs = describ_div.find_all(['p', 'div'])
                    description_text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

            if not claims_text:
                claims_div = soup.find('section', {'itemprop': 'claims'})
                if claims_div:
                    ps = claims_div.find_all('p')
                    claims_text = "\n\n".join(p.get_text(strip=True) for p in ps if p.get_text(strip=True))
                else:
                    fallback_claims = soup.find_all("div", class_="claim-text")
                    if fallback_claims:
                        claims_text = "\n\n".join(c.get_text(strip=True) for c in fallback_claims)

            # Compose result
            full_text = ""
            if abstract_text:
                full_text += "ABSTRACT:\n" + abstract_text + "\n\n"
            if description_text:
                full_text += "DESCRIPTION:\n" + description_text + "\n\n"
            if claims_text:
                full_text += "CLAIMS:\n" + claims_text + "\n"

            if not full_text.strip():
                print("⚠️ Failed to extract meaningful text from the page.")

            return full_text.strip()

        except Exception as e:
            print(f"❌ Text extraction failed: {e}")
            return ""

    def download_high_quality_images(self, patent_number, image_folder):
        try:
            print(f"🖼️ Downloading high-quality images for patent {patent_number}...")
            sources = [
                self._get_uspto_images,
                self._get_google_patents_images,
                self._get_freepatentsonline_images
            ]
            downloaded_images = []

            for source_func in sources:
                try:
                    images = source_func(patent_number, image_folder)
                    downloaded_images.extend(images)
                    if images:
                        break
                except Exception as e:
                    print(f"⚠️ Source failed: {e}")
                    continue

            return downloaded_images

        except Exception as e:
            print(f"❌ Image download failed: {e}")
            return []

    def _get_uspto_images(self, patent_number, image_folder):
        images = []
        try:
            # For USPTO images, the numeric part is needed in the URL
            numeric_part = re.search(r'\d+', patent_number)
            if not numeric_part:
                return []
            number = numeric_part.group()

            base_url = f"https://patentimages.storage.googleapis.com/{number[:2]}/{number}/US{number}-"

            for fig_num in range(1, 21):
                for format_type in ['D1', 'D2', 'D3']:
                    img_url = f"{base_url}{fig_num:02d}-{format_type}.png"
                    try:
                        response = self.session.get(img_url, timeout=10)
                        if response.status_code == 200 and len(response.content) > 1000:
                            filename = f"fig_{fig_num:02d}_{format_type}.png"
                            filepath = os.path.join(image_folder, filename)
                            with open(filepath, 'wb') as f:
                                f.write(response.content)
                            images.append(filepath)
                            print(f"✅ Downloaded: {filename}")
                            time.sleep(0.5)
                    except:
                        continue
            return images

        except Exception as e:
            print(f"⚠️ USPTO images failed: {e}")
            return []

    def _get_google_patents_images(self, patent_number, image_folder):
        images = []
        try:
            api_url = f"https://patents.google.com/patent/{patent_number}/en"
            response = self.session.get(api_url)
            if response.status_code == 200:
                img_pattern = r'https://patentimages\.storage\.googleapis\.com/[^"]+\.png'
                img_urls = re.findall(img_pattern, response.text)

                for i, img_url in enumerate(img_urls[:10]):
                    try:
                        img_response = self.session.get(img_url, timeout=10)
                        if img_response.status_code == 200:
                            filename = f"google_fig_{i + 1:02d}.png"
                            filepath = os.path.join(image_folder, filename)
                            with open(filepath, 'wb') as f:
                                f.write(img_response.content)
                            images.append(filepath)
                            print(f"✅ Downloaded: {filename}")
                            time.sleep(0.5)
                    except:
                        continue
            return images

        except Exception as e:
            print(f"⚠️ Google Patents images failed: {e}")
            return []

    def _get_freepatentsonline_images(self, patent_number, image_folder):
        images = []
        try:
            base_url = f"http://www.freepatentsonline.com/{patent_number}.html"
            response = self.session.get(base_url)
            if response.status_code == 200:
                img_pattern = fr'http://www\.freepatentsonline\.com/{re.escape(patent_number)}-\d+\.png'
                img_urls = re.findall(img_pattern, response.text)

                for i, img_url in enumerate(img_urls[:10]):
                    try:
                        img_response = self.session.get(img_url, timeout=10)
                        if img_response.status_code == 200:
                            filename = f"fpo_fig_{i + 1:02d}.png"
                            filepath = os.path.join(image_folder, filename)
                            with open(filepath, 'wb') as f:
                                f.write(img_response.content)
                            images.append(filepath)
                            print(f"✅ Downloaded: {filename}")
                            time.sleep(0.5)
                    except:
                        continue
            return images

        except Exception as e:
            print(f"⚠️ FreePatentsOnline images failed: {e}")
            return []

    def extract_patent_data(self, patent_number):
        # Use full patent number as is, don't normalize (except for images needing the numeric part extracted internally)
        print(f"🔍 Starting extraction for patent: {patent_number}")

        folders = self.create_patent_folder(patent_number)
        patent_text = self.extract_patent_text(patent_number)

        if patent_text:
            text_file = os.path.join(folders['text'], f"patent_{patent_number}_text.txt")
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(patent_text)
            print(f"📄 Text saved to: {text_file}")
        else:
            text_file = None

        image_files = self.download_high_quality_images(patent_number, folders['images'])

        if not image_files:
            print("⚠️ No images downloaded. Trying alternative method...")

        result = {
            'patent_number': patent_number,
            'folders': folders,
            'text_file': text_file,
            'image_files': image_files,
            'text_content': patent_text
        }

        info_file = os.path.join(folders['main'], 'extraction_info.json')
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump({
                'patent_number': patent_number,
                'extraction_time': datetime.now().isoformat(),
                'images_downloaded': len(image_files),
                'text_extracted': bool(patent_text),
                'folders': folders
            }, f, indent=2)

        print(f"✅ Extraction completed for patent {patent_number}")
        print(f"📁 Main folder: {folders['main']}")
        print(f"🖼️ Images downloaded: {len(image_files)}")

        return result


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        patent_no = sys.argv[1].strip()
    else:
        patent_no = input("Enter patent number (e.g., KR102518694B1 or US9861170B1): ").strip()

    if patent_no:
        extractor = PatentExtractor()
        extractor.extract_patent_data(patent_no)
    else:
        print("❌ No patent number provided. Please run the script again with a valid patent number.")
