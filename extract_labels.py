import re
import spacy
import inflect
from pathlib import Path


class TextAnalyzer:
    def __init__(self):
        # Load spaCy model (fall back if large model unavailable)
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:0
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ No spaCy model found. Please install: python -m spacy download en_core_web_sm")
            self.nlp = None

        self.inflect_engine = inflect.engine()

        # Regex to extract phrase + numbers
        self.pattern = re.compile(
            r'(?:[\w\s\-,;:\(\)]*?\s(?:indicated\s+(?:generally\s+)?as|identified\s+as|as|no\.?|reference\s+numeral|shown\s+as)\s+)?'
            r'([\w\s\-\.,;:\(\)]+?)\s*(\d{1,4}(?:,\s*\d{1,4})*)'
        )

        # Stopwords/noise to remove from phrases
        self.stopwords_re = re.compile(
            r'\b(?:wherein|each|the|and|a|an|when|all|of|may be|is|are|with|such as|general(?:ly)?|indicated|identified|numeral|no|shown|'
            r'that defines|controlled be|may be made of|or includes|i\.e\.|e\.g\.|as by|in use|considering again|be it|some other|one embodiment|'
            r'roughly|such that|whether by|to the extent that|as suggested by|mounted|attached|respectively|similarly|or|this|that|these|those|'
            r'some|any|all|every|each|either|neither|both|few|many|much|more|most|other|such|what|however|with|within|without)\b',
            re.IGNORECASE
        )

    def normalize_phrase(self, phrase):
        """Normalize and clean a phrase to extract the main noun"""
        if not self.nlp:
            return phrase.lower().strip()

        phrase = self.stopwords_re.sub('', phrase.lower()).strip()
        phrase = re.sub(r'\bfigs?\.?\s*\d+\w*\b', '', phrase).strip()
        phrase = re.sub(r'\s+', ' ', phrase).strip(" ,.-:;")

        doc = self.nlp(phrase)

        # Find meaningful noun chunk from end
        main_noun = ""
        for chunk in reversed(list(doc.noun_chunks)):
            if chunk.root.pos_ in ("NOUN", "PROPN") and chunk.root.text.lower() not in {
                "it", "access", "extent", "width", "ends", "structure", "point", "form", "define", "has", "portion",
                "side", "area"
            }:
                main_noun = chunk.text
                break

        if not main_noun:
            for token in reversed(doc):
                if token.pos_ in ("NOUN", "PROPN") and token.text.lower() not in {
                    "it", "access", "extent", "width", "ends", "structure", "point", "form", "define", "has", "portion"
                }:
                    main_noun = token.text
                    break

        if not main_noun:
            return ""

        doc2 = self.nlp(main_noun.lower())
        words = []
        for token in doc2:
            if token.pos_ in ("NOUN", "PROPN"):
                singular = self.inflect_engine.singular_noun(token.text)
                words.append(singular if singular else token.text)
            elif token.pos_ == "ADJ":
                words.append(token.text)

        label = ' '.join(words)
        label = re.sub(r'\b(\w+)\b(?: \1\b)+', r'\1', label)  # Remove duplicates like 'pad pad'

        return label if len(label) > 1 else ""

    def extract_number_descriptions_from_text(self, text):
        """Extract number descriptions from patent text"""
        candidate_labels = {}

        for match in self.pattern.finditer(text):
            phrase, nums = match.group(1), match.group(2)

            # Skip figure references
            if re.search(r'\bfigs?\.?\s*\d+\w*\b', phrase, re.IGNORECASE):
                continue

            phrase = phrase.lower()
            label = self.normalize_phrase(phrase)
            if not label:
                continue

            for num in nums.split(','):
                num = num.strip()
                if num.isdigit():
                    candidate_labels.setdefault(num, []).append(label)

        # Pick shortest label for each number (most concise)
        final_labels = {}
        for num, labels in candidate_labels.items():
            if labels:
                final_labels[num] = sorted(labels, key=len)[0]

        return final_labels

    def analyze_patent_text_file(self, text_file_path):
        """Analyze patent text file and extract number descriptions"""
        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            print(f"📄 Analyzing text file: {text_file_path}")
            number_descriptions = self.extract_number_descriptions_from_text(text)

            print(f"✅ Found {len(number_descriptions)} number descriptions")

            return number_descriptions

        except Exception as e:
            print(f"⚠️ Error analyzing text file: {e}")
            return {}

    def analyze_patent_text_content(self, text_content):
        """Analyze patent text content directly"""
        try:
            print(f"📄 Analyzing text content ({len(text_content)} characters)")
            number_descriptions = self.extract_number_descriptions_from_text(text_content)

            print(f"✅ Found {len(number_descriptions)} number descriptions")

            return number_descriptions

        except Exception as e:
            print(f"⚠️ Error analyzing text content: {e}")
            return {}

    def save_number_descriptions(self, number_descriptions, output_path):
        """Save number descriptions to a file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Patent Number Descriptions\n")
                f.write("# Generated automatically from patent text\n\n")

                f.write("number_descriptions = {\n")
                for num in sorted(number_descriptions.keys(), key=int):
                    label = number_descriptions[num].replace('"', "'")
                    f.write(f'    "{num}": "{label}",\n')
                f.write("}\n")

            print(f"💾 Number descriptions saved to: {output_path}")
            return True

        except Exception as e:
            print(f"⚠️ Error saving number descriptions: {e}")
            return False

    def print_number_descriptions(self, number_descriptions):
        """Print number descriptions in a formatted way"""
        if not number_descriptions:
            print("⚠️ No number descriptions found")
            return

        print("\n📋 Number Descriptions:")
        print("=" * 50)

        for num in sorted(number_descriptions.keys(), key=int):
            print(f"  {num:>4} → {number_descriptions[num]}")

        print("=" * 50)
        print(f"Total: {len(number_descriptions)} descriptions")


if __name__ == "__main__":
    # Test the analyzer
    analyzer = TextAnalyzer()

    # Test text
    test_text = """
    FIG. 1 shows an overall frontal view of Embodiment 1 001 of the present invention noting a flexible main body 100, front flap 120, insulated compartment flap 250, and non-insulated compartment flap 350. The figure also shows the relative positions of rigid insulated compartment 200 and rigid non-insulated compartment 300, as well as a plurality of fixed carry handles 600.
    """

    descriptions = analyzer.analyze_patent_text_content(test_text)
    analyzer.print_number_descriptions(descriptions)