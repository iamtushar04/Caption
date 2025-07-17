import re
import spacy
import inflect

# Load spaCy model (fall back if large model unavailable)
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    nlp = spacy.load("en_core_web_sm")

p = inflect.engine()

text = """
FIG. 1 shows an overall frontal view of Embodiment 1 001 of the present invention noting a flexible main body 100, front flap 120, insulated compartment flap 250, and non-insulated compartment flap 350. The figure also shows the relative positions of rigid insulated compartment 200 and rigid non-insulated compartment 300, as well as a plurality of fixed carry handles 600. Also shown are a plurality of accessories for customizing the invention, including a detachable shoulder strap 610 and a logo tag 650.
FIG. 2 shows an overall frontal view of Embodiment 1 001 of the present invention noting a flexible main body 100, front flap 120, front pocket 400, insulated compartment flap 250, and non-insulated compartment flap 350. The flexible main body 100 comprises an outer surface 110, an upper edge 112, and a lower edge 113. The figure also shows the relative positions of rigid insulated compartment 200 and rigid non-insulated compartment 300. Also shown are a plurality of zippers 620, used for closing a plurality of compartments comprising the embodiment.
FIG. 3 shows an overall frontal view of Embodiment 1 001 of the present invention noting a front flap 120, front pocket 400, insulated compartment flap 250, and non-insulated compartment flap 350. Also shown are a plurality of surfaces comprising a rigid insulated compartment 200, notably an insulated upper surface 201, an insulated lower surface 202, an insulated lateral surface 203, an insulated medial surface 204, and an insulated posterior surface 206.
FIG. 4 shows an overall rear view of Embodiment 1 001 of the present invention noting a rear flap 130, insulated compartment flap 250, and non-insulated compartment flap 350. The non-insulated compartment flap 350 comprises a non-insulated upper edge 351 and a non-insulated lower edge 352. Also shown are a plurality of surfaces comprising a rigid non-insulated compartment 300, notably a non-insulated upper surface 301, a non-insulated lower surface 302, a non-insulated medial surface 304, and a non-insulated anterior surface 305.
FIG. 5 shows a frontal interior view of Embodiment 1 001 of the present invention noting a rear flap 130 and a plurality of fixed carry handles 600. The figure also shows drinks bottle pocket 162, phone pocket 163, credit card slips 164, large pocket 165, small pocket 166, and keys clip 167. Also shown are a plurality of accessories for customizing the invention, including a shoulder strap ring 611 and a zipper 620.
FIG. 6 shows a rearward interior view of Embodiment 1 001 of the present invention noting front flap 120, front pocket 400, and a plurality of fixed carry handles 600. The figure also shows an inner surface 151 of a main compartment interior lining 150, a laptop pocket 160, and a plurality of pen pockets 161. Also shown are a plurality of accessories for customizing the invention, including a shoulder strap ring 611, a zipper 620, and a lock 630.
FIG. 7 shows an overall bottom view of Embodiment 1 001 of the present invention noting a base 500 and metal feet 640. The base 500 comprises a front edge 501, back edge 502, first side 503, and second side 504.
FIG. 8 shows an overall frontal view of Embodiment 2 002 of the present invention noting a flexible main body outer surface 110, front pocket 400, insulated compartment flap 250, and non-insulated compartment flap 350. The figure also shows the relative positions of rigid insulated compartment 200 and rigid non-insulated compartment 300. Also shown are a plurality of zippers 620, used for closing a plurality of compartments comprising the embodiment, and a plurality of attached shoulder straps 660.
FIG. 9 shows an overall frontal view of Embodiment 3 003 of the present invention noting a flexible main body outer surface 110, rear flap 130, insulated compartment flap 250, and non-insulated compartment flap 350. The figure also shows the relative positions of rigid insulated compartment 200 and rigid non-insulated compartment 300. Also shown are a plurality of shoulder strap rings 611.
FIG. 8 shows a logo tag 650.
"""

# Regex to extract phrase + numbers
pattern = re.compile(
    r'(?:[\w\s\-,;:\(\)]*?\s(?:indicated\s+(?:generally\s+)?as|identified\s+as|as|no\.?|reference\s+numeral|shown\s+as)\s+)?'
    r'([\w\s\-\.,;:\(\)]+?)\s*(\d{1,4}(?:,\s*\d{1,4})*)'
)

# Stopwords/noise to remove from phrases
stopwords_re = re.compile(
    r'\b(?:wherein|each|the|and|a|an|when|all|of|may be|is|are|with|such as|general(?:ly)?|indicated|identified|numeral|no|shown|'
    r'that defines|controlled be|may be made of|or includes|i\.e\.|e\.g\.|as by|in use|considering again|be it|some other|one embodiment|'
    r'roughly|such that|whether by|to the extent that|as suggested by|mounted|attached|respectively|similarly|or|this|that|these|those|'
    r'some|any|all|every|each|either|neither|both|few|many|much|more|most|other|such|what|however|with|within|without)\b',
    re.IGNORECASE
)

def normalize_phrase(phrase):
    phrase = stopwords_re.sub('', phrase.lower()).strip()
    phrase = re.sub(r'\bfigs?\.?\s*\d+\w*\b', '', phrase).strip()
    phrase = re.sub(r'\s+', ' ', phrase).strip(" ,.-:;")

    doc = nlp(phrase)

    # Find meaningful noun chunk from end
    main_noun = ""
    for chunk in reversed(list(doc.noun_chunks)):
        if chunk.root.pos_ in ("NOUN", "PROPN") and chunk.root.text.lower() not in {
            "it", "access", "extent", "width", "ends", "structure", "point", "form", "define", "has", "portion", "side", "area"
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

    doc2 = nlp(main_noun.lower())
    words = []
    for token in doc2:
        if token.pos_ in ("NOUN", "PROPN"):
            singular = p.singular_noun(token.text)
            words.append(singular if singular else token.text)
        elif token.pos_ == "ADJ":
            words.append(token.text)

    label = ' '.join(words)
    label = re.sub(r'\b(\w+)\b(?: \1\b)+', r'\1', label)  # Remove duplicates like 'pad pad'

    return label if len(label) > 1 else ""

candidate_labels = {}

for match in pattern.finditer(text):
    phrase, nums = match.group(1), match.group(2)
    if re.search(r'\bfigs?\.?\s*\d+\w*\b', phrase, re.IGNORECASE):
        continue

    phrase = phrase.lower()
    label = normalize_phrase(phrase)
    if not label:
        continue

    for num in nums.split(','):
        num = num.strip()
        if num.isdigit():
            candidate_labels.setdefault(num, []).append(label)

final_labels = {}
for num, labels in candidate_labels.items():
    # Pick shortest label assuming it's the most concise
    if labels:
        final_labels[num] = sorted(labels, key=len)[0]

# ---------- ✅ OUTPUT IN REQUIRED DICTIONARY FORMAT ----------
print("\nnumber_descriptions = {")
for num in sorted(final_labels, key=int):
    label = final_labels[num].replace('"', "'")  # avoid breaking quotes
    print(f'    "{num}": "{label}",')
print("}")
