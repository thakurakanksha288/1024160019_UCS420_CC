import cv2
import easyocr
import re

# Initialize the EasyOCR reader for English text
reader = easyocr.Reader(['en'])

def parse_medicine_label(text_list):
    full_text = " ".join(text_list)
    print("\n" + "=" * 50)
    print("      COGNITIVE MEDICINE ASSISTANT REPORT      ")
    print("=" * 50)
    print(f"📄 Raw Extracted Text: {full_text}\n")
    
    # Extract Expiration Date
    exp_pattern = r'(EXP|EXPIRATION|MFG|DATE)?\s*[:.-]?\s*(\d{2}[/-]\d{2}[/-]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b)'
    exp_match = re.search(exp_pattern, full_text, re.IGNORECASE)
    
    if exp_match:
        print(f"📅 Expiry/Mfg Date Detected: {exp_match.group(0)}")
    else:
        print("📅 Expiry Date: Not automatically detected (check label manually)")

    # Extract Dosage Units (mg, ml, mcg, tablets, etc.)
    dosage_pattern = r'\b\d+\s*(?:mg|ml|g|mcg|tablets|capsules)\b'
    dosages = re.findall(dosage_pattern, full_text, re.IGNORECASE)
    
    if dosages:
        print(f"💊 Detected Dosage Strengths: {', '.join(set(dosages))}")
    else:
        print("💊 Dosage Strength: Not explicitly found in text")

    # Cognitive Advice Engine
    print("\n💡 Assistant Recommendations:")
    print(" • Verify dosage instructions with your physician or pharmacist.")
    print(" • Keep out of reach of children and store at room temperature.")
    print(" • Ensure safety seals are intact prior to use.")
    print("=" * 50)

def process_medicine_bottle(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at '{image_path}'. Check the file path.")
        return

    # Extract text and bounding boxes
    results = reader.readtext(image_path)
    extracted_texts = []

    for (bbox, text, prob) in results:
        if prob > 0.25:  # Confidence threshold filter
            extracted_texts.append(text)
            
            # Draw bounding box and text overlays
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(img, text, (top_left[0], max(20, top_left[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Process extracted text through cognitive rules
    parse_medicine_label(extracted_texts)

    # Display visualized output
    cv2.imshow("Medicine Label Reader - Cognitive Vision", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Place your medicine bottle image in the same workspace folder
    sample_image = "medicine_bottle.jpg"
    process_medicine_bottle(sample_image)