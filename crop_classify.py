import cv2
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- CONFIGURATION ---
IMAGE_PATH = "bedroom_panorama.png"
JSON_PATH = "bedroom_staff_analysis.json"
OUTPUT_JSON_PATH = "bedroom_staff_labeled.json"

# Define the "vocabulary" for your room. 
# CLIP will choose the best match from this list.
CANDIDATE_LABELS = [
    "bed",
    "messy pile of clothes",
    "folded clothes",
    "wooden nightstand",
    "lamp",
    "books",
    "trash or clutter",
    "shoe",
    "window",
    "door",
    "wall",
    "floor"
]

def load_clip_model():
    print("⏳ Loading CLIP model (approx 500MB)...")
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor

def classify_crop(model, processor, cv2_image, labels):
    # Convert OpenCV (BGR) to PIL (RGB)
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    # Prepare inputs
    inputs = processor(
        text=labels, 
        images=pil_image, 
        return_tensors="pt", 
        padding=True
    )

    # Run Inference (No Gradients needed)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Calculate probabilities
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    # Get the highest confidence label
    best_idx = probs.argmax().item()
    best_label = labels[best_idx]
    confidence = probs[0][best_idx].item()

    return best_label, confidence

def main():
    # 1. Load Data
    model, processor = load_clip_model()
    image = cv2.imread(IMAGE_PATH)
    
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    print(f"🔍 Classifying {len(data['items'])} detected items...")

    # 2. Iterate and Classify
    labeled_count = 0
    for item in data['items']:
        # Skip tiny items (noise) to save time
        if item['area_px'] < 500:
            item['label'] = "too_small"
            continue

        # Extract coordinates
        x, y, w, h = item['bbox_xywh']
        
        # Safety check for image boundaries
        if w <= 0 or h <= 0: continue
        
        # Crop the object
        crop = image[y:y+h, x:x+w]
        
        if crop.size == 0: continue

        # Classify
        label, conf = classify_crop(model, processor, crop, CANDIDATE_LABELS)
        
        # Update the JSON object
        item['label'] = label
        item['label_confidence'] = round(conf, 4)
        
        # Optional: Print progress for interesting items
        if conf > 0.5:
            print(f"   ID {item['item_id']}: Found '{label}' ({int(conf*100)}%)")
        
        labeled_count += 1

    # 3. Save Updated JSON
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"✅ Finished! labeled {labeled_count} items.")
    print(f"📂 Saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()