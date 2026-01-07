import cv2
import json
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def analyze_room_safe(image_path, checkpoint_path, model_type="vit_h"):
    # 1. Force Clean Slate
    torch.set_default_dtype(torch.float32)

    # 2. Load Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not find image at {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 3. CRITICAL: Force CPU to avoid the MPS Float64 Crash
    # The 'AutomaticMaskGenerator' algorithms require float64, which Mac GPUs cannot do.
    # We must use CPU here. It will be slower (approx 1 min) but stable.
    print("⚠️  Mac GPU (MPS) does not support the 64-bit precision required for Mask Filtering.")
    print("⚙️  Switching to CPU mode for stability...")
    device = torch.device("cpu")

    # 4. Load Model
    print("⏳ Loading SAM model... (This might take a moment)")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device, dtype=torch.float32)

    # 5. Initialize Generator
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    print("🧠 Analyzing room... please wait (~45-60 seconds)...")
    
    # Run Generation (No 'no_grad' needed on CPU usually, but good practice)
    with torch.no_grad():
        masks = mask_generator.generate(image)

    # 6. Process results
    room_data = {
        "room_type": "Bedroom",
        "total_items_detected": len(masks),
        "items": []
    }

    for i, mask in enumerate(masks):
        x, y, w, h = mask['bbox']
        item_metadata = {
            "item_id": i,
            "center_point": [int(x + w/2), int(y + h/2)],
            "bbox_xywh": [int(x), int(y), int(w), int(h)],
            "area_px": int(mask['area']),
            "confidence": float(mask['predicted_iou'])
        }
        room_data["items"].append(item_metadata)

    # 7. Save
    output_file = 'bedroom_staff_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(room_data, f, indent=4)
    
    print(f"✅ Success! Saved {len(masks)} items to {output_file}")

# Run the analysis
if __name__ == "__main__":
    analyze_room_safe("bedroom_panorama.png", "sam_vit_h_4b8939.pth")