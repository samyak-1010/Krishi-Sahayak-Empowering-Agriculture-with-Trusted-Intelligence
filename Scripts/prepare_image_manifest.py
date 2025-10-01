import os
import csv

# Root PlantVillage directory
PLANT_DIR = os.path.join(os.getcwd(),"Capital-One-Launchpad--25----Team-Tech_Pulse-", "datasets", "PlantVillage")  
OUT_CSV = os.path.join(os.getcwd(),"Capital-One-Launchpad--25----Team-Tech_Pulse-", "datasets", "plant_images_manifest.csv")

def build_manifest(root_dir, out_csv):
    rows = []
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append([os.path.join(cls_dir, fname), cls])

    print(f"Found {len(rows)} images across {len(classes)} classes.")

    # write csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(rows)

    print(f"Manifest saved to: {out_csv}")

if __name__ == "__main__":
    if not os.path.isdir(PLANT_DIR):
        raise SystemExit(f"PlantVillage folder not found at {PLANT_DIR}. Edit the PLANT_DIR path in this script.")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    build_manifest(PLANT_DIR, OUT_CSV)
