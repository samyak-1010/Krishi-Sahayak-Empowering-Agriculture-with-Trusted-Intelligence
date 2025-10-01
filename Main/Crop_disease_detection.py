import os
import numpy as np
import tensorflow as tf
import logging
import absl.logging
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from datasets.disease_info import DISEASE_INFO

# Handle ImageDataGenerator import for new TF versions
try:
    from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
except ImportError:
    from keras.preprocessing.image import load_img, img_to_array # type: ignore

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = all messages, 1 = info, 2 = warnings, 3 = errors
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)


# Suppress TensorFlow warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')


CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite"
]

# Load your trained model
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "img_model_best.h5")
model = tf.keras.models.load_model(MODEL_PATH)

def parse_class_name(class_name):
    """Split into crop name and disease name (or Healthy)."""
    # Replace different separators with uniform
    name = class_name.replace("__", "_")
    parts = name.split("_", 1)
    crop = parts[0]
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

    if "healthy" in disease.lower():
        disease = "Healthy"
    return crop, disease

def predict_image(image_path, output_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    full_class_name = CLASS_NAMES[class_idx]
    crop, disease_name = parse_class_name(full_class_name)

    info = DISEASE_INFO.get(full_class_name, {
        "causes": "N/A",
        "symptoms": "N/A",
        "lifecycle": "N/A",
        "prevention": "N/A",
        "chemical_control": "N/A",
        "organic_solutions": "N/A",
        "fertilizer": "N/A",
        "additional_tips": "N/A"
    })

    # Risk level
    if "healthy" in disease_name.lower():
        risk_level = "Low"
        severity_color = "#C8E6C9"  # light green
    elif confidence > 80:
        risk_level = "High"
        severity_color = "#FFCDD2"  # light red
    else:
        risk_level = "Medium"
        severity_color = "#FFE0B2"  # light orange

    # Terminal output
    print("\n🌾 PLANT DISEASE DETECTION REPORT 🌾")
    print(f"🌱 Crop: {crop}")
    print(f"⚠ Disease: {disease_name}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print(f"🔥 Risk Level: {risk_level}\n")
    print(f"⚠ Causes: {info.get('causes', 'N/A')}")
    print(f"🩺 Symptoms: {info.get('symptoms', 'N/A')}")
    print(f"🔄 Lifecycle: {info.get('lifecycle', 'N/A')}")
    print(f"🛡 Prevention: {info.get('prevention', 'N/A')}")
    print(f"🧪 Chemical Control: {info.get('chemical_control', 'N/A')}")
    print(f"🌱 Organic Solutions: {info.get('organic_solutions', 'N/A')}")
    print(f"🌿 Fertilizer: {info.get('fertilizer', 'N/A')}")
    print(f"💡 Additional Tips: {info.get('additional_tips', 'N/A')}")
  
    print("-" * 70)

    # Matplotlib visualization
    fig, ax = plt.subplots(1, 2, figsize=(20, 14), gridspec_kw={'width_ratios': [1, 1.5]})
    fig.patch.set_facecolor('#DAFABF')  # very light green background

    # Left: Uploaded image
    ax[0].imshow(mpimg.imread(image_path))
    ax[0].axis("off")
    ax[0].set_title("Uploaded Image", fontsize=16, fontweight='bold', color="#2E8B57")

    # Right: Table with information
    ax[1].axis("off")

    # Background box for right panel
    ax[1].add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax[1].transAxes,
        facecolor=severity_color, alpha=0.3, zorder=-1
    ))
    ax[1].set_title("Results", fontsize=16, fontweight='bold', color="#2E8B57", pad=20)



    # Table data
    data = [
    ["Crop", crop],
    ["Disease", disease_name],
    ["Risk Level", risk_level],
    ["Causes", info.get('causes', 'N/A')],
    ["Symptoms", info.get('symptoms', 'N/A')],
    ["Lifecycle", info.get('lifecycle', 'N/A')],
    ["Prevention", info.get('prevention', 'N/A')],
    ["Chemical Control", info.get('chemical_control', 'N/A')],
    ["Organic Solutions", info.get('organic_solutions', 'N/A')],
    ["Fertilizer", info.get('fertilizer', 'N/A')],
    ["Additional Tips", info.get('additional_tips', 'N/A')]
]

     # Create table
    table = ax[1].table(
        cellText=data,
        colWidths=[0.25, 0.99],  # Adjusted column widths
        loc='center',
        cellLoc='left',
        bbox=[0.0, 0.03, 1.0, 0.96]   # Adjusted table width and height
    )

    # Styling the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)  # Adjusted row height

    # Bold the headings
    for (row, col), cell in table.get_celld().items():
        if col == 0:
            cell.set_text_props(weight='bold', fontsize=11)

    fig.suptitle("Plant Disease Detection Report\n\n\n\n",
                 fontsize=20, fontweight="bold", color="#2E8B57", y=0.95)
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    input_folder = os.path.join(PROJECT_ROOT, "input")
    output_folder = os.path.join(PROJECT_ROOT, "output")   # folder where you put images
    output_folder = r"D:\CapitalOne\Capital-One-Launchpad--25----Team-Tech_Pulse-\output" # folder where results will be saved
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_report.png")
            predict_image(image_path, output_path)
            print(f"✅ Processed {filename} → saved report to {output_path}")