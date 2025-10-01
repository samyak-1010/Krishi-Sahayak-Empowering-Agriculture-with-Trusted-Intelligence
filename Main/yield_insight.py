import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- Load dataset ---
DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "enhanced_crop_dataset.csv")
df = pd.read_csv(DATA_PATH)

# --- Numeric features with units and example values ---
NUMERIC_FEATURES_INFO = {
    "temp_max": {"unit": "°F", "example": 95},
    "temp_min": {"unit": "°F", "example": 26},
    "rainfall": {"unit": "mm", "example": 100},
    "humidity": {"unit": "%", "example": 70},
    "ph": {"unit": "pH", "example": 6.5},
    "altitude": {"unit": "m", "example": 200},
    "crop_year": {"unit": "year", "example": 2023},
}

# --- Load saved ensemble model ---
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "crop_yield_recommender_ensemble.pkl")

model = joblib.load(MODEL_PATH)

# --- Predict function using the loaded dict ---
def predict_yield(features: dict):
    feature_names = model["feature_names"]
    x = np.array([[features.get(f, 0.0) for f in feature_names]], dtype=float)
    x[np.isnan(x)] = 0.0

    out = {}
    if model.get("xgb"): out["XGBoost"] = float(max(0.0, model["xgb"].predict(x)[0]))
    if model.get("rf"): out["RandomForest"] = float(max(0.0, model["rf"].predict(x)[0]))
    if model.get("et"): out["ExtraTrees"] = float(max(0.0, model["et"].predict(x)[0]))
    if model.get("cat"): out["CatBoost"] = float(max(0.0, model["cat"].predict(x)[0]))

    weights = model["weights"]
    denom = 0.0
    ens = 0.0
    for key, w in weights.items():
        if key == "xgb" and "XGBoost" in out: ens += w*out["XGBoost"]; denom += w
        elif key == "rf" and "RandomForest" in out: ens += w*out["RandomForest"]; denom += w
        elif key == "et" and "ExtraTrees" in out: ens += w*out["ExtraTrees"]; denom += w
        elif key == "cat" and "CatBoost" in out: ens += w*out["CatBoost"]; denom += w
    out["Ensemble"] = float(max(0.0, ens / denom if denom>0 else np.mean(list(out.values()))))
    return out

# ------------------------
# User input section with emojis and examples
# ------------------------
print("🌱 Welcome to the Crop Yield Predictor! 🌾\n")
crop_name = input(" 1️⃣  Enter crop name (e.g., rice, sugarcane, banana, wheat , garlic, potato, maize etc): ").strip().lower()
state_name = input(" 2️⃣  Enter state (e.g., Uttar Pradesh, ODISHA, MAHARASHTRA, BIHAR, ARUNACHAL PRADESH, ASSAM, JHARKHAND, KERALA, HARYANA etc ): ").strip()

# Season input with emoji and example options
season_options = [
    "AUTUMN 🍂",
    "KHARIF 🌾",
    "RABI 🌱",
    "WHOLE YEAR 🌞",
    "SUMMER ☀️",
    "WINTER ❄️"
]
season_clean = ["AUTUMN", "KHARIF", "RABI", "WHOLE YEAR","SUMMER","WINTER"]

while True:
    season_input = input(f" 4️⃣  Enter season (options: {', '.join(season_options)}): ").strip().upper()
    if season_input in season_clean:
        break
    print(f"❌ Invalid input. Please choose from {', '.join(season_clean)}.")

user_features = {}
for feature, info in NUMERIC_FEATURES_INFO.items():
    while True:
        try:
            value = float(input(
                f"📝 Enter {feature.replace('_',' ')} ({info['unit']}, e.g., {info['example']}): "
            ))
            user_features[feature] = value
            break
        except ValueError:
            print("❌ Please enter a valid number.")

# Add crop info in features
user_features["crop"] = crop_name
user_features["state"] = state_name
user_features["season"] = season_input

# ------------------------
# Get prediction
# ------------------------
preds_dict = predict_yield(user_features)

# Display results
print("\n🌟 Predictions (tons/ha):")
for model_name, pred in preds_dict.items():
    print(f"  {model_name:<10s}: {pred:.3f}")

print("\n✅ Prediction complete!")

# ------------------------
# Visualization: Input vs Prediction
# ------------------------
import matplotlib.pyplot as plt

def display_input_vs_prediction(user_features, preds_dict):
    # --- Figure ---
    fig, ax = plt.subplots(1, 2, figsize=(18, 9), facecolor="#DAFABF")  # light green bg
    fig.subplots_adjust(wspace=0.25)
    
    # ------------------------
    # LEFT: Provided Input (Wider Box)
    # ------------------------
    ax[0].axis("off")
    input_text = (
        f"Provided Input\n\n"
        f"Crop: {user_features['crop'].upper()}\n"
        f"State: {user_features['state']}\n"
        f"Season: {user_features['season']}\n\n"
    )
    for feat, val in NUMERIC_FEATURES_INFO.items():
        input_text += f"{feat.replace('_',' ').title()}: {user_features[feat]} {val['unit']}\n"

    ax[0].text(
        0.14, 0.35, input_text,
        ha="left", va="center",
        fontsize=13, fontweight="bold",
        bbox=dict(facecolor="#D6EAF8", edgecolor="navy", boxstyle="round,pad=4.0"),  # wider padding
        wrap=True
    )
    ax[0].text(
    0.02, 0.78, "Provided Input",   # position above the box
    ha="left", va="bottom",
    fontsize=16, fontweight="bold", color="navy"
)

    # ------------------------
    # RIGHT: Predictions (Bar Chart + Highlight Ensemble)
    # ------------------------
    ax[1].set_facecolor("#F4F6F6")
    models = list(preds_dict.keys())
    values = list(preds_dict.values())

    colors = ["#5DADE2", "#58D68D", "#F5B041", "#AF7AC5", "#E74C3C"]  # Ensemble highlighted in red

    ax[1].barh(models, values, color=colors, edgecolor="black")
    ax[1].set_xlabel("Yield (tons/ha)", fontsize=14, fontweight="bold")
    ax[1].set_title("\nPredicted Output", fontsize=18, fontweight="bold", color="darkgreen")

    # Annotate values
    for i, v in enumerate(values):
        ax[1].text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=12, fontweight="bold")

    # ------------------------
    # Title
    # ------------------------
    plt.suptitle("Crop Yield Prediction – Input vs Output", fontsize=17, fontweight="bold", color="black")
    plt.show()

display_input_vs_prediction(user_features, preds_dict)