# 🌱 Krishi-Sahayak: Agentic AI for Smart Agriculture

**A prototype for an AI-powered agricultural advisory platform, addressing yield prediction, climate/agronomy insights, and plant disease detection.**

---

## 📌 Project Overview
**Krishi-Sahayak** is designed to bridge information gaps in Indian agriculture by providing **data-driven, AI-powered insights** for farmers. This prototype focuses on three key modules:
1. **Crop Yield Prediction** (ensemble ML models)(`yield_insight.py`)
2. **Climate & Agronomy Advisor** (hyperlocal insights)(`Crops_insight.py`)
3. **Plant Disease Detection** (deep learning-based diagnosis)(`Crops_disease_detection.py`)

*Built as part of Capital One Launchpad 2025 Hackathon under the theme: **Exploring and Building Agentic AI Solutions for Agriculture**.*

---

## 🌟 Implemented Features

### 1. Crop Climate & Agronomy Advisor (`Crops_insight.py`)
The tool is able to work on a total of 144 different types of crops (potato, maize, rice, sugarcane etc.) and 6 seasons (Winter, Autumn, Summer, Rabi, Kharif, Whole year) 
- **Agent Role**: *Retriever + Explainer Agent* (context-aware recommendations)
- **Input**: Crop name and season.
- **Output**:
  - **Terminal Report**: Optimal Climate conditions, soil type, NPK ratio, irrigation tips, pests/diseases, and planting/harvesting time.
  - **Visual Report**: Bar charts, pie charts, and tables for easy comprehension.
- **Key Innovation**: Hyperlocal, explainable insights with transparent reasoning.

**Example Output:**
<img width="1919" height="887" alt="image" src="https://github.com/user-attachments/assets/e86a4579-f379-4f14-b05e-61d4ab745024" />


---

### 2. Plant Disease Detection (`Crops_disease_detection.py`)
The model is trained on 20,000+ crop leaf disease images and can classify over 15 different types of disease in crops like potato, tomato etc.
- **Agent Role**: *Planner + Explainer Agent* (multi-modal analysis)
- **Input**: Images of plant leaves (tomato, potato, pepper). Inside a input folder
- **Output**:
  - **Terminal Report**: Disease name, confidence score, risk level, causes, symptoms, and solutions.
  - **Visual Report**: Image with annotated disease details and actionable tips. Report saved in output folder
- **Key Innovation**: Multi-modal (image + text) reasoning with source citations.

**Example Output:**
<img width="1905" height="927" alt="image" src="https://github.com/user-attachments/assets/81a6f7bd-5e5e-4836-b5bd-eaa017f92a1a" />


---

### 3. Crop Yield Predictor (`yield_insight.py`)
- **Agent Role**: *Reasoner Agent* (predictive analytics)
- **Input**: Crop, state, season, and environmental parameters (temp, rainfall, humidity, pH, altitude).
- **Output**: Yield predictions (tons/ha) from XGBoost, RandomForest, ExtraTrees, CatBoost, and an ensemble model.
- **Key Innovation**: Ensemble approach for robust, fact-grounded yield forecasts.
- **User Interaction**: CLI with emoji-guided prompts for accessibility.

**Example Output:**

<img width="1849" height="926" alt="image" src="https://github.com/user-attachments/assets/f409a608-6ece-4cd0-b864-cd113c68d51b" />


Terminal output 

🌱 Welcome to the Crop Yield Predictor! 🌾

1️⃣  Enter crop name (e.g., rice, sugarcane, banana, wheat , garlic, potato, maize etc): maize


2️⃣  Enter state (e.g., Uttar Pradesh, ODISHA, MAHARASHTRA, BIHAR, ARUNACHAL PRADESH, ASSAM, JHARKHAND, KERALA, HARYANA etc ): ODISHA


4️⃣  Enter season (options: AUTUMN 🍂, KHARIF 🌾, RABI 🌱, WHOLE YEAR 🌞, SUMMER ☀️, WINTER ❄️): WINTER


📝 Enter temp max (°F, e.g., 95): 100


📝 Enter temp min (°F, e.g., 26): 40


📝 Enter rainfall (mm, e.g., 100): 200


📝 Enter humidity (%, e.g., 70): 78


📝 Enter ph (pH, e.g., 6.5): 4.2


📝 Enter altitude (m, e.g., 200): 300


📝 Enter crop year (year, e.g., 2023): 2021


---------------------------------------------------


🌟 Predictions (tons/ha):
  XGBoost   : 1.304

  RandomForest: 1.000

  ExtraTrees: 1.000

  CatBoost  : 1.306

  Ensemble  : 1.152

✅ Prediction complete!

-----------------------

## 🛠 Technical Stack
| Component          | Technology Stack                                                                 |
|--------------------|----------------------------------------------------------------------------------|
| **Backend**        | Python, TensorFlow, Scikit-learn, XGBoost, CatBoost                              |
| **Data Processing**| Pandas, NumPy, Joblib                                                             |
| **Visualization**  | Matplotlib, PIL                                                                   |
| **Datasets**       | `enhanced_crop_dataset.csv`, PlantVillage (via `img_model_best.h5`)              |
| **Offline Support**| Local model inference (no internet required for predictions/reports)             |

---

## 🚀 Alignment with Synopsis Vision
| Synopsis Feature               | Prototype Implementation                                          |
|--------------------------------|-------------------------------------------------------------------|
| Multi-Agent Architecture       | Modular scripts for yield, climate, and disease agents.           |
| Hyperlocal Context Awareness   | State/season/crop-specific insights in `Crops_insight.py`.       |
| Multi-Modal Inputs             | Image uploads for disease detection (`Crops_disease_detection.py`).|
| Explainable AI                 | Visual tables, confidence scores, and risk levels in all outputs.|
| Offline-First Design           | Local model files; no cloud dependency.                           |
| Fact-Grounded Recommendations  | Ensemble models + disease knowledge base (`DISEASE_INFO`).       |

---

## 📂 Installation & Usage

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies
pip install -r requirements.txt


   
