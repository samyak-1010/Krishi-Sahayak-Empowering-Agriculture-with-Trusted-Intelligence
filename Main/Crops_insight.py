import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets", "enhanced_crop_dataset.csv")

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Make sure 'season' and 'crop' are strings, replace NaN with empty string
df['season'] = df['season'].fillna("").astype(str)
df['crop'] = df['crop'].fillna("").astype(str)

def get_crop_climate(season, crop):
    # Filter dataset for the given season and crop (case-insensitive)
    filtered_df = df[
        (df['season'].str.upper() == season.upper()) &
        (df['crop'].str.upper() == crop.upper())
    ]

    if filtered_df.empty:
        return f"❌ No data found for crop '{crop}' in season '{season}'."

    # Compute average climatic conditions
    climate_data = {
        "🌡️ Avg Max Temp (°F)": filtered_df['temp_max'].mean(),
        "🌡️ Avg Min Temp (°F)": filtered_df['temp_min'].mean(),
        "☔ Avg Rainfall (mm)": filtered_df['rainfall'].mean(),
        "💧 Avg Humidity (%)": filtered_df['humidity'].mean(),
        "🧪 Avg pH": filtered_df['ph'].mean(),
        "⛰️ Avg Altitude (m)": filtered_df['altitude'].mean(),
        "🌱 Soil Type": filtered_df['soil_type'].mode()[0],
        "💩 NPK Ratio": filtered_df['npk_ratio'].mode()[0],
        "🚰 Irrigation Tips": filtered_df['irrigation_tips'].mode()[0],
        "🐜 Pests/Diseases": filtered_df['pests_diseases'].mode()[0],
        "🌿 Companion Plants": filtered_df['companion_plants'].mode()[0],
        "📅 Planting Time": filtered_df['planting_time'].mode()[0],
        "📅 Harvesting Time": filtered_df['harvesting_time'].mode()[0],
        "🌾 Expected Yield": filtered_df['expected_yield'].mode()[0]
    }

    return climate_data

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Creative visualization function for terminal
def display_terminal_output(crop, season, climate_data):
    print(f"\n{Colors.BOLD}{Colors.HEADER}🌱🌾 CROP CLIMATE & AGRONOMY REPORT 🌾🌱{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.YELLOW}📜 Report for: {crop.upper()} in {season.upper()} season{Colors.ENDC}\n")

    # Top border
    print(f"{Colors.GREEN}╔{'═' * 50}╗{Colors.ENDC}")

    # Climate section
    print(f"{Colors.GREEN}║{Colors.ENDC} {Colors.BOLD}{Colors.BLUE}🌦️ CLIMATE CONDITIONS:{Colors.ENDC}")
    print(f"{Colors.GREEN}║{Colors.ENDC}   🌡️ Avg Max Temp: {climate_data['🌡️ Avg Max Temp (°F)']:.1f}°C")
    print(f"{Colors.GREEN}║{Colors.ENDC}   🌡️ Avg Min Temp: {climate_data['🌡️ Avg Min Temp (°F)']:.1f}°C")
    print(f"{Colors.GREEN}║{Colors.ENDC}   ☔ Avg Rainfall: {climate_data['☔ Avg Rainfall (mm)']:.1f} mm")
    print(f"{Colors.GREEN}║{Colors.ENDC}   💧 Avg Humidity: {climate_data['💧 Avg Humidity (%)']:.1f}%")
    print(f"{Colors.GREEN}║{Colors.ENDC}   🧪 Avg pH: {climate_data['🧪 Avg pH']:.1f}")
    print(f"{Colors.GREEN}║{Colors.ENDC}   ⛰️ Avg Altitude: {climate_data['⛰️ Avg Altitude (m)']:.1f} m")

    # Middle border
    print(f"{Colors.GREEN}║{'─' * 50}║{Colors.ENDC}")

    # Agronomy section
    print(f"{Colors.GREEN}║{Colors.ENDC} {Colors.BOLD}{Colors.BLUE}🌱 AGRONOMY GUIDELINES:{Colors.ENDC}")
    print(f"{Colors.GREEN}║{Colors.ENDC}   🌱 Soil Type: {climate_data['🌱 Soil Type']}")
    print(f"{Colors.GREEN}║{Colors.ENDC}   💩 NPK Ratio: {climate_data['💩 NPK Ratio']}")
    print(f"{Colors.GREEN}║{Colors.ENDC}   🚰 Irrigation: {climate_data['🚰 Irrigation Tips']}")
    print(f"{Colors.GREEN}║{Colors.ENDC}   🐜 Pests/Diseases: {climate_data['🐜 Pests/Diseases']}")
    print(f"{Colors.GREEN}║{Colors.ENDC}   🌿 Companions: {climate_data['🌿 Companion Plants']}")
    print(f"{Colors.GREEN}║{Colors.ENDC}   📅 Planting: {climate_data['📅 Planting Time']}")
    print(f"{Colors.GREEN}║{Colors.ENDC}   📅 Harvesting: {climate_data['📅 Harvesting Time']}")
    print(f"{Colors.GREEN}║{Colors.ENDC}   🌾 Yield: {climate_data['🌾 Expected Yield']}")

    # Bottom border
    print(f"{Colors.GREEN}╚{'═' * 50}╝{Colors.ENDC}\n")

    # Visual tips
    print(f"{Colors.BOLD}{Colors.YELLOW}💡 TIPS:{Colors.ENDC}")
    print(f"   • Use {climate_data['💩 NPK Ratio']} fertilizer for optimal growth.")
    print(f"   • {climate_data['🚰 Irrigation Tips']}")
    print(f"   • Watch for {climate_data['🐜 Pests/Diseases']} and use organic pesticides if needed.")
    print(f"   • Plant with {climate_data['🌿 Companion Plants']} for natural pest control.\n")

def display_matplotlib_visualizations(crop, season, climate_data):
    # Light green background for figure
    fig = plt.figure(figsize=(15, 10), facecolor='#DAFABF')
    fig.suptitle(f"Visual Report for {crop.upper()} in {season.upper()} Season\n\n\n", 
                 fontsize=16, fontweight='bold')

    # 1. Climate Conditions Bar Chart
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_facecolor('honeydew')
    climate_keys = ["Avg Max Temp (°F)", "Avg Min Temp (°F)", "Avg Rainfall (mm)", 
                    "Avg Humidity (%)", "Avg pH", "Avg Altitude (m)"]
    climate_values = [
        climate_data["🌡️ Avg Max Temp (°F)"],
        climate_data["🌡️ Avg Min Temp (°F)"],
        climate_data["☔ Avg Rainfall (mm)"],
        climate_data["💧 Avg Humidity (%)"],
        climate_data["🧪 Avg pH"],
        climate_data["⛰️ Avg Altitude (m)"]
    ]
    colors = plt.cm.viridis(np.linspace(0, 1, len(climate_keys)))
    bars = ax1.bar(climate_keys, climate_values, color=colors, width=0.6)
    ax1.set_title("Optimal Climatic Conditions", fontsize=12, fontweight='bold',color='#2E8B57')
    ax1.set_xticklabels(climate_keys, rotation=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom')

    # 2. NPK Ratio Pie Chart
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_facecolor('honeydew')
    npk = climate_data["💩 NPK Ratio"].split('-')
    npk_labels = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
    npk_values = [int(val) for val in npk]
    ax2.pie(npk_values, labels=npk_labels, autopct='%1.1f%%', startangle=90, 
            colors=['#ff9999','#66b3ff','#99ff99'])
    ax2.set_title("Optimal NPK Ratio Breakdown", fontsize=12, fontweight='bold',color='#2E8B57')

    # 3. Agronomy Summary Table
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_facecolor('honeydew')
    table_data = [
        ["Pests/Diseases", climate_data["🐜 Pests/Diseases"]],
        ["Companion Plants", climate_data["🌿 Companion Plants"]],
        ["Planting Time", climate_data["📅 Planting Time"]],
        ["Harvesting Time", climate_data["📅 Harvesting Time"]],
        ["Expected Yield", climate_data["🌾 Expected Yield"]]
    ]
    ax3.axis('off')
    table = ax3.table(cellText=table_data, colLabels=["Attribute", "Details"], 
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')

    ax3.set_title("\n\n\n\n\n\nAgronomy Summary", fontsize=12, fontweight='bold',color='#2E8B57')

        # 4. Soil, Irrigation & Tips
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_facecolor('honeydew')

    soil_info = (
        f"Soil Type: {climate_data['🌱 Soil Type']}\n"
        f"\nIrrigation: {climate_data['🚰 Irrigation Tips']}"
    )

    tips_text = (
        f"\n TIPS:\n\n"

        f"• Use {climate_data['💩 NPK Ratio']} fertilizer for optimal growth.\n"
        f"• {climate_data['🚰 Irrigation Tips']}\n"
        f"• Watch for {climate_data['🐜 Pests/Diseases']} and use organic pesticides if needed.\n"
        f"• Plant with {climate_data['🌿 Companion Plants']} for natural pest control."
    )

    # Soil box (top position)
    ax4.text(
        0, 1.45,  # x, y position (top)
        soil_info,
        ha='center',
        va='center',
        fontsize=12,
        fontweight='bold',
        bbox=dict(facecolor='lightgreen', alpha=0.5, boxstyle='round,pad=0.5'),
        transform=ax4.transAxes
    )

    # Tips box (below soil box with gap)
    ax4.text(
        0.5, 0.75,  # x, y position (below soil box with gap)
        tips_text,
        ha='center',
        va='top',
        fontsize=10,
        fontweight='bold',
        bbox=dict(facecolor='lightyellow', alpha=0.5, boxstyle='round,pad=0.5'),
        transform=ax4.transAxes
    )

    ax4.set_title("Tips for better growth", fontsize=12, fontweight='bold', color='#2E8B57')
    ax4.axis('off')



    plt.tight_layout()
    plt.show()

# Interactive input
print(f"{Colors.BOLD}{Colors.HEADER}🌱 Welcome to the Crop Climate & Agronomy Info Tool! 🌾{Colors.ENDC}\n")
season_input = input(f"{Colors.BOLD}🌞 Enter season (e.g., AUTUMN, KHARIF, RABI, SUMMER, WINTER):{Colors.ENDC} ").strip()
crop_input = input(f"{Colors.BOLD}🌾 Enter crop name (e.g., rice, wheat, banana, maize):{Colors.ENDC} ").strip()

# Get climate and agronomy data
result = get_crop_climate(season_input, crop_input)

# Display results
if isinstance(result, str):
    print(f"{Colors.RED}{result}{Colors.ENDC}")
else:
    # Display terminal output
    display_terminal_output(crop_input, season_input, result)
    # Display Matplotlib visualizations
    display_matplotlib_visualizations(crop_input, season_input, result)
