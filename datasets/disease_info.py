# This data is collected from various dataset sources from the internet

DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "causes": """
Caused by the bacterium Xanthomonas campestris pv. vesicatoria.
Spread through rain, overhead irrigation, contaminated tools, seeds, and plant debris.
Favorable conditions include warm (24–29°C), humid environments with frequent rainfall.
        """,
        "symptoms": """
Small, water-soaked spots on leaves, which turn brown or black and may have a yellow halo.
Spots may coalesce, leading to leaf drop.
Fruit lesions are raised, scabby, and can cause cracking.
        """,
        "lifecycle": """
Bacteria overwinter in seed, plant debris, and weeds. Secondary spread occurs
via water splash and mechanical means.
        """,
        "prevention": """
Use certified disease-free seeds and resistant varieties such as Aristotle or Heritage.
Apply copper-based bactericides like Bordeaux mixture or copper hydroxide preventatively.
Avoid overhead irrigation and use drip irrigation instead.
        """,
        "chemical_control": """
Copper compounds such as Kocide or Champ.
Streptomycin, where permitted by local regulations.
Actigard (acibenzolar-S-methyl) for induced resistance.
        """,
        "organic_solutions": """
Neem oil/baking soda sprays (1 tablespoon baking soda, 1 teaspoon oil, and 1 lit of water.)
Compost tea to boost plant immunity.
Beneficial microbes like Bacillus subtilis.
        """,
        "fertilizer": """
Use a 10-10-10 NPK fertilizer with added calcium and magnesium to strengthen cell walls.
Avoid excessive nitrogen, which promotes succulent, disease-susceptible growth.
Micronutrients like zinc and manganese support enzyme function.
        """,
        "additional_tips": """
Mulch with straw to reduce soil splash.
Prune lower leaves to improve airflow.
Monitor fields regularly for early symptoms.
        """
    },
    "Pepper__bell___healthy": {
        "causes": "No disease present. Plant is in optimal health.",
        "symptoms": "Vibrant green leaves, strong stems, and robust fruit development.",
        "prevention": """
Use drip irrigation to avoid wetting foliage.
Maintain balanced fertilization with micronutrients.
Monitor for pests like aphids, whiteflies, and mites.
        """,
        "fertilizer": """
Use a 15-5-15 NPK fertilizer for optimal growth and fruit production.
Micronutrients like boron for fruit set and iron for chlorophyll synthesis are beneficial.
Organic amendments such as compost or worm castings improve soil health.
        """,
        "additional_tips": """
Rotate crops annually to prevent soil-borne diseases.
Use row covers to protect from pests in early growth stages.
        """
    },
    "Potato___Early_blight": {
        "causes": """
Caused by the fungus Alternaria solani.
Spread by wind, rain, infected seed tubers, and plant debris.
Favorable conditions (75–85°F/24–29°C), humid weather with alternating wet/dry periods.
        """,
        "symptoms": """
Brown, concentric ringed spots on lower or older leaves.
Stem lesions are dark, sunken, and may girdle stems.
Tubers develop dry, leathery spots.
        """,
        "lifecycle": """
Fungus overwinters in infected tubers/debris. Spores are wind-dispersed through wounds.
        """,
        "prevention": """
Plant resistant varieties like Defender or Elba.
Treat seed tubers with fungicide dips such as mancozeb.
Practice crop rotation for 3 or more years away from solanaceous crops.
Apply fungicides like chlorothalonil or mancozeb.
Hill potatoes to bury tubers and avoid overhead irrigation.
        """,
        "chemical_control": """
Protectant fungicides like mancozeb or chlorothalonil (e.g., Bravo, Daconil).
Systemic fungicides like azoxystrobin (Quadris) or pyraclostrobin.
        """,
        "organic_solutions": """
Baking soda spray made with 1 tablespoon baking soda, 1 teaspoon oil, and 1 liter of water.
Compost tea to suppress fungal growth.
Neem oil for its fungistatic properties.
        """,
        "fertilizer": """
Use a 5-10-10 NPK fertilizer with added potassium to enhance disease resistance.
Avoid excessive nitrogen, which promotes lush, susceptible foliage.
Calcium reduces tuber bruising and disease entry.
        """,
        "additional_tips": """
Remove and destroy infected foliage before harvest.
Harvest during dry weather to minimize tuber infection.
        """
    },
    "Potato___healthy": {
        "causes": "No disease present. Plant is in optimal health.",
        "symptoms": "Uniform green foliage, strong stems, and healthy tuber development.",
        "prevention": """
Test soil pH (ideal: 5.0–6.5) and amend as needed.
Provide consistent moisture and avoid waterlogging.
Monitor for pests like Colorado potato beetle, aphids, and wireworms.
        """,
        "fertilizer": """
Use a 10-10-10 NPK fertilizer for overall plant health.
Micronutrients like magnesium for photosynthesis and sulfur for protein synthesis are good.
Organic matter such as compost or aged manure improves soil structure.
        """,
        "additional_tips": """
Plant certified disease-free seed potatoes.
Use mulch to conserve moisture and suppress weeds.
        """
    },
    "Potato___Late_blight": {
        "causes": """
Caused by the oomycete Phytophthora infestans.
Spread by sporangia via wind and water, and infected seed tubers.
Favorable conditions include cool (60–70°F/15–21°C), wet weather with high humidity (>90%).
        """,
        "symptoms": """
Water-soaked, greasy green-black spots on leaves.
White fungal growth on the undersides of leaves in humid conditions.
Stem lesions are dark and irregular, and tubers develop reddish-brown, irregular lesions.
        """,
        "lifecycle": """
Overwinters in infected tubers and volunteer plants. Sporangia germinate in free water,
 releasing zoospores that infect foliage.
        """,
        "prevention": """
Plant resistant varieties like Elba, Kennebec, or Sarpo Mira.
Treat seed tubers with hot water or fungicide dips.
Practice crop rotation for 3 or more years and avoid planting near tomatoes or eggplants.
Apply fungicides like chlorothalonil or mefenoxam.
Ensure proper drainage and avoid overhead irrigation.
        """,
        "chemical_control": """
Protectant fungicides like chlorothalonil (Bravo) or mancozeb.
Systemic fungicides like mefenoxam (Ridomil) or dimethomorph (Acrobat).
Biological control using Bacillus subtilis (e.g., Serenade).
        """,
        "organic_solutions": """
Copper sprays such as Bordeaux mixture.
Garlic or horseradish spray for natural fungicidal properties.
Remove volunteers and cull piles to eliminate inoculum.
        """,
        "fertilizer": """
Use a 10-20-10 NPK fertilizer with added phosphorus to promote root and tuber development.
Potassium enhances plant vigor and disease resistance.
Calcium reduces tuber susceptibility to infection.
        """,
        "additional_tips": """
Scout fields daily during wet periods.
Destroy cull piles and volunteer plants.
Harvest tubers 2–3 weeks after vine kill to allow skin set.
        """
    },
    "Tomato__Target_Spot": {
        "causes": """
        Caused by the fungus Corynespora cassiicola.
        Spread by wind, rain, infected seed, and plant debris.
        Favorable conditions include warm (75–85°F/24–29°C), humid environments.
        """,
        "symptoms": """
        Small, circular, dark brown spots with concentric rings on leaves.
        Stem lesions are elongated and may girdle stems.
        Fruit lesions are sunken, dark, and may crack.
        """,
        "lifecycle": """
        Fungus overwinters in seed and plant debris. Spores are wind-dispersed
          and infect through wounds or stomata.
        """,
        "prevention": """
        Plant resistant varieties like Mountain Merit or Quinte.
        Treat seeds with hot water or fungicide soaks.
        Practice crop rotation for 2 or more years away from solanaceous crops.
        Apply fungicides like chlorothalonil, mancozeb, or copper-based sprays.
        Prune for airflow and avoid overhead irrigation.
        """,
        "chemical_control": """
        Protectant fungicides like chlorothalonil or mancozeb.
        Systemic fungicides like azoxystrobin or pyraclostrobin.
        """,
        "organic_solutions": """
        Neem oil or copper sprays.
        Compost tea to suppress fungal growth.
        Remove infected debris promptly.
        """,
        "fertilizer": """
        Use a 15-5-15 NPK fertilizer with added magnesium for photosynthesis.
        Calcium reduces fruit cracking.
        Avoid excessive nitrogen, which promotes disease-susceptible growth.
        """,
        "additional_tips": """
        Stake or trellis plants to improve airflow.
        Monitor for early symptoms, especially after rain.
        """
    },
    "Tomato__Tomato_mosaic_virus": {
        "causes": """ 
Caused by Tomato mosaic virus (ToMV), a tobamovirus.
Spread through mechanical transmission (tools), aphids, and infected seed/plant debris.
Thrives in a wide range of temperatures with no specific environmental triggers.
        """,
        "symptoms": """
Mottled yellow and green leaves, creating a mosaic pattern.
Stunted growth and distorted leaves, often called "fern leaf" symptom.
Reduced fruit size and quality, with possible internal browning.
        """,
        "lifecycle": """
Virus overwinters in infected plant debris, weeds, and perennial hosts.
Spread mechanically or via aphids.
        """,
        "prevention": """
Plant resistant varieties like Mountain Spring or Iron Lady.
Sanitize tools and wash hands after handling plants.
Practice crop rotation for 2 or more years away from solanaceous crops.
Control aphids using reflective mulch or insecticidal soap.
Remove infected plants immediately.
        """,
        "chemical_control": """
No chemical controls are effective for the virus. Focus on vector (aphid) control.
Use insecticidal soaps or pyrethrin for aphids.
        """,
        "organic_solutions": """
Reflective mulch to repel aphids.
Neem oil as an aphid deterrent.
Remove weeds, which serve as alternate hosts for aphids and the virus.
        """,
        "fertilizer": """
Use a 10-10-10 NPK fertilizer with added boron for cell wall integrity.
Potassium enhances plant resilience.
Avoid high nitrogen, which attracts aphids.
        """,
        "additional_tips": """
Avoid smoking or handling tobacco near plants, as TMV can spread via hands.
Use floating row covers to exclude aphids.
        """
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "causes": """
Caused by Tomato yellow leaf curl virus (TYLCV), a begomovirus.
Spread by whiteflies (Bemisia tabaci) in a persistent manner.Favorable conditions include 
warm climates (75–90°F/24–32°C), where whitefly populations thrive in dry, warm weather.
        """,
        "symptoms": """
Upward curling and yellowing of leaves.
Stunted growth and reduced leaf size.
Poor fruit set, with fruits often small and deformed.
        """,
        "lifecycle": """
Virus overwinters in whiteflies and alternate hosts like weeds. Transmitted during
 feeding, with whiteflies remaining viruliferous for life.
        """,
        "prevention": """
Use resistant varieties (e.g., Tygress, Volgogradskiy).
Control whiteflies with traps, soaps, or imidacloprid.
Apply reflective mulch, remove infected plants, and rotate crops.
        """,
        "chemical_control": """
Neonicotinoids like imidacloprid or thiamethoxam as soil drenches.
Insect growth regulators like pyriproxyfen.
Biological control using Beauveria bassiana, a fungal pathogen of whiteflies.
        """,
        "organic_solutions": """
Neem oil or kaolin clay to deter whiteflies.
Introduce natural predators like ladybugs and lacewings.
Interplant with basil, which repels whiteflies.
        """,
        "fertilizer": """
Use a 20-10-10 NPK fertilizer with added micronutrients like zn,fe,mg to support plant vigor.
Silica strengthens plant cell walls against whiteflies.
Avoid excessive nitrogen, which attracts whiteflies.
        """,
        "additional_tips": """
Monitor whitefly populations with yellow sticky traps.
Introduce parasitic wasps like Encarsia formosa for biological control.
        """
    },
    "Tomato_Bacterial_spot": {
        "causes": """
Caused by Xanthomonas spp. (X. perforans, X. vesicatoria, X. euvesicatoria).
Spread by water splash, contaminated tools, seed, and plant debris.
Favorable conditions include warm (80–85°F/27–29°C), humid weather with frequent rain.
        """,
        "symptoms": """
Small, water-soaked spots on leaves that turn brown or black with yellow halos.
Fruit lesions are raised, scabby, and may crack.
Defoliation can occur in severe cases.
""",
        "lifecycle": """
Bacteria overwinter in seed, plant debris, and weeds. Secondary spread occurs
 via water splash and mechanical transmission.
        """,
        "prevention": """
Plant resistant varieties like Mountain Spring or Amelia.
Treat seeds with hot water or chlorine soak.
Practice crop rotation for 2 or more years away from solanaceous crops.
Apply copper-based bactericides like Kocide or Champ.
Avoid overhead irrigation and use drip irrigation instead.
Sanitize tools and stakes with a 10% bleach solution.
        """,
        "chemical_control": """
Copper compounds like copper hydroxide or copper oxychloride.
Bactericides like streptomycin, where permitted by local regulations.
Actigard (acibenzolar-S-methyl) for induced resistance.
        """,
        "organic_solutions": """
Neem oil or baking soda sprays.
Compost tea to boost plant immunity.
Remove infected debris promptly.
        """,
        "fertilizer": """
Use a 10-10-20 NPK fertilizer with added potassium and calcium to strengthen cell walls.
Avoid high nitrogen, which promotes succulent, susceptible growth.
Micronutrients like zinc and manganese support enzyme function.
        """,
        "additional_tips": """
Prune lower leaves to reduce humidity and splash.
Mulch with straw to prevent soil splash.
        """
    },
    "Tomato_Early_blight": {
        "causes": """
Caused by the fungus Alternaria solani.
Spread by wind, rain, infected seed, and plant debris.
Favorable conditions include warm (75–85°F/24–29°C), humid weather with wet/dry periods.
        """,
        "symptoms": """
Brown, concentric ringed spots on lower or older leaves.
Stem lesions are dark, sunken, and may girdle stems.
Fruit lesions are leathery, with concentric rings.
        """,
        "lifecycle": """
Fungus overwinters in infected plant debris and seed. Spores are wind-dispersed
and infect through wounds or natural openings.
        """,
        "prevention": """
Plant resistant varieties like Mountain Merit or Defiant.
Treat seeds with hot water or fungicide soaks.
Practice crop rotation for 2 or more years away from solanaceous crops.
Apply fungicides like chlorothalonil, mancozeb, or copper-based sprays.
Prune for airflow and avoid overhead irrigation.
        """,
        "chemical_control": """
Protectant fungicides like chlorothalonil (Bravo) or mancozeb.
Systemic fungicides like azoxystrobin (Quadris) or pyraclostrobin.
        """,
        "organic_solutions": """
Baking soda spray made with 1 tablespoon baking soda, 1 teaspoon oil, and 1 liter of water.
Neem oil or compost tea.
Remove infected debris promptly.
        """,
        "fertilizer": """
Use a 5-15-15 NPK fertilizer with added phosphorus for root development.
Potassium enhances disease resistance.
Calcium reduces fruit susceptibility to infection.
        """,
        "additional_tips": """
Mulch with black plastic to reduce soil splash.
Scout fields weekly for early symptoms.
        """
    },
    "Tomato_healthy": {
        "causes": "No disease present. Plant is in optimal health.",
 "symptoms": "Dark green leaves, strong stems, and uniform fruit development.",
        "prevention": """
Use drip irrigation to avoid wetting foliage.
Maintain balanced fertilization with micronutrients.
Monitor for pests like aphids, whiteflies, and hornworms.
Remove suckers and lower leaves for airflow.
        """,
        "fertilizer": """
Use a 15-15-15 NPK fertilizer for balanced growth.
Micronutrients like magnesium for photosynthesis and boron for fruit set are beneficial.
Organic amendments such as compost or fish emulsion improve soil health.
""",
        "additional_tips": """
Rotate crops annually to prevent soil-borne diseases.
Use row covers to protect from early-season pests.
        """
    },
    "Tomato_Late_blight": {
        "causes": """
        Caused by the oomycete Phytophthora infestans.
        Spread by sporangia via wind and water, and infected seed tubers and plant debris.
        Favorable conditions include cool (15–21°C), wet weather with more humidity (>90%).
        """,
        "symptoms": """
Water-soaked, greasy green-black spots on leaves.
White fungal growth on the undersides of leaves in humid conditions.
Stem lesions are dark and irregular, and tubers develop reddish-brown rot.
        """,
        "lifecycle": """
Overwinters in infected tubers and plant debris. Sporangia germinate
 in free water, releasing zoospores that infect foliage.
        """,
        "prevention": """
Plant resistant varieties like Mountain Magic or Defiant PhR.
Treat seed tubers with hot water or fungicide dips.
Practice crop rotation for 3 or more years and avoid planting near potatoes.
Apply fungicides like chlorothalonil or mefenoxam.
Space plants for airflow and avoid overhead irrigation.
        """,
        "chemical_control": """
Protectant fungicides like chlorothalonil (Bravo) or mancozeb.
Systemic fungicides like mefenoxam (Ridomil) or dimethomorph (Acrobat).
Biological control using Bacillus subtilis (Serenade).
        """,
        "organic_solutions": """
Copper sprays like Bordeaux mixture.
Garlic or horseradish spray for natural fungicidal properties.
Remove volunteers and cull piles to eliminate inoculum.
        """,
        "fertilizer": """
Use a 10-20-20 NPK fertilizer with added potassium and calcium for plant health.
Avoid excessive nitrogen, which promotes susceptible growth.
Silica strengthens cell walls against infection.
        """,
        "additional_tips": """
Scout fields daily during wet periods.
Apply fungicides preventatively before symptoms appear.
Destroy cull piles and volunteer plants.
        """
    },
    "Tomato_Leaf_Mold": {
        "causes": """
Caused by the fungus Cladosporium fulvum (syn. Fulvia fulva).
Spread by wind, splashing water, and infected plant debris.
Favorable conditions include high humidity (>85%), moderate temp (20–25°C), poor airflow.
        """,
        "symptoms": """
Yellow spots on upper leaf surfaces.
Olive-green to brown velvety growth (fungal sporulation) on lower leaf surfaces.
Leaf drop and reduced fruit quality in severe cases.
        """,
        "lifecycle": """
Fungus overwinters in plant debris and greenhouse structures.
 Spores are wind-dispersed and infect through stomata.
        """,
        "prevention": """
Plant resistant varieties like Capitano or Lemance.
Sanitize greenhouse surfaces and remove debris.
Maintain low humidity (<80%) with fans or vents.
Apply fungicides like chlorothalonil, mancozeb, or sulfur-based sprays.
Prune for airflow and avoid overhead irrigation.
        """,
        "chemical_control": """
Protectant fungicides like chlorothalonil or mancozeb.
Systemic fungicides like azoxystrobin or difenoconazole.
        """,
        "organic_solutions": """
Sulfur sprays for organic production.
Neem oil for its fungistatic properties.
Baking soda spray made with 1 tablespoon baking soda, 1 teaspoon oil, and 1 liter of water.
        """,
        "fertilizer": """
Use a 10-10-10 NPK fertilizer with added calcium for cell wall strength.
Magnesium prevents leaf yellowing.
Avoid high nitrogen, which promotes dense foliage and humidity.
        """,
        "additional_tips": """
Use drip irrigation to reduce leaf wetness.
Monitor humidity with a hygrometer.
        """
    },
    "Tomato_Septoria_leaf_spot": {
        "causes": """
Caused by the fungus Septoria lycopersici.
Spread by wind, rain, infected plant debris, and contaminated tools.
Favorable conditions include warm (24–29°C), wet weather with prolonged leaf wetness.
        """,
        "symptoms": """
Small, circular spots with gray centers and dark borders on lower leaves.
Spots may coalesce, causing leaf yellowing and drop.
Severe defoliation exposes fruit to sunscald.
        """,
        "lifecycle": """
Fungus overwinters in infected plant debris. Spores are splash-dispersed
 and infect through stomata or wounds.
        """,
        "prevention": """
Plant resistant varieties like Mountain Supreme or Legend.
Practice crop rotation for 2 or more years away from tomatoes.
Apply fungicides like chlorothalonil, mancozeb, or copper-based sprays.
Prune for airflow and avoid overhead irrigation.
Remove infected debris and disinfect stakes and cages.
        """,
        "chemical_control": """
Protectant fungicides like chlorothalonil (Bravo) or mancozeb.
Systemic fungicides like azoxystrobin or pyraclostrobin.
        """,
        "organic_solutions": """
Copper sprays like Bordeaux mixture.
Neem oil or baking soda sprays.
Compost tea to suppress fungal growth.
        """,
        "fertilizer": """
        Use a 12-12-17 NPK fertilizer with added sulfur to enhance disease resistance.
Potassium promotes plant vigor.
Calcium strengthens cell walls.
        """,
        "additional_tips": """
Mulch with straw to reduce soil splash.
Space plants 2–3 feet apart for airflow.
        """
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "causes": """
Caused by Tetranychus urticae (two-spotted spider mite).
Spread by wind, infested plants, and mechanical transport.
Favorable conditions include hot, dry weather (80–90°F/27–32°C) and low humidity.
        """,
        "symptoms": """
Tiny yellow or white stipples on leaves from feeding damage.
Fine webbing on the undersides of leaves in severe infestations.
Leaf bronzing, curling, and drop, leading to reduced plant vigor.
        """,
        "lifecycle": """
Mites overwinter as adults in plant debris or weeds. Females lay eggs on leaf undersides,
 and the lifecycle completes in 5–20 days depending on temperature.
        """,
        "prevention": """
Plant resistant varieties like Plum Regal or Solar Fire.
Monitor with yellow sticky traps and regular leaf checks, especially on the undersides.
Avoid drought stress and use overhead irrigation to disrupt mites.
Introduce biological control agents like Phytoseiulus persimilis (predatory mites).
Apply miticides like abamectin or bifenthrin, or insecticidal soap.
        """,
        "chemical_control": """
Miticides like abamectin (Avid) or bifenthrin (Talstar).
Insecticidal soaps or oils to suffocate mites, such as Safer Soap or horticultural oil.
        """,
        "organic_solutions": """
Neem oil or rosemary oil for their miticidal properties.
Diatomaceous earth for ground-dwelling mites.
Introduce natural predators like ladybugs and lacewings.
        """,
        "fertilizer": """
Use a 20-5-10 NPK fertilizer with added micronutrients like zn,fe,mg to support -
- plant resilience. Silica strengthens plant cell walls against mites.
Avoid excessive nitrogen, which attracts mites.
        """,
        "additional_tips": """
Introduce predatory mites like Phytoseiulus persimilis early in the season.
Avoid broad-spectrum pesticides that kill natural enemies.
Use reflective mulch to repel mites.
        """
    }
}