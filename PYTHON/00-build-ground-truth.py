#!/usr/bin/env python3
"""
Build Ground Truth Dataset for LLM Benchmark
=============================================
Constructs a structured ground-truth reference for 175 GBD diseases using
data from the Global Burden of Disease Study and WHO, focused on sub-Saharan
Africa. This provides the evaluation baseline for scoring LLM responses.

Outputs:
    ../DATA/ground-truth/disease_ground_truth.csv

Author: Manuel Corpas
Date: February 2026
"""

import csv
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HEIM_METRICS = Path(
    "/Users/superintelligent/Library/Mobile Documents/"
    "com~apple~CloudDocs/PUBLICATIONS/07-EHR-LINKED-BIOBANKS/"
    "ANALYSIS/05-03-SEMANTIC-METRICS/disease_metrics.csv"
)
OUTPUT_DIR = Path(__file__).parent.parent / "DATA" / "ground-truth"
OUTPUT_CSV = OUTPUT_DIR / "disease_ground_truth.csv"
OUTPUT_JSON = OUTPUT_DIR / "disease_ground_truth.json"

# ---------------------------------------------------------------------------
# GBD disease category mapping (broad groupings for colour-coding figures)
# ---------------------------------------------------------------------------
DISEASE_CATEGORIES = {
    # Neglected Tropical Diseases
    "African_trypanosomiasis": "NTD",
    "Onchocerciasis": "NTD",
    "Lymphatic_filariasis": "NTD",
    "Guinea_worm_disease": "NTD",
    "Chagas_disease": "NTD",
    "Cystic_echinococcosis": "NTD",
    "Leishmaniasis": "NTD",
    "Schistosomiasis": "NTD",
    "Trachoma": "NTD",
    "Cysticercosis": "NTD",
    "Food-borne_trematodiases": "NTD",
    "Leprosy": "NTD",
    "Rabies": "NTD",
    "Dengue": "NTD",
    "Yellow_fever": "NTD",
    "Other_neglected_tropical_diseases": "NTD",
    # Infectious Diseases
    "Malaria": "Infectious",
    "Tuberculosis": "Infectious",
    "HIV_AIDS": "Infectious",
    "COVID-19": "Infectious",
    "Ebola": "Infectious",
    "Zika_virus": "Infectious",
    "Measles": "Infectious",
    "Pertussis": "Infectious",
    "Diphtheria": "Infectious",
    "Tetanus": "Infectious",
    "Meningitis": "Infectious",
    "Encephalitis": "Infectious",
    "Acute_hepatitis": "Infectious",
    "Lower_respiratory_infections": "Infectious",
    "Upper_respiratory_infections": "Infectious",
    "Diarrheal_diseases": "Infectious",
    "Typhoid_and_paratyphoid": "Infectious",
    "Invasive_Non-typhoidal_Salmonella_(iNTS)": "Infectious",
    "Sexually_transmitted_infections_excluding_HIV": "Infectious",
    "Varicella_and_herpes_zoster": "Infectious",
    "Other_intestinal_infectious_diseases": "Infectious",
    "Other_unspecified_infectious_diseases": "Infectious",
    # NCDs — Cancers
    "Breast_cancer": "NCD-Cancer",
    "Cervical_cancer": "NCD-Cancer",
    "Prostate_cancer": "NCD-Cancer",
    "Colon_and_rectum_cancer": "NCD-Cancer",
    "Stomach_cancer": "NCD-Cancer",
    "Liver_cancer": "NCD-Cancer",
    "Esophageal_cancer": "NCD-Cancer",
    "Pancreatic_cancer": "NCD-Cancer",
    "Ovarian_cancer": "NCD-Cancer",
    "Bladder_cancer": "NCD-Cancer",
    "Kidney_cancer": "NCD-Cancer",
    "Thyroid_cancer": "NCD-Cancer",
    "Non-Hodgkin_lymphoma": "NCD-Cancer",
    "Hodgkin_lymphoma": "NCD-Cancer",
    "Leukemia": "NCD-Cancer",
    "Multiple_myeloma": "NCD-Cancer",
    "Malignant_skin_melanoma": "NCD-Cancer",
    "Non-melanoma_skin_cancer": "NCD-Cancer",
    "Lip_and_oral_cavity_cancer": "NCD-Cancer",
    "Larynx_cancer": "NCD-Cancer",
    "Nasopharynx_cancer": "NCD-Cancer",
    "Other_pharynx_cancer": "NCD-Cancer",
    "Gallbladder_and_biliary_tract_cancer": "NCD-Cancer",
    "Mesothelioma": "NCD-Cancer",
    "Testicular_cancer": "NCD-Cancer",
    "Uterine_cancer": "NCD-Cancer",
    "Eye_cancer": "NCD-Cancer",
    "Brain_and_central_nervous_system_cancer": "NCD-Cancer",
    "Neuroblastoma_and_other_peripheral_nervous_cell_tumors": "NCD-Cancer",
    "Soft_tissue_and_other_extraosseous_sarcomas": "NCD-Cancer",
    "Malignant_neoplasm_of_bone_and_articular_cartilage": "NCD-Cancer",
    "Other_neoplasms": "NCD-Cancer",
    "Other_malignant_neoplasms": "NCD-Cancer",
    "Tracheal": "NCD-Cancer",
    # NCDs — Cardiovascular
    "Ischemic_heart_disease": "NCD-CVD",
    "Stroke": "NCD-CVD",
    "Hypertensive_heart_disease": "NCD-CVD",
    "Atrial_fibrillation_and_flutter": "NCD-CVD",
    "Aortic_aneurysm": "NCD-CVD",
    "Non-rheumatic_valvular_heart_disease": "NCD-CVD",
    "Cardiomyopathy_and_myocarditis": "NCD-CVD",
    "Endocarditis": "NCD-CVD",
    "Rheumatic_heart_disease": "NCD-CVD",
    "Pulmonary_Arterial_Hypertension": "NCD-CVD",
    "Lower_extremity_peripheral_arterial_disease": "NCD-CVD",
    "Other_cardiovascular_and_circulatory_diseases": "NCD-CVD",
    # NCDs — Other
    "Diabetes_mellitus": "NCD-Other",
    "Chronic_kidney_disease": "NCD-Other",
    "Chronic_obstructive_pulmonary_disease": "NCD-Other",
    "Asthma": "NCD-Other",
    "Cirrhosis_and_other_chronic_liver_diseases": "NCD-Other",
    "Alzheimer's_disease_and_other_dementias": "NCD-Other",
    "Parkinson's_disease": "NCD-Other",
    "Idiopathic_epilepsy": "NCD-Other",
    "Multiple_sclerosis": "NCD-Other",
    "Motor_neuron_disease": "NCD-Other",
    "Inflammatory_bowel_disease": "NCD-Other",
    "Pancreatitis": "NCD-Other",
    "Gallbladder_and_biliary_diseases": "NCD-Other",
    "Osteoarthritis": "NCD-Other",
    "Rheumatoid_arthritis": "NCD-Other",
    "Gout": "NCD-Other",
    # Mental Health
    "Depressive_disorders": "Mental",
    "Anxiety_disorders": "Mental",
    "Bipolar_disorder": "Mental",
    "Schizophrenia": "Mental",
    "Eating_disorders": "Mental",
    "Autism_spectrum_disorders": "Mental",
    "Attention-deficit_hyperactivity_disorder": "Mental",
    "Conduct_disorder": "Mental",
    "Idiopathic_developmental_intellectual_disability": "Mental",
    "Drug_use_disorders": "Mental",
    "Alcohol_use_disorders": "Mental",
    "Other_mental_disorders": "Mental",
    # Injuries
    "Road_injuries": "Injury",
    "Falls": "Injury",
    "Drowning": "Injury",
    "Interpersonal_violence": "Injury",
    "Self-harm": "Injury",
    "Conflict_and_terrorism": "Injury",
    "Police_conflict_and_executions": "Injury",
    "Exposure_to_forces_of_nature": "Injury",
    "Other_transport_injuries": "Injury",
    "Foreign_body": "Injury",
    "Poisonings": "Injury",
    "Environmental_heat_and_cold_exposure": "Injury",
    "Exposure_to_mechanical_forces": "Injury",
    "Animal_contact": "Injury",
    "Other_unintentional_injuries": "Injury",
    "Adverse_effects_of_medical_treatment": "Injury",
    # Maternal / Neonatal / Nutritional
    "Maternal_disorders": "MNCH",
    "Neonatal_disorders": "MNCH",
    "Protein-energy_malnutrition": "Nutritional",
    "Dietary_iron_deficiency": "Nutritional",
    "Iodine_deficiency": "Nutritional",
    "Vitamin_A_deficiency": "Nutritional",
    "Other_nutritional_deficiencies": "Nutritional",
}

# Fallback category for diseases not explicitly mapped
DEFAULT_CATEGORY = "Other"


def load_heim_diseases():
    """Load all 175 diseases and metrics from HEIM disease_metrics.csv."""
    diseases = []
    with open(HEIM_METRICS, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            disease = row["disease"]
            diseases.append({
                "disease": disease,
                "disease_display": disease.replace("_", " "),
                "n_papers": int(row["n_papers"]),
                "sii": float(row["sii"]),
                "ktp": float(row["ktp"]),
                "rcc": float(row["rcc"]),
                "mean_drift": float(row["mean_drift"]),
                "category": DISEASE_CATEGORIES.get(disease, DEFAULT_CATEGORY),
            })
    return diseases


def build_ground_truth(diseases):
    """
    Build ground-truth reference data for each disease.

    NOTE: In a full implementation, this would pull from the GBD Results Tool
    API (https://vizhub.healthdata.org/gbd-results/) and WHO Global Health
    Observatory. For now, we create the structure and flag that ground-truth
    data should be populated from these sources before running the evaluation.
    """
    ground_truth = []
    for d in diseases:
        gt = {
            "disease": d["disease"],
            "disease_display": d["disease_display"],
            "category": d["category"],
            "n_papers": d["n_papers"],
            "sii": d["sii"],
            "ktp": d["ktp"],
            "rcc": d["rcc"],
            # Ground-truth fields to be populated from GBD/WHO
            "ssa_dalys_2021": None,          # DALYs in sub-Saharan Africa (GBD 2021)
            "ssa_deaths_2021": None,          # Deaths in SSA (GBD 2021)
            "ssa_prevalence_2021": None,      # Prevalence in SSA (GBD 2021)
            "ssa_incidence_2021": None,       # Incidence in SSA (GBD 2021)
            "global_rank_dalys": None,        # Global rank by DALYs
            "ssa_rank_dalys": None,           # SSA rank by DALYs
            "who_priority": None,             # WHO priority disease? (bool)
            "ntd_classification": d["category"] == "NTD",
            "key_risk_factors_ssa": None,     # Top risk factors in SSA (list)
            "diagnostic_availability_ssa": None,  # POC diagnostics available? (str)
            "treatment_access_ssa": None,     # Treatment access level (str)
            "prevention_interventions": None, # Key prevention strategies (list)
            "key_african_studies": None,      # Notable African research (list)
            "policy_frameworks": None,        # Relevant AU/WHO policies (list)
            "data_source_notes": "Populate from GBD 2021 Results Tool and WHO GHO",
        }
        ground_truth.append(gt)
    return ground_truth


def main():
    print("Loading HEIM disease metrics...")
    diseases = load_heim_diseases()
    print(f"  Loaded {len(diseases)} diseases")

    # Category summary
    from collections import Counter
    cats = Counter(d["category"] for d in diseases)
    print("\n  Category distribution:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    print("\nBuilding ground-truth structure...")
    ground_truth = build_ground_truth(diseases)

    # Write CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ground_truth[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ground_truth)
    print(f"  Saved CSV: {OUTPUT_CSV}")

    # Write JSON (more flexible for downstream scripts)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, default=str)
    print(f"  Saved JSON: {OUTPUT_JSON}")

    # Summary statistics
    sii_values = [d["sii"] for d in diseases]
    print(f"\n  SII range: {min(sii_values):.6f} — {max(sii_values):.6f}")
    print(f"  SII mean:  {sum(sii_values)/len(sii_values):.6f}")

    ntd_sii = [d["sii"] for d in diseases if d["category"] == "NTD"]
    non_ntd_sii = [d["sii"] for d in diseases if d["category"] != "NTD"]
    print(f"  NTD mean SII:     {sum(ntd_sii)/len(ntd_sii):.6f} (n={len(ntd_sii)})")
    print(f"  Non-NTD mean SII: {sum(non_ntd_sii)/len(non_ntd_sii):.6f} (n={len(non_ntd_sii)})")

    print("\nNOTE: Ground-truth epidemiological fields (DALYs, deaths, prevalence)")
    print("      are placeholder NULLs. Populate from GBD Results Tool before")
    print("      running 03-evaluate-responses.py.")


if __name__ == "__main__":
    main()
