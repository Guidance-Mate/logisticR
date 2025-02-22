import re
import requests
import csv
import joblib
import random
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
import pandas as pd
# Initialize FastAPI app
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and MultiLabelBinarizer
model = joblib.load("mental_health_tool_recommendation_logreg_model.pkl")
mlb = joblib.load("mental_health_tool_recommendation_mlb.pkl")

# URLs for assessment data
PHQ9_URL = "https://docs.google.com/spreadsheets/d/1fQ8lRGPvNg3gcM0JkAMrdixGsj0CrK6T31K149D-jMI/export?format=csv"
ASQ_URL = "https://docs.google.com/spreadsheets/d/1TiU8sv5cJg30ZL3fqPSmBwJJbB7h2xv1NNbKo4ZIydU/export?format=csv"
BAI_URL = "https://docs.google.com/spreadsheets/d/1f7kaFuhCv6S_eX4EuIrlhZFDR7W5MhQpJSXHznlpJEk/export?format=csv"

# Load phrases from JSON files
def load_phrases(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


phrases_phq9 = load_phrases("phrases_phq9.json")
phrases_bai = load_phrases("phrases_bai.json")
phrases_asq = load_phrases("phrases_asq.json")

# Response mappings for PHQ9 and BAI
response_mapping_phq9 = {
    "Not at all": 0,
    "Several Days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}

response_mapping_bai = {
    "Not at all": 0,
    "Mildly, but it didn't bother me much": 1,
    "Moderately - it wasn't pleasant at times": 2,
    "Severely - it bothered me a lot": 3,
}

# Mapping for categorical variables for the ML model
phq9_mapping = {
    "Minimal or None (0-4)": 0,
    "Mild Depression (5-9)": 1,
    "Moderate Depression (10-14)": 2,
    "Moderately Severe Depression (15-19)": 3,
    "Severe Depression (20-27)": 4
}

bai_mapping = {
    "Low Anxiety (0-21)": 0,
    "Moderate Anxiety (22-35)": 1,
    "Severe Anxiety (36+)": 2
}

suicide_risk_mapping = {
    "Non-Acute Positive Screen": 0,
    "Acute Positive Screen": 1
}

# Function to map PHQ9 total score to a category
def get_phq9_category(score):
    if score <= 4:
        return "Minimal or None (0-4)"
    elif score <= 9:
        return "Mild Depression (5-9)"
    elif score <= 14:
        return "Moderate Depression (10-14)"
    elif score <= 19:
        return "Moderately Severe Depression (15-19)"
    else:
        return "Severe Depression (20-27)"

# Function to map BAI total score to a category
def get_bai_category(score):
    if score <= 21:
        return "Low Anxiety (0-21)"
    elif score <= 35:
        return "Moderate Anxiety (22-35)"
    else:
        return "Severe Anxiety (36+)"

@app.get("/analyze")
def analyze_assessments(first_name: str, last_name: str, middle_name: str = "", suffix: str = ""):
    input_name = f"{first_name} {middle_name} {last_name} {suffix}".strip()
    input_name_clean = re.sub(r'\s+', ' ', input_name).lower()

    try:
        results = {}

        # --- PHQ9 Processing ---
        response = requests.get(PHQ9_URL)
        response.raise_for_status()
        data = response.text.splitlines()
        reader = csv.reader(data)
        header = next(reader)  # Read the header row

        # ✅ Find the correct column indexes dynamically
        first_name_index = header.index("First name") if "First name" in header else -1
        middle_name_index = header.index("Middle name") if "Middle name" in header else -1
        last_name_index = header.index("Last name") if "Last name" in header else -1
        suffix_index = header.index(
            "Suffix  (e.g., Jr., Sr., III)") if "Suffix  (e.g., Jr., Sr., III)" in header else -1

        # ✅ Find the first PHQ9 response column
        response_start_index = header.index("Little interest or pleasure in doing things")

        found_phq9 = False

        for row in reader:
            if not row:
                continue

            # ✅ Ensure indexes exist before accessing
            first_name_row = row[first_name_index].strip() if first_name_index != -1 and len(row) > first_name_index and \
                                                              row[first_name_index] else ''
            middle_name_row = row[middle_name_index].strip() if middle_name_index != -1 and len(
                row) > middle_name_index and row[middle_name_index] else ''
            last_name_row = row[last_name_index].strip() if last_name_index != -1 and len(row) > last_name_index and \
                                                            row[last_name_index] else ''
            suffix_row = row[suffix_index].strip() if suffix_index != -1 and len(row) > suffix_index and row[
                suffix_index] else ''

            print(
                f"Checking PHQ9 row: First Name = '{first_name_row}', Middle Name = '{middle_name_row}', Last Name = '{last_name_row}', Suffix = '{suffix_row}'")

            # ✅ Normalize names for comparison
            full_name = f"{first_name_row} {middle_name_row} {last_name_row} {suffix_row}".strip().lower()
            alt_name = f"{first_name_row} {last_name_row}".strip().lower()

            # ✅ Debugging: Check name matching
            print(f"Comparing input: '{input_name_clean}' with row names: '{full_name}', '{alt_name}'")

            if input_name_clean == full_name or input_name_clean == alt_name:
                # ✅ Extract PHQ-9 responses from the correct column range
                response_end_index = response_start_index + 9  # Ensure 9 PHQ9 responses
                responses = row[response_start_index:response_end_index]

                # ✅ Debugging: Print extracted PHQ9 responses
                print(f"Extracted PHQ9 responses for {input_name}: {responses}")

                # ✅ Compute total PHQ9 score
                total_score = sum(
                    response_mapping_phq9.get(r.strip(), 0) for r in responses if r.strip() in response_mapping_phq9
                )

                # ✅ Debugging: Print final computed score
                print(f"Final Computed PHQ9 Score for {input_name}: {total_score}")

                primary_impression = get_phq9_category(total_score)

                results["phq9"] = {
                    "client_name": input_name.title(),
                    "total_score": total_score,
                    "Interpretation": primary_impression,
                    "primary_impression": random.choice(phrases_phq9["Depression"]),
                    "additional_impressions": [
                        random.choice(phrases_phq9["Physical Symptoms"]),
                        random.choice(phrases_phq9["Well-Being"])
                    ]
                }
                found_phq9 = True
                break

        if not found_phq9:
            raise HTTPException(status_code=404, detail="Client not found in PHQ9 data")

        # --- ASQ Processing ---
        response = requests.get(ASQ_URL)
        response.raise_for_status()
        data = response.text.splitlines()
        reader = csv.reader(data)
        next(reader)

        found_asq = False
        for row in reader:
            if not row:
                continue

            first_name_row = row[8].strip()
            middle_name_row = row[9].strip()
            last_name_row = row[10].strip()

            full_name = f"{first_name_row} {middle_name_row} {last_name_row}".strip().lower()
            alt_name = f"{first_name_row} {last_name_row}".strip().lower()

            selected_options = row[2].strip() if len(row) > 2 else ""
            have_you_ever_tried = row[3].strip() if len(row) > 3 else "N/A"
            how_and_when = row[4].strip() if len(row) > 4 else "N/A"
            are_you_having_thoughts = row[6].strip() if len(row) > 6 else "N/A"
            please_describe = row[7].strip() if len(row) > 7 else "N/A"

            if input_name_clean == full_name or input_name_clean == alt_name:
                ask_suicide_risk = "Acute Positive Screen" if "Yes" in row[5].strip() else "Non-Acute Positive Screen"

                results["asq"] = {
                    "Interpretation": ask_suicide_risk,
                    "primary_impression": random.choice(phrases_asq[ask_suicide_risk]),
                    "additional_impressions": [random.choice(phrases_asq[ask_suicide_risk]) for _ in range(2)],
                    "selected_options": selected_options,
                    "have_you_ever_tried to kill yourself?": have_you_ever_tried,
                    "how_and_when": how_and_when,
                    "Are you having thoughts of killing yourself right now?": are_you_having_thoughts,
                    "please_describe": please_describe
                }

                found_asq = True
                break

        if not found_asq:
            raise HTTPException(status_code=404, detail="Client not found in ASQ data")

        # --- BAI Processing ---
        response = requests.get(BAI_URL)
        response.raise_for_status()
        data = response.text.splitlines()
        reader = csv.reader(data)
        next(reader)

        found_bai = False
        for row in reader:
            if not row:
                continue

            first_name_row = row[-4].strip()
            middle_name_row = row[-3].strip()
            last_name_row = row[-2].strip()

            full_name = f"{first_name_row} {middle_name_row} {last_name_row}".strip().lower()
            alt_name = f"{first_name_row} {last_name_row}".strip().lower()

            if input_name_clean == full_name or input_name_clean == alt_name:
                responses = row[1:-4]
                total_score = sum(response_mapping_bai.get(r.strip(), 0) for r in responses)  # ✅ FIXED SYNTAX ERROR
                primary_impression_bai = get_bai_category(total_score)

                results["bai"] = {
                    "client_name": input_name.title(),
                    "total_score": total_score,
                    "Interpretation": primary_impression_bai,
                    "primary_impression": random.choice(phrases_bai["Anxiety"]),
                    "additional_impressions": [
                        random.choice(phrases_bai["Trauma & PTSD"]),
                        random.choice(phrases_bai["Youth Mental Health Test"])
                    ]
                }
                found_bai = True
                break

        if not found_bai:
            raise HTTPException(status_code=404, detail="Client not found in BAI data")

        # --- Predict Recommended Tools ---

        feature_vector = pd.DataFrame([[
            results["phq9"]["total_score"], phq9_mapping[results["phq9"]["Interpretation"]],
            results["bai"]["total_score"], bai_mapping[results["bai"]["Interpretation"]],
            suicide_risk_mapping[results["asq"]["Interpretation"]]
        ]], columns=["Total_PHQ9_Score", "Primary_Impression_PHQ9",
                     "Total_BAI_Score", "Primary_Impression_BAI", "Ask_Suicide_Risk"])

        predicted_labels = model.predict(feature_vector)
        recommended_tools = mlb.inverse_transform(predicted_labels)[0]

        results["recommended_tools"] = recommended_tools

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {e}")

handler = Mangum(app)
