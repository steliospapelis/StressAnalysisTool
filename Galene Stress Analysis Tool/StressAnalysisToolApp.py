import streamlit as st
import os
import shutil
import json
from segmentAnalysis import analyze_participant
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from annotationAnalysis import analyze_annotation
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


# App config
st.set_page_config(page_title="Galene's Stress Analysis Tool", page_icon="🧠", layout="centered")

if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_participant" not in st.session_state:
    st.session_state.selected_participant = None

# Data Folder to save each participant's logs
DATA_FOLDER = "data"

# Change session(page)
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_add_participant():
    st.session_state.page = "add"

def go_to_home():
    st.session_state.page = "home"

def go_to_graph_page(participant_id):
    st.session_state.selected_participant = participant_id
    st.session_state.page = "graph"


# Display λίστα με participants στο home
def list_participants():
    """List participant IDs (folders with 5-letter names)."""
    if not os.path.exists(DATA_FOLDER):
        return []
    return sorted([name for name in os.listdir(DATA_FOLDER)
                   if os.path.isdir(os.path.join(DATA_FOLDER, name)) and len(name) == 5])

# Σελίδα για να προσθέσεις participant δίνοντας το ID του και τα 3 αρχεία
def add_participant_page():
    st.title("➕ Add Participant")

    
    participant_id = st.text_input("Participant ID (exactly 5 letters)", max_chars=5).upper()

    tutorial_log = st.file_uploader("Upload Tutorial Log (.csv)", type=["csv"])
    game_log = st.file_uploader("Upload Game Log (.csv)", type=["csv"])
    stress_annotation = st.file_uploader("Upload Stress Annotation (.txt)", type=["txt"])

    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit"):
            errors = []

            #Έλεγχοι για σωστό submission

            # Validate ID
            if len(participant_id) != 5 or not participant_id.isalpha():
                errors.append("Participant ID must be exactly 5 letters (A–Z).")

            # Check if ID already exists
            participant_folder = os.path.join(DATA_FOLDER, participant_id)
            if os.path.exists(participant_folder):
                errors.append(f"Participant '{participant_id}' already exists.")

            # Validate files
            if not tutorial_log:
                errors.append("Tutorial log not uploaded.")
            if not game_log:
                errors.append("Game log not uploaded.")
            if not stress_annotation:
                errors.append("Stress annotation not uploaded.")

            if errors:
                st.error("Please fix the following:\n" + "\n".join(errors))
            else:
                # Φτιάχνουμε folder για τον νέο participant
                os.makedirs(participant_folder, exist_ok=True)

                # Αποθηκεύουμε εκεί τα αρχεία του
                with open(os.path.join(participant_folder, "tutorial_log.csv"), "wb") as f:
                    shutil.copyfileobj(tutorial_log, f)

                with open(os.path.join(participant_folder, "game_log.csv"), "wb") as f:
                    shutil.copyfileobj(game_log, f)

                with open(os.path.join(participant_folder, "stress_annotation.txt"), "wb") as f:
                    shutil.copyfileobj(stress_annotation, f)

                st.success(f"Participant '{participant_id}' added successfully! ✅")

                # Και κάνουμε την πρώτη ανάλυση (segment analysis script)
                analysis_result = analyze_participant(participant_folder)

                with open(os.path.join(participant_folder, "analysis.json"), "w", encoding='utf-8') as f:
                    json.dump(analysis_result, f, indent=4)

                st.success(f"Participant '{participant_id}' added and analyzed successfully! ✅")
                st.session_state.page = "home"
                st.rerun()

    with col2:
        st.button("Back to Home", on_click=go_to_home)



#Σελίδα με χαρακτηριστικά κάθε participant(γράφημα και features) όπως προκύπτουν από το annotation analysis script

def participant_analysis_page(participant_id):
    participant_folder = os.path.join(DATA_FOLDER, participant_id)
    analysis_json_path = os.path.join(participant_folder, "analysis.json")
    if not os.path.exists(analysis_json_path):
        st.error("Analysis data not found for this participant.")
        return

    with open(analysis_json_path, "r", encoding="utf-8") as f:
        analysis_result = json.load(f)

    normalized_trace, calm_ranges, stressed_ranges, breathing_ranges, final_means = analyze_annotation(participant_folder, analysis_result)

    st.title("📊 Annotation Curve")

    # Πλοτάρουμε τη συνεχή, κανονικοποιημένη συνάρτηση του annotation
    times = [t for t, v in normalized_trace]
    values = [v for t, v in normalized_trace]

    plt.figure(figsize=(12,4))
    plt.plot(times, values, label='Normalized annotation')

    # Πλοτάρουμε και οριζόντιες γραμμές κάτω κάτω για να φαίνεται σε τι state ήταν
    for start, end in calm_ranges:
        plt.fill_betweenx([-0.1, -0.05], start, end, color='green', alpha=0.6)
    for start, end in stressed_ranges:
        plt.fill_betweenx([-0.1, -0.05], start, end, color='red', alpha=0.6)
    for start, end in breathing_ranges:
        plt.fill_betweenx([-0.1, -0.05], start, end, color='yellow', alpha=0.6)

    plt.ylim(-0.2, 1.1)
    plt.xlabel("Time (sec)")
    plt.ylabel("Normalized Stress")
    plt.legend()
    st.pyplot(plt)
    st.subheader("Mean features:")
    st.json(final_means)

    if st.button("🔙 Back to home"):
        st.session_state.page = "home"
        st.rerun()


#Αρχική οθόνη με λίστα participants

def home_page():
    st.title("🧠 Galene's Stress Analysis Tool")
    participants = list_participants()
    count = len(participants)
    st.markdown(f"**Total participants:** `{count}`")
    if participants:
        st.markdown("### Participants:")
        
        for participant_id in participants:
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.markdown(f"✅ `{participant_id}`")
            with col2:
                if st.button(f"📊 {participant_id}"):
                    st.session_state.selected_participant = participant_id
                    st.session_state.page = "participant_analysis"
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"delete_{participant_id}"):
                    participant_folder = os.path.join("data", participant_id)
                    if os.path.exists(participant_folder):
                        shutil.rmtree(participant_folder)
                        st.success(f"Deleted participant {participant_id}")
                        st.rerun()
    else:
        st.info("No participants added yet.")
    
    st.button("➕ Add Participant", on_click=go_to_add_participant)

    if st.button("Run Wilcoxon Test on All Participants"):
        test_results = run_wilcoxon_test()
        st.subheader("Wilcoxon Test Results")
        for feature, result in test_results.items():
            st.markdown(f"**{feature.capitalize()}**")
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.write(f"Statistic: `{result['statistic']}`, p-value: `{result['p_value']}`")

    

#Συγκρίνουμε ανά ζεύγη για κάθε participant τις calm και stressed τιμές του ανά ζεύγος
def run_wilcoxon_test():
    calm_values = {'mean': [], 'area': [], 'amplitude': [], 'gradient': []}
    stressed_values = {'mean': [], 'area': [], 'amplitude': [], 'gradient': []}

    participants = list_participants()
    for participant_id in participants:
        features_path = os.path.join(DATA_FOLDER, participant_id, "annotation_features.json")
        if not os.path.exists(features_path):
            continue
        with open(features_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            try:
                for feature in calm_values.keys():
                    calm_values[feature].append(data["calm"][feature])
                    stressed_values[feature].append(data["stressed"][feature])
            except KeyError as e:
                print(f"Missing key for participant {participant_id}: {e}")

    #Wilcoxon test για κάθε feature (4)
    results = {}
    for feature in calm_values.keys():
        if len(calm_values[feature]) < 2:
            results[feature] = {"error": "Not enough data for test"}
            continue
        try:
            print(calm_values)
            print(stressed_values)
            stat, p = wilcoxon(stressed_values[feature], calm_values[feature])
            results[feature] = {
                "statistic": stat,
                "p_value": p
            }
        except Exception as e:
            results[feature] = {"error": str(e)}

    return results


# Main - Ξεκινάμε από το home page 
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "add":
    add_participant_page()
elif st.session_state.page == "participant_analysis":
    participant_analysis_page(st.session_state.selected_participant)
