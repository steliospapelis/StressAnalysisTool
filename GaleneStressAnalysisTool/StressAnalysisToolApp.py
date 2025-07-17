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
from scipy.stats import wilcoxon, shapiro, ttest_rel


# App config
st.set_page_config(page_title="Galene's Stress Analysis Tool", page_icon="🧠", layout="centered")

if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_participant" not in st.session_state:
    st.session_state.selected_participant = None

# Data Folder to save each participant's logs
DATA_FOLDER = "data"

# Folder for statistical test results
STAT_RESULTS_FOLDER = "stat_results"


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
                    
                # Κάνουμε και την ανάλυση του annotation
                analyze_annotation(participant_folder, analysis_result)

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

    if st.button("Run Statistical Tests (Wilcoxon + t-test)"):
        test_results = run_statistical_tests()

        st.subheader("📊 Shapiro-Wilk Normality Test Results")
        shapiro_table = []
        normal_features = []
        for feature, result in test_results["shapiro"].items():
            if "error" in result:
                st.error(f"{feature.capitalize()}: {result['error']}")
                continue
            shapiro_p = result["shapiro_p_value"]
            shapiro_table.append([feature.capitalize(), f"{shapiro_p:.4f}"])
            if shapiro_p > 0.05:
                normal_features.append(feature.capitalize())

        if shapiro_table:
            st.table(pd.DataFrame(
                shapiro_table, columns=["Feature", "Shapiro-Wilk p-value"]
            ))

        if normal_features:
            st.success(f"✅ Features showing normality (p > 0.05) where t-test applies: {', '.join(normal_features)}.")
        else:
            st.warning("❗ No features show normality. Wilcoxon is more appropriate.")

        st.subheader("📊 Wilcoxon Signed-Rank Test Results")
        wilcoxon_table = []
        for feature, result in test_results["wilcoxon"].items():
            if "error" in result:
                st.error(f"{feature.capitalize()}: {result['error']}")
                continue
            wilcoxon_table.append([
                feature.capitalize(),
                f"{result['wilcoxon_statistic']:.4f}",
                f"{result['wilcoxon_p_value']:.4f}"
            ])
        if wilcoxon_table:
            st.table(pd.DataFrame(
                wilcoxon_table, columns=["Feature", "Wilcoxon Statistic", "Wilcoxon p-value"]
            ))

        st.subheader("📊 Paired t-test Results")
        t_test_table = []
        for feature, result in test_results["t_test"].items():
            if "error" in result:
                st.error(f"{feature.capitalize()}: {result['error']}")
                continue
            t_test_table.append([
                feature.capitalize(),
                f"{result['t_statistic']:.4f}",
                f"{result['t_p_value']:.4f}"
            ])
        if t_test_table:
            st.table(pd.DataFrame(
                t_test_table, columns=["Feature", "t-statistic", "t-test p-value"]
            ))

        st.success(f"Results saved to `{STAT_RESULTS_FOLDER}/results.json`")

    if st.button("🧮 Perform Full Classification Evaluation (all thresholds)"):
        # Run evaluations
        avg_agg_metrics = run_classification_evaluation()
        avg_pointwise_metrics = run_pointwise_classification_evaluation()
        avg_memory_metrics_short = run_memory_classification_evaluation(memory_type='short')
        avg_memory_metrics_long = run_memory_classification_evaluation(memory_type='long')

        # Show Aggregate metrics
        if avg_agg_metrics is None:
            st.warning("⚠️ No aggregate metrics found.")
        else:
            print_metrics_table("📊 Average Segment Classification Metrics Across Participants (by threshold)", avg_agg_metrics)

        if avg_pointwise_metrics is None:
            st.warning("⚠️ No pointwise metrics found.")
        else:
            print_metrics_table("📊 Average pointwise Classification Metrics Across Participants (by threshold)", avg_agg_metrics)

        # Show Pointwise metrics
        if avg_memory_metrics_short:
            st.subheader("🧠 Average Short Memory Metrics Across Participants (by threshold)")
            print_metrics_table("Short Memory Metrics", avg_memory_metrics_short)

        if avg_memory_metrics_long:
            st.subheader("🧠 Average Long Memory Metrics Across Participants (by threshold)")
            print_metrics_table("Long Memory Metrics", avg_memory_metrics_long)

        experiments = {
            "Aggregate": avg_agg_metrics,
            "Pointwise": avg_pointwise_metrics,
            "Short Memory": avg_memory_metrics_short,
            "Long Memory": avg_memory_metrics_long
        }

        metric_names = ["accuracy", "precision", "recall", "f1", "undefined_percentage"]

        st.subheader("📊 Classification Metrics by Experiment and Threshold")
        for experiment_name, metrics_dict in experiments.items():
            if metrics_dict is None:
                st.warning(f"⚠️ No metrics found for {experiment_name}.")
                continue

            st.markdown(f"### 🔬 {experiment_name} Experiment")

            # metrics_dict is expected to be {threshold: {metric: value, ...}, ...}
            thresholds = sorted(metrics_dict.keys())  # make sure thresholds are sorted
            for threshold in thresholds:
                threshold_metrics = metrics_dict[threshold]

                # Create lists to hold values for each metric
                values = [threshold_metrics.get(metric, 0)*100 for metric in metric_names]  # *100 for percentage

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=metric_names,
                    y=values,
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],  # different colors
                    text=[f"{v:.2f}%" for v in values],
                    textposition='auto'
                ))

                fig.update_layout(
                    title=f"Threshold: {threshold}",
                    yaxis_title="Percentage",
                    xaxis_title="Metric",
                    yaxis=dict(range=[0, 100]),
                    bargap=0.4
                )
                st.plotly_chart(fig, use_container_width=True)





# Εκτελούμε στατιστική ανάλυση (Wilcoxon, Shapiro-Wilk, t-test) για όλα τα χαρακτηριστικά (mean, area, amplitude, gradient)
# Συλλέγουμε για κάθε συμμετέχοντα τις μέσες τιμές στα calm και stressed διαστήματα (μία τιμή για κάθε κατάσταση και χαρακτηριστικό).
def run_statistical_tests():
    calm_values = {'mean': [],'median': [], 'max': [], 'area': [], 'amplitude': [], 'gradient': []}
    stressed_values = {'mean': [],'median': [], 'max': [], 'area': [], 'amplitude': [], 'gradient': []}

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

    wilcoxon_results = {}
    shapiro_results = {}
    t_test_results = {}

    for feature in calm_values.keys():
        if len(calm_values[feature]) < 2:
            wilcoxon_results[feature] = {"error": "Not enough data for test"}
            shapiro_results[feature] = {"error": "Not enough data for test"}
            t_test_results[feature] = {"error": "Not enough data for test"}
            continue

        try:
            diffs = [stressed_values[feature][i] - calm_values[feature][i] for i in range(len(calm_values[feature]))]
            shapiro_stat, shapiro_p = shapiro(diffs)
            shapiro_results[feature] = {"shapiro_p_value": shapiro_p}

            w_stat, w_p = wilcoxon(stressed_values[feature], calm_values[feature])
            wilcoxon_results[feature] = {
                "wilcoxon_statistic": w_stat,
                "wilcoxon_p_value": w_p
            }

            t_stat, t_p = ttest_rel(stressed_values[feature], calm_values[feature])
            t_test_results[feature] = {
                "t_statistic": t_stat,
                "t_p_value": t_p
            }

        except Exception as e:
            wilcoxon_results[feature] = {"error": str(e)}
            shapiro_results[feature] = {"error": str(e)}
            t_test_results[feature] = {"error": str(e)}

    # Save results to folder
    os.makedirs(STAT_RESULTS_FOLDER, exist_ok=True)
    results_dict = {
        "participants_used": len(participants),
        "shapiro": shapiro_results,
        "wilcoxon": wilcoxon_results,
        "t_test": t_test_results
    }
    with open(os.path.join(STAT_RESULTS_FOLDER, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)

    return results_dict

#Συγκρίνουμε την τιμή που έχει δώσει ο αισθητήρας σε αυτό το segment( calm ή stressed ) με το αν έχει μέση τιμή μεγαλύτερη από τη συνολική μέση τιμή (τότε label stressed ενώ αν έχει μικρότερη calm)
def run_classification_evaluation():
    metrics_per_threshold = {}

    participants = list_participants()
    for pid in participants:
        feat_path = os.path.join(DATA_FOLDER, pid, "annotation_features.json")
        if not os.path.exists(feat_path):
            continue

        with open(feat_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            agg_metrics = data.get("segment_metrics", {})
            for threshold, metrics in agg_metrics.items():
                if threshold not in metrics_per_threshold:
                    metrics_per_threshold[threshold] = []
                metrics_per_threshold[threshold].append(metrics)

    if not metrics_per_threshold:
        return None

    # Compute mean per metric for each threshold
    avg_metrics_per_threshold = {}
    for threshold, metrics_list in metrics_per_threshold.items():
        df = pd.DataFrame(metrics_list)
        avg_metrics_per_threshold[threshold] = df.mean().to_dict()

    return avg_metrics_per_threshold



#Κάνουμε το ίδιο αλλά για κάθε σημείο αντί για segment
def run_pointwise_classification_evaluation():
    metrics_per_threshold = {}

    participants = list_participants()
    for pid in participants:
        feat_path = os.path.join(DATA_FOLDER, pid, "annotation_features.json")
        if not os.path.exists(feat_path):
            continue

        with open(feat_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            pw_metrics = data.get("pointwise_metrics", {})
            for threshold, metrics in pw_metrics.items():
                if threshold not in metrics_per_threshold:
                    metrics_per_threshold[threshold] = []
                metrics_per_threshold[threshold].append(metrics)

    if not metrics_per_threshold:
        return None

    avg_metrics_per_threshold = {}
    for threshold, metrics_list in metrics_per_threshold.items():
        df = pd.DataFrame(metrics_list)
        avg_metrics_per_threshold[threshold] = df.mean().to_dict()

    return avg_metrics_per_threshold


def run_memory_classification_evaluation(memory_type='short'):
    metrics_per_threshold = {}

    participants = list_participants()
    for pid in participants:
        feat_path = os.path.join(DATA_FOLDER, pid, "annotation_features.json")
        if not os.path.exists(feat_path):
            continue

        with open(feat_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            mem_metrics = data.get("memory_metrics", {}).get(memory_type, {})
            for threshold, metrics in mem_metrics.items():
                if threshold not in metrics_per_threshold:
                    metrics_per_threshold[threshold] = []
                metrics_per_threshold[threshold].append(metrics)

    if not metrics_per_threshold:
        return None

    avg_metrics_per_threshold = {}
    for threshold, metrics_list in metrics_per_threshold.items():
        df = pd.DataFrame(metrics_list)
        avg_metrics_per_threshold[threshold] = df.mean().to_dict()

    return avg_metrics_per_threshold

def print_metrics_table(title, metrics_per_threshold):
    st.subheader(title)
    if all(isinstance(v, dict) for v in metrics_per_threshold.values()):
        # Case: thresholds as keys
        for threshold, metrics_dict in sorted(metrics_per_threshold.items(), key=lambda x: float(x[0])):
            st.markdown(f"**🔹 Threshold = {threshold}**")
            table = [[metric.capitalize(), f"{value:.4f}"] for metric, value in metrics_dict.items()]
            df_table = pd.DataFrame(table, columns=["Metric", "Average Value"])
            st.table(df_table)
    else:
        # Flat metrics dict
        table = [[metric.capitalize(), f"{value:.4f}"] for metric, value in metrics_per_threshold.items()]
        df_table = pd.DataFrame(table, columns=["Metric", "Average Value"])
        st.table(df_table)


# Main - Ξεκινάμε από το home page 
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "add":
    add_participant_page()
elif st.session_state.page == "participant_analysis":
    participant_analysis_page(st.session_state.selected_participant)
