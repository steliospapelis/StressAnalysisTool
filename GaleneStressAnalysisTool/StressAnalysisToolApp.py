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
import seaborn as sns
from scipy.stats import wilcoxon
from scipy.stats import binomtest,probplot
from statistics import stdev
import numpy as np



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

    if st.button("🔍 Analyze Spearman Correlation "):
        spearman_results = run_spearman_directionality_analysis()
        
        if not spearman_results:
            st.warning("⚠️ No valid spearman data found.")
        else:
            st.subheader("📈 Spearman Correlation Summary")

            # Count table
            count_table = pd.DataFrame([
                [feat.capitalize(), count]
                for feat, count in spearman_results["significant_counts"].items()
            ], columns=["Feature", "Participants with ρ > 0 and p < 0.05"])
            st.table(count_table)

            # Mean correlation bar plot
            st.subheader("📊 Mean Correlation Across Participants")
            mean_corrs = spearman_results["mean_correlations"]
            sd_corrs = spearman_results["sd_correlations"]

            summary_table = pd.DataFrame([
            [
                feat.capitalize(),
                f"{mean_corrs[feat]:.4f}",
                f"{sd_corrs[feat]:.4f}"
            ]
            for feat in mean_corrs.keys()
            ], columns=["Feature", "Mean Spearman ρ", "SD"])
            st.table(summary_table)

            fig, ax = plt.subplots()
            sns.barplot(x=list(mean_corrs.keys()), y=list(mean_corrs.values()), ax=ax)
            ax.axhline(0, linestyle="--", color="gray")
            ax.set_ylabel("Mean Spearman ρ")
            st.pyplot(fig)

            # Wilcoxon results
            st.subheader("🧪 Wilcoxon Signed-Rank Test (H₀: median ρ = 0)")
            wilcoxon_table = pd.DataFrame([
                [feat.capitalize(), f"{stat:.4f}", f"{p:.4f}"]
                for feat, (stat, p) in spearman_results["wilcoxon"].items()
            ], columns=["Feature", "Wilcoxon Statistic", "p-value"])
            st.table(wilcoxon_table)

            st.subheader("🧪 Binomial Test (H₀: ρ equally likely to be > 0 or < 0)")

            binom_table = pd.DataFrame([
                [
                    feat.capitalize(),
                    f"{res['count']} / {res['n']}",
                    f"{res['p_value']:.4f}"
                ]
                for feat, res in spearman_results["binom"].items()
            ], columns=["Feature", "# Positive ρ", "p-value"])

            st.table(binom_table)
   


    if st.button("🧮 Perform Full Classification Evaluation (all thresholds)"):
        # Run evaluations
        avg_agg_metrics = run_classification_evaluation()
        avg_pointwise_metrics = run_pointwise_classification_evaluation()
        avg_memory_metrics_short = run_memory_classification_evaluation(memory_type='short')
        avg_memory_metrics_long = run_memory_classification_evaluation(memory_type='long')
        avg_window_metrics = run_window_classification_evaluation()

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

        if avg_window_metrics:
            st.subheader("🧠 Average Window Metrics Across Participants (by threshold)")
            print_metrics_table("Window Metrics", avg_memory_metrics_long)

        import plotly.graph_objects as go

        experiments = {
            "Segment": avg_agg_metrics,
            "Pointwise": avg_pointwise_metrics,
            "Short Memory": avg_memory_metrics_short,
            "Long Memory": avg_memory_metrics_long,
            "Window":avg_window_metrics
        }

        metric_names = ["accuracy", "precision", "recall", "f1", "Undefined Percentage"]

        st.subheader("📊 Classification Metrics by Experiment and Threshold")

        for experiment_name, metrics_dict in experiments.items():
            if metrics_dict is None:
                st.warning(f"⚠️ No metrics found for {experiment_name}.")
                continue

            st.markdown(f"### 🔬 {experiment_name} Experiment")

            thresholds = sorted(metrics_dict.keys())
            
            
            max_cols_per_row = 2
            for i in range(0, len(thresholds), max_cols_per_row):
                batch = thresholds[i:i+max_cols_per_row]
                cols = st.columns(len(batch))
                
                for col, threshold in zip(cols, batch):
                    threshold_metrics = metrics_dict[threshold]
                    values = [threshold_metrics.get(metric, 0)*100 for metric in metric_names]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=metric_names,
                        y=values,
                        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                        text=[f"{v:.2f}%" for v in values],
                        textposition='auto',
                        marker_line_width=0.5
                    ))

                    fig.update_layout(
                        title=f"Threshold: {threshold}",
                        yaxis_title="Percentage",
                        xaxis_title="Metric",
                        yaxis=dict(range=[0, 100]),
                        bargap=0.3,
                        margin=dict(l=40, r=40, t=40, b=30),
                        height=350,
                        width=400,  
                        font=dict(size=12),
                    )

                    col.plotly_chart(fig, use_container_width=False)


def analyze_segment_durations():
    calm_durations_all = []
    stressed_durations_all = []
    calm_counts = []
    stressed_counts = []

    participants = list_participants()
    for participant_id in participants:
        annotation_path = os.path.join(DATA_FOLDER, participant_id, "analysis.json")
        if not os.path.exists(annotation_path):
            continue
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        calm_segments = data.get("calm", [])
        stressed_segments = data.get("stressed", [])

        calm_durations = [end - start for start, end in calm_segments]
        stressed_durations = [end - start for start, end in stressed_segments]

        if calm_durations:
            calm_durations_all.append(np.mean(calm_durations))
            calm_counts.append(len(calm_durations))

        if stressed_durations:
            stressed_durations_all.append(np.mean(stressed_durations))
            stressed_counts.append(len(stressed_durations))

    print("Calm Segments:")
    print(f"  Mean of per-participant average durations: {np.mean(calm_durations_all):.2f} s")
    print(f"  Std Dev of per-participant average durations: {np.std(calm_durations_all):.2f} s")
    print(f"  Mean number of calm segments: {np.mean(calm_counts):.2f}")
    print(f"  Std Dev of calm segment counts: {np.std(calm_counts):.2f}")

    print("\nStressed Segments:")
    print(f"  Mean of per-participant average durations: {np.mean(stressed_durations_all):.2f} s")
    print(f"  Std Dev of per-participant average durations: {np.std(stressed_durations_all):.2f} s")
    print(f"  Mean number of stressed segments: {np.mean(stressed_counts):.2f}")
    print(f"  Std Dev of stressed segment counts: {np.std(stressed_counts):.2f}")



# Εκτελούμε στατιστική ανάλυση (Wilcoxon, Shapiro-Wilk, t-test) για όλα τα χαρακτηριστικά (mean, area, amplitude, gradient)
# Συλλέγουμε για κάθε συμμετέχοντα τις μέσες τιμές στα calm και stressed διαστήματα (μία τιμή για κάθε κατάσταση και χαρακτηριστικό).
def run_statistical_tests():
    analyze_segment_durations()
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

            # Create a figure
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # Histogram
            axes[0].hist(diffs, bins='auto', alpha=0.7, color='skyblue', edgecolor='black')
            axes[0].set_title(f'Histogram of Differences\nFeature: {feature}')
            axes[0].set_xlabel('Difference (stressed - calm)')
            axes[0].set_ylabel('Frequency')

            # Q-Q plot
            probplot(diffs, dist="norm", plot=axes[1])
            axes[1].set_title(f'Q-Q Plot of Differences\nFeature: {feature}')

            plt.tight_layout()

            # Display the figure in Streamlit
            st.pyplot(fig)

            # Optionally: close the figure to free memory
            plt.close(fig)

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

    feature_names = [f for f in calm_values.keys() if f != 'max']
    calm_means = [np.mean(calm_values[feature]) for feature in feature_names]
    stressed_means = [np.mean(stressed_values[feature]) for feature in feature_names]

    x = np.arange(len(feature_names))  # the label locations
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, calm_means, width, label='Calm', color='green')
    bars2 = ax.bar(x + width/2, stressed_means, width, label='Stressed', color='red')

    # Add labels and title
    ax.set_ylabel('Average Feature Value')
    ax.set_title('Average Feature Values in Calm vs Stressed States')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45)
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

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
        # ✅ filter out participants with undefined_percentage == 1
        filtered = [m for m in metrics_list if m.get('Undefined Percentage', 0) != 1]
        if filtered:
            df = pd.DataFrame(filtered)
            avg_metrics_per_threshold[threshold] = df.mean().to_dict()
        else:
            # fallback: no valid data left → set mean to zeros
            avg_metrics_per_threshold[threshold] = {k: 0 for k in metrics_list[0].keys()}

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
        filtered = [m for m in metrics_list if m.get('Undefined Percentage', 0) != 1]
        if filtered:
            df = pd.DataFrame(filtered)
            avg_metrics_per_threshold[threshold] = df.mean().to_dict()
        else:
            avg_metrics_per_threshold[threshold] = {k: 0 for k in metrics_list[0].keys()}

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
        filtered = [m for m in metrics_list if m.get('Undefined Percentage', 0) != 1]
        if filtered:
            df = pd.DataFrame(filtered)
            avg_metrics_per_threshold[threshold] = df.mean().to_dict()
        else:
            avg_metrics_per_threshold[threshold] = {k: 0 for k in metrics_list[0].keys()}

    return avg_metrics_per_threshold


def run_window_classification_evaluation():
    metrics_per_threshold = {}

    participants = list_participants()
    for pid in participants:
        feat_path = os.path.join(DATA_FOLDER, pid, "annotation_features.json")
        if not os.path.exists(feat_path):
            continue

        with open(feat_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            agg_metrics = data.get("window_metrics", {})
            for threshold, metrics in agg_metrics.items():
                if threshold not in metrics_per_threshold:
                    metrics_per_threshold[threshold] = []
                metrics_per_threshold[threshold].append(metrics)

    if not metrics_per_threshold:
        return None

    # Compute mean per metric for each threshold
    avg_metrics_per_threshold = {}
    for threshold, metrics_list in metrics_per_threshold.items():
        # ✅ filter out participants with undefined_percentage == 1
        filtered = [m for m in metrics_list if m.get('Undefined Percentage', 0) != 1]
        if filtered:
            df = pd.DataFrame(filtered)
            avg_metrics_per_threshold[threshold] = df.mean().to_dict()
        else:
            # fallback: no valid data left → set mean to zeros
            avg_metrics_per_threshold[threshold] = {k: 0 for k in metrics_list[0].keys()}

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


def run_spearman_directionality_analysis():
    participants = list_participants()
    features = ["mean_rank", "gradient_rank", "amplitude_rank", "area_rank","median_rank"]
    correlations = {f: [] for f in features}
    significant_positive_counts = {f: 0 for f in features}

    for pid in participants:
        final_path = os.path.join(DATA_FOLDER, pid, "annotation_features.json")
        if not os.path.exists(final_path):
            continue
        try:
            with open(final_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            spearman_0 = data.get("spearman_metrics", {}).get("0", {})
            for feat in features:
                metric = spearman_0.get(feat)
                if metric:
                    rho = metric.get("correlation", 0)
                    p = metric.get("p_value", 1)
                    correlations[feat].append(rho)
                    if rho > 0 and p < 0.05:
                        significant_positive_counts[feat] += 1
        except Exception as e:
            print(f"Error reading Spearman data for {pid}: {e}")

    # Mean correlations
    mean_correlations = {
        feat: sum(corrs)/len(corrs) if corrs else 0
        for feat, corrs in correlations.items()
    }

    # Standard deviation of correlations
    sd_correlations = {}
    for feat, corrs in correlations.items():
        if len(corrs) >= 2:
            try:
                sd_correlations[feat] = stdev(corrs)
            except Exception as e:
                print(f"Error computing SD for {feat}: {e}")
                sd_correlations[feat] = 0
        else:
            sd_correlations[feat] = 0

    # Wilcoxon test
    wilcoxon_results = {}
    for feat in features:
        try:
            if len(correlations[feat]) >= 2:
                stat, p = wilcoxon(correlations[feat])
                wilcoxon_results[feat] = (stat, p)
            else:
                wilcoxon_results[feat] = (0, 1)
        except Exception as e:
            print(f"Error running Wilcoxon for {feat}: {e}")
            wilcoxon_results[feat] = (0, 1)

    positive_counts = {feat: sum(1 for v in corrs if v > 0) for feat, corrs in correlations.items()}

    # Binomial tests
    binom_results = {}
    for feat in features:
        n = len(correlations[feat])
        k = positive_counts[feat]
        if n > 0:
            result = binomtest(k, n, p=0.5, alternative='greater')
            binom_results[feat] = {"count": k, "n": n, "p_value": result.pvalue}
        else:
            binom_results[feat] = {"count": 0, "n": 0, "p_value": 1.0}

    return {
        "significant_counts": significant_positive_counts,
        "mean_correlations": mean_correlations,
        "sd_correlations": sd_correlations,
        "wilcoxon": wilcoxon_results,
        "binom": binom_results
    }


        
# Main - Ξεκινάμε από το home page 
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "add":
    add_participant_page()
elif st.session_state.page == "participant_analysis":
    participant_analysis_page(st.session_state.selected_participant)
