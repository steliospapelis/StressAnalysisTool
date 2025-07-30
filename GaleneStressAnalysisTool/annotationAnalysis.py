import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import spearmanr
import pandas as pd




#Ανοίγουμε το αρχείο με τα annotatons και σκιπάρουμε τόσο χρόνο όσο διαρκεί το tutorial (διά 2 γιατί το βίντεο το έβλεπαν x2 ταχύτητα) 
# (το έχουμε αποθηκεύσει στο αρχείο analysis ως delta_seconds)

def parse_annotation_file(annotation_file, delta_seconds):
    """Parse the annotation file and return list of (relative_time_in_seconds, value) after adjusted base time."""
    data = []
    base_time = None
    last_known_value = None
    last_known_time = None
    inserted_base_point = False

    with open(annotation_file, encoding='utf-8') as f:
        next(f)  
        for line in f:
            if "time:" not in line:
                continue
            parts = line.strip().split()
            try:
                value = float(parts[0])
                time_str = parts[-1].replace('time:', '').strip()

                # Parse time format HH:MM.S
                if ':' not in time_str or '.' not in time_str:
                    continue

                hour_str, minute_sec_str = time_str.split(':')
                minute_str, sec_str = minute_sec_str.split('.')

                current_time = timedelta(
                    hours=int(hour_str), minutes=int(minute_str), seconds=int(sec_str)
                )

                # Επειδή κάποιες τιμές (δευτερόλεπτα λείπουν, αν αυτό που θέλουμε ως base δεν υπάρχει, παίρνουμε το προηγούμενο)
                if base_time is None:
                    base_time = current_time + timedelta(seconds=delta_seconds / 2)

                
                if current_time < base_time:
                    last_known_time = current_time
                    last_known_value = value
                    continue

                if not inserted_base_point and last_known_value is not None:
                    relative_seconds = 0.0
                    data.append((relative_seconds*2, last_known_value))
                    inserted_base_point = True

                relative_seconds = (current_time - base_time).total_seconds()
                data.append((relative_seconds*2, value))

            except Exception:
                continue  
    return data

#Κάνουμε τη συνάρτηση συνεχούς χρόνου βάζοντας σημεία ανά 0.01 (minimum frame time)
def interpolate_continuous_trace(trace, resolution=0.01):
    """
    Linearly interpolate points between existing trace values.
    `resolution` determines the time step size (in seconds) for interpolation.
    """
    if len(trace) < 2:
        return trace  
    interpolated = []
    for i in range(len(trace) - 1):
        t0, v0 = trace[i]
        t1, v1 = trace[i + 1]
        dt = t1 - t0
        steps = int(dt / resolution)
        for s in range(steps):
            interp_t = t0 + s * resolution
            interp_v = v0 + (v1 - v0) * ((interp_t - t0) / dt)
            interpolated.append((interp_t, interp_v))
    interpolated.append((t1, v1))
    return interpolated

#Κάνουμε κανονικοποίηση των τιμών ψάχνοντας min και max μόνο στα stressed και calm segments
def normalize_trace(continuous_data, calm_ranges, stressed_ranges):
    values_in_segments = []
    for start, end in calm_ranges + stressed_ranges:
        for t, v in continuous_data:
            if start <= t <= end:
                values_in_segments.append(v)
    if not values_in_segments:
        return []  
    

    min_val = min(values_in_segments)
    max_val = max(values_in_segments)
    range_val = max_val - min_val if max_val != min_val else 1

    normalized = [ (t, (v - min_val) / range_val) for t, v in continuous_data ]

    normalized_values_in_segments = []
    for start, end in calm_ranges + stressed_ranges:
        for t, norm_v in normalized:
            if start <= t <= end:
                normalized_values_in_segments.append(norm_v)
    global_mean_val = np.mean(normalized_values_in_segments) if normalized_values_in_segments else 0
    return normalized, global_mean_val

#Για κάθε segment (stressed και calm) υπολογίζουμε 4 χαρακτηριστικά :
def compute_features(normalized_trace, segments,global_mean_val,are_stressed,threshold):
    features = []
    
    for start, end in segments:
        seg_values = [v for t, v in normalized_trace if start <= t <= end]
        seg_times = [t for t, v in normalized_trace if start <= t <= end]
        if len(seg_values) < 2:
            continue
        seg_values = np.array(seg_values)
        seg_times = np.array(seg_times)
        duration = seg_times[-1] - seg_times[0]

        mean_val = np.mean(seg_values)
        area = np.trapz(seg_values, seg_times) / duration if duration > 0 else 0
        amplitude = np.max(seg_values) - np.min(seg_values)
        gradients = np.diff(seg_values) / np.diff(seg_times)
        avg_gradient = np.mean(gradients)
        max_val = np.max(seg_values)
        median_val = np.median(seg_values)

        if mean_val > global_mean_val + threshold:
            label_is_stressed = 1
        elif mean_val < global_mean_val - threshold:
            label_is_stressed = 0
        else:
            label_is_stressed = 'undefined'
        prediction_is_stressed = are_stressed

        

        features.append({
            'start':start,
            'end':end,
            'mean': mean_val,
            'median': median_val,
            'max': max_val,
            'area': area,
            'amplitude': amplitude,
            'gradient': avg_gradient,
            'label': label_is_stressed,
            'prediction': prediction_is_stressed
        })


   
        

    return features

#Βρίσκουμε μέσο όρο για όλα τα segments ίδιου είδους (όλων των calm και όλων των stressed δηλαδή)
def aggregate_features(features_list):
    if not features_list:
        return {'mean': 0, 'area': 0, 'amplitude': 0, 'gradient': 0}
    return {
        'mean': np.mean([f['mean'] for f in features_list]),
        'median': np.mean([f['median'] for f in features_list]),
        'max': np.mean([f['max'] for f in features_list]),
        'area': np.mean([f['area'] for f in features_list]),
        'amplitude': np.mean([f['amplitude'] for f in features_list]),
        'gradient': np.mean([f['gradient'] for f in features_list]),
    }

def compute_classification_metrics(all_features):
    tp = fp = tn = fn = 0
    undefined_count = 0
    for f in all_features:
        if f['label'] == 'undefined':
            undefined_count += 1
            continue
        pred = f['prediction']
        label = f['label']
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and label:
            fn += 1
        else: 
            tn += 1
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'total': tp + fp + tn + fn+undefined_count,
        'Stressed Predicted': tp+fp,
        'Calm Predicted':tn+fn,
        'Stressed Percentage':(tp+fp)/(tp + fp + tn + fn+undefined_count),
        'Calm Percentage':(tn+fn)/(tp + fp + tn + fn+undefined_count),
        'Undefined Percentage':undefined_count/(tp + fp + tn + fn+undefined_count)
    }


def compute_pointwise_classification(normalized_trace, calm_segments, stressed_segments, global_mean,threshold):
    classification = []

    # Process calm segments
    for start, end in calm_segments:
        pred = 0
        segment_points = [ (t, v) for t, v in normalized_trace if start <= t <= end ]
        for t, v in segment_points:
            if v > global_mean + threshold:
                label = 1
            elif v < global_mean - threshold:
                label = 0
            else:
                label = 'undefined'
            classification.append({
                'timestamp': t,
                'value': v,
                'prediction': pred,
                'label': label
            })

    # Process stressed segments
    for start, end in stressed_segments:
        pred = 1
        segment_points = [ (t, v) for t, v in normalized_trace if start <= t <= end ]
        for t, v in segment_points:
            if v > global_mean + threshold:
                label = 1
            elif v < global_mean - threshold:
                label = 0
            else:
                label = 'undefined'
            classification.append({
                'timestamp': t,
                'value': v,
                'prediction': pred,
                'label': label
            })

    return classification


def compute_pointwise_metrics(pointwise_results):
    tp = fp = tn = fn = 0
    undefined_count = 0

    for p in pointwise_results:
        label = p['label']
        prediction = p['prediction']
        if label == 'undefined':
            undefined_count+=1;
            continue
        if prediction == 1 and label == 1:
            tp += 1
        elif prediction == 1 and label == 0:
            fp += 1
        elif prediction == 0 and label == 1:
            fn += 1
        elif prediction == 0 and label == 0:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'total defined': tp + fp + tn + fn,
        'Stressed Points Predicted': tp+fp,
        'Calm Points':tn+fn,
        'Stressed Percentage':(tp+fp)/(tp + fp + tn + fn+ undefined_count),
        'Calm Percentage':(tn+fn)/(tp + fp + tn + fn+ undefined_count),
        'Undefined Percentage':undefined_count/(tp + fp + tn + fn+undefined_count)

    }

def compute_memory_based_labels(all_features, memory_type='short', threshold=0.1):
    """
    Label segments as stressed or calm based on short or long memory.
    
    memory_type: 'short' -> use last 2 segments; 'long' -> use all previous segments.
    """
    # Sort by start time
    sorted_features = sorted(all_features, key=lambda f: f['start'])
    new_features = []

    for i, f in enumerate(sorted_features):
        if memory_type == 'short':
            previous = sorted_features[max(0, i-2):i]
        elif memory_type == 'long':
            previous = sorted_features[:i]
        else:
            raise ValueError("memory_type must be 'short' or 'long'")

        # If no previous segments, keep label as undefined
        if not previous or i<2:
            new_label = 'undefined'
        else:
            prev_means = [p['mean'] for p in previous]
            local_mean = np.mean(prev_means)

            if f['mean'] > local_mean + threshold:
                new_label = 1  # stressed
            elif f['mean'] < local_mean - threshold:
                new_label = 0  # calm
            else:
                new_label = 'undefined'

        # Copy and update the feature dict
        new_f = f.copy()
        new_f['memory_label'] = new_label
        if not previous or i<2:
            new_f['memory_local_mean'] = None 
        else:
            new_f['memory_local_mean'] =  local_mean
        new_features.append(new_f)

    return new_features

def compute_memory_classification_metrics(features_with_memory_labels):
    tp = fp = tn = fn = 0
    undefined_count=0
    for f in features_with_memory_labels:
        if f['memory_label'] == 'undefined':
            undefined_count+=1
            continue
        pred = f['prediction']
        label = f['memory_label']
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and label:
            fn += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'total defined': total,
        'Undefined Percentage':undefined_count/(tp + fp + tn + fn+undefined_count)

    }


def split_trace_into_windows(normalized_trace, window_size):
    """
    Split the normalized trace into non-overlapping windows of `window_size` seconds.
    Returns list of (start_time, end_time, values_in_window).
    """
    if not normalized_trace:
        return []

    start_time = normalized_trace[0][0]
    end_time = normalized_trace[-1][0]
    windows = []

    current_start = start_time
    while current_start + window_size <= end_time:
        current_end = current_start + window_size
        window_values = [v for t, v in normalized_trace if current_start <= t < current_end]
        windows.append((current_start, current_end, window_values))
        current_start = current_end

    return windows

def label_windows(windows, global_mean, epsilon):
    """
    For each window, compute mean and assign label based on threshold.
    """
    labeled_windows = []
    for start, end, values in windows:
        if not values:
            label = 'undefined'
        else:
            mean_val = np.mean(values)
            if mean_val > global_mean + epsilon:
                label = 1
            elif mean_val < global_mean - epsilon:
                label = 0
            else:
                label = 'undefined'
        labeled_windows.append({'start': start, 'end': end, 'label': label})
    return labeled_windows

def predict_window_labels(labeled_windows, calm_segments, stressed_segments):
    """
    If window is fully inside a calm segment → prediction=0.
    If fully inside a stressed segment → prediction=1.
    Else → prediction='undefined'.
    """
    predictions = []
    for win in labeled_windows:
        start, end = win['start'], win['end']
        prediction = 'undefined'

        for c_start, c_end in calm_segments:
            if c_start <= start and end <= c_end:
                prediction = 0
                break
        for s_start, s_end in stressed_segments:
            if s_start <= start and end <= s_end:
                prediction = 1
                break

        predictions.append({
            'start': start,
            'end': end,
            'label': win['label'],
            'prediction': prediction
        })
        
    return predictions

def compute_window_metrics(window_results):
    tp = fp = tn = fn = 0
    undefined_count = 0
    for item in window_results:
        label = item['label']
        prediction = item['prediction']
        if label == 'undefined' or prediction == 'undefined':
            undefined_count += 1
            continue
        if prediction == 1 and label == 1:
            tp += 1
        elif prediction == 1 and label == 0:
            fp += 1
        elif prediction == 0 and label == 1:
            fn += 1
        elif prediction == 0 and label == 0:
            tn += 1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'undefined_windows': undefined_count,
        'total_windows': total + undefined_count,
        'Undefined Percentage': undefined_count / (total + undefined_count) if (total + undefined_count) else 0
    }

#Entry point της ανάλυσης
def analyze_annotation(participant_folder, analysis_result):
    """
    Complete pipeline:
    - parse annotation
    - normalize
    - compute features per segment
    - aggregate
    - save to file
    """

    #Καλούμε με τη σειρά του παραπάνω pipeline τις παραπάνω συναρτήσεις
    annotation_file = os.path.join(participant_folder, "stress_annotation.txt")
    delta_time = analysis_result['delta_seconds']
    

    annotation_data = parse_annotation_file(annotation_file, delta_time)
    if not annotation_data:
        print("No valid annotation data found.")
        return None

    continuous = interpolate_continuous_trace(annotation_data, resolution=0.01)

    calm_ranges = analysis_result['calm']
    stressed_ranges = analysis_result['stressed']

    normalized, global_mean_val = normalize_trace(continuous, calm_ranges, stressed_ranges)
    
    thresholds = [0, 0.05, 0.1, 0.2]

    segment_metrics = {}
    pointwise_metrics = {}
    memory_metrics_short = {}
    memory_metrics_long = {}
    window_metrics={}
    spearman_metrics={}


    for threshold in thresholds:
       
        calm_features = compute_features(normalized, calm_ranges, global_mean_val, False, threshold=threshold)
        stressed_features = compute_features(normalized, stressed_ranges, global_mean_val, True, threshold=threshold)
        all_features = calm_features + stressed_features
       

        df = pd.DataFrame(all_features)

        df['mean_rank'] = df['mean'].rank(method='average', ascending=True)
        df['gradient_rank'] = df['gradient'].rank(method='average', ascending=True)
        df['amplitude_rank'] = df['amplitude'].rank(method='average', ascending=True)
        df['area_rank'] = df['area'].rank(method='average', ascending=True)
        df['median_rank'] = df['median'].rank(method='average', ascending=True)

        print(df)

        corr_results = {}

        for metric in ['mean_rank', 'gradient_rank', 'amplitude_rank', 'area_rank', 'median_rank']:
            corr, p = spearmanr(df['prediction'], df[metric])
            corr_results[metric] = {'correlation': corr, 'p_value': p}

        # Save correlation results for this threshold
        spearman_metrics[threshold] = corr_results

        # Segment metrics
        metrics = compute_classification_metrics(all_features)
        segment_metrics[threshold] = metrics

        calm_agg = aggregate_features(calm_features)
        stressed_agg = aggregate_features(stressed_features)

        # Pointwise metrics
        pointwise_results = compute_pointwise_classification(normalized, calm_ranges, stressed_ranges, global_mean_val, threshold=threshold)
        pw_metrics = compute_pointwise_metrics(pointwise_results)
        pointwise_metrics[threshold] = pw_metrics

        # Memory-based metrics
        features_short = compute_memory_based_labels(all_features, memory_type='short', threshold=threshold)
        features_long = compute_memory_based_labels(all_features, memory_type='long', threshold=threshold)

        metrics_short = compute_memory_classification_metrics(features_short)
        metrics_long = compute_memory_classification_metrics(features_long)

        memory_metrics_short[threshold] = metrics_short
        memory_metrics_long[threshold] = metrics_long

        window_size = 10 
        epsilon = threshold

        windows = split_trace_into_windows(normalized, window_size)
        labeled_windows = label_windows(windows, global_mean_val, epsilon)
        predicted_windows = predict_window_labels(labeled_windows, calm_ranges, stressed_ranges)
        window_metrics[threshold] = compute_window_metrics(predicted_windows)

        
    

    # Combine all into final result
    final_result = {
        'calm': calm_agg,
        'stressed': stressed_agg,
        'segment_metrics': segment_metrics,
        'pointwise_metrics': pointwise_metrics,
        'memory_metrics': {
            'short': memory_metrics_short,
            'long': memory_metrics_long
        },
        'window_metrics':window_metrics,
        'spearman_metrics':spearman_metrics
    }

    # Save to JSON
    out_file = os.path.join(participant_folder, "annotation_features.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2)

    return normalized, calm_ranges, stressed_ranges, analysis_result.get('breathing_ranges', []), final_result
