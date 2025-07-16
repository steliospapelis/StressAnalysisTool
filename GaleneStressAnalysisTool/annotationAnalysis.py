import os
import json
import numpy as np
from datetime import datetime, timedelta


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
def compute_features(normalized_trace, segments,global_mean_val,are_stressed):
    features = []
    print(global_mean_val)
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

        if mean_val > global_mean_val + 0.1:
            label_is_stressed = 1
        elif mean_val < global_mean_val - 0.1:
            label_is_stressed = 0
        else:
            label_is_stressed = 'undefined'
        prediction_is_stressed = are_stressed

        

        features.append({
            'mean': mean_val,
            'median': median_val,
            'max': max_val,
            'area': area,
            'amplitude': amplitude,
            'gradient': avg_gradient,
            'label': label_is_stressed,
            'prediction': prediction_is_stressed
        })


    print(features)
        

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
    for f in all_features:
        if f['label'] == 'undefined':
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
        'total': tp + fp + tn + fn,
        'StressedPoints': tp+fn,
        'CalmPoints':tn+fp,
        'Stressed Percentage':(tp+fn)/(tp + fp + tn + fn),
        'Calm Percentage':(tn+fp)/(tp + fp + tn + fn),
    }


def compute_pointwise_classification(normalized_trace, calm_segments, stressed_segments, global_mean):
    classification = []

    # Process calm segments
    for start, end in calm_segments:
        pred = 0
        segment_points = [ (t, v) for t, v in normalized_trace if start <= t <= end ]
        for t, v in segment_points:
            if v > global_mean + 0.1:
                label = 1
            elif v < global_mean - 0.1:
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
            if v > global_mean + 0.1:
                label = 1
            elif v < global_mean - 0.1:
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
    for p in pointwise_results:
        label = p['label']
        prediction = p['prediction']
        if label == 'undefined':
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
        'total': tp + fp + tn + fn,
        'StressedPoints': tp+fn,
        'CalmPoints':tn+fp,
        'Stressed Percentage':(tp+fn)/(tp + fp + tn + fn),
        'Calm Percentage':(tn+fp)/(tp + fp + tn + fn),

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
    

    calm_features = compute_features(normalized, calm_ranges,global_mean_val, False)
    stressed_features = compute_features(normalized, stressed_ranges,global_mean_val, True)

    calm_agg = aggregate_features(calm_features)
    stressed_agg = aggregate_features(stressed_features)

    all_features = calm_features+stressed_features
    metrics = compute_classification_metrics(all_features)

    pointwise_results = compute_pointwise_classification(normalized, calm_ranges, stressed_ranges, global_mean_val)
    

    pointwise_metrics = compute_pointwise_metrics(pointwise_results)
    print("Pointwise metrics:", pointwise_metrics)
    


    final_result = {'calm': calm_agg, 'stressed': stressed_agg,'metrics': metrics,'pointwise_metrics': pointwise_metrics}

    out_file = os.path.join(participant_folder, "annotation_features.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2)

    return normalized, calm_ranges, stressed_ranges, analysis_result.get('breathing_ranges', []), final_result
