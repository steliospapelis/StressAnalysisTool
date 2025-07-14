# analysis.py
import csv
import datetime



#Helper function για να παίρνουμε το timestamp της αρχής από το csv

def parse_start_time(first_cell):
    """
    Parse StartTime from string like 'StartTime: 2025-07-09 16:48:25'
    Ignores extra text after the timestamp.
    """
    prefix = "StartTime:"
    if first_cell.startswith(prefix):
        time_str_full = first_cell[len(prefix):].strip()
        time_str = time_str_full[:19]  # keep only 'YYYY-MM-DD HH:MM:SS'
        return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    else:
        raise ValueError(f"Invalid start time format: '{first_cell}'")



def analyze_participant(participant_folder):
    """
    Performs analysis for the participant.
    Returns a dict with: tutorial_start, game_start, delta_seconds, ranges.
    """
    # Κρατάμε πότε άρχισε το tutorial
    tutorial_file = f"{participant_folder}/tutorial_log.csv"
    with open(tutorial_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        first_row = next(reader)
        tutorial_start = parse_start_time(first_row[0])

    # Κρατάμε πότε άρχισε το level 1
    game_file = f"{participant_folder}/game_log.csv"
    with open(game_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        first_row = next(reader)  
        game_start = parse_start_time(first_row[0])

        header = next(reader)  
        timestamp_idx = header.index("Timestamp")
        state_idx = header.index("State")
        breathing_idx = header.index("Breathing")

        
        data = []
        for row in reader:
            ts_str = row[timestamp_idx].replace(',', '.')
            ts = float(ts_str)
            state = int(row[state_idx])
            breathing = int(row[breathing_idx])
            data.append((ts, state, breathing))

    # Βρίσκουμε χρόνο μεταξύ tutorial start και level 1 start για συγχρονισμό με annotation tool
    delta = (game_start - tutorial_start).total_seconds()

    # Χωρίζουμε σε διαστήματα που ήταν calm , stressed ή έκανε breathing
    calm_ranges = []
    stressed_ranges = []
    breathing_ranges = []

    if not data:
        return {
            "tutorial_start": tutorial_start.isoformat(),
            "game_start": game_start.isoformat(),
            "delta_seconds": delta,
            "calm": [],
            "stressed": [],
            "breathing": []
        }

    
    current_state = None
    start_time = data[0][0]

    #Φτιάχνουμε κλειστό διάστημα και το προσθέτουμε στην κατάλληλη λίστα
    def close_range(end_time, kind):
        if kind == 'calm':
            calm_ranges.append((start_time, end_time))
        elif kind == 'stressed':
            stressed_ranges.append((start_time, end_time))
        elif kind == 'breathing':
            breathing_ranges.append((start_time, end_time))

    for i, (ts, state, breathing) in enumerate(data):
        if breathing != 0:
            new_state = 'breathing'
        elif state == 0:
            new_state = 'calm'
        else:
            new_state = 'stressed'

        if current_state is None:
            current_state = new_state
            start_time = ts
        elif new_state != current_state:
            
            close_range(ts, current_state)
        
            current_state = new_state
            start_time = ts

    
    close_range(data[-1][0], current_state)

    #Επιστρέφουμε τα παρακάτω στο analysis αρχείο στο φάκελο του participant
    return {
        "tutorial_start": tutorial_start.isoformat(),
        "game_start": game_start.isoformat(),
        "delta_seconds": delta,
        "calm": calm_ranges,
        "stressed": stressed_ranges,
        "breathing": breathing_ranges
    }
