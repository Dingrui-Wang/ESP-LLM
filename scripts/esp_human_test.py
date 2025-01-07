import streamlit as st
from pathlib import Path
import json
import os

# Function to load ground truth (taken from esp_prediction_evaluation.py)
def load_groundtruth(mons_dir, pred_filename):
    """Load groundtruth from metadata.json in mons folder"""
    uuid = pred_filename.split('.')[0]
    metadata_path = Path(mons_dir) / uuid / 'metadata.json'
    
    if not metadata_path.exists():
        return 0
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if not metadata:
        return 0
        
    for vehicle_id in metadata:
        if metadata[vehicle_id].get('cut_in', 0) == 1:
            return 1
    return 0

# Initialize session state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'ground_truths' not in st.session_state:
    st.session_state.ground_truths = {}

# Paths
image_dir = 'seq_balanced_100_4F'
mons_dir = 'tokens_by_mons/mons'
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

# Calculate metrics
def calculate_metrics():
    TP = TN = FP = FN = 0
    for uuid, pred in st.session_state.predictions.items():
        gt = st.session_state.ground_truths.get(uuid, 0)
        if pred == 1 and gt == 1:
            TP += 1
        elif pred == 0 and gt == 0:
            TN += 1
        elif pred == 1 and gt == 0:
            FP += 1
        elif pred == 0 and gt == 1:
            FN += 1
    
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    return accuracy, precision, recall

# UI Layout
st.title("Cut-in Detection Evaluation")

# Display current image
current_file = image_files[st.session_state.current_index]
uuid = current_file.split('.')[0]
image_path = os.path.join(image_dir, current_file)
st.image(image_path, caption=f"Image: {current_file}")

# Load ground truth
gt = load_groundtruth(mons_dir, current_file)
st.session_state.ground_truths[uuid] = gt

# Prediction buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("YES (Cut-in)"):
        st.session_state.predictions[uuid] = 1
        # Automatically go to next image if available
        if st.session_state.current_index < len(image_files) - 1:
            st.session_state.current_index += 1
        st.rerun()
with col2:
    if st.button("NO (No Cut-in)"):
        st.session_state.predictions[uuid] = 0
        # Automatically go to next image if available
        if st.session_state.current_index < len(image_files) - 1:
            st.session_state.current_index += 1
        st.rerun()

# Navigation buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Previous") and st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.rerun()

# Display metrics
accuracy, precision, recall = calculate_metrics()
st.subheader("Current Metrics")
st.write(f"Accuracy: {accuracy:.4f}")
st.write(f"Precision: {precision:.4f}")
st.write(f"Recall: {recall:.4f}")

# Display progress
progress = (st.session_state.current_index + 1) / len(image_files)
st.progress(progress)
st.write(f"Image {st.session_state.current_index + 1} of {len(image_files)}")