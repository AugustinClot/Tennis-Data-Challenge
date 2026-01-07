import json
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
#import pandas as pd
#from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from xgboost import XGBClassifier

import os

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj) 

def extract_features(ball_data, window_size=5):
    """
    Extract interpretable physics-based features for each frame.
    
    Feature categories:
    1. Kinematic: position, velocity, acceleration
    2. Trajectory shape: curvature, direction changes
    3. Temporal context: what happened before/after
    """
    
    # Parse data
    frames = sorted([int(k) for k in ball_data.keys()])
    
    # Extract visible frames
    x_pos, y_pos, visible_frames, labels = [], [], [], []
    for frame in frames:
        if ball_data[str(frame)]['visible']:
            x_pos.append(ball_data[str(frame)]['x'])
            y_pos.append(ball_data[str(frame)]['y'])
            visible_frames.append(frame)
            labels.append(ball_data[str(frame)].get('action', 'air'))
    
    if len(x_pos) < window_size:
        return None, None, None
    
    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)
    visible_frames = np.array(visible_frames)
    
    # Smooth trajectories
    window = min(11, len(x_pos) if len(x_pos) % 2 == 1 else len(x_pos)-1)
    if window < 5:
        return None, None, None
        
    x_smooth = savgol_filter(x_pos, window_length=window, polyorder=2)
    y_smooth = savgol_filter(y_pos, window_length=window, polyorder=2)
    
    # Calculate derivatives
    dt = np.diff(visible_frames)
    dt[dt == 0] = 1
    
    vx = np.diff(x_smooth) / dt
    vy = np.diff(y_smooth) / dt
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    ax = np.diff(vx) / dt[:-1]
    ay = np.diff(vy) / dt[:-1]
    a_magnitude = np.sqrt(ax**2 + ay**2)
    
    # Normalize positions (0-1 range)
    y_normalized = (y_pos - y_pos.min()) / (y_pos.max() - y_pos.min() + 1e-6)
    
    # Build feature matrix
    features = []
    valid_indices = []
    
    half_window = window_size // 2
    
    for i in range(half_window, len(visible_frames) - half_window):
        # Skip if we don't have enough derivative data
        if i >= len(v_magnitude) or i-1 >= len(a_magnitude):
            continue
            
        feat = []
        
        # === POSITION FEATURES ===
        feat.append(x_smooth[i])  # Current x position
        feat.append(y_smooth[i])  # Current y position
        feat.append(y_normalized[i])  # Normalized height (0=top, 1=bottom)
        
        # === VELOCITY FEATURES ===
        feat.append(vx[i-1])  # Horizontal velocity
        feat.append(vy[i-1])  # Vertical velocity
        feat.append(v_magnitude[i-1])  # Speed
        
        # Velocity direction (angle)
        velocity_angle = np.arctan2(vy[i-1], vx[i-1])
        feat.append(velocity_angle)
        
        # === ACCELERATION FEATURES ===
        if i-1 < len(a_magnitude):
            feat.append(ax[i-2])  # Horizontal acceleration
            feat.append(ay[i-2])  # Vertical acceleration
            feat.append(a_magnitude[i-2])  # Total acceleration
        else:
            feat.extend([0, 0, 0])
        
        # === TRAJECTORY SHAPE (local window) ===
        start_idx = i - half_window
        end_idx = i + half_window + 1
        
        # Vertical range in window (how much y changes)
        y_window = y_smooth[start_idx:end_idx]
        feat.append(y_window.max() - y_window.min())
        feat.append(np.std(y_window))  # Vertical variation
        
        # Horizontal range
        x_window = x_smooth[start_idx:end_idx]
        feat.append(x_window.max() - x_window.min())
        feat.append(np.std(x_window))  # Horizontal variation
        
        # === VELOCITY CHANGES (before vs after) ===
        if i >= 2 and i < len(v_magnitude) - 1:
            v_before = v_magnitude[i-2]
            v_current = v_magnitude[i-1]
            v_after = v_magnitude[i] if i < len(v_magnitude) else v_current
            
            feat.append(v_current - v_before)  # Velocity change (deceleration/acceleration)
            feat.append(abs(v_after - v_before))  # Total velocity change magnitude
            
            # Direction change
            if i-2 < len(vx) and i < len(vx):
                angle_before = np.arctan2(vy[i-2], vx[i-2])
                angle_after = np.arctan2(vy[i], vx[i]) if i < len(vy) else angle_before
                angle_change = abs(angle_after - angle_before)
                feat.append(angle_change)
            else:
                feat.append(0)
        else:
            feat.extend([0, 0, 0])
        
        # === CURVATURE (trajectory bending) ===
        if i >= 2 and i < len(y_smooth) - 2:
            # Simple curvature approximation using 5 points
            y_curve = y_smooth[i-2:i+3]
            x_curve = x_smooth[i-2:i+3]
            
            # Second derivative of y (how much trajectory curves)
            if len(y_curve) == 5:
                curvature = abs(y_curve[0] - 2*y_curve[2] + y_curve[4])
                feat.append(curvature)
            else:
                feat.append(0)
        else:
            feat.append(0)
        
        features.append(feat)
        valid_indices.append(i)
    
    features = np.array(features)
    valid_labels = [labels[i] for i in valid_indices]
    valid_frames = [visible_frames[i] for i in valid_indices]
    
    return features, valid_labels, valid_frames

def supervised_hit_bounce_detection(json_path, model_path='models/xgb_model.pkl', encoder_path = "models/xgb_label_encoder.pkl", output_path = "predictions/ball_data_{}_predicted.json"):
    """
    Predict hits and bounces using trained model. (XGboost is used as it gave the best results)

    Note: You can use supervised_predict_and_save_multiple (available in helper.py) to repeat this process multiple times automatically. 
    """
    
    # Load model
    model = joblib.load(model_path)
    
    # Load data
    with open(json_path, 'r') as f:
        ball_data = json.load(f)
    
    # Extract features
    features, _, valid_frames = extract_features(ball_data)
    
    if features is None:
        # Not enough data, return all 'air'
        output_data = {}
        for frame_str, data in ball_data.items():
            output_data[frame_str] = data.copy()
            output_data[frame_str]['pred_action'] = 'air'
        return output_data
    
    # Predict
    predictions = model.predict(features)
    # In the case of XGboost we need to return to original labels
    if encoder_path is not None:
        encoder = joblib.load(encoder_path)
        try:
            predictions = encoder.inverse_transform(predictions.astype(int))
        except Exception:
            # Not an XGBoost-style numeric classifier or incompatible encoder
            pass
    
    # Map predictions back to frames
    frame_predictions = dict(zip(valid_frames, predictions))
    
    # Create output
    output_data = {}
    for frame_str, data in ball_data.items():
        frame = int(frame_str)
        output_data[frame_str] = data.copy()
        output_data[frame_str]['pred_action'] = frame_predictions.get(frame, 'air') # default to air if ball not visible
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=_json_default)
    return output_data


def supervised_predict_and_save_multiple(test_numbers, model_path='models/xgb_model.pkl', 
                                base_input_path="Data hit & bounce/per_point_v2/ball_data_{}.json",
                                base_output_path="predictions/ball_data_{}_predicted.json",
                             ):
    """
    Run predictions on multiple test files and save results.
    
    Args:
        test_numbers: List of test file numbers (e.g., [369, 145, 78])
        model_path: Path to trained model
        base_input_path: Input path template with {} for number
        base_output_path: Output path template with {} for number
    
    Returns:
        List of output file paths
    """
    
    
    # Create output directory if needed
    output_dir = os.path.dirname(base_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"RUNNING PREDICTIONS ON {len(test_numbers)} TEST FILES")
    
    output_files = []
    
    for i, test_num in enumerate(test_numbers, 1):
        input_path = base_input_path.format(test_num)
        output_path = base_output_path.format(test_num)
        
        print(f"\n[{i}/{len(test_numbers)}] Processing ball_data_{test_num}.json...")
        
        try:
            # Run prediction
            result = supervised_hit_bounce_detection(
                json_path=input_path,
                model_path=model_path
            )
            
            # Save predictions
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=_json_default)
            
            output_files.append(output_path)
            print(f"  ✓ Saved to: {output_path}")
            
        except FileNotFoundError as e:
            print(f"  ✗ ERROR: File not found: {e.filename}")
            print(f"    Input attempted: {os.path.abspath(input_path)}")
            print(f"    Model attempted: {os.path.abspath(model_path)}")

        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
    
    print(f"\n✓ Completed {len(output_files)}/{len(test_numbers)} predictions")
    return output_files

def train_xgboost_model(data_folder, output_model_path='models/xgb_model.pkl',
                        output_encoder_path='models/xgb_label_encoder.pkl'):
    """Train XGBoost model with consistent label encoding."""
    
    all_features = []
    all_labels = []
    
    json_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]
    for json_file in json_files:
        json_path = os.path.join(data_folder, json_file)
        with open(json_path, 'r') as f:
            ball_data = json.load(f)

        features, labels, _ = extract_features(ball_data)
        if features is not None and len(labels) > 0:
            all_features.append(features)
            all_labels.extend(labels)

    if not all_features:
        raise ValueError(f"No features extracted from folder: {data_folder}")

    X = np.vstack(all_features)
    y_str = np.array(all_labels)

    # --- Encode labels ---
    le = LabelEncoder()
    y = le.fit_transform(y_str)   # e.g. air->0, bounce->1, hit->2 (order depends on sorting)

    # --- Sample weights (balanced per class) ---
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    count_map = dict(zip(unique, counts))
    sample_weights = np.array([max_count / count_map[cls] for cls in y], dtype=float)

    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
    )
    model.fit(X, y, sample_weight=sample_weights)

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(model, output_model_path)
    joblib.dump(le, output_encoder_path)

    print(f"✓ XGBoost model saved to {output_model_path}")
    print(f"✓ Label encoder saved to {output_encoder_path}")
    print("Class mapping:", {cls: int(idx) for idx, cls in enumerate(le.classes_)})
    return model, le