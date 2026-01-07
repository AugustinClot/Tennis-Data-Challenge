import json
import joblib
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import os
from helper import supervised_hit_bounce_detection, extract_features, _json_default

def unsupervised_hit_bounce_detection(json_path, output_path = "ball_data_{}_unsupervised_predicted.json"):
    """
    Detect hits and bounces using physics-based trajectory analysis.
    
    Key principles:
    - Bounces: Sharp change in vertical velocity (y-direction reversal)
    - Hits: Sudden acceleration in horizontal movement + possible direction change
    """
    
    # Load the data
    with open(json_path, 'r') as f:
        ball_data = json.load(f)
    
    # Convert to sorted arrays
    frames = sorted([int(k) for k in ball_data.keys()])
    
    # Extract positions for visible frames
    x_pos, y_pos, visible_frames = [], [], []
    for frame in frames:
        if ball_data[str(frame)]['visible']:
            x_pos.append(ball_data[str(frame)]['x'])
            y_pos.append(ball_data[str(frame)]['y'])
            visible_frames.append(frame)
    
    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)
    visible_frames = np.array(visible_frames)
    
    # Smooth the trajectories to reduce noise
    if len(x_pos) > 5:
        x_smooth = savgol_filter(x_pos, window_length=min(11, len(x_pos) if len(x_pos) % 2 == 1 else len(x_pos)-1), polyorder=2)
        y_smooth = savgol_filter(y_pos, window_length=min(11, len(y_pos) if len(y_pos) % 2 == 1 else len(y_pos)-1), polyorder=2)
    else:
        x_smooth, y_smooth = x_pos, y_pos
    
    # Calculate velocities (first derivative)
    dt = np.diff(visible_frames)
    dt[dt == 0] = 1  # Avoid division by zero
    
    vx = np.diff(x_smooth) / dt
    vy = np.diff(y_smooth) / dt
    
    # Calculate accelerations (second derivative)
    if len(vx) > 1:
        ax = np.diff(vx) / dt[:-1]
        ay = np.diff(vy) / dt[:-1]
    else:
        ax, ay = np.array([]), np.array([])
    
    # Detect bounces
    # Bounce = sharp upward change in y-velocity (ball going down, then up)
    bounce_frames = []
    if len(ay) > 0:
        # Look for strong upward acceleration (negative y is down in image coords)
        ay_threshold = np.percentile(np.abs(ay), 85)
        
        for i in range(len(ay)):
            # Strong upward acceleration + ball was moving downward
            if ay[i] < -ay_threshold and i > 0 and vy[i] > 0:
                bounce_frames.append(visible_frames[i+1])
    
    # Detect hits
    # Hit = sudden change in horizontal velocity + possible vertical component
    hit_frames = []
    if len(ax) > 0:
        # Combined acceleration magnitude
        a_magnitude = np.sqrt(ax**2 + ay**2)
        a_threshold = np.percentile(a_magnitude, 90)
        
        for i in range(len(a_magnitude)):
            # High acceleration but not a bounce
            frame = visible_frames[i+1]
            if a_magnitude[i] > a_threshold and frame not in bounce_frames:
                # Additional check: significant horizontal component
                if np.abs(ax[i]) > 0.3 * a_magnitude[i]:
                    hit_frames.append(frame)
    
    # Create output with predictions
    output_data = {}
    for frame_str, data in ball_data.items():
        frame = int(frame_str)
        output_data[frame_str] = data.copy()
        
        if frame in bounce_frames:
            output_data[frame_str]['pred_action'] = 'bounce'
        elif frame in hit_frames:
            output_data[frame_str]['pred_action'] = 'hit'
        else:
            output_data[frame_str]['pred_action'] = 'air'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
        print(f"Saved predictions to {output_path}")
    return output_data

def supervised_hit_bounce_detection(json_path, model_path='models/xgb_model.pkl', encoder_path = "models/xgb_label_encoder.pkl", output_path = "predictions/ball_data_{}_supervised_predicted.json"):
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
        print(f"Saved predictions to {output_path}")
    return output_data
