# Tennis-Data-Challenge
Assignment for Quantum Sports Analytics.

This repository contains two methods to detect **tennis ball hits and bounces** from ball-tracking data extracted from the Roland-Garros 2025 Final.

The goal is to classify each video frame as one of:

* `air`
* `hit`
* `bounce`

and to output a JSON file identical to the input structure, enriched with a new key:

```json
"pred_action": "air" | "hit" | "bounce"
```

---

## Methods

### 1. Unsupervised Method (Physics-Based)

A rule-based approach relying on physical properties of the ball trajectory:

* velocity and acceleration changes
* vertical motion patterns
* characteristic rebound and racket-contact signatures

This method does not use ground-truth labels.

**Function:**
`unsupervised_hit_bounce_detection(json_path)`

---

### 2. Supervised Method (Machine Learning)

A supervised pipeline trained using the provided `action` labels:

* feature engineering from the (x, y) time series (velocities, accelerations, dynamics)
* multi-class classification using an **XGBoost classifier**

The trained model and label encoder are provided in the repository.

**Function:**
`supervised_hit_bounce_detection(json_path, model_path)`

---

## Repository Structure

```
Tennis-Data-Challenge/
│
├── main.py               # Entry point: supervised & unsupervised detection methods
├── helper.py             # Helper functions (feature extraction, utilities)
├── requirements.txt      # Required Python packages
├── README.md
│
├── models/
│   ├── xgb_model.pkl           # Trained XGBoost classifier
│   └── xgb_label_encoder.pkl   # Label encoder for class decoding
```

---

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run detection from Python:

```python
from main import supervised_hit_bounce_detection, unsupervised_hit_bounce_detection

result = supervised_hit_bounce_detection("ball_data_369.json")
```

The output is a dictionary matching the input JSON format, with an added `pred_action` field for each frame.

---

## Notes

* The supervised model was trained on the provided labeled dataset.
* The unsupervised method is fully self-contained and does not rely on labels.
* All outputs are JSON-serializable and ready for evaluation or visualization.

