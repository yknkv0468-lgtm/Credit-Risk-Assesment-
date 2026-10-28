Credit Risk Terminal

A production-grade Streamlit application that scores retail loan applicants using a tuned, monotonic-constrained XGBoost model. It evaluates default risk against an F1-optimal decision cutoff and returns an automated approve/decline verdict with real-time financial telemetry.

Live demo:https://yknkv0468-lgtm-credit-risk-assesment--credit-48qnug.streamlit.app/

🚀 Key Engineering Features

Probability Calibration: Raw XGBoost scores were calibrated using CalibratedClassifierCV (Platt Scaling) so that output predictions reflect true historical default frequencies rather than arbitrary margin scores.

Optimal Decision Threshold: Analyzed the Precision-Recall curve to find the F1-optimal threshold at 0.759, maximizing default detection while preventing excessive false rejections.

Monotonic Constraints: Enforced logical boundaries directly into the algorithm (e.g., risk must rise with higher loan-to-income ratios, and must fall with higher income).

Interactive Dashboard: Dark-mode finance UI featuring Plotly risk gauges, instant benchmark profiles (Prime, Borderline, High Risk), and auto-calculated LTI ratios.

📊 Model Performance & Metrics

Trained on 28,500+ cleaned records (~21.7% default rate). The model handles class imbalance using a scale_pos_weight of 3.6.

Measured on a 20% hold-out test set (5,699 loans):

ROC-AUC: 0.934

PR-AUC: 0.887

Precision: 0.982

Recall: 0.730

F1-Score: 0.837

🛠️ Tech Stack

App & UI: Streamlit 1.43.2, Plotly 6.0.0, Custom CSS

Machine Learning Core: XGBoost 2.1.4, scikit-learn 1.6.1

Data Processing: pandas 2.2.3, NumPy 2.2.3, joblib 1.4.2

📂 Project Structure

.
├── credit.py                # Main Streamlit app and UI frontend
├── train_model.py           # Data cleaning, training, and threshold optimization
├── credit_risk_model.pkl    # Serialized preprocessing & calibrated XGBoost model
├── best_threshold.pkl       # Saved F1-optimal decision cutoff scalar
├── requirements.txt         # Pinned production dependencies
├── runtime.txt              # Python runtime specification (3.11.11)
└── .gitignore               # Exclusions for data and virtual environments


⚡ Run Locally

git clone https://github.com/yknkv0468-lgtm/Credit-Risk-Assesment-.git
cd Credit-Risk-Assesment-

# 1. Set up virtual environment
python -m venv .venv

# 2. Activate virtual environment
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies and run
pip install -r requirements.txt
streamlit run credit.py
