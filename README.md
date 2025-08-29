### AI Safety Models Proof of Concept (POC)###
Overview
This project is a Proof of Concept (POC) for a suite of AI Safety Models aimed at enhancing user safety in conversational AI platforms, such as chat applications or social media messaging. The POC demonstrates end-to-end functionality including data preprocessing, model inference, and real-time interaction through a Streamlit web interface.

key Features
- Abuse Language Detection: Real-time identification of harmful, threatening, or inappropriate content.
- Escalation Pattern Recognition: Detects when conversations become emotionally dangerous (e.g., repeated aggression or intensifying negativity).
- Crisis Intervention: Recognizes severe emotional distress or self-harm indicators, triggering potential human intervention.
- Content Filtering: Age-appropriate content filtering for guardian-supervised accounts.

### Project Structure ###
The project is organized as follows:

AI_safety_POC/
│
├── data/
│   ├── raw/             # Original datasets (CSV files)
│   └── processed/       # Cleaned and preprocessed datasets
│
├── models/              # Saved ML models (baseline TF-IDF + Logistic Regression)
│
├── src/                 # Python scripts for preprocessing, training, and utilities
│   ├── datapreprocessing.py
│   ├── train_model.py
│   └── ...
│
├── app.py               # Streamlit app for real-time AI safety demo
├── requirements.txt     # Python dependencies
└── README.md            # This file

Installation
1. Clone the repository:

git clone https://github.com/Jasminesonia/AI_safety_poc.git
cd AI_safety_poc

2. Create a virtual environment:

python -m venv venv

3. Activate the virtual environment:
- Windows:

venv\Scripts\activate


4. Install dependencies:

pip install -r requirements.txt

Usage
1. Prepare your dataset:
- Place your raw CSV files in data/raw/
- Run preprocessing:

python src/datapreprocessing.py

2. Train the baseline model:

python src/train_model.py

3. Run the Streamlit app:

streamlit run app.py

- Open the URL shown in the terminal (usually http://localhost:8501)
- Enter sample text to test safe/unsafe detection, escalation detection, crisis intervention, and content filtering.

### AI Model Details ###
- Baseline Model: TF-IDF vectorizer + Logistic Regression for abuse detection.
- Escalation & Crisis Detection: Pretrained Hugging Face NLP pipelines for real-time inference.
- Content Filtering: Rule-based checks for age-appropriate content.
- Integration: Models combined in the Streamlit app simulate a real-time safety system.

### Evaluation ###
- The baseline model outputs precision, recall, and F1-score after training.
- Metrics are printed in the terminal after running train_model.py.
- Edge cases (slang, misspellings, multilingual text) may require further fine-tuning for production.

Notes
- The virtual environment (venv/) is ignored in Git. Use requirements.txt to recreate dependencies.
- The project is designed for modularity, allowing easy extension with additional models or APIs.
- Ethical considerations such as bias mitigation and fairness should be addressed in real deployments.

License
This project uses publicly available datasets and is intended for educational purposes.
