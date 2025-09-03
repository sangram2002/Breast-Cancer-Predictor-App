# Breast-Cancer-Predictor-App
An interactive machine learning app using Logistic Regression and Streamlit to predict breast cancer (benign vs malignant) with visualization support.

# 🩺 Breast Cancer Prediction App

An interactive **machine learning web app** that predicts whether a breast mass is **benign** or **malignant** based on cell nuclei measurements.  
Built with **Python, Scikit-learn, Streamlit, and Plotly**.

---

## 📌 Features
- Logistic Regression model trained on the Breast Cancer Wisconsin dataset.
- Data preprocessing: feature scaling, categorical encoding, and cleaning.
- Interactive **Streamlit dashboard** with sliders for all cell measurement features.
- **Radar chart visualization** of mean, standard error, and worst feature values.
- Real-time predictions with probabilities (benign vs malignant).
- Model persistence with `pickle` for reproducibility.

---

## 🚀 Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn)
- **Streamlit** (for UI)
- **Plotly** (for radar charts)
- **Pickle** (for model saving/loading)

---

## 📊 Model Performance
- Logistic Regression achieved **high accuracy** on the test dataset.
- Evaluation metrics include accuracy score, classification report, and confusion matrix.

---

## 📂 Project Structure
├── data.csv          # Dataset
├── model.pkl         # Trained ML model
├── scaler.pkl        # Scaler object
├── app.py            # Streamlit app
├── train.py          # Model training script
├── style.css         # Custom styles
└── README.md         # Project documentation
⚡ How to Run

Clone the repository:
bashgit clone https://github.com/yourusername/breast-cancer-predictor.git
cd breast-cancer-predictor

Install dependencies:
bashpip install -r requirements.txt

Run the Streamlit app:
bashstreamlit run app.py


🧑‍⚕️ Disclaimer
This tool is intended for educational purposes only. It can support medical professionals but should not replace professional medical diagnosis.
📸 Demo Screenshot
(Add a screenshot/gif of your Streamlit app here!)
🙌 Acknowledgements

Dataset: 
[Link Text](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic "Breast Cancer Wisconsin (Diagnostic) Dataset - UCI ML Repository")
Inspired by: Real-world applications of machine learning in healthcare
