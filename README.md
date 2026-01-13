# 🔮 Customer Churn Prediction System

## 📌 Project Overview
The **Customer Churn Prediction System** is an end-to-end Machine Learning solution designed to identify customers at risk of leaving (churning) and provide actionable insights for retention. Built using Python, Scikit-Learn, XGBoost, and Streamlit, this project demonstrates professional data science workflows from data processing to deployment.

## 🚀 Business Problem
Customer churn is a critical metric for subscription-based businesses. Acquiring a new customer can cost **5-25x more** than retaining an existing one. This system helps telecom companies:
1. **Predict** which customers are likely to churn.
2. **Understand** key drivers of churn (e.g., contract type, tenure).
3. **Take Action** by targeting high-risk segments with retention offers.

## 🛠️ Data Source
- **Dataset**: Telco Customer Churn (Kaggle)
- **Features**: 21 columns including Demographics, Service Details, and Account Information.
- **Target**: `Churn` (Yes/No)

## 📂 Project Structure
```
customer-churn-prediction/
├── data/                  # Dataset directory
├── notebooks/             # PROTOTYPING & ANALYSIS
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/                   # PRODUCTION CODE
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── model_evaluation.py
├── models/                # Saved models (.pkl)
├── app/                   # WEB DASHBOARD
│   └── streamlit_app.py
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## ⚙️ Installation & Setup

1. **Clone the repository** (or download files):
   ```bash
   git clone <repo_url>
   cd customer-churn-prediction
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Get Data**:
   - Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle.
   - Place it in the `data/` folder.

4. **Run Notebooks** (Optional, to retrain):
   - Navigate to `notebooks/` and run `03_model_training.ipynb` to generate `models/best_model.pkl`.

5. **Launch Dashboard**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## 📊 Model Performance
The best performing model (XGBoost) achieved:
- **AUC-ROC**: ~0.87
- **Accuracy**: ~82%
- **Recall**: ~78% (High recall ensures we catch most churners)

## 💡 Key Insights
- **Contract Type**: Month-to-month customers have the highest churn rate.
- **Tenure**: New customers (0-12 months) are most vulnerable.
- **Fiber Optic**: Customers with Fiber Optic internet churn more than DSL users.
- **Electronic Check**: This payment method is associated with higher churn.

## 🔮 Future Improvements
- Deploy to AWS/Heroku/Streamlit Cloud.
- Integrate with live database for real-time predictions.
- Add A/B testing framework for retention offers.

## 👨‍💻 Author
**Your Name**  
Data Scientist | Machine Learning Engineer  
[LinkedIn Profile] | [GitHub Profile]
