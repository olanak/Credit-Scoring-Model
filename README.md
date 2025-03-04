# **Credit Scoring Model for Bati Bank**

## **Overview**
This repository contains the code and resources for a credit scoring model developed as part of the Week 6 Challenge at 10 Academy. The goal of this project is to enable Bati Bank to offer a buy-now-pay-later service by assessing the creditworthiness of customers using data provided by an eCommerce platform.

The model leverages advanced machine learning techniques, including feature engineering, Weight of Evidence (WoE) binning, Information Value (IV) analysis, and model training with hyperparameter tuning. Additionally, a REST API is implemented to serve real-time predictions.

---

## **Business Context**
Bati Bank is partnering with an eCommerce company to provide customers with the ability to purchase products on credit if they qualify for the service. To ensure financial stability, the bank requires a robust credit scoring system that can classify users as high-risk or low-risk based on their transaction history.

The definition of "default" aligns with Basel II guidelines, which emphasize Probability of Default (PD), Loss Given Default (LGD), Exposure at Default (EAD), and Expected Loss (EL). The model focuses on PD estimation to predict the likelihood of customer default.

---

## **Key Features**
- **RFMS Framework**: Recency, Frequency, Monetary, and Status scores are calculated to classify users as high-risk or low-risk.
- **Feature Engineering**: New features such as total transaction amount, average transaction amount, standard deviation of transactions, and temporal features (hour, day, month, year) are created.
- **Weight of Evidence (WoE) Binning**: Categorical variables are transformed into numerical representations using WoE binning to improve model interpretability.
- **Model Training**: Multiple models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) are trained and evaluated using metrics like Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
- **Hyperparameter Tuning**: Grid Search is used to optimize the Random Forest model's performance.
- **REST API**: A Flask-based API serves the trained model for real-time predictions.
- **Deployment**: The API and frontend are deployed on Render for accessibility.

---

## **Dataset**
The dataset provided by the eCommerce platform contains transactional data with the following key fields:
- **TransactionId**: Unique identifier for each transaction.
- **BatchId**: Identifier for batches of transactions.
- **AccountId**: Unique customer account identifier.
- **SubscriptionId**: Unique subscription identifier.
- **CustomerId**: Unique identifier for customers.
- **CurrencyCode**: Currency used in the transaction.
- **CountryCode**: Numerical code representing the country.
- **ProductId**: Item name being bought.
- **ProductCategory**: Broader category of the product.
- **ChannelId**: Identifies the channel used (web, Android, iOS, pay later, etc.).
- **Amount**: Transaction amount (positive for debits, negative for credits).
- **Value**: Absolute value of the transaction amount.
- **TransactionStartTime**: Timestamp of when the transaction started.
- **PricingStrategy**: Pricing category for merchants.
- **FraudResult**: Fraud status of the transaction (`1` for fraud, `0` for no fraud).

---

## **Project Structure**
The repository is organized into the following directories:

### **1. notebooks/**  
Contains Jupyter notebooks for exploratory data analysis (EDA), feature engineering, RFMS calculations, WoE binning, and model training.  
- `task_1_understanding_credit_risk.ipynb`: Understanding credit risk concepts and references.
- `task_2_exploratory_data_analysis.ipynb`: EDA, summary statistics, distributions, and correlations.
- `task_3_feature_engineering.ipynb`: Creation of aggregate and temporal features.
- `task_4_default_estimator_and_woe_binning.ipynb`: RFMS framework, WoE binning, and IV calculations.
- `task_5_model_selection_and_training.ipynb`: Model training, evaluation, and hyperparameter tuning.
- `task_6_model_serving_api_call.ipynb`: API design and testing.

### **2. data/**  
Stores raw and processed datasets.  
- **raw/**: Original dataset provided by the eCommerce platform.
- **processed/**: Cleaned and engineered dataset ready for modeling.

### **3. models/**  
Holds the trained machine learning models and scaler used during preprocessing.  
- `best_random_forest_model.pkl`: Final tuned Random Forest model.
- `scaler.pkl`: StandardScaler instance for numeric feature scaling.

### **4. scripts/**  
Includes Python scripts for deploying the API.  
- `api_server.py`: Flask-based REST API for serving predictions.

### **5. static/**  
Stores the frontend files for user interaction.  
- `index.html`: Main HTML file for the credit scoring app.
- `style.css`: CSS file for styling the frontend.

---

## **Installation**
To run the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/olanak/credit-scoring-model.git
   cd credit-scoring-model
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Models from Google Drive**:
   - Replace `YOUR_MODEL_FILE_ID` and `YOUR_SCALER_FILE_ID` with the actual file IDs from Google Drive.
   - Use the following commands to download the models:
     ```bash
     gdown https://drive.google.com/uc?id=1TV29ZdUREXSB3rnX1mxbiLeKnV1H9BTj -O models/best_random_forest_model.pkl
     gdown https://drive.google.com/uc?id=1IsZ3ed0tGfjnK_wnrSFyrFaX_bN9Msoc -O models/scaler.pkl
     ```

5. **Run the API**:
   ```bash
   python scripts/api_server.py
   ```
   The API will be accessible at `http://localhost:5000/predict`.

6. **Open the Frontend**:
   Open `static/index.html` in your browser to interact with the credit scoring app.

---

## **Usage**
### **Frontend**
- Access the application via the deployed URL (e.g., `https://credit-scoring-app.onrender.com`).
- Enter transaction-related details (e.g., total transaction amount, frequency, year, etc.) and submit the form.
- The app will display the predicted credit risk probability and classification (low-risk or high-risk).

### **API**
- Use the `/predict` endpoint to send POST requests with JSON input data.
- Example request:
  ```bash
  curl -X POST https://credit-scoring-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '[
      {
          "TotalTransactionAmount": 5000,
          "AverageTransactionAmount": 500,
          "TransactionCount": 10,
          "StdDevTransactionAmount": 100,
          "TransactionYear": 2023,
          "ProductCategory_financial_services_woe": 0,
          "ChannelId_ChannelId_3_woe": 0,
          "ProviderId_encoded_woe": 0
      }
  ]'
  ```
- Example response:
  ```json
  {
      "risk_probability": [0.1234]
  }
  ```

---

## **Results**
### **Model Performance**
| Model              | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Decision Tree      | 0.9935   | 0.9830    | 0.9764 | 0.9797   | 0.9866  |
| Random Forest      | 0.9809   | 0.9745    | 0.9051 | 0.9385   | 0.9503  |

After hyperparameter tuning, the Random Forest model achieved the best performance with an **ROC-AUC score of 0.9978**.

---

## **Technologies Used**
- **Python Libraries**: Pandas, NumPy, Scikit-learn, Flask, Joblib, Gdown.
- **Machine Learning**: Logistic Regression, Decision Trees, Random Forest, Gradient Boosting Machines (GBM).
- **Feature Engineering**: RFMS framework, WoE binning, Information Value (IV) analysis.
- **API Development**: Flask for creating the REST API.
- **Deployment**: Render for hosting the API and frontend.

---

## **Contributions**
This project was developed as part of the 10 Academy Week 6 Challenge. Contributions were made by:
- **Mahlet**: Introduction to the challenge.
- **Elias**: Feature engineering and WoE/IV calculations.
- **Rediet**: Model serving and deployment.
- **Emtinan**: Model training, hyperparameter tuning, and evaluation.

---

## **References**
1. **Credit Risk Analysis**:
   - [Corporate Finance Institute - Credit Risk](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)
   - [Basel II Capital Accord](https://www.bis.org/bcbs/ca.htm)

2. **Feature Engineering**:
   - [Weight of Evidence (WoE) and Information Value (IV)](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
   - [Scorecardpy Documentation](https://scorecardpy.readthedocs.io/en/latest/)

3. **Model Deployment**:
   - [Flask Documentation](https://flask.palletsprojects.com/en/2.3.x/)
   - [Render Deployment Guide](https://render.com/docs/deploy-flask)


## **Future Work**
1. **Improve Model Interpretability**:
   - Implement SHAP or LIME to explain model predictions.
2. **Enhance Feature Set**:
   - Incorporate additional features such as customer demographics or external economic indicators.
3. **Optimize Deployment**:
   - Containerize the API using Docker for consistent deployment across environments.
4. **Expand Business Use Cases**:
   - Develop models for predicting optimal loan amounts and durations.

---

## **Contact**
For questions or feedback, feel free to reach out:
- Email: your-email@example.com
- LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/olana-kenea)

---

## **License**
This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code as needed.

---
