# 💳 Transactional Fraud Detection using Machine Learning

---

## 📌 Project Overview

Financial fraud is a major global challenge, causing billions of dollars in losses each year and significantly impacting customer trust in digital financial systems. With the rapid growth of online transactions, detecting fraudulent activities has become increasingly complex, as fraudulent transactions are rare and often hidden within a vast volume of legitimate transactions.

This project develops a machine learning-based fraud detection system using historical credit card transaction data. The system is designed to analyze transaction patterns, identify anomalies, and distinguish fraudulent transactions from normal ones.

A key focus of this project is handling the **highly imbalanced nature of the dataset**, where fraudulent transactions represent only a small fraction of total data. The solution emphasizes **real-world applicability**, prioritizing the detection of fraud (high recall) while maintaining a reasonable balance with precision.

The final outcome is a predictive model that can be integrated into real-time fraud detection systems, enabling organizations to proactively flag suspicious activities and reduce financial losses.

---

## 🎯 Objective

The primary objective of this project is to design and develop a robust machine learning model capable of accurately identifying fraudulent financial transactions.

### 🔍 Specific Goals:

- Analyze transaction data to uncover patterns and characteristics of fraudulent behavior  
- Handle extreme class imbalance effectively using appropriate techniques  
- Build and compare multiple classification models  
- Optimize model performance with a strong focus on **recall**, ensuring fraudulent transactions are not missed  
- Maintain a balance between recall and precision to minimize false alarms  
- Evaluate models using comprehensive performance metrics  
- Select the most suitable model for real-world deployment  
- Provide actionable insights to improve fraud detection strategies  

---

## 📊 Dataset

- **Source:** Kaggle Credit Card Fraud Dataset
- Kaggle Credit Card Fraud Detection Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (An excellent, standard dataset
for this problem).
- **Total Transactions:** 284,807  
- **Fraud Cases:** 492 (0.17%)  

Pandas Documentation: https://pandas.pydata.org/docs/
● Seaborn Tutorial: https://seaborn.pydata.org/tutorial.html
● Scikit-learn Guide to Classification:
https://scikit-learn.org/stable/supervised_learning.html#supervised-learning


### ⚠️ Key Challenge:
> The dataset is highly imbalanced, making fraud detection a complex classification problem.

---

## 🧹 Data Preprocessing

The dataset was prepared using the following steps:

- Data cleaning and consistency checks  
- Handling missing values  
- Feature scaling and normalization  
- Addressing class imbalance  
- Removing irrelevant features  

---

## 📈 Exploratory Data Analysis (EDA)

EDA was performed to identify patterns and anomalies in transaction data.

### 🔍 Key Insights:

- Fraudulent transactions are extremely rare compared to normal transactions  
- Fraud often involves unusual transaction amounts or patterns  
- Behavioral anomalies play a significant role in detecting fraud  
- Certain features show strong correlation with fraudulent activity  

Visualizations such as histograms, boxplots, and heatmaps were used to validate these findings.

---

## 🤖 Machine Learning Models

The following models were implemented:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Extra Trees Classifier  
- XGBoost ⭐ *(Best Performing Model)*  

---

## ⚙️ Model Evaluation

Models were evaluated using multiple performance metrics:

- Precision  
- Recall ⭐ *(Most important metric)*  
- F1-score  
- ROC-AUC  
- PR-AUC  
- MCC (Matthews Correlation Coefficient)  

### 📌 Why Recall is Important?

In fraud detection:
> Missing a fraudulent transaction (False Negative) results in direct financial loss.

Therefore, maximizing recall is critical.

---

## 🏆 Results

| Metric | XGBoost |
|--------|--------|
| Precision | 0.94 |
| Recall | 0.76 |
| F1-score | 0.84 |
| ROC-AUC | 0.9723 |

👉 XGBoost achieved the best balance between fraud detection and minimizing false positives.

---
POWER BI DASHBOARD LINK(.pbix):
https://drive.google.com/drive/folders/1akA7jgVMJIIlCExDYtB4Z1zjzAWv4gNH?usp=sharing

## 🔍 Feature Importance

Key features influencing fraud detection include:

- Transaction Amount  
- Time-based patterns  
- Behavioral anomalies  
- Complex feature interactions  

---

## 💼 Business Impact

This system provides practical value by:

- Detecting fraudulent transactions in near real-time  
- Reducing financial losses  
- Improving security systems  
- Enhancing customer trust  
- Supporting automated fraud monitoring  

---

## 🚀 How It Works

1. Input transaction data  
2. Model analyzes transaction patterns  
3. Predicts probability of fraud  
4. Flags suspicious transactions  

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost  

---

## ▶️ Installation & Usage

```bash
pip install -r requirements.txt
jupyter notebook
