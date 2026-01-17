## Machine Learning Assignment – 2  

## a. Problem Statement

The objective of this project is to design and implement an **end-to-end machine learning classification system** to predict the **presence or absence of heart disease** based on patient clinical and physiological attributes.

The task involves building multiple classification models, evaluating their performance using standard metrics, and deploying the trained models through an interactive **Streamlit web application**. The system allows users to upload unseen test data, select a trained model, and view prediction performance along with evaluation metrics and a confusion matrix.

This project demonstrates the complete machine learning lifecycle, including data preprocessing, feature engineering, model training, evaluation, and deployment.

---

## b. Dataset Description
 
The dataset used in this project was sourced from **Kaggle** and consists of patient-level medical records containing both **numerical** and **categorical** features relevant to heart disease diagnosis. The dataset is structured for a **binary classification problem**, where the target variable indicates the presence or absence of heart disease.

### Key Characteristics:
- **Data source:** Kaggle  
- **Number of input features:** 13  
- **Target variable:** Binary  
  - `0` → No heart disease  
  - `1` → Presence of heart disease  

### Feature Types:
- **Numerical features:**  
  Age, resting blood pressure, cholesterol level, maximum heart rate, ST depression (oldpeak)

- **Categorical features:**  
  Sex, chest pain type, fasting blood sugar, resting ECG, exercise-induced angina, slope of ST segment, number of vessels colored by fluoroscopy, thalassemia

Categorical features were transformed using **One-Hot Encoding**, while numerical features were standardized using **StandardScaler**. All preprocessing steps were integrated into a unified pipeline to ensure consistent transformation during both training and deployment.

The dataset was split into training and testing subsets, and the test split was saved as a separate CSV file and reused in the Streamlit application to ensure consistent evaluation between offline experiments and deployment.

---

## c. Models Used 

The following six classification models were implemented using the same dataset and preprocessing pipeline:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

Each model was evaluated using the following metrics:
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

### Model Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.873 | 0.944  | 0.855  | 0.904 | 0.897 | 0.747 |
| Decision Tree | 0.985 | 0.986 | 1.000 | 0.971 | 0.985 | 0.971 |
| KNN | 0.863 | 0.970 | 0.847 | 0.895 | 0.870 | 0.727 |
| Naive Bayes | 0.824 | 0.870 | 0.811 | 0.857 | 0.833  | 0.650 |
| Random Forest (Ensemble) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| XGBoost (Ensemble) | 1.000 | 1.000 | 1.000 | 1.000 |1.000 | 1.000 |

All models were trained and evaluated using the same train-test split to ensure a fair comparison.

---

## d. Observations on Model Performance 

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Logistic Regression achieved strong baseline performance with high AUC and recall, indicating effective class separation and good detection of positive cases. Its balanced precision and recall make it a reliable and interpretable model. |
| Decision Tree | The Decision Tree model delivered very high accuracy and recall, showing its ability to capture complex non-linear patterns. However, its near-perfect performance may indicate potential overfitting. |
| KNN | KNN demonstrated good predictive performance with a high AUC and balanced precision–recall. Its performance depends heavily on feature scaling and the choice of k value. |
| Naive Bayes | Naive Bayes showed moderate but consistent performance. Despite its strong independence assumptions, it provided reasonable accuracy and recall with low computational cost. |
| Random Forest (Ensemble) | Random Forest achieved perfect performance across all metrics, highlighting the effectiveness of ensemble learning. However, such results may suggest overfitting and should be interpreted cautiously. |
| XGBoost (Ensemble) | XGBoost also produced perfect scores, demonstrating its powerful gradient boosting mechanism and ability to model complex feature interactions, though the possibility of overfitting exists. |
