# Bank Marketing Campaign Analysis

## Project Overview
This project aims to compare the performance of K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines classifiers on a dataset related to marketing bank products over the telephone. The goal is to optimize the efficiency and effectiveness of the bank's telemarketing campaigns.

Our dataset originates from the UCI Machine Learning repository ([link](https://archive.ics.uci.edu/ml/datasets/bank+marketing)). It comprises data from a Portuguese banking institution, detailing the results of multiple marketing campaigns. The analysis is informed by the accompanying article [CRISP-DM-BANK.pdf] for detailed context on the data and features. The data represents **17** distinct marketing campaigns.

## Business Objective
The primary business objective is to **optimize the efficiency and effectiveness of the bank's telemarketing campaigns for term deposits by identifying potential clients who are most likely to subscribe.** This involves developing a predictive model to accurately identify characteristics of clients likely to subscribe, thereby enabling targeted marketing efforts, reducing operational costs, improving conversion rates, and enhancing client experience.

### Machine Learning Task and Target Variable
*   **Machine Learning Task**: Binary classification.
*   **Target Variable**: `y` (client subscribed to a term deposit: 'yes' or 'no').
*   **Business Context of a 'Positive' Prediction**: A 'positive' prediction (`y='yes'`) signifies a client **will subscribe**, representing a successful outcome for the bank.

### Business Value and Impact
Successfully predicting potential subscribers offers substantial value through:
*   **Reduced Marketing Costs**: Targeting only promising leads minimizes resource expenditure.
*   **Improved Campaign Efficiency**: Strategic allocation of resources for higher success rates.
*   **Increased Term Deposit Subscriptions**: Boosting conversion rates and revenue.
*   **Better Allocation of Human Resources**: Focusing sales teams on high-probability leads.

### Deployment and Actionability
The predictive model will primarily be used for **pre-campaign targeting** to generate prioritized lists of potential subscribers, optimizing resource allocation. It could also provide **in-campaign guidance** by highlighting key customer segments or suggesting talking points based on client profiles. The `duration` feature has been excluded to prevent data leakage for realistic predictive models.

### Business Success Metrics
Success will be measured by:
*   **Increase in Term Deposit Subscription Rate (Conversion Rate)**
*   **Improved Return on Investment (ROI) for Telemarketing Campaigns**
*   **Reduction in Cost Per Acquisition (CPA)**
*   **Enhanced Telemarketing Resource Efficiency**

## Data Source
**Dataset**: `bank-additional-full.csv` from the UCI Machine Learning repository: [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing).

## Key Data Analysis Findings
*   **Prevalence of 'Unknown' Values:** The 'default' feature exhibits 20.87% 'unknown' values, while 'education' has 4.20%, and 'housing'/'loan' both have 2.40%. 'job' and 'marital' have lower percentages.
*   **Target Variable Imbalance:** The target variable 'y' is highly imbalanced, with 88.73% 'no' and 11.27% 'yes', indicating the 'no' class is almost 8 times more frequent.
*   **Numerical Feature Correlations:** 'duration' shows a strong positive correlation with 'y' (0.41), but was discarded due to data leakage. 'euribor3m', 'emp.var.rate', and 'nr.employed' show strong negative correlations with 'y'. Several economic indicators are highly inter-correlated.
*   **Discarding 'duration' Feature:** The `duration` feature was discarded to ensure a realistic predictive model, as its value is only known after the target outcome 'y' has occurred.

## Methodology and Approach
1.  **Data Loading and Initial Exploration**: The `bank-additional-full.csv` dataset was loaded into a Pandas DataFrame. Initial checks for missing values (explicit NaNs) were performed, and categorical features with 'unknown' values were identified and quantified.
2.  **Feature Understanding and Visualization**: Categorical and numerical feature distributions were visualized, and correlations among numerical features and with the target variable were analyzed. The severe class imbalance in the target variable `y` was identified.
3.  **Feature Engineering (Bank Client Data Only)**: Only 'bank client data' features (`age`, `job`, `marital`, `education`, `default`, `housing`, `loan`) were selected for initial modeling. Categorical features were one-hot encoded, treating 'unknown' as distinct categories. The 'age' feature was scaled using `StandardScaler`.
4.  **Train/Test Split**: The processed features (`X_processed`) and the target variable (`y_encoded`) were split into training and testing sets (80/20 ratio) using `train_test_split` with `stratify=y` to preserve class distribution due to imbalance, and a `random_state` for reproducibility.
5.  **Baseline Model**: A baseline accuracy was established by predicting the majority class, which was 0.8873.
6.  **Initial Model Comparison**: Logistic Regression, K Nearest Neighbor, Decision Tree, and Support Vector Machine models were trained with default settings (and `max_iter=1000` for SVM). Initial evaluation using accuracy showed all models struggled to outperform the baseline, indicating issues with class imbalance and a need for more appropriate metrics.
7.  **Performance Metric Adjustment and Hyperparameter Tuning**: Performance metrics were shifted to focus on Precision, Recall, F1-score, and AUC-ROC for the minority ('yes') class. Hyperparameter tuning was performed using `GridSearchCV` for each model, incorporating `class_weight='balanced'` for Logistic Regression, Decision Tree, and SVM to address class imbalance.

## Model Performance Summary

| Model | Train Time | Precision (Yes) | Recall (Yes) | F1-score (Yes) | AUC-ROC |
|:-----------------------------|-----------:|----------------:|-------------:|---------------:|--------:|
| Tuned Logistic Regression |   0.073678 |        0.157351 |     0.640086 |       0.252605 |  0.649509 |
| Tuned Decision Tree |   0.087331 |        0.167960 |     0.536638 |       0.255844 |  0.636091 |
| Tuned K Nearest Neighbor |   0.013101 |        0.223301 |     0.099138 |       0.137313 |  0.571774 |
| Tuned Support Vector Machine |  56.869225 |        0.112649 |     1.000000 |       0.202487 |  0.510285 |

### Key Insights from Comparison
*   **Best F1-score for Minority Class**: The **Tuned Decision Tree** achieved the highest F1-score (0.2558) for the 'yes' class, closely followed by **Tuned Logistic Regression** (0.2526). These models offer the best balance between precision and recall for identifying subscribers.
*   **AUC-ROC Performance**: **Tuned Logistic Regression** had the highest AUC-ROC (0.6495), indicating the best overall discriminatory power among the models.
*   **Recall vs. Precision Trade-off**: Tuned SVM achieved perfect recall (1.00) but at an extremely low precision (0.11), making it impractical due to massive false positives. Tuned KNN showed very low recall (0.099).
*   **Training Time**: Tuned Logistic Regression and Tuned Decision Tree offered fast training times, while Tuned SVM was significantly slower (56.87 seconds).

## Conclusion and Recommendations
Based on these evaluations, the **Tuned Logistic Regression** and **Tuned Decision Tree** models are the most promising. They provide the best balance of F1-score and AUC-ROC for the minority class while maintaining efficient training times. The `class_weight='balanced'` approach significantly improved minority class recall, though precision remains a challenge.

## Next Steps
*   **Focus on Logistic Regression and Decision Tree**: Concentrate further optimization efforts on these two models.
*   **Address Precision/Recall Trade-off**: Explore more sophisticated resampling techniques (e.g., SMOTE, ADASYN) or cost-sensitive learning to improve the balance between precision and recall, considering the business costs of false positives vs. false negatives.
*   **Enhance Feature Engineering**: Incorporate other feature groups from the dataset (contact information, other attributes, social and economic context, *excluding* `duration`) and perform robust feature selection to potentially improve model performance.

[Bank Marketing Campaign Models Analysis](https://github.com/tahirp/classification_benchmarking/blob/main/classifier_comparison.ipynb)
