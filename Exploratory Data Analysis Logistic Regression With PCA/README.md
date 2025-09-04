 # EDA and Predictive Modeling using Logistic Regression with PCA
📌 Project Overview

This project applies Exploratory Data Analysis (EDA) and Logistic Regression with Principal Component Analysis (PCA) on the Adult Income Dataset. The goal is to predict whether an individual earns >50K or ≤50K per year based on demographic attributes.

📂 Dataset

Source: adult.csv

Shape: 32,561 rows × 15 columns

Target Variable: income (<=50K or >50K)

Features:

Age, Workclass, Education, Marital Status, Occupation, Relationship, Race, Sex, Capital Gain, Capital Loss, Hours per Week, Native Country, etc.

🔍 Exploratory Data Analysis (EDA)

Checked dataset shape, datatypes, and summary statistics.

Handled missing values (workclass, occupation, native.country) using mode imputation.

Replaced ? with NaN.

Encoded categorical variables with Label Encoding.

Applied Standard Scaling to normalize features.

🧮 Modeling
Logistic Regression (Baseline)

Trained using all features.

Accuracy Score: 0.8218

Logistic Regression with PCA

Reduced dimensionality using PCA.

Preserved 90% variance with 12 components.

Accuracy comparison with reduced features:

13 features → 0.8213

12 features → 0.8227

11 features → 0.8186

📊 Best Result: Logistic Regression with 12 PCA components gave accuracy 0.8227.

📈 Results & Insights

Logistic Regression performed well even with dimensionality reduction.

PCA helped optimize performance with fewer features while maintaining similar accuracy.

Key finding: 12 principal components are enough to preserve 90% of dataset variance.

# Run Jupyter Notebook
jupyter notebook eda-logistic-regression-pca.ipynb

🛠️ Tech Stack

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn (Logistic Regression, PCA, Preprocessing)
