# Exploratory-Data-Analysis

🧹 Exploratory Data Analysis – Employee Dataset
📌 Project Overview
This project demonstrates Exploratory Data Analysis (EDA) and Data Cleaning on an employee dataset with inconsistent, missing, and noisy values. The notebook walks through data preprocessing, handling missing values, type conversions, and visualization using Python libraries like Pandas, NumPy, Matplotlib, and Seaborn.

✅ Key Objectives
Load and inspect raw employee data

Clean noisy text (remove unwanted characters)

Extract meaningful numeric values from mixed data

Handle missing values with mean and mode imputation

Convert data types (object → int, category)

Visualize salary distribution & experience vs salary trends

Prepare data for further analysis or modeling

📊 Dataset Details
Column	Description	Issues Found
Name	Employee Name	Extra symbols (e.g. Mike#$)
Domain	Work domain (e.g. Data Science)	Noisy text like Datascience#$
Age	Employee age in years	Missing & mixed format (45'yr)
Location	Employee city	Missing values
Salary	Employee salary	Special characters (10%%000)
Exp	Work experience in years	Missing & symbols (4>yrs)

🗂 Sample Raw Data
Name	Domain	Age	Location	Salary	Exp
Mike	Datascience#$	34 years	Mumbai	5^00#0	2+
Teddy^	Testing	45' yr	Bangalore	10%%000	<3
Umar#r	Dataanalyst^^#	NaN	NaN	1$5%000	4> yrs

🛠 Steps Performed
Data Inspection

Checked shape, column names, and data types

Identified missing values & noisy data

Data Cleaning

Removed special characters using regex

Extracted numeric values for Age, Salary, and Experience

Standardized categorical values

Missing Value Treatment

Filled Age & Experience using mean imputation

Filled Location using mode (most frequent city)

Data Type Conversion

Converted Age, Salary, Exp → int

Converted Name, Domain, Location → category

Visualization

Salary distribution (sns.distplot)

Salary vs Experience (sns.lmplot)

📈 Final Cleaned Dataset
Name	Domain	Age	Location	Salary	Exp
Mike	Datascience	34	Mumbai	5000	2
Teddy	Testing	45	Bangalore	10000	3
Umar	Dataanalyst	50	Bangalore	15000	4
Jane	Analytics	50	Hyderbad	20000	4
Uttam	Statistics	67	Bangalore	30000	5
Kim	NLP	55	Delhi	60000	10

📂 Folder Structure
bash
Copy
Edit
📁 Exploratory-Data-Analysis
 ├── 📄 Rawdata.xlsx        # Original messy data
 ├── 📄 clean_data.csv      # Cleaned dataset
 ├── 📓 eda_notebook.ipynb  # Jupyter notebook with EDA steps
 ├── 📊 salary_dist.png     # Visualization: Salary Distribution
 └── README.md             # Project Documentation
🖥️ Tech Stack
Python 3.x

Libraries: Pandas, NumPy, Matplotlib, Seaborn

🚀 How to Run
Clone this repo

Install required libraries

pip install pandas numpy matplotlib seaborn
Open the Jupyter Notebook

jupyter notebook eda_notebook.ipynb
Run cells step by step

📌 Future Improvements
Automate cleaning using custom functions

Apply outlier detection

Add more visualizations
