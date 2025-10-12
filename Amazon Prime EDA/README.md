# 🎬 Amazon Prime EDA

An interactive Exploratory Data Analysis (EDA) dashboard built with Streamlit, providing engaging and insightful visualizations on the Amazon Prime Video content dataset.
This project uncovers patterns in content types, genres, release trends, global distributions, and ratings — all through a sleek, user-friendly interface.

# 🚀 Features

✅ Interactive Dashboard

Real-time filters for content type (Movies / TV Shows)

Sidebar statistics and quick insights

# ✅ Visual Analytics

📊 Type Distribution (Movies vs TV Shows)

🌍 Top Content-Producing Countries

📅 Release Year Trends (1990–2024)

🧠 Word Cloud of Popular Themes & Descriptions

⭐ Top Rated Movies (Numeric Analysis)

🎞️ Maturity Rating Distribution

🧭 Amazon Historical Timeline Visualization

📈 Cumulative Growth of Movie Releases

# ✅ Dynamic Word Cloud

Uses a custom mask image (prime.jpeg) for an aesthetic design

Styled with an attractive blue color palette

# ✅ Performance Optimized

Data loading cached using @st.cache_data

Exception handling and fallback visualizations

# 🧩 Tech Stack
Tool / Library	Purpose
Python	Core language
Streamlit	Interactive dashboard
Pandas, NumPy	Data manipulation
Matplotlib, Seaborn	Data visualization
WordCloud, PIL	Word cloud generation
Collections, re, os	Data parsing and utilities


# ⚙️ Installation & Usage

🔹 Step 1: Install Dependencies
pip install -r requirements.txt

🔹 Step 2: Run the Streamlit App
streamlit run app.py

🔹 Step 3: Open in Browser

👉 http://localhost:8501

# 📊 Dataset Overview

The dataset amazon_prime_titles.csv contains metadata for thousands of Amazon Prime titles, including:

Title, Type, Director, Cast, Country

Release Year, Rating, Duration

Genre(s), Description

# 🧠 Insights Derived

Movies dominate over TV Shows in the Prime Video catalog.

USA, India, and UK are top producers of Prime content.

2015–2021 saw a sharp growth in new content releases.

13+ and 16+ are the most common maturity ratings.

The Word Cloud reveals recurring themes like Drama, Comedy, and Documentary.

# 🖥️ Screenshots
Visualization	Description
🎞️ Type Distribution	Comparison of Movies vs TV Shows
🌍 Top Countries	Top 10 content-producing countries
☁️ Word Cloud	Popular themes and keywords
📈 Yearly Trends	Growth of releases over time
🧩 Maturity Analysis	Distribution of content ratings
📦 Requirements
streamlit
pandas
numpy
matplotlib
seaborn
Pillow
wordcloud

# 💡 Future Enhancements

Add interactive genre filters

Integrate IMDb ratings API for live updates

Deploy app on Streamlit Cloud / Render

Include AI-powered content recommendations


<img width="1918" height="932" alt="Screenshot (185)" src="https://github.com/user-attachments/assets/51bd9b48-ce0d-4c73-ad79-a91e01b54a42" />

