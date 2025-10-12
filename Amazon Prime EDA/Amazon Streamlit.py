import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from PIL import Image 
import warnings
from collections import Counter
import re
import os
import matplotlib.lines as lines

st.set_page_config(
    page_title="🎬 Amazon Prime EDA",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Loads the dataset, performs initial cleaning, and converts the 'rating' column."""
    
    FILE_PATH = r"C:\Users\JANHAVI\Desktop\FSDS1\Dataset\amazon_prime_titles.csv" 
    
    try:
        if not os.path.exists(FILE_PATH):
             st.error(f"Error: File not found. The application is looking for the file at: **{FILE_PATH}**")
             return None
             
        df = pd.read_csv(FILE_PATH)
        
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the data: {e}")
        return None

    # Data Cleaning and Preprocessing
    df['director'] = df['director'].fillna('No Director')
    df['cast'] = df['cast'].fillna('No Cast')
    df['country'] = df['country'].fillna('Unknown')
    df['rating'] = df['rating'].fillna('Unknown') 
    
    df['numeric_rating'] = pd.to_numeric(df['rating'], errors='coerce')

    df['date_added'] = df['date_added'].fillna('January 1, 1900') 
    df['first_genre'] = df['listed_in'].apply(lambda x: x.split(',')[0].strip())

    return df

# --- Utility Functions ---

def get_top_n_elements(df, column, n=10):
    """Extracts all comma-separated elements, counts them, and returns the top N."""
    if column not in df.columns:
        return pd.DataFrame({column: [], 'Count': []})

    all_elements = df[column].dropna().astype(str).str.split(',\s*').explode()
    
    if column == 'country':
        all_elements = all_elements.replace('United States', 'USA')
    
    element_counts = Counter(all_elements)
    top_n = pd.DataFrame(element_counts.most_common(n), columns=[column, 'Count'])
    return top_n

# -------------------------------------------------------------------
# --- VISUALIZATION FUNCTIONS  ---
# -------------------------------------------------------------------

def plot_type_distribution(df):
    type_counts = df['type'].value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    colors = ['#08AAE3', '#233D53']
    
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        type_counts['Count'], labels=type_counts['Type'], autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'}
    )
    for autotext in autotexts: autotext.set_color('white')
    ax.axis('equal') 
    ax.set_title('Content Type Distribution', fontsize=16, color='#233D53')
    st.pyplot(fig)

def plot_top_countries(df, n=10):
    df_countries = get_top_n_elements(df, 'country', n=n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x='Count', y='country', data=df_countries, palette=sns.light_palette("#08AAE3", n_colors=n, reverse=True), ax=ax
    )
    ax.set_title(f'Top {n} Content-Producing Countries', fontsize=16, color='#233D53')
    ax.set_xlabel('Number of Titles', fontsize=12)
    ax.set_ylabel('Country', fontsize=12)
    st.pyplot(fig)

def plot_yearly_trend(df):
    yearly_count = df['release_year'].value_counts().sort_index()
    yearly_count = yearly_count[yearly_count.index >= 1990] 
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly_count.index, yearly_count.values, marker='o', linestyle='-', color='#233D53')
    ax.set_title('Content Release Trend (Titles Per Year)', fontsize=16, color='#233D53')
    ax.set_xlabel('Release Year', fontsize=12)
    ax.set_ylabel('Number of Titles', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def generate_custom_wordcloud(df):
    """Generates a highly-styled word cloud, using the 'prime.jpeg' mask."""
    text = " ".join(df['listed_in'].dropna().astype(str)) + " " + " ".join(df['description'].dropna().astype(str))
    
    custom_stopwords = STOPWORDS.union({'show', 'movie', 'series', 'season', 'drama', 'film', 's', 'one', 'new', 't', 'life', 'world', 'story', 'can', 'amazon', 'prime'})
    colors = ['#233D53', '#08AAE3', '#5684AE'] 
    custom_colormap = plt.cm.colors.LinearSegmentedColormap.from_list("", colors)
    
    mask_array = None
    mask_path = 'prime.jpeg' 
    try:
        mask_array = np.array(Image.open(mask_path))
    except FileNotFoundError:
        pass
    
    wordcloud = WordCloud(
        stopwords=custom_stopwords, background_color="white", width=1000, height=627, max_words=350,
        mask=mask_array, contour_color='#233D53', colormap=custom_colormap, collocations=True
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Content Word Cloud (Popular Themes & Genres)', fontsize=18, color='#233D53')
    plt.tight_layout(pad=0)
    st.pyplot(fig)

def plot_amazon_timeline():
    """Plots the historical timeline of Amazon and Prime Video."""
    tl_dates = ['1994\nAmazon founded', '1998\nAcquires IMDB', '2005\nAmazon Prime Launched',
                '2013\nAmazon launches in India', '2022\nAmazon acquires MGM', '2024\n230 million Subscriptions']
    tl_x = [1, 2.5, 4.3, 6.3, 8.2, 10]
    tl_sub_x = [1.7, 3.4, 5, 7, 9]
    tl_sub_times = ["1997", "2003", "2007", "2014", '2023']
    tl_text = ["Amazon Ipo launched", "A9.com was Launched", "Amazon Music is launched", 
               "Acquires Twitch", "Prime Content Reached 210 countires"]

    fig, ax = plt.subplots(figsize=(15, 4), constrained_layout=True)
    ax.set_ylim(-2, 1.75)
    ax.set_xlim(0, 11)

    ax.axhline(0, xmin=0.1, xmax=0.9, color='#CEA968', zorder=1)
    ax.scatter(tl_x, np.zeros(len(tl_x)), s=120, color='#146eb4', zorder=2)
    ax.scatter(tl_x, np.zeros(len(tl_x)), s=30, color='#fafafa', zorder=3)
    ax.scatter(tl_sub_x, np.zeros(len(tl_sub_x)), s=50, color='#233D53', zorder=4)

    for x, date in zip(tl_x, tl_dates):
        ax.text(x, -0.70, date, ha='center', fontfamily='serif', fontweight='bold', color='#4a4a4a', fontsize=12)
    
    levels = np.zeros(len(tl_sub_x))
    levels[::2] = 0.5
    levels[1::2] = 0.3
    markerline, stemline, baseline = ax.stem(tl_sub_x, levels)
    plt.setp(baseline, zorder=0)
    plt.setp(markerline, marker=',', color='#233D53')
    plt.setp(stemline, color='#4a4a4a')

    for x, time, txt in zip(tl_sub_x, tl_sub_times, tl_text):
        ax.text(x, 1.3 - 0.5, time, ha='center', fontfamily='serif', fontweight='bold', color='#4a4a4a', fontsize=11)
        ax.text(x, 1.3 - 0.6, txt, va='top', ha='center', fontfamily='serif', color='#4a4a4a')

    for spine in ["left", "top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title("Amazon Prime through the years", fontweight="bold", fontfamily='serif', fontsize=16, color='#233D53')
    ax.text(2.4, 1.57, "From Book Store to a global audience of over 230m people - Delivering Infinite Possibilities, One Click at a Time", fontfamily='serif', fontsize=12, color='#000000')

    st.pyplot(fig)


def plot_top_rated_movies(df):
    """Plots the Top 10 Rated Movies (using numeric_rating)."""
    df_movies = df[df['type'] == 'Movie'].copy()
    top_movies = df_movies.groupby('title')['numeric_rating'].mean().sort_values(ascending=False).head(10)
    
    color_map = ['#CEA968'] * 10
    if len(top_movies) >= 3:
        color_map[0] = color_map[1] = color_map[2] = '#233D53'

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.barh(top_movies.index, top_movies, height=0.5, edgecolor='darkgrey', linewidth=0.6, color=color_map)

    for i in top_movies.index:
        if pd.notna(top_movies[i]):
             ax.annotate(f'{top_movies[i]:.2f}', xy=(top_movies[i], i), va='center', ha='left', fontweight='light', fontfamily='serif')

    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)

    ax.set_yticklabels(top_movies.index, fontfamily='serif', rotation=0)
    fig.text(0.09, 1, 'Top 10 Movies on Prime', fontsize=15, fontweight='bold', fontfamily='serif')
    fig.text(0.09, 0.95, 'The three most Rated Movies have been highlighted.', fontsize=12, fontweight='light', fontfamily='serif')
    
    ax.grid(axis='x', linestyle='-', alpha=0.4)
    grid_x_ticks = np.arange(0, 10, 1)
    ax.set_xticks(grid_x_ticks)
    ax.set_axisbelow(True)
    plt.axvline(x=0, color='black', linewidth=1.3, alpha=.9)
    ax.tick_params(axis='both', which='major', labelsize=12)

    l1 = lines.Line2D([1, 1], [0, 1], transform=fig.transFigure, figure=fig, color='black', lw=0.7)
    fig.lines.extend([l1])
    ax.tick_params(axis=u'both', which=u'both', length=0)
    
    st.pyplot(fig)


def plot_rating_by_country(df):
    """Plots the Average Rating by Country as a Polar Bar Chart, or falls back to content count."""
    
    country_ratings = df.groupby('country')['numeric_rating'].mean().dropna().sort_values(ascending=False).head(10)
    num_bars = len(country_ratings)

    if num_bars == 0:
        #  Plotting Top 10 Countries by Content Count
        st.warning("Average Rating by Country: Insufficient numeric rating data. Displaying Top 10 Content Countries by Volume as fallback.")
        df_counts = get_top_n_elements(df, 'country', n=10)
        
        if df_counts.empty:
            st.error("No country data available for current filter.")
            return

        counts = df_counts['Count'].to_list()
        country_names = df_counts['country'].to_list()
        
        num_bars = len(counts)
        angles = np.linspace(0, 2 * np.pi, num_bars, endpoint=False).tolist()
        width = np.pi / num_bars
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw=dict(polar=True))
        ax.axis('off')
        
        colors = ['#233D53'] * num_bars
        max_count = max(counts) if counts else 1
        
        for angle, value, country_name in zip(angles, counts, country_names):
            ax.bar(angle, value, width=width, color=colors[0], alpha=0.7)
            ax.text(angle, max_count * 1.05, country_name.replace(' ', '\n'), ha='center', va='bottom', fontsize=8, fontfamily='serif', color='black', fontweight='bold')
            ax.text(angle, value + 5, str(value), ha='center', va='bottom', fontsize=9, fontfamily='serif', color='black')

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_title('Top 10 Content Countries (By Volume - Fallback)', fontsize=12, fontweight='bold', fontfamily='serif', loc='center')
        st.pyplot(fig)
        return

    # Original Polar Chart plotting logic 
    angles = np.linspace(0, 2 * np.pi, num_bars, endpoint=False).tolist()
    width = np.pi / num_bars

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw=dict(polar=True))
    ax.axis('off')

    for angle, value, country_name in zip(angles, country_ratings, country_ratings.index):
        rating_label = f'{value:.1f}'
        bar_color = '#08AAE3' if value > country_ratings.mean() else '#233D53'
        
        ax.bar(angle, value, width=width, color=bar_color)
        ax.text(angle, 5, country_name.replace(' ', '\n'), ha='center', va='bottom', fontsize=8, fontfamily='serif', color='white', fontweight='bold')
        ax.text(angle, value + 0.5, rating_label, ha='center', va='bottom', fontsize=9, fontfamily='serif', color='black')

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(country_ratings.index, fontsize=10, fontfamily='serif')

    l1 = lines.Line2D([1, 1], [0, 1], transform=fig.transFigure, figure=fig, color='black', lw=0.7)
    fig.lines.extend([l1])
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.set_title('Average Rating by Top 10 Countries', fontsize=12, fontweight='bold', fontfamily='serif', loc='center')
    
    st.pyplot(fig)


def plot_cumulative_releases(df):
    """Plots the Cumulative Distribution of Movie Releases over time."""
    df_movies = df[df['type'] == 'Movie'].copy()
    
    df_1961_2000 = df_movies[df_movies['release_year'].between(1961, 2000)]
    df_2000_2021 = df_movies[df_movies['release_year'].between(2000, 2021)]

    count_by_year_1961_2000 = df_1961_2000.groupby('release_year').size().cumsum()
    count_by_year_2000_2021 = df_2000_2021.groupby('release_year').size().cumsum()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(count_by_year_1961_2000.index, count_by_year_1961_2000, color='#08AAE3', alpha=0.5, label='1961-2000')
    ax.fill_between(count_by_year_2000_2021.index, count_by_year_2000_2021, color='#233D53', alpha=0.5, label='2000-2021')

    ax.set_title('Cumulative Distribution of Movie Releases (1961-2021)', fontsize=14, fontweight='bold', fontfamily='serif')
    ax.set_xlabel('Year of Release', fontsize=12, fontfamily='serif')

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(0.13, 0.85, 'Movies added over time [Cumulative Total]', fontsize=15, fontweight='bold', fontfamily='serif')

    fig.text(0.3, 0.2, "Movies - 90's", fontweight="bold", fontfamily='serif', fontsize=15, color='#233D53')
    fig.text(0.7, 0.2, "Movies - 2000's", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')

    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.yaxis.tick_right()
    plt.tight_layout()
    st.pyplot(fig)


def plot_maturity_ratings(df):
    """Plots the distribution of Maturity Ratings."""
    df_movies = df[df['type'] == 'Movie'].copy()
    rating = df_movies.groupby('rating')['title'].count()

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors = ['#233D53' if rating.index[i] != '13+' else '#08AAE3' for i in range(len(rating))]
    ax.bar(rating.index, rating, width=0.5, color=colors, alpha=0.8, label='Movie')

    for rating_label, count in rating.items():
        ax.annotate(f"{count}",
                    xy=(rating_label, count + 60),
                    va='center', ha='center', fontweight='light', fontfamily='serif',
                    color='#4a4a4a')

    ax.yaxis.set_visible(False)

    for s in ['top', 'left', 'right', 'bottom']:
        ax.spines[s].set_visible(False)

    ax.legend().set_visible(False)
    fig.text(0.16, 1, 'Exploring Movie Maturity Ratings: A Comprehensive Analysis of Rating Distribution', fontsize=15, fontweight='bold', fontfamily='serif')

    st.pyplot(fig)


# -------------------------------------------------------------------
# --- Main Application Logic ---
# -------------------------------------------------------------------

def main():
    st.title("🎬 Amazon Prime Video Content Analysis")
    st.markdown("A deep dive into the Amazon Prime Video content library, featuring custom visualizations and a styled word cloud.")
    
    df = load_data()
    if df is None:
        return

    # --- Sidebar for Filtering/Controls ---
    st.sidebar.header("Data Filter")
    content_type = st.sidebar.radio(
        "Select Content Type:",
        ('All', 'Movie', 'TV Show'),
        index=0
    )
    
    if content_type != 'All':
        filtered_df = df[df['type'] == content_type].copy()
    else:
        filtered_df = df.copy()
        
    movie_df = df[df['type'] == 'Movie'].copy()

    st.sidebar.markdown("---")
    st.sidebar.metric("Total Titles Analyzed", len(filtered_df))

    # ---  Amazon Prime Historical Timeline ---
    st.header("1. Amazon Prime Historical Timeline")
    st.markdown("Key milestones in Amazon's history leading up to its streaming service.")
    plot_amazon_timeline() 

    st.markdown("---")

    # --- Distribution Overview (Existing EDA) ---
    st.header("2. Content Distribution & Basic EDA")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Type Distribution (Movies vs. TV Shows)")
        plot_type_distribution(filtered_df)
    
    with col2:
        st.subheader("Top Genres")
        df_genres = get_top_n_elements(filtered_df, 'first_genre', n=10)
        st.dataframe(df_genres.style.background_gradient(cmap='Blues'), use_container_width=True)
        st.markdown(f"**Total titles in view:** **{len(filtered_df)}**")

    st.markdown("---")

    # ---  Geographic and Temporal Analysis (All Over Visualization) ---
    st.header("3. Geographic and Temporal Insights")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Top Content-Producing Countries")
        plot_top_countries(filtered_df, n=10)
        
    with col4:
        st.subheader("Content Release Trend Over Time")
        plot_yearly_trend(filtered_df)

    st.markdown("---")

    # ---  Word Cloud (All Over Wordcloud) ---
    st.header("4. Custom Word Cloud: Popular Themes")
    st.markdown("This visualization aggregates the most frequent words from content genres and descriptions, using the attractive blue color palette.")
    generate_custom_wordcloud(filtered_df)

    st.markdown("---")

    # --- Advanced Rating & Release Analysis (New Plots) ---
    st.header("5. Advanced Content Analysis")
    
    st.subheader("Top 10 Rated Movies (Based on Available Data)")
    plot_top_rated_movies(movie_df)

    st.markdown("---")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("Average Rating by Top 10 Countries")
        plot_rating_by_country(movie_df) 
    
    with col6:
        st.subheader("Movie Maturity Rating Distribution")
        plot_maturity_ratings(movie_df)
        st.markdown('''
        **Insight:** The majority of movie ratings are for ages 13 and above, 
        with other age categories being evenly distributed.
        ''')
        
    st.markdown("---")
    st.subheader("Cumulative Movie Release Growth")
    plot_cumulative_releases(movie_df)

    st.markdown("---")

    # --- Raw Data Snapshot ---
    st.header("6. Raw Data Snapshot")
    st.dataframe(df.head())


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()