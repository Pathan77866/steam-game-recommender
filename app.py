import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration
st.set_page_config(page_title="Game Recommender", page_icon="🎮", layout="centered")

# Hide Streamlit's default watermark menu to keep it looking like a custom app
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 2. Caching the Heavy Math 
@st.cache_data
def load_engine():
    df = pd.read_csv('clean_games.csv')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['metadata_soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return df, cosine_sim

# 3. Build the UI Header 
st.title("🎮 Game Recommender")
st.subheader("What Should I Play Next?")
st.markdown("Select a game you love, and our engine will find similar titles based on genres and player tags.")
st.divider()

# Load the data behind the scenes
with st.spinner("Loading Recommendation Engine..."):
    df, cosine_sim = load_engine()

# 4. User Interface Controls 
game_list = sorted(df['Name'].tolist())
selected_game = st.selectbox("Search for a game:", game_list)

# Slider for number of recommendations
num_recs = st.slider("How many recommendations do you want?", min_value=1, max_value=10, value=5)

# 5. The Output Logic
if st.button("🔮 Find My Next Game"):
    
    # Bulletproof search: force both sides to be clean strings to ignore weird CSV formatting
    safe_names = df['Name'].astype(str).str.strip()
    search_term = str(selected_game).strip()
    
    try:
        # Run the math safely
        idx = safe_names[safe_names == search_term].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_matches = sim_scores[1:num_recs+1]
        
        st.subheader(f"Top Matches for **{selected_game}**:")
        st.write("")
        
        # Display the results with upgraded visual hierarchy
        for rank, match in enumerate(top_matches, 1):
            game_index = match[0]
            match_score = match[1]  # The raw decimal number (e.g., 0.85)
            match_percentage = match_score * 100
            
            # Pulling basic text data safely
            rec_title = df['Name'].iloc[game_index]
            rec_genre = str(df['Primary_Genre'].iloc[game_index]).title()
            
            # The fix for missing tags
            raw_tags = str(df['All_Tags'].iloc[game_index])
            if raw_tags == 'nan':
                rec_tags = "No tags available"
            else:
                rec_tags = raw_tags.replace(',', ', ')
            
            # Visual Container for each game (Classic vertical stack)
            with st.container():
                st.markdown(f"### #{rank}: {rec_title}")
                st.markdown(f"**Genre:** {rec_genre}")
                
                # Using the professional metric widget and progress bar
                st.metric(label="Match Confidence", value=f"{match_percentage:.1f}%")
                st.progress(float(match_score))
                
                # Hide the messy tags inside a clean, clickable expander
                with st.expander("View Game Tags"):
                    st.caption(rec_tags)
                
                st.write("---")

    except IndexError:
        st.error(f"❌ Error: Could not process '{selected_game}'. The dataset formatting might be corrupted for this specific row.")