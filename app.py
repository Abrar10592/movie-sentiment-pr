import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

st.set_page_config(layout="wide")
st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("**Upload reviews for ONE movie to see sentiment trends over time**")

tab1, tab2 = st.tabs(["📝 Instant Review", "📊 Single Movie Trends"])

with tab1:
    st.subheader("Test single review instantly")
    review = st.text_area("Paste movie review:", height=150)
    if st.button("🔍 Analyze", type="primary") and review:
        pred = model.predict([review])[0]
        prob = model.predict_proba([review]).max()
        st.success(f"**Sentiment**: {pred} (confidence: {prob:.1%})")

with tab2:
    st.subheader("📈 Upload CSV for one movie (needs 'review' and 'date' columns)")
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Choose CSV file", type='csv')
    with col2:
        movie_name = st.text_input("Movie name (for title):", "My Movie")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Assume columns: review, date (YYYY-MM-DD or similar)
        if 'review' not in df.columns:
            st.error("CSV must have 'review' column!")
        else:
            df['sentiment'] = model.predict(df['review'])
            df['prob'] = model.predict_proba(df['review']).max(axis=1)
            
            # Display sample
            st.dataframe(df[['review', 'sentiment', 'prob']].head(10), use_container_width=True)
            
            colA, colB = st.columns(2)
            
            with colA:
                # Sentiment distribution
                sentiment_counts = df['sentiment'].value_counts()
                fig_pie = px.pie(values=sentiment_counts.values, 
                                names=sentiment_counts.index, 
                                title=f"Overall Sentiment - {movie_name}")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with colB:
                # Single movie trend (requires date column)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df.dropna(subset=['date'])
                    
                    # Daily average sentiment score (0-1, higher=more positive)
                    trend_df = df.groupby(df['date'].dt.date)['prob'].agg(['mean', 'count']).reset_index()
                    trend_df.columns = ['date', 'avg_score', 'review_count']
                    
                    fig_line = px.line(trend_df, x='date', y='avg_score', 
                                     title=f"Sentiment Trend Over Time - {movie_name}",
                                     hover_data=['review_count'])
                    fig_line.update_yaxis(title="Avg Confidence (Positive)", range=[0,1])
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.warning("Add 'date' column (YYYY-MM-DD) to CSV for time trends!")
            
            # Download
            st.download_button("💾 Download results with predictions", 
                             df.to_csv(index=False), f"{movie_name}_analysis.csv")
