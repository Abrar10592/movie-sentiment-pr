import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

st.title("🎬 Movie Review Sentiment Analysis")

# Tab 1: Instant review
tab1, tab2 = st.tabs(["📝 Instant Review", "📊 Batch & Trends"])

with tab1:
    review = st.text_area("Paste a movie review:")
    if st.button("Analyze") and review:
        prediction = model.predict([review])[0]
        probability = model.predict_proba([review]).max()
        st.success(f"Sentiment: **{prediction}** (confidence: {probability:.2%})")

with tab2:
    uploaded_file = st.file_uploader("Upload CSV (columns: review, date)", type='csv')
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['sentiment'] = model.predict(df['review'])
        df['score'] = model.predict_proba(df['review']).max(axis=1)
        
        st.dataframe(df.head())
        
        # Sentiment pie chart
        fig_pie = px.pie(df, names='sentiment', title="Sentiment Distribution")
        st.plotly_chart(fig_pie)
        
        # Trend line (if date column exists)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            trend = df.groupby(df['date'].dt.to_period('D'))['score'].mean().reset_index()
            trend['date'] = trend['date'].astype(str)
            fig_line = px.line(trend, x='date', y='score', title="Sentiment Trend Over Time")
            st.plotly_chart(fig_line)
        
        st.download_button("Download results", df.to_csv(index=False), "results.csv")
