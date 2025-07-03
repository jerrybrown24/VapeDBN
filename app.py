
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os

# Load assets
@st.cache_data
def load_data():
    trends = pd.read_csv('flavor_trends.csv', parse_dates=['date'])
    users = pd.read_csv('users_synthetic.csv')
    rules = pd.read_csv('compliance_rules.csv')
    return trends, users, rules

trends_df, users_df, rules_df = load_data()

# Inject CSS
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
st.markdown('<div id="smoke-overlay"></div>', unsafe_allow_html=True)

st.title('🚀 VaporIQ Galaxy Dashboard')
st.caption('Demo platform with synthetic data')

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ['Flavor Forecast', 'TasteDNA', 'MoodSync', 'Limited Drops', 'Compliance'])

# Flavor Forecast tab
with tab1:
    st.header('🔮 Flavor Forecast')
    flavor = st.selectbox('Choose a flavor', trends_df['flavor'].unique(),
                          index=list(trends_df['flavor'].unique()).index('Custard Kunafa'))
    dsub = trends_df[trends_df['flavor'] == flavor].sort_values('date')
    # naive forecast
    last_mentions = dsub['mentions'].values[-1]
    forecast = [last_mentions + (i*2 if flavor == 'Custard Kunafa' else i) for i in range(1,15)]
    future_dates = [dsub['date'].max() + timedelta(days=i) for i in range(1,15)]
    forecast_df = pd.DataFrame({'date': list(dsub['date']) + future_dates,
                                'mentions': list(dsub['mentions']) + forecast})
    fig = px.line(forecast_df, x='date', y='mentions',
                  title=f"Trend & Forecast for {flavor}")
    st.plotly_chart(fig, use_container_width=True)

# TasteDNA tab
with tab2:
    st.header('🧬 TasteDNA Quiz')
    fruity = st.slider('Fruity affinity',0,5,3)
    dessert = st.slider('Dessert affinity',0,5,3)
    menthol = st.slider('Menthol affinity',0,5,2)
    if st.button('Generate Recommendations'):
        vec = np.array([[fruity,dessert,menthol]])
        km = KMeans(n_clusters=3, random_state=42).fit(
            users_df[['TasteDNA_Fruity','TasteDNA_Dessert','TasteDNA_Menthol']])
        cluster = km.predict(vec)[0]
        st.success(f"You belong to Taste Cluster {cluster}")
        recs = ['Custard Kunafa','Mango Breeze','Icy Mint']
        st.write('Recommended flavours:', ', '.join(recs))

# MoodSync tab
with tab3:
    st.header('💫 MoodSync')
    mood = st.selectbox('Current mood', ['Energised','Neutral','Stressed','Tired'])
    hour = st.slider('Hour of Day',0,23, datetime.now().hour)
    if st.button('Suggest Blend'):
        chosen = 'Custard Kunafa' if mood=='Stressed' else 'Berry Blast'
        st.subheader(chosen)

# Limited Drops
with tab4:
    st.header('🔥 Limited Drops')
    today = trends_df['date'].max()
    today_trends = trends_df[trends_df['date']==today]
    top = today_trends.sort_values('mentions', ascending=False).head(5)
    st.table(top[['flavor','mentions']])
    st.markdown("### Today's Exclusive")
    st.subheader('Custard Kunafa')

# Compliance
with tab5:
    st.header('🔐 Compliance Checker')
    country = st.selectbox('Country', rules_df['Country'])
    age = st.number_input('Your age',18,100,25)
    rule = rules_df[rules_df['Country']==country].iloc[0]
    st.write(f"Min age: {rule['MinAge']}, Max nicotine: {rule['MaxNicotine_mgml']} mg/mL")
    if age < rule['MinAge']:
        st.error('Too young to vape here.')
    else:
        st.success('Age verified.')
