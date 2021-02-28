import streamlit as st
# import yfinance as yf
import pandas as pd
import numpy as np

st.title("Stock Options' Volatility Prediction")
st.header("QMIND - Group 21 - March 6th, 2021")
st.subheader("Alex Le Blanc :coffee:, Smeet Schheda :100:, Andrew Brown :raised_hands:, Tanner Dunn :sunglasses:")

st.sidebar.title("Navigation")
sideBarOptions = ['Introduction','Options Description', 'ARIMA Description', 'Our Solution']
navigation = st.sidebar.selectbox('Go To', sideBarOptions, index = 0)

if navigation == 'Introduction':
    st.write("")

if navigation == 'Options Description':
    st.write("""
        ## Options Description
    """)

if navigation == 'ARIMA Description':
    st.write("""
    ## Arima Description
    """)



if navigation == 'Our Solution':
    st.write("""## Our Solution""")
    
    st.write("")
    st.write("**Our ARIMA Model**")
    st.write("After conducting our stationarity test using the Augmented-Dickey Fuller Test, analysing the ACF and PACF plots, we have determined that the optimal p,d,q hyperparameters for the VIX dataset are (1,0,0).")

    st.write("**Snippet of ARIMA Code**")
    code_snippet = '''# Build Model
model = ARIMA(train, order=(1, 0, 0))  
fitted = model.fit(disp=-1)  

# Forecast
fc, se, conf = fitted.forecast(num_lags, alpha=0.25)  # 75% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show();'''
    
    st.code(code_snippet,language='python')
    
    st.write("**ARIMA Model Prediction on VIX**")


    st.write("**Interactive Demo**")

    forecast_period = st.slider('How far in the future would you like to forecast', min_value=1, max_value=10, value=5, step=1)
    time_to_expiry = st.slider('What time to expiry would you like on the options', min_value=7, max_value=28, value=14, step=7)

