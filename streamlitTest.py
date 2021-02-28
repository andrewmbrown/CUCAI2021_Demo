import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

st.title("""

    Options Volatility Analysis
""")
sideBarOptions = ['Introduction','Options Description', 'ARIMA Description', 'Our Solution']

sidebar = st.sidebar.selectbox("Navigation",sideBarOptions)

marketState = st.sidebar.write("""
### Current State of the Market:

""")

spyData = yf.Ticker('SPY')
spyDf = spyData.history(period = '1d', interval = '1m')
spyState = st.sidebar.line_chart(spyDf.Close, height = 200)
if sidebar == 'Introduction':
    
    st.write("""
        ## Introduction
    """)

    st.write('A table:')

    st.write(pd.DataFrame({
        'first column:': [1, 2, 3, 4],
        'second column': [12, 23, 424, 123]

    }))

    st.write("""
    #The volume and closing price of the SPY:

    test

    """)

    tickerSymbol = 'SPY'

    tickerData =  yf.Ticker(tickerSymbol)

    tickerDf = tickerData.history(period='1d', start='2020-4-1', end = '2021-2-10')

    st.line_chart(tickerDf.Close)

    st.line_chart(tickerDf.Volume)

if sidebar == 'Options Description':
    st.write("""
        ## Options Description
    """)

if sidebar == 'ARIMA Description':
    st.write("""
    ## Arima Description
    """)

if sidebar == 'Our Solution':
    st.write("""
    ## Our Solution
    """)