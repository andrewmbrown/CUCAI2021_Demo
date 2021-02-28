import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.title("""
    Options Volatility Analysis

""")

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
    st.write("""
    ## Our Solution
    """)
