import streamlit as st
import yfinance as yf
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
    st.write("""
    ## Our Solution
    """)
