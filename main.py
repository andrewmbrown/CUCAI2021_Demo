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
    st.write("""
        ## Introduction

        Inflation and the subsequent increased cost of living has left several people struggling to make end’s meet; 
        more than ever before, individuals must establish multiple streams of income to support themselves and their 
        families. The stock market has been a money-making tool for millions of people around the world, and its increased 
        volatility during the pandemic presents several opportunities for traders and investors to generate considerable 
        income. The SPY, an Exchange Traded Fund (ETF) compromised of securities included in the S&P 500, is up 76% from 
        its early pandemic lows; however, there have been periods of weakness in conjunction withperiods of strength. 
        Thus, to maximize potential returns and incomes, traders must take advantage of both upward and downward trends 
        in the market. Option contracts can be utilized to benefit from both the adverse and favorable movements. This 
        study researched and analyzed the Chicago Board Options Exchange (CBOE) Volatility Index, ticker symbol: VIX, a 
        volatility index whose price is obtained from the implied volatility of various options contracts belonging to 
        securities in the S&P 500. Stationarity and seasonality tests were conducted on the VIX dataset and accordingly 
        an Auto-Regressive Integrated Moving Average (ARIMA) model (Figures 1 and 2) was implemented and fine-tuned to 
        forecast the VIX. Using the VIX as an indicator of the S&P 500’s volatility, our model utilizes the forecasted 
        volatility to establish which options trading strategy (out of a predetermined list) will yield the greatest future 
        returns.

    """)

if navigation == 'Options Description':
    st.write("""
        ## What are options?

        Stock market volatility has been a heavily 
        discussed topic ever since the inception of the market in the early 1600s. Most people assume that the only time
        anyone ever makes money is when stocks go up; however, this is far from the case. A profit can be made even when a stock gets
        bad news or sees itself in the midst of a \'market crash\'. A majority stocks have
        options contracts whose price is derived from the price of the stock. Options contracts present investors with the ability to 
        benefit from adverse movement's in an underlying securities\' price.            
    """)
    st.write("""
        ## Types of Options Contracts
        Regardless of the type of contract, all options contracts will have the two following things:
        - Strike Price
        - Expiry Date
        - Premium
        The strike price is the price at which the contract agreement is executed, whereas the expiry date
        acts the same way it does on a coupon. A buyer can, but is not obligated to, execute the contract at any point
        prior to expiry; however, once the contract has reached expiry, it is worthless. The premium can be thought of as the price
        of the contract. As the underlying stock price moves favorably with respect to the contract terms, the premium of the
        contract increases. 
        ### Calls
        When a trader 
        
    
    """)
    
    st.write("""
    #
    """)


if navigation == 'ARIMA Description':
    st.write("""
    ## ARIMA Description
    An ARIMA model stands for 'Auto Regressive Integrated Moving Average' it has 3 hyper-paramters:
    """)
    st.markdown('- p: the order of the AR term')
    st.markdown('- q: the order of the MA term')
    st.markdown('- d: the number of differencing required for stationarity')
    st.write("ARIMA model are used for time series forecasting, a class of models that explains a given time series based on its own passed values")
    st.write("The team decided to apply ARIMA models to forecast the VIX, a popular known volatility index of the S&P 500")

    tik = '^VIX'
    vixData = yf.Ticker(tik)
    vixDF = vixData.history(period='1d', start='2010-5-31', end='2020-2-26')

    st.write("## VIX Price Since 2010")
    st.line_chart(vixDF.Close)

    st.write('')


if navigation == 'Our Solution':
    st.write("""
    ## Our Solution
    """)
