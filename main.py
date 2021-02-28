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

