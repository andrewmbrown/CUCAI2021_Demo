import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pylab as plt  
from statsmodels.tsa.arima_model import ARIMA

st.title("Stock Options' Volatility Prediction")
st.header("QMIND - Group 21 - March 6th, 2021")
st.subheader("Alex Le Blanc :coffee:, Smeet Chheda :100:, Andrew Brown :raised_hands:, Tanner Dunn :sunglasses:")

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
    
    st.markdown("""____""")
    st.write("""## :white_check_mark: Our Solution""")
    
    st.write("")
    st.write("**Determining Our ARIMA Model Parameters**")
    st.write("""After conducting our stationarity test using the Augmented-Dickey 
        Fuller Test, and analysing the ACF and PACF plots, we have determined that the 
        optimal p, d, and q hyperparameters for the VIX dataset are (1, 0, 0), respectively.""")
    
    st.write("- *show augmented dickey-fuller test results -> discuss how this gives us d*")
    st.write("- *show PACF plots -> discuss how this gives us p*")
    st.write("- *show ACF plots -> discuss how this gives us q*")
    st.write("- *Discuss how to handle over/under differencing by adjust p or q -> discuss how this gives us p?*")
    st.write("- *show ARIMA summary results*")
    st.write("- *show past value prediction plots*")
    st.write("- *show future value forecast plots*")
    st.write("- *show accuracy and other metrics results?*")


    st.write("**ARIMA Model Prediction on VIX**")
    st.write("- *put arima code here and show prediction plot*")
    st.write("- *show past value prediction plots*")
    st.write("- *show future value forecast plots*")
    st.write("- *show accuracy and other metrics results?*")
    st.write("- *show options strategy recommendation*")


    st.write("**Snippet of ARIMA Code**")
    code_snippet = '''from statsmodels.tsa.arima_model import ARIMA

# Build Model
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
    


    st.write("")
    st.write("")
    st.write("**Interactive Demo**")

    forecast_period = st.slider('How far in the future would you like to forecast?', min_value=1, max_value=10, value=5, step=1)
    time_to_expiry = st.slider('What time to expiry would you like on the options?', min_value=7, max_value=28, value=14, step=7)
    lookback_period = st.slider('How many days in the past would you like the model to lookback in order to make its predictions? **(hmmm... idk about this)**', min_value=200, max_value=3000, value=365, step=10)
    
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    p_val = st.radio("What p value would you like to use?", (0, 1, 2, 3))
    d_val = st.radio("What d value would you like to use?", (0, 1, 2, 3))
    q_val = st.radio("What q value would you like to use?", (0, 1, 2, 3))
    
    todays_date = datetime.date.today()
    start_date = todays_date - datetime.timedelta(days=lookback_period)
    start_date_text = start_date.strftime('%B %d, %Y')


    tik = '^VIX'
    vixData = yf.Ticker(tik)
    vixDF = vixData.history(period='1d', start=start_date, end=todays_date).drop(['Volume','Dividends','Stock Splits'], axis = 1)

    st.write(f"## VIX Prediction on data since {start_date_text}")
    st.line_chart(vixDF.Close)


    vixDF.rename(columns = {'Close':'Volatility'}, inplace = True) 
    df = pd.DataFrame(vixDF['Volatility']).dropna()
    

    # Create Training and Test
    num_lags = 35
    start_bound = 0

    dataset_size = len(df)
    split_idx = len(df) - num_lags

    train = df[start_bound:split_idx]
    test = df[split_idx:]

    # Build Model
    model = ARIMA(train, order=(1, 0, 0))  
    fitted = model.fit(disp=-1) 

    # Forecast
    fc, se, conf = fitted.forecast(num_lags, alpha=0.25)  # 75% conf

    # Make as pandas series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)


    # Plot
    # fig, ax = plt.plot(range(len(train)),train, label='training')
    # ax.plot(test, label='actual')
    # ax.plot(fc_series, label='forecast')
    # ax.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    # ax.title('Forecast vs Actuals')
    # ax.legend(loc='upper left', fontsize=8)
    # ax.show() 

    # st.line_chart(plt.plot(train, label='training'))