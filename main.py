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
sideBarOptions = ['Introduction','Options Description', 'Strategies', 'ARIMA Description', 'Our Solution']
navigation = st.sidebar.selectbox('Go To', sideBarOptions, index = 0)

if navigation == 'Introduction':
    st.write("""
        ## Introduction

        The stock market has been a money-making tool for millions of people around the world, and its increased volatility 
        during the pandemic presents several opportunities for traders and investors to generate considerable income.

        ## Our Goal
        Our goal was to implement a model that can track these trends and pick up on patterns within the stock market. Our 
        model is designed to analyse the recent volatility of SPY to forecast the future volatility and determine which options 
        trading strategy (from a predetermined subset) will yield the greatest return.

        The SPY, an Exchange Traded Fund (ETF) compromised of securities included in the S&P 500, is up 76% from its early pandemic 
        lows; however, there have been periods of weakness in conjunction with periods of strength. Thus, to maximize potential 
        returns and incomes, traders must take advantage of both upward and downward trends in the market. Option contracts can be 
        utilized to benefit from both the adverse and favorable movements.

        ## Our Solution
        Our team researched and analyzed the Chicago Board Options Exchange (CBOE) Volatility Index (VIX). The price of the VIX is 
        obtained from the implied volatility of various options contracts belonging to securities in the S&P 500. 

    """)
    st.image('src/VIXvsSPY.JPG')
    st.write("""

        From our analysis of the VIX and SPY, we found that when VIX sees a lage percent gain, SPY tends to be weaker. This allows
        us to predict an options trading strategy for SPY if we are able to forcast the VIX.

        Stationarity and seasonality tests were conducted on the VIX dataset and accordingly an Auto-Regressive Integrated Moving 
        Average (ARIMA) model was implemented and fine-tuned to forecast the VIX. From there, we can suggest a strategy that will 
        yield the greatest return.
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
        When a trader enters into a call contract, they are agreeing to purchase 100 shares of the underlying
        security at the chosen strike price. Again, the contract can be executed at any time before the expiry date
        but the trader is not obligated to execute.

        Call contracts can net profit when the underlying stock price goes above the purchased strike price; intuitively,
        think of it as purchasing a stock at a discount, and then selling it at market value.
        ### Puts
        Alternatively, a put contract represents the agreement to sell 100 shares of the underlying security at the
        chosen strike price. Thus, a profit can be made when the stock price goes below the strike, as it essentially
        allows the contract holder to sell a stock for more than its worth.
        
        ## Buying v.s Selling Options
        The notion of making profit described above in both puts and calls is in the context of buying an options contract; however, in order
        for there to be a buyer, there must also be a seller. The seller of the contract collects the premiums of the contract and is \'assigned\' the
        the obligation to sell/buy the shares should the buyer choose to execute. In this context, the seller makes the most profit as the 
        premium of the contract gets closer to zero.

        For instance, if a trader sold another trader a put contract with a strike of 100 while the underlying asset
        was trading at 95, then they would net the most profit if the asset price goes above 100 close to expiration, essentially
        rendering the contract worthless.
    
    """)
    

if navigation == 'Strategies':
    st.write("""
        ## Strategies

        The benefit of using options can be seen in the various strategies that are listed below. Their main point of attraction
        is the ability to hedge downside risk which, in other words, means to limit the amount of money one can lose
        if the price of the underlying stock was to move adversely to the contract conditions. The following strategies are based on
        volatility predictions on the VIX.

        Note that:

        - Bullish: Optimistic on a stock's future outlook
        - Bearish: Pessimistic on a stock's future outlook 
    """)

    st.write("""
        ### Fairly Low Volatility (Slightly Bullish on SPY)

        This strategy is called the **Long At-The-Money Call Vertical**. If the ARIMA model is forecasting the VIX will
        have relatively the same, or slightly lower, volatility as the previous period, then one can take a slightly bullish stance on
        SPY and deploy this strategy

        It consists of buying a **CALL** at a strike price (C1) that is below the current SPY price, and selling a **CALL**  with a strike price (C2) that
        is above the current price. This setup allows the owner of the portfolio to limit the amount of money they can lose; if the SPY goes up then the call
        that was bought will become more valuable. Conversely, the if SPY goes down, then the call that was sold expires worthlessly and
        the premiums collected from it will subsidize downside losses from the bought call.
    """)

    st.image('./src/lowVolBullish.PNG')

    st.write("""
        ### Fairly High Volatility
    """)
    

if navigation == 'ARIMA Description':
    # TODO
    # finding p,q,d pacf plots...
    # auto arima
    # Augmented dickie fuller
    st.markdown("""____""")
    st.write("""
    ## ARIMA Description
    An ARIMA model stands for 'Auto Regressive Integrated Moving Average' it has 3 hyper-paramters:
    """)
    st.markdown('- p: the order of the AR term')
    st.markdown('- q: the order of the MA term')
    st.markdown('- d: the number of differencing required for stationarity')
    st.write("ARIMA model are used for time series forecasting, a class of models that explains a given time series based on its own passed values")
    

    st.write('To explain ARIMA, we explain **Stationarity**, **AR** only model, **MA** only model, and **d** (differencing) term')
    st.subheader('Stationarity')
    st.write('Stationarity means that the statistical properties of a process generating a time series do not change over time. In technical terms it must:')
    st.markdown('- Have a constant mean')
    st.markdown('- Have constant variance/standard deviation')
    st.markdown('- Auto-covariance should not depend on time')
    st.write('This is utmost importance for using time-series modelling, and presents a huge concern when looking at financial data. Is it stationary?')
    st.write("The team decided to apply ARIMA models to forecast the VIX, a popular known volatility index of the S&P 500")

    tik = '^VIX'
    vixData = yf.Ticker(tik)
    vixDF = vixData.history(period='1d', start='2010-5-31', end='2020-2-26')

    st.write("### VIX Price Since 2010")
    st.line_chart(vixDF.Close)
    st.write('We must preprocess this data, giving it a constant mean, variance, and standard deviation')
    
    st.subheader('AR Only Model')
    st.write('AR (Auto Regressive) models forecast only depending on its own lags')
    st.markdown('$Y_t=a+B_1Y_{t-1}+B_2Y_{t-2}+...+B_pY_{t-p}+\epsilon_1$')
    st.write('Where $Y_{t-1}$ is the lag1 of the series with beta being the coeficient. $a$ is the intercept term')
    
    st.subheader('MA Only Model')
    st.write('MA (Moving Average) models forecast on the errors(residuals) of the previous forecasts you made to make current forecasts')
    st.markdown('$Y_t=a+\epsilon_t+\phi_1\epsilon_{t-1}++\phi_2\epsilon_{t-2}+...+\phi_q\epsilon_{t-q}$')

    st.subheader('Differencing Parameter d')
    st.write('We difference a time series to make it stationary')
    st.markdown('This is usually through taking values and subtracting them from previous values. Depending on how far back the values are differenced, this is called first, second, third... differencing')
    st.image('./src/differencing.png')

    st.subheader('ARIMA - Combining the AR, MA, and d')
    st.write('We achieve a model that predicts using its own lags *and* using its errors of previous forecasts')
    st.markdown('$Y_t=a+B_1Y_{t-1}+...+B_pY_{t-p}\epsilon_t+\phi_1\epsilon_{t-1}+\phi_2\epsilon_{t-2}...+\phi_q\epsilon_{t-q}$')
    st.markdown('Predicted Yt = Constant + Linear combination Lags of Y (upto p lags) + Linear Combination of Lagged forecast errors (upto q lags)')


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