import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pylab as plt  
from statsmodels.tsa.arima_model import ARIMA

st.title("Stock Options' Volatility Prediction :chart_with_upwards_trend:")
st.header("This is strictly **educational content only**")
st.header("QMIND - Group 21 - March 6th, 2021")
st.subheader("Alex Le Blanc :coffee:, Smeet Chheda :100:, Andrew Brown :raised_hands:, Tanner Dunn :sunglasses:")

st.sidebar.title("Navigation")
sideBarOptions = ['Introduction','Options Description', 'Strategies', 'ARIMA Description', 'Our Solution', 'Interactive Demo!']
navigation = st.sidebar.selectbox('Go To', sideBarOptions, index = 0)

if navigation == 'Introduction':
    st.markdown("""____""")
    st.write("""
        ## Introduction

        The stock market has been a money-making tool for millions of people around the world, and its increased volatility 
        during the pandemic presents several opportunities for traders and investors to generate considerable income.

        ## The Power of Options!
        - options are contracts
        - expire at certain times
        - By utilizing options trading strategies,.... Volatility matters! not direction.

        ## Our Goal - Tanner
        Our goal was to implement a model that can track these trends and pick up on patterns within the stock market. Our 
        model is designed to analyse the recent volatility of VIX to forecast the future volatility and determine which options 
        trading strategy (from a predetermined subset) will yield the greatest return.

        The SPY, an Exchange Traded Fund (ETF) compromised of securities included in the S&P 500, is up 76% from its early pandemic 
        lows; however, there have been periods of weakness in conjunction with periods of strength. Thus, to maximize potential 
        returns and incomes, traders must take advantage of both upward and downward trends in the market. Option contracts can be 
        utilized to benefit from both the adverse and favorable movements.

        ## Our Solution - Andrew
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
    st.markdown("""____""")

    st.write("""
        ## :thinking_face: What are options?

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
        
        ## Buying vs Selling Options
        The notion of making profit described above in both puts and calls is in the context of buying an options contract; however, in order
        for there to be a buyer, there must also be a seller. The seller of the contract collects the premiums of the contract and is \'assigned\' the
        the obligation to sell/buy the shares should the buyer choose to execute. In this context, the seller makes the most profit as the 
        premium of the contract gets closer to zero.

        For instance, if a trader sold another trader a put contract with a strike of 100 while the underlying asset
        was trading at 95, then they would net the most profit if the asset price goes above 100 close to expiration, essentially
        rendering the contract worthless.
    
    """)
    

if navigation == 'Strategies':
    st.markdown("""____""")

    st.write("""
        ## :brain: Strategies

        The benefit of using options can be seen in the various strategies that are listed below. Their main point of attraction
        is the ability to hedge downside risk which, in other words, means to limit the amount of money one can lose
        if the price of the underlying stock was to move adversely to the contract conditions. The following strategies are based on
        volatility predictions on the VIX.

        Note that:

        - Bullish: Optimistic on a stock's future outlook
        - Bearish: Pessimistic on a stock's future outlook 
    """)

    st.write("""
        ### Low Volatility (Bullish on SPY)

        This strategy is called the **Married Put**. If the ARIMA is forecasting the VIX to have lower volatility in the chosed period,
        then one can expect the SPY to be strong in that same period. In this case, the Married Put is provides the best way to hedge against the downside.

        This set up will be slightly different than the ones to follow in that it involves the purchase of **shares** alongside the option contract.
        The trader deploying this stratgy will purchase **shares** of SPY with the expectation that they will grow in value, while also
        purchasing **puts** with a strike price near the price at which the shares were purchased.
    """)

    st.image('./src/marriedPut.png')

    st.write("""
        The picture above illustrates the benefit of using this stratgy as opposed to purchasing the shares or puts alone.
        The dotted line labelled \'stock only\' shows the profit and loss scenario when purchasing just shares; the downside loss
        is not bounded and leaves investors with the potential to lose their entire position. The put, on the other hand would not make 
        sense, if one were to have a bullish position on SPY.  By purchasing both, the trader will net profit if the VIX forecast is accurate,
        but will protect his downside by earning profit via the put if the forecast proves to be inaccurate.
    """)
    
    st.write("""
        ### Fairly Low Volatility (Slightly Bullish on SPY)

        This strategy is called the **Long At-The-Money Call Vertical**. If the ARIMA model is forecasting the VIX will
        have relatively the same, or slightly lower, volatility as the previous period, then one can take a slightly bullish stance on
        SPY and deploy this strategy

        It consists of buying a **call** at a strike price (C1) that is below the current SPY price, and selling a **call**  with a strike price (C2) that
        is above the current price. This setup allows the owner of the portfolio to limit the amount of money they can lose; if the SPY goes up then the call
        that was bought will become more valuable. Conversely, the if SPY goes down, then the call that was sold expires worthlessly and
        the premiums collected from it will subsidize downside losses from the bought call.
    """)

    st.image('./src/lowVolBullish.PNG')

    st.write("""
        ### High Volatility (Bearish on SPY)

        This strategy is refered to as the **Long At-The-Money Put Vertical**. As opposed to the strategy above, this strategy
        would be useful when the VIX is forecasted to be high relative to the prior week, and thus indicating that SPY may see
        some bad market days in the coming forecasting period.

        The set up is nearly identical to the **call** version of this stratgy in that one must do 2 transactions:
        *buy* a **put** with a strike price (P1) higher than the current price of SPY, and *sell* a **put** with a strike price (P2) under the
        price of SPY.

        Risk-reward in this situation is similar to above, but with a bearish outlook. The trader can hedge his downside losses,
        in the case of SPY seeing percentage gains in the contract period, with the premiums they collect from selling a put with a strike
        lower than what they owned.
    """)
    
    st.image('./src/longAtmPutVert.png')

    

if navigation == 'ARIMA Description':
    st.markdown("""____""")
    st.write("""
    ## ARIMA Description :woman-raising-hand: :man-raising-hand:
    An ARIMA model stands for 'Auto Regressive Integrated Moving Average' it has 3 hyper-paramters:
    """)
    st.markdown('- p: the order of the AR term')
    st.markdown('- q: the order of the MA term')
    st.markdown('- d: the number of differencing required for stationarity')
    st.write("We must find these 3 parameters to use ARIMA for time series forecasting, a class of models that explains a given time series based on its own passed values")
    

    st.write('To explain ARIMA, we explain **Stationarity**, **AR** only model, **MA** only model, and **d** (differencing) term')
    st.subheader('Stationarity :bar_chart:')
    st.write('Stationarity means that the statistical properties of a process generating a time series do not change over time. In technical terms it must:')
    st.markdown('- Have a constant mean')
    st.markdown('- Have constant variance/standard deviation')
    st.markdown('- Auto-covariance should not depend on time')
    st.write('This is of utmost importance for using time-series modelling, and presents a huge concern when looking at financial data. Is it stationary?')
    st.write('To check for stationarity we can either use **Rolling Statistics (visual analysis)** or a **Stationary Dickie-Fuller Test**')
    st.write('**Rolling Statistics**: graphing the data and visually checking for 3 criteria of stationarity')
    st.write('**Dickie-Fuller Test**: Set a Null hypothesis says that a data is non-stationary. Using statistics libraries we check for confidence in stationarity. if p<0.05 we reject H0 and assume stationary')
    st.write("The team decided to apply ARIMA models to forecast the VIX, a popular known volatility index of the S&P 500")

    tik = '^VIX'
    vixData = yf.Ticker(tik)
    vixDF = vixData.history(period='1d', start='2010-5-31', end='2020-2-26')

    st.write("### VIX Price Since 2010")
    st.line_chart(vixDF.Close)
    st.write('We must preprocess this data, giving it a constant mean, variance, and standard deviation')
    
    st.subheader('AR Only Model :chart_with_upwards_trend:')
    st.write('AR (Auto Regressive) models forecast only depending on its own lags')
    st.markdown('$Y_t=a+B_1Y_{t-1}+B_2Y_{t-2}+...+B_pY_{t-p}+\epsilon_1$')
    st.write('Where $Y_{t-1}$ is the lag1 of the series with beta being the coeficient. $a$ is the intercept term')
    
    st.subheader('MA Only Model :chart_with_downwards_trend:')
    st.write('MA (Moving Average) models forecast on the errors(residuals) of the previous forecasts you made to make current forecasts')
    st.markdown('$Y_t=a+\epsilon_t+\phi_1\epsilon_{t-1}++\phi_2\epsilon_{t-2}+...+\phi_q\epsilon_{t-q}$')

    st.subheader('Differencing Parameter d :microscope:')
    st.write('We difference a time series to make it stationary')
    st.markdown('This is usually through taking values and subtracting them from previous values. Depending on how far back the values are differenced, this is called first, second, third... differencing')

    st.subheader('ARIMA - Combining the AR, MA, and d :world_map:')
    st.write('We achieve a model that predicts using its own lags *and* using its errors of previous forecasts')
    st.markdown('$Y_t=a+B_1Y_{t-1}+...+B_pY_{t-p}\epsilon_t+\phi_1\epsilon_{t-1}+\phi_2\epsilon_{t-2}...+\phi_q\epsilon_{t-q}$')
    st.markdown('Predicted Yt = Constant + Linear combination Lags of Y (upto p lags) + Linear Combination of Lagged forecast errors (upto q lags)')

    st.write('Now we understand that we understand the model, we must find **p**, **q**, **d** to use ARIMA')
    st.subheader('Finding p and q :question:')
    st.write('To find p and q we use two functions:')
    st.markdown('- ACF (Auto Correlation Function) to find p')
    st.markdown('- PACF (Partial Auto Correlation Function) to find q')
    st.write('Plots of the ACF and PACF show us when which value in x-axis, our line plot drops to 0 in y-axis for 1st time.')
    st.image('./src/acf_pacf_example.png')
    st.write('This figure above shows an example of when p and q are both 2. The line plots cross x=0 at y=2')

    st.subheader('Finding d :question:')
    st.write('Finding d (order of differencing) is trickier as it is more dependent on the nature of the data')
    st.write('If our data is already stationary, we do not need any differencing (d=0)')
    st.write('If the data is not stationary, we attempt to take the first, second, third... difference in the data to achieve stationarity')
    st.write('After applying first, second, or third differencing, we use the previously mentioned Rolling Statistics or Dickie-Fuller test to check for stationarity')
    st.image('./src/differencing.png')
    st.write('An example of applying first and second differencing. The higher we difference the better the stationarity gets **but** the more intuitiveness we lose in the data for prediction')

    st.write('After finding p, q, d we are good to go! Happy time-series forecasting :smile:')
    st.write(':100::100::100::100::100::100::100::100::100::100::100::100::100::100::100::100::100::100::100::100::100::100::100::100::100::100::100::100::100:')

if navigation == 'Our Solution':
    
    st.markdown("""____""")
    st.write("""## :white_check_mark: Our Solution""")
    st.write("")
    
    st.write("**Determining Our ARIMA Model Parameters**")
    st.write("""After conducting our stationarity test using the Augmented-Dickey 
        Fuller Test, and analysing the ACF and PACF plots, we have determined that the 
        optimal **p**, **d**, and **q** hyperparameters for the VIX dataset are (1, 0, 0), respectively.""")
    
    st.write("""The following image displays the results of the Augmented Dickey-Fuller Test. It was deployed in order to check stationarity and determine the **d** hyperparameter;
    if it displays a p-value below 0.05, one may assume the data the test was conducted upon is stationary according to the
    definition stated in the \'ARIMA Description\' section. The number of differencing required to reach stationarity reflects an
    appropriate value for the **d** hyperparameter. The results for the unaltered VIX historical data can be seen below:""")
    st.image('./src/ADFResults.PNG')
    st.write("""
        The unaltered data had a p-value of 0.000789, indicating that no differencing was necessary in order to achieve stationarity. Thus, a value of 1 was assgined to **d** going forward    
    """)

    st.subheader('Finding Order of AR Term (p)')
    st.image('./src/acf_plot.png')
    st.write('Lag 1 seems to be within limit. Therefore we can reasonably set p=0')
    st.subheader('Finding Order of MA Term (q)')
    st.image('./src/pacf_plot.png')
    st.write('Lag 1 seems to be within limit. Therefore we can reasonably set q=0')

    st.write('The right order of differencing is the minimum differencing required to get a near-stationary series which roams around a defined mean and the ACF plot reaches to zero fairly quick.')
    st.write('We determined that p=0 and q=0 are a good tradeoff between near-stationarity and predictability in the data')

    st.write("")
    st.write("**ARIMA Model Prediction on VIX**")

    ## Predictions and plots
    max_time_to_expiry = 21
    time_to_expiry = 14
    lookback_period = 500
    
    p_val = 1
    d_val = 0
    q_val = 0
    
    todays_date = datetime.date.today()
    start_date = todays_date - datetime.timedelta(days=lookback_period)
    start_date_text = start_date.strftime('%B %d, %Y')

    tik = '^VIX'
    vixData = yf.Ticker(tik)
    vixDF = vixData.history(period='1d', start=start_date, end=todays_date).drop(['Volume','Dividends','Stock Splits'], axis = 1)

    vixDF.rename(columns = {'Close':'Volatility'}, inplace = True) 
    df = pd.DataFrame(vixDF['Volatility']).dropna()
    
    start_plot_idx = 365

    const_fig_size=(9,4)

    ## ARIMA for Predicting past data
    # Create Training and Test
    dataset_size = len(df)
    split_idx = len(df) - time_to_expiry

    train = df[:split_idx+1]
    test = df[split_idx:]

    # Build Model
    model = ARIMA(train, order=(p_val, d_val, q_val))  
    fitted = model.fit(disp=-1) 

    # Forecast
    fc, se, conf = fitted.forecast(time_to_expiry, alpha=0.25)  # 75% conf

    # Make as pandas series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)

    todays_date_text = todays_date.strftime('%B %d, %Y')
    end_date = todays_date + datetime.timedelta(days=time_to_expiry)
    end_date_text = end_date.strftime('%B %d, %Y')
    past_start_date = todays_date - datetime.timedelta(days=time_to_expiry)
    past_start_date_text = past_start_date.strftime('%B %d, %Y')

    # Plot
    st.write(f'### Past VIX Forecast vs Actuals from {past_start_date_text} to {todays_date_text}')
    figure1, axes1 = plt.subplots(nrows=1, ncols=1, figsize=const_fig_size)
    axes1.plot(train[-start_plot_idx:], label='Training',color='C0')
    axes1.plot(test, label='Actual',color='g')
    axes1.plot(fc_series, label='Forecast',color='C1')
    axes1.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    axes1.legend(loc='upper left', fontsize=8)
    st.pyplot(figure1)


    ## ARIMA for Forecasting Future data
    # Build Model
    model = ARIMA(df, order=(p_val, d_val, q_val))  
    fitted = model.fit(disp=-1)  

    # Forecast
    fc2, se, conf = fitted.forecast(time_to_expiry, alpha=0.25)  # 75% conf

    forecast_idx = pd.date_range(df.index[-1], periods=time_to_expiry)

    # Make as pandas series
    fc_series = pd.Series(fc2, index=forecast_idx)
    lower_series = pd.Series(conf[:, 0], index=forecast_idx)
    upper_series = pd.Series(conf[:, 1], index=forecast_idx)

    # Plot
    st.write(f'### Future VIX Forecast from {todays_date_text} to {end_date_text}')
    figure2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=const_fig_size)
    axes2.plot(df[-start_plot_idx:], label='Past VIX',color='C0')
    axes2.plot(fc_series, label='Forecast',color='C1')
    axes2.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    axes2.legend(loc='upper left', fontsize=8)
    st.pyplot(figure2)


    # Decision/Recommendation Algorithm
    st.write("")
    st.write(f"### Trading Strategy Recommendation for options expiring in {time_to_expiry} days:")
    volatility_difference = fc2[-1] - fc2[0]
    volatility_range = df[-50:].mean()[0]
    time_to_expiry_volatility_factor = time_to_expiry/(max_time_to_expiry+10) + 1
    volatility_score = np.abs(2 * volatility_difference * time_to_expiry_volatility_factor / volatility_range)
    
    if volatility_score > 0.35:
        st.write("## :point_right: *Use a **Long At-The-Money Put Vertical** (High Volatility Strategy)!*")
    elif volatility_score > 0.12:
        st.write("## :point_right: *Use a **Long At-The-Money Call Vertical** (Fairly Low Volatility Strategy)!*")
    else:
        st.write("## :point_right: *Use a **Married Put** (Low Volatility Strategy)!*")

    
    st.write("___")
    st.write("**Snippet of ARIMA Code**")
    code_snippet = '''# Download VIX Dataset
tik = '^VIX'
vixData = yf.Ticker(tik)
vixDF = vixData.history(period='1d', start=start_date, end=todays_date).drop(['Volume','Dividends','Stock Splits'], axis = 1)

vixDF.rename(columns = {'Close':'Volatility'}, inplace = True) 
df = pd.DataFrame(vixDF['Volatility']).dropna()

# Create Training and Test
dataset_size = len(df)
time_to_expiry = 14
split_idx = len(df) - time_to_expiry

train = df[:split_idx+1]
test = df[split_idx:]

# Build ARIMA(p=1,d=0,q=0) Model
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
    


if navigation == 'Interactive Demo!':

    st.markdown("""____""")
    st.write("""## :video_game: Interactive Demo!""")
    st.write("")

    max_time_to_expiry = 28

    time_to_expiry = st.slider('How far in the future would you like to forecast (i.e.: what is your desired time to expiry on your contracts)?', min_value=7, max_value=max_time_to_expiry, value=7, step=7)
    lookback_period = st.slider('How many days in the past would you like the model to lookback in order to make its predictions?', min_value=200, max_value=3000, value=700, step=10)
    
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    p_val = st.radio("What p value would you like to use?", (0, 1, 2, 3), index=1)
    d_val = st.radio("What d value would you like to use?", (0, 1, 2), index=0)
    q_val = st.radio("What q value would you like to use?", (0, 1, 2, 3), index=0)
    
    todays_date = datetime.date.today()
    start_date = todays_date - datetime.timedelta(days=lookback_period)
    start_date_text = start_date.strftime('%B %d, %Y')


    tik = '^VIX'
    vixData = yf.Ticker(tik)
    vixDF = vixData.history(period='1d', start=start_date, end=todays_date).drop(['Volume','Dividends','Stock Splits'], axis = 1)

    vixDF.rename(columns = {'Close':'Volatility'}, inplace = True) 
    df = pd.DataFrame(vixDF['Volatility']).dropna()
    
    start_plot_idx = st.radio("How many days would you like to see plotted?", (200,365, 'All of them'), index=2)

    if start_plot_idx == 'All of them':
       start_plot_idx = len(df) 

    const_fig_size=(10,5)

    ## ARIMA for Predicting past data
    # Create Training and Test
    dataset_size = len(df)
    split_idx = len(df) - time_to_expiry

    train = df[:split_idx+1]
    test = df[split_idx:]

    # Build Model
    model = ARIMA(train, order=(p_val, d_val, q_val))  
    fitted = model.fit(disp=-1) 

    # Forecast
    fc, se, conf = fitted.forecast(time_to_expiry, alpha=0.25)  # 75% conf

    # Make as pandas series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)

    todays_date_text = todays_date.strftime('%B %d, %Y')

    end_date = todays_date + datetime.timedelta(days=time_to_expiry)
    end_date_text = end_date.strftime('%B %d, %Y')

    past_start_date = todays_date - datetime.timedelta(days=time_to_expiry)
    past_start_date_text = past_start_date.strftime('%B %d, %Y')

    # Plot
    st.write(f'### Past VIX Forecast vs Actuals from {past_start_date_text} to {todays_date_text}')
    figure1, axes1 = plt.subplots(nrows=1, ncols=1, figsize=const_fig_size)
    axes1.plot(train[-start_plot_idx:], label='Training',color='C0')
    axes1.plot(test, label='Actual',color='g')
    axes1.plot(fc_series, label='Forecast',color='C1')
    axes1.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    axes1.legend(loc='upper left', fontsize=8)
    st.pyplot(figure1)


    ## ARIMA for Forecasting Future data
    # Build Model
    model = ARIMA(df, order=(p_val, d_val, q_val))  
    fitted = model.fit(disp=-1)  

    # Forecast
    fc2, se, conf = fitted.forecast(time_to_expiry, alpha=0.25)  # 75% conf

    forecast_idx = pd.date_range(df.index[-1], periods=time_to_expiry)

    # Make as pandas series
    fc_series = pd.Series(fc2, index=forecast_idx)
    lower_series = pd.Series(conf[:, 0], index=forecast_idx)
    upper_series = pd.Series(conf[:, 1], index=forecast_idx)

    # Plot
    st.write(f'### Future VIX Forecast from {todays_date_text} to {end_date_text}')
    figure2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=const_fig_size)
    axes2.plot(df[-start_plot_idx:], label='Past VIX',color='C0')
    axes2.plot(fc_series, label='Forecast',color='C1')
    axes2.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    axes2.legend(loc='upper left', fontsize=8)
    st.pyplot(figure2)


    # Decision/Recommendation Algorithm
    st.write("")
    st.write(f"### Trading Strategy Recommendation for options expiring in {time_to_expiry} days:")
    volatility_difference = fc2[-1] - fc2[0]
    volatility_range = df[-50:].mean()[0]
    time_to_expiry_volatility_factor = time_to_expiry/(max_time_to_expiry+10) + 1
    volatility_score = np.abs(2 * volatility_difference * time_to_expiry_volatility_factor / volatility_range)
    
    if volatility_score > 0.35:
        st.write("## :point_right: *Use a **Long At-The-Money Put Vertical** (High Volatility Strategy)!*")
    elif volatility_score > 0.12:
        st.write("## :point_right: *Use a **Long At-The-Money Call Vertical** (Fairly Low Volatility Strategy)!*")
    else:
        st.write("## :point_right: *Use a **Married Put** (Low Volatility Strategy)!*")
