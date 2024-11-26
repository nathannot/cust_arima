import streamlit as st
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize


@st.cache_data
def load_stock_data(ticker, start, end):
    """Fetch historical stock data."""
    return yf.download(ticker, start=start, end=end)['Adj Close']

st.header('ARIMA Forecasting for Stocks')
st.write('This app uses a custom ARIMA algorithm to forecast stock prices.')

# Stock selection
stocks = st.selectbox('Select from the following stocks', ('Apple', 'Google', 'Tesla', 'Meta', 'Amazon', 'Microsoft'))
code = {'Apple': 'AAPL', 'Google': 'GOOG', 'Tesla': 'TSLA', 'Meta': 'META', 'Amazon': 'amzn', 'Microsoft': 'MSFT'}
ticker = code[stocks]

# Date selection
st.write('Select date range for historical data')
start = st.date_input('Start date', datetime.date(2024, 1, 1))
end = st.date_input('End date', datetime.date(2024, 11, 1))

# Load data
x = load_stock_data(ticker, start, end)
if x.empty:
    st.warning("No data available for the selected date range. Please adjust the date range.")
    st.stop()

st.write('Select forecast period')

h = st.slider('Forecast period',1,60,5)

def arima(x, p,d,q,h):
    y = x.values
    n = len(y)
    z = np.diff(y, n=d)
    zt = np.zeros(len(y))
    zt[d:] = z

    def f(params, z,p,q):
        ps = params[:p]
        th = params[p:p+q]
        s = params[-1]
        n = len(z)
        if s <=0:
            return np.inf
        e = np.zeros(n)
        for t in range(1,n):
            ar_term = ps.dot(np.array([z[t-i] for i in range(1,p+1)]))
            ma_term = th.dot(np.array([e[t-i] for i in range(1,q+1)]))
            e[t] = z[t] - ar_term - ma_term
         
        e = np.clip(e, -1e10, 1e10)
        SSE = np.sum(e**2)
        like = (n/2)*np.log(2*np.pi)+(n/2)*np.log(s)+(1/(2*s))*SSE
        return like
    
    init_params = [p*[0],q*[0], np.var(zt)]
    init =  np.hstack(init_params).tolist()
    
    constraints = (
        {'type':'ineq', 'fun': lambda params:1-abs(params[p])},
        {'type':'ineq', 'fun': lambda params:1-abs(params[p:p+q])},
        {'type':'ineq', 'fun': lambda params:abs(params[-1])}
    
    )
    
    result = minimize(f, init, args=(zt,p,q), constraints = constraints)
    params = result.x
    ps = params[:p]
    th = params[p:p+q]
    s = params[-1]

    e = np.zeros(n)
    for t in range(1,n):
        ar_term = ps.dot(np.array([zt[t-i] for i in range(1,p+1)]))
        ma_term = th.dot(np.array([e[t-i] for i in range(1,q+1)]))
        e[t] = zt[t]-ar_term-ma_term
    
    zt1 = np.zeros(n)
    for t in range(1,n):
        ar_term = ps.dot(np.array([zt[t-i] for i in range(1,p+1)]))
        ma_term = th.dot(np.array([e[t-i] for i in range(1,q+1)]))
        zt1[t]= ar_term + ma_term
    #fine y by differencing
    yt = np.zeros(len(y))
    yt[0]=y[0]
    for t in range(1,len(y)):
        yt[t] = y[t-1] +zt1[t]

    np.random.seed(1)

    fut = np.zeros(h)  
    ztf = np.hstack([zt, np.zeros(h)])  
    residual_variance = np.var(e) 
    future_residuals = np.random.normal(0, np.sqrt(residual_variance), size=h) 
    ztf[len(zt)] = ps.dot(np.array([zt[-i] for i in range(1,p+1)])) +th.dot(np.array([e[-i] for i in range(1,q+1)]))
    fut[0] = y[-1]+ztf[len(zt)]+future_residuals[0]
    
    for i in range(1,h):
        ar_term = ps.dot(np.array([ztf[len(zt)-j+i] for j in range(1, p+1)])) 
        ma_term = th.dot([future_residuals[i - j] for j in range(q)]) 
        ztf[len(zt) + i] = ar_term + ma_term
        fut[i] = fut[0] + np.sum(ztf[len(zt):len(zt) + i + 1])
        
    return yt, fut

test = x[-h:]
y = x[:-h]
st.write('Select parameters for ARIMA model, if unsure leave default')
p = st.slider('Select p value',1,5,1)
d = st.slider('Select d value',1,3,1)
q = st.slider('Select q value',1,5,1)


fit, fut = arima(y, p,d,q,h)

rmse = np.sqrt((np.sum(y.values-fit)**2)/len(y)).round(4)
rmse_f = np.sqrt((np.sum(test.values-fut)**2)/len(fut)).round(4)

fig = make_subplots(2,1,shared_xaxes = False, row_heights=[0.5,0.5])

fig.add_trace(go.Scatter(x = y.index, y = y, name= 'actual'), row =1, col=1)
fig.add_trace(go.Scatter(x = y.index, y = fit, name= f'arima {p,d,q}'), row =1, col=1)
fig.add_trace(go.Scatter(x = test.index, y = test, name= 'actual'), row =1, col=1)
fig.add_trace(go.Scatter(x = test.index, y = fut, name= 'forecast'), row =1, col=1)
fig.add_trace(go.Scatter(x = test.index, y = test, name= 'actual close up'), row =2, col=1)
fig.add_trace(go.Scatter(x = test.index, y = fut, name= 'forecast close up'), row =2, col=1)

fig.update_layout(hovermode='x', height = 600, width =800, title=f'ARIMA {p,d,q}  Model for {stocks}: RMSE {rmse} Forecast RMSE {rmse_f}')


st.plotly_chart(fig)
st.write(pd.DataFrame({'Date':test.index,
                       'Actual Vale': test.values,
                       'Forecasted Values':fut}))
