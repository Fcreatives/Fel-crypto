import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
import time
import datetime
import seaborn as sns
import streamlit as st
from prophet import Prophet
import pickle

tickers_list = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'USDC-USD', 'HEX-USD',
'XRP-USD',
'LUNA1-USD',
'SOL-USD',
'ADA-USD',
'UST-USD',
'BUSD-USD',
'DOGE-USD',
'AVAX-USD',
'DOT-USD',
'SHIB-USD',
'WBTC-USD',
'STETH-USD',
'DAI-USD',
'MATIC-USD']

duration_type_list = ["day", "week", "month", "quarter"]
duration_map = {"day":1, "week":7, "month":30, "quarter":90}
models_list = ["fb-prophet"]

st.set_page_config(
page_title='Real-Time Data Science Dashboard',
page_icon='✅',
layout='wide'
)

st.title("SOLiGence Real-Time Cryptocurrency Dashboard")

currency_filter = st.selectbox("Select the Currency", tickers_list)

def fbprophet_model(data, currency_filter, data_prediction):
    model_ = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    data_train = pd.DataFrame()
    data_train['ds'] = data.reset_index()['Date']
    data_train['y'] = data.reset_index()[currency_filter]['Close']
    data_train.dropna(inplace=True)
    model_.fit(data_train)
    data_pred = pd.DataFrame(data_prediction, columns=['ds'])
    return model_.predict(data_pred)['yhat']

def find_profit(data, times, forecast, profit, currency):
    current_value = data[currency]['Close'][-1]
    df_ = pd.DataFrame(columns=["Currency", "Date", "Expected Profit"])
    for i in range(len(forecast)):
        if (forecast[i]-current_value) >= profit:
            df_.loc[len(df_.index)] = [currency, times[i], forecast[i]-current_value]
            return df_, current_value

def get_prediction_data(data, duration):
    current_date = data.index[-1]
    times = [(current_date.to_pydatetime() + datetime.timedelta(days=x)) for x in range(duration_map[duration])]
    return times

placeholder = st.empty()

with placeholder.container():
    df = yf.download(tickers_list, period="max", interval="1d", group_by='tickers')
    idx = pd.IndexSlice
    df_tmp = df.loc[:,idx[:,'Close']]
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.markdown("### First Chart")
        fig = px.line(df[currency_filter], y="Open")
        st.write(fig)
    with fig_col2:
        st.markdown("### Second Chart")
        fig2 = px.line(df[currency_filter], y="Close")
        st.write(fig2)
        fig_col3, fig_col4 = st.columns(2)
    with fig_col3:
        st.markdown("### Third Chart")
        fig3 = px.line(df[currency_filter], y="High")
        st.write(fig3)
    with fig_col4:
        st.markdown("### Fourth Chart")
        fig4 = px.line(df[currency_filter], y="Low")
        st.write(fig4)
        fig_col5, fig_col6 = st.columns(2)
    with fig_col5:
       st.markdown("### Fifth Chart")
       fig5 = px.line(df[currency_filter], y="Adj Close")
       st.write(fig5)
    with fig_col6:
       st.markdown("### Sixth Chart")
       fig6 = px.line(df[currency_filter], y="Volume")
       st.write(fig6)
       fig_col7, fig_col8 = st.columns(2)
    with fig_col7:
       st.markdown("### Seventh Chart")
       fig7 = plt.figure()
       sns.heatmap(df_tmp.corr())
       st.write(fig7)
    with fig_col8:
       st.markdown("### Eighth Chart")
       df_copy = pd.DataFrame()
       df_copy['Close'] = df[currency_filter]['Close']
       df_copy = df_copy.reset_index()
       df_copy['rolling_mean'] = df_copy['Close'].rolling(7).mean()
       fig8 = plt.figure()
       sns.lineplot(x='Date', y='Close', data=df_copy, label='Close Values')
       sns.lineplot(x='Date', y='rolling_mean', data=df_copy, label='Rolling Close Values')
       st.write(fig8)


with st.expander("Find your bet!!"):
    duration_filter = st.selectbox("Select the Duration", duration_type_list)
    profit_filter = int(st.number_input('Insert the required profit'))
    model_filter = st.selectbox("Select the Predictor Model", models_list)
    if st.button('Submit'):
        with st.spinner('In progress...'):
            pred_data = get_prediction_data(df, duration_filter)
            if model_filter=="fb-prophet":
                forecast = fbprophet_model(df, currency_filter, pred_data)
            output, current_value = find_profit(df, pred_data, forecast, profit_filter, currency_filter)
            st.write("Output Dataframe is generated!! with current value as " + str(current_value))
            for i in range(len(pred_data)):
                st.write("" + str(pred_data[i]) + "\t\t" + str(forecast[i]))
            st.dataframe(output)
            if output.shape[0]!=0:
                fig3 = px.bar(output, x="Date", y="Expected Profit")
                st.write(fig3)
            else:
                st.write("No dates found!!")
#Save the model using pickle
model_data = {
'tickers_list': tickers_list,
'duration_type_list': duration_type_list,
'duration_map': duration_map,
'models_list': models_list
}
with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)






