'''
S = price of non-dividend paying stock
X = exercise price
t = time to expiration (in years)
r = annual interest rate in %
sigma = annual st. dev (volatility) of stock price in %
'''

import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import yfinance as yf
from streamlit_extras.stylable_container import stylable_container

st.title("European Option Pricing using Black-Scholes-Merton Model")
default_stock = yf.Ticker("SPX").history()
current_stock_price = round(default_stock['Close'].iloc[-1],2)
treasury_ticker = yf.Ticker("^TNX")
current_treasury_yield = round(treasury_ticker.history(period='1d')['Close'].iloc[-1],2)

st.sidebar.header("Model Inputs")
S = st.sidebar.number_input('Stock Price (default is SPX)', value = current_stock_price)
X = st.sidebar.number_input('Exercise Price', value = current_stock_price *1.05)
t = (-(date.today() - st.sidebar.date_input('Expiration Date', date.today(),min_value=date.today()))).days/365
r = (st.sidebar.number_input('Risk-free rate (%)', value = current_treasury_yield))/100
sigma = (st.sidebar.number_input('Annual Volatility (in %)', value = 20))/100
option_type = None

st.sidebar.divider()

st.sidebar.header("Heatmap Parameters")
sigma_range_min = (st.sidebar.slider("Minimum Volatility",0.01,100.00,49.00))/100
sigma_range_max = (st.sidebar.slider("Maximum Volatility",0.01,100.00,51.00))/100
sigma_range = (sigma_range_min,sigma_range_max)
S_range_min = st.sidebar.number_input("Minimum Underlying Price",value = current_stock_price*0.99)
S_range_max = st.sidebar.number_input("Maximum Underlying Price",value = current_stock_price*1.01)
S_range = (S_range_min,S_range_max)

class blackscholesmodel:
    def __init__(self,S,X,t,r,sigma):
        self.S = S
        self.X = X
        self.t = t
        self.r = r
        self.sigma = sigma
    def d1(self):
        return (np.log(self.S / self.X) + (self.r + 0.5 * self.sigma ** 2) * self.t) / (self.sigma * np.sqrt(self.t))
    def d2(self):
        return self.d1() - (self.sigma * np.sqrt(self.t))
    def call_price(self):
        return (self.S * norm.cdf(self.d1(), 0, 1) - self.X * np.exp(-self.r * self.t) * norm.cdf(self.d2(), 0, 1))
    def put_price(self):
        return (self.X * np.exp(-self.r * self.t) * norm.cdf(-self.d2(), 0, 1) - self.S * norm.cdf(-self.d1(), 0, 1))

def black_scholes(S,X,t,r,sigma,option_type):
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))

    if option_type == 'call':
        option_price = S * norm.cdf(d1, 0, 1) - X * np.exp(-r * t) * norm.cdf(d2, 0, 1)
    elif option_type == 'put':
        option_price = X * np.exp(-r * t) * norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1)
    else:
        raise ValueError("Type must be call / put")
    return option_price

def generate_heatmap(S,X,t,r,sigma_range, S_range,option_type):
    sigma_values = np.linspace(*sigma_range, 10)
    S_values = np.linspace(*S_range,10)

    price_matrix = np.zeros((len(S_values),len(sigma_values)))

    for i, S_val in enumerate(S_values):
        for j, sigma_val in enumerate(sigma_values):
            price_matrix[i,j] = black_scholes(S_val,X,t,r,sigma_val,option_type)
    plt.figure(figsize=(10,8))
    sns.heatmap(price_matrix, xticklabels=np.round(S_values,2), yticklabels=np.round(sigma_values, 2), cmap='RdYlGn',annot=True,fmt='.3g')
    plt.xlabel('Stock Price')
    plt.ylabel('Volatility')
    plt.title(f'{option_type.capitalize()} Option Price Heatmap')
    st.pyplot(plt.gcf())

class greeks(blackscholesmodel):
    def delta_call(self):
        return norm.cdf(self.d1(),0,1)
    def delta_put(self):
        return -norm.cdf(-self.d1(),0,1)
    def gamma(self):
        return norm.pdf(self.d1(),0,1) / (self.S * self.sigma * np.sqrt(self.t))
    def theta_call(self):
        return (-self.S * norm.pdf(self.d1(), 0, 1) * self.sigma / (2 * np.sqrt(self.t)) + self.r * self.X * np.exp(-self.r * self.t) * norm.cdf(-self.d2(), 0, 1))
    def theta_put(self):
        return (-self.S * norm.pdf(self.d1(), 0, 1) * self.sigma / (2 * np.sqrt(self.t)) + self.r * self.X * np.exp(-self.r * self.t) * norm.cdf(-self.d2(), 0, 1))
    def vega(self):
        return self.S * norm.pdf(self.d1(), 0, 1) * np.sqrt(self.t)
    def rho_call(self):
        return self.X * self.t * np.exp(-self.r * self.t) * norm.cdf(self.d2(), 0, 1)
    def rho_put(self):
        return -self.X * self.t * np.exp(-self.r * self.t) * norm.cdf(-self.d2(), 0, 1)

class plot_greeks(blackscholesmodel):
    '''
    def delta_call_plot(self):
        S_range = (0.8*self.S,1.2*self.S)
        stock_prices = np.linspace(*S_range,num=1000)
        deltas = [greeks(S=S, X=X, t=t, r=r, sigma=sigma).delta_call() for S in stock_prices]
        plt.figure(figsize=(10,5))
        plt.plot(stock_prices,deltas)
        plt.title('Delta of a Call Option as Underlying hanges')
        plt.xlabel('Stock Price')
        plt.ylabel('Delta')
        plt.grid(True)
    '''
    def delta_call_plot(self):
        S_range = (0.8*self.S,1.2*self.S)
        stock_prices = np.linspace(*S_range,num=1000)
        deltas = [greeks(S=S, X=X, t=t, r=r, sigma=sigma).delta_call() for S in stock_prices]
        fig,ax = plt.subplots(figsize=(10,5))
        ax.plot(stock_prices,deltas)
        ax.set_title('Delta of a Call Option as Underlying Changes')
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('Delta')
        ax.grid()
        return plt.gcf()
    def delta_put_plot(self):
        S_range = (0.8*self.S,1.2*self.S)
        stock_prices = np.linspace(*S_range, num=1000)
        deltas = [greeks(S=S, X=X, t=t, r=r, sigma=sigma).delta_put() for S in stock_prices]
        fig,ax = plt.subplots(figsize=(10,5))
        ax.plot(stock_prices,deltas)
        ax.set_title('Delta of a Put Option as Underlying Changes')
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('Delta')
        ax.grid()
        return plt.gcf()
    def gamma_plot(self):
        S_range = (0.8*self.S,1.2*self.S)
        stock_prices = np.linspace(*S_range,num=1000)
        gammas = [greeks(S=S, X=X, t=t, r=r, sigma=sigma).gamma() for S in stock_prices]
        fig,ax = plt.subplots(figsize=(10,5))
        ax.plot(stock_prices,gammas)
        ax.set_title('Gamma of an Option as Underlying Changes')
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('Gamma')
        ax.grid()
        return plt.gcf()
    def theta_call_plot(self):
        S_range = (0.8*self.S,1.2*self.S)
        stock_prices = np.linspace(*S_range, num=1000)
        thetas = [greeks(S=S, X=X, t=t, r=r, sigma=sigma).theta_call() for S in stock_prices]
        fig,ax = plt.subplots(figsize=(10,5))
        ax.plot(stock_prices,thetas)
        ax.set_title('Theta of a Call Option as Underlying Changes')
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('Theta')
        ax.grid()
        return plt.gcf()
    def theta_put_plot(self):
        S_range = (0.8*self.S,1.2*self.S)
        stock_prices = np.linspace(*S_range, num=1000)
        thetas = [greeks(S=S, X=X, t=t, r=r, sigma=sigma).theta_put() for S in stock_prices]
        fig,ax = plt.subplots(figsize=(10,5))
        ax.plot(stock_prices,thetas)
        ax.set_title('Theta of a Put Option as Underlying Changes')
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('Theta')
        ax.grid()
        return plt.gcf()
    def vega_plot(self):
        S_range = (0.8*self.S,1.2*self.S)
        stock_prices = np.linspace(*S_range,num = 1000)
        vegas = [greeks(S=S, X=X, t=t, r=r, sigma=sigma).vega() for S in stock_prices]
        fig,ax = plt.subplots(figsize=(10,5))
        ax.plot(stock_prices,vegas)
        ax.set_title('Vega of an Option as Underlying Changes')
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('Vega')
        ax.grid()
        return plt.gcf()
    def rho_call_plot(self):
        S_range = (0.8*self.S,1.2*self.S)
        stock_prices = np.linspace(*S_range, num=1000)
        rhos = [greeks(S=S, X=X, t=t, r=r, sigma=sigma).rho_call() for S in stock_prices]
        fig,ax = plt.subplots(figsize=(10,5))
        ax.plot(stock_prices,rhos)
        ax.set_title('Rho of a Call Option as Underlying Changes')
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('Rho')
        ax.grid()
        return plt.gcf()
    def rho_put_plot(self):
        S_range = (0.8*self.S,1.2*self.S)
        stock_prices = np.linspace(*S_range, num=1000)
        rhos = [greeks(S=S, X=X, t=t, r=r, sigma=sigma).rho_put() for S in stock_prices]
        fig,ax = plt.subplots(figsize=(10,5))
        ax.plot(stock_prices,rhos)
        ax.set_title('Rho of a Put Option as Underlying Changes')
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('Rho')
        ax.grid()
        return plt.gcf()

price_call = blackscholesmodel(S,X,t,r,sigma).call_price()
price_put = blackscholesmodel(S,X,t,r,sigma).put_price()

col1, col2 = st.columns(2)

with col1:
    with stylable_container(
        key="call_cont",
        css_styles=["""
        {
            text-align: center;
            background-color: green;
            border-radius: 1em;
            padding: 0.5em;
        }
        """]
    ):

        st.metric(label='CALL VALUE', value=f'${round(price_call, 2)}')

with col2:
    with stylable_container(
            key="put_cont",
            css_styles=["""
            {
                text-align: center;
                background-color: red;
                border-radius: 1em;
                padding: 0.5em;
            }
            """]
    ):
        st.metric(label = "PUT VALUE", value = f'${round(price_put,2)}')

st.divider()

st.write('### Option Price Heatmap')
generate_heatmap(S,X,t,r,sigma_range, S_range,option_type='call')
generate_heatmap(S,X,t,r,sigma_range, S_range,option_type='put')
st.divider()

st.write('### Greeks Visualization')
col3,col4 = st.columns(2)
col3.pyplot(plot_greeks.delta_call_plot(self=blackscholesmodel(S,X,t,r,sigma)))
col4.pyplot(plot_greeks.delta_put_plot(self=blackscholesmodel(S,X,t,r,sigma)))
col5,col6 = st.columns(2)
col5.pyplot(plot_greeks.gamma_plot(self=blackscholesmodel(S,X,t,r,sigma)))
col6.pyplot(plot_greeks.vega_plot(self=blackscholesmodel(S,X,t,r,sigma)))
col7,col8 = st.columns(2)
col7.pyplot(plot_greeks.theta_call_plot(self=blackscholesmodel(S,X,t,r,sigma)))
col8.pyplot(plot_greeks.theta_put_plot(self=blackscholesmodel(S,X,t,r,sigma)))
col9,col10 = st.columns(2)
col9.pyplot(plot_greeks.rho_call_plot(self=blackscholesmodel(S,X,t,r,sigma)))
col10.pyplot(plot_greeks.rho_put_plot(self=blackscholesmodel(S,X,t,r,sigma)))
