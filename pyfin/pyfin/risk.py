import pandas as pd
import numpy as np
import yfinance as yf  
import scipy.stats as scs



def get_equity_data(tickers, start, end):
    """
    Download adjusted close prices and compute simple daily returns.

    Parameters
    ----------
    tickers : list or str
        Ticker symbol(s).
    start : str
        Start date (e.g. '2013-01-01').
    end : str
        End date (e.g. '2024-01-01').

    Returns
    -------
    prices : pd.DataFrame
        Adjusted close prices (columns ordered as input tickers).
    returns : pd.DataFrame
        Simple daily returns (same column order).
    """

    # Ensure tickers is a list
    if isinstance(tickers, str):
        tickers = [tickers]

    # Download data (adjusted for splits/dividends)
    data = yf.download(tickers, start=start, end=end, ignore_tz = True, progress=False)

    # Extract Close prices
    prices = data["Close"].copy()
    
    # Sort dates
    prices = prices.sort_index()

    # Reorder columns to match input ticker order
    prices = prices[tickers]

    # Compute simple daily returns
    returns = prices.pct_change()

    return prices, returns


    
def annualised_ret(ret, freq):
    """
    Computes annualised return for each ticker that is in the returns data. 
    
    Inputs
    ===========
    ret    : returns data as a DataFrame or a Series. The returns data can be in any time frame such as daily, monthly, etc... 
    freq   : frequency of the return in ret in terms of the number of periods per year. 12 for monthly returns, 252 for daily returns, etc...
    
    Returns
    ===========
    annualised ret: as a Scalar (for Series input) or a Series (for DataFrame input)
    """
    
    num_of_obs = ret.shape[0]
    return (1+ret).prod()**(freq/num_of_obs) -1

def annualised_vol(ret, freq):
    """
    Computes annualised volatility for each ticker that is in the returns data. 
    
    Inputs
    ===========
    ret    : returns data as a DataFrame or a Series. The returns data can be in any time frame such as daily, monthly, etc... 
    freq   : frequency of the return in ret in terms of the number of periods per year. 12 for monthly returns, 252 for daily returns, etc...
    
    Returns
    ===========
    annualised volatility: as a Scalar (for Series input) or a Series (for DataFrame input)
    """
    
    
    return ret.std(ddof=1) * (freq ** 0.5)

def rolling_vol(ret, window, freq):
    """
    Computes rolling annualised volatility for each ticker in the returns data.
    
    Inputs
    ===========
    ret      : returns data as a DataFrame or a Series.
               The returns data can be in any time frame such as daily, monthly, etc.
    window   : rolling window length in number of observations.
               For example, 12 for 12 months, 60 for 60 months, etc.
    freq     : frequency of the return in ret in terms of the number of periods per year.
               12 for monthly returns, 252 for daily returns, etc.
    
    Returns
    ===========
    rolling annualised volatility:
        - as a Series (for Series input)
        - as a DataFrame (for DataFrame input)
    """
    
    return ret.rolling(window).std() * (freq ** 0.5)

def max_drawdown(ret):
    """
    Computes the maximum drawdown of each ticker in the input data and when these events occurred.
    
    Inputs
    =========== 
    ret    : returns data as a DataFrame or a Series. The returns data can be in any time frame such as daily, monthly, etc... 
    
    Returns
    ===========
    (maximum drawdown, index of the max drawdown occurence) : a tuple that contains 2 items.
    When ret is a DataFrame, each item is a Series.
    When ret is a Series, each item is a Scalar.
    """
    
    wealth_index = (1 + ret).cumprod()
    peak = wealth_index.cummax() 
    drawdown = (wealth_index - peak)/peak
    return round(drawdown.min(), 4), drawdown.idxmin()
        
def VaR_historical(ret, alpha=5):
    """
    Computes the 1-period Value at Risk using historical data, at a specified confidence level.
   
    Inputs
    ===========
    ret    : returns data as a DataFrame or a Series. The returns data can be in any time frame such as daily, monthly, etc... 
    alpha  : confidence level
    
    Returns
    ===========
    Value at risk: as a Scalar (for Series input) or a Series (for DataFrame input)
    """
    
    return -ret.quantile(alpha/100)
        
def CVaR_historical(ret, alpha=5):
    """
    Computes the 1-period Conditional Value at Risk using historical data, at a specified confidence level
   
    Inputs
    ===========
    ret    : returns data as a DataFrame or a Series. The returns data can be in any time frame such as daily, monthly, etc... 
    alpha  : confidence level
    
    Returns
    ===========
    Conditional Value at risk: as a Scalar (for Series input) or a Series (for DataFrame input)
    """
    
    z = ret.quantile(alpha/100)
    return -ret[ret < z].mean()
   
def VaR_normal(ret, alpha=5):
    """
    Computes the Value at Risk assuming normal distribution in data, at a specified confidence level
        
    Inputs
    ===========
    ret    : returns data as a DataFrame or a Series. The returns data can be in any time frame such as daily, monthly, etc... 
    alpha  : confidence level
    
    Returns
    ===========
    Value at risk: as a Scalar (for Series input) or a Series (for DataFrame input)
    """
    
    mu = ret.mean()
    sigma = ret.std()
    z = scs.norm.ppf(alpha/100)
    return -(mu + z*sigma)


def sharpe_ratio(ret, freq, rf=0.03):
    """
    Computes annualised Sharpe ratio for each ticker that is in the returns data.
    
    Inputs
    ===========
    ret    : returns data as a DataFrame or a Series. The returns data can be in any time frame such as daily, monthly, etc...
    freq   : frequency of the return in ret in terms of the number of periods per year. 12 for monthly returns, 252 for daily returns, etc...
    rf     : risk-free rate per annum as a float, defaulted to be 0.03
    
    Returns
    ===========
    annualised Sharpe ratio: as a Scalar (for Series input) or a Series (for DataFrame input)
    """
    
    mean_excess_ret = ret.mean()*freq - rf;
    stdev = ret.std()*np.sqrt(freq)  
    return mean_excess_ret/stdev 

def tear_sheet(ret, freq=12, alpha=5, rf=0.03):
    """
    Builds a standard tear sheet of historical performance & risk measures.
    Assumes ret is a BALANCED panel (aligned dates, no missing values).

    Inputs
    ===========
    ret   : returns as a DataFrame (rows=time, cols=assets)
    freq  : periods per year (12 monthly, 252 daily, etc.)
    alpha : tail probability for historical VaR/CVaR (percent, e.g. 5)
    rf    : annual risk-free rate for Sharpe ratio

    Returns
    ===========
    tear sheet: DataFrame indexed by asset
    """
    
    # Check if there is any nan in input
    if ret.isna().any().any():
        raise ValueError("tear_sheet assumes balanced data: ret contains missing values. Align/clean first.")
    
    # Creat a Dictionary
    
    tear = {
        'ann_ret': annualised_ret(ret, freq = freq),
        'ann_vol': annualised_vol(ret, freq = freq),
        'max_dd': max_drawdown(ret)[0],
        f'VaR {alpha}': VaR_historical(ret, alpha = alpha),
        f'CVaR {alpha}': CVaR_historical(ret, alpha = alpha),
        'sharpe': sharpe_ratio(ret, freq = freq, rf = rf)
        
    }
    
    # Convert to DataFrame then output
    return pd.DataFrame(tear)
    