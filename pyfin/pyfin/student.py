import numpy as np
import pandas as pd

from pyfin import (
    annualised_ret,
    annualised_vol,
    max_drawdown,
    VaR_historical,
    CVaR_historical,
    sharpe_ratio
)

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

    # TODO: write your code here
    
    # Creat a Dictionary
    
    tear = {
        'ann_ret': annualised_ret(ret, freq = freq),
        'ann_vol': annualised_vol(ret, freq = freq),
        'max_dd': max_drawdown(ret)[0],
        f'VaR {alpha}': VaR_historical(ret, alpha = alpha),
        f'CVaR {alpha}': CVaR_historical(ret, alpha = alpha),
        'sharpe': sharpe_ratio(ret, freq = freq, rf = rf)
        
       
    }
    
    return pd.DataFrame(tear)

    