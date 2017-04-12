import arrow
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
from pprint import pprint
import pudb
from sklearn.linear_model import LinearRegression

# db imports
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# application level imports
from ..models.quanta import Base, PortfolioReturn
from ..utils.plottools import plot_ts, plot_histogram
from ..utils.quantatools import quanta_login

DATA_SRC = 'application/csv/in_data.csv'
RESULTS_DIR = 'results/'
N_STOCKS = 100
N_DATES = 1003




def merge_dicts(*dict_args):
    ''' Merge multiple dictionaries into 1.
    '''
    res = {}
    for dictionary in dict_args:
        res.update(dictionary)
    return res







def get_drawdown(cumrp):
    ''' Calculates the maximum drawdown.
    '''
    prev = 0
    drawdown = 0
    duration = 0
    startDate = 0
    endDate = 0
    maxLoss = 0
    maxStartDate = 0
    maxEndDate = 0
    for t in range(1,len(cumrp)):
        if drawdown == 0:
            if cumrp[t] <= cumrp[t-1]:
                startDate = t
                drawdown = 1
        else:
            if cumrp[t] > cumrp[t-1]:
                drawdown = 0 
                endDate = t
                loss = cumrp[startDate] - cumrp[endDate]
                if loss > maxLoss:
                    maxStartDate = startDate
                    maxEndDate = endDate
                    maxLoss = loss
                    duration = maxEndDate - maxStartDate
    summary = {}
    summary['start'] = maxStartDate
    summary['end'] = maxEndDate
    summary['loss'] = maxLoss
    summary['duration'] = duration
    return summary

def get_alpha_beta(pr, naivePr):
    mdl = LinearRegression()
    X = np.array([naivePr]).T
    mdl.fit(X, pr)
    beta = mdl.coef_[0]
    alpha = mdl.intercept_
    return (alpha, beta) 

def plot_results(allPortfolios, fname):
    ''' Generates all of the required plots:
        * long-short return PR(t)
        * cumulative long-short return in natural logarithm
        * average weights (fill adjusted) for every day
        
        Generates summary statistics:
        * mean and standard deviation of daily log returns ln(1+PR(t))
        * sharpe ratio (annualized)
        * skewness and excess kurtosis
        * maximum drawdown: the largest cumulative loss incurred (# of days and peak-to-trough return)
        * correlation of the strategy return with the return on the corresponding equal weighted long-only portfolio
    '''
    df = pd.read_csv(DATA_SRC, header=None)
    allDates = df[0].tolist()
    allDates = map(lambda x: arrow.get(str(x), 'YYYYMMDD').date(), allDates)
    allDates = list(allDates)


    # plot long-short return
    plotData = [ (allDates, portfolio.pr, portfolio.name) for portfolio in allPortfolios ]
    plotKwargs = { 'title': 'Daily Return',
            'rotation': 45,
            'alpha': 0.5,
            'legend': True,
            'save': '{}{}_rp'.format(RESULTS_DIR, fname)}
    plot_ts(plotData, **plotKwargs)

    # plot histogram of long-short returns
    plotData = [ (portfolio.pr, portfolio.name) for portfolio in allPortfolios ]
    plotKwargs = { 'title': 'Histogram of Daily Returns',
            'bins': 50,
            'alpha': 0.5,
            'legend': True,
            'save': '{}{}_hist'.format(RESULTS_DIR, fname)}
    plot_histogram(plotData, **plotKwargs)
    
    # plot cumulative long-short return in natural logarithm
    prs = [ portfolio.pr for portfolio in allPortfolios ]
    cumPrs = [ np.cumsum(np.log(1 + pr)) for pr in prs ]
    plotData = [ (allDates, cumPr, portfolio.name) for cumPr, portfolio in zip(cumPrs, allPortfolios) ]
    plotKwargs = { 'title': 'Cumulative Return (Log)',
            'rotation': 45,
            'legend': True,
            'save': '{}{}_cumrp'.format(RESULTS_DIR, fname)}
    plot_ts(plotData, **plotKwargs)

    #drawdowns = [ get_drawdown(cumPr) for cumPr in cumPrs ]  
    mkt = [ p for p in allPortfolios if p.name == 'naive' ]
    calculate_alphas = 0
    if len(mkt) == 1:
        calculate_alphas = 1
        mkt = mkt[0]
        allABs = [ get_alpha_beta(portfolio.pr, mkt.pr) for portfolio in allPortfolios ]
    
    stats = {}
    for portfolio in allPortfolios:
        pname = portfolio.name
        pr = portfolio.pr
        stats['{} Sharpe ratio'.format(pname)] = np.mean(pr) / np.std(pr)
        stats['{} Annualized Sharpe ratio'.format(pname)] = np.sqrt(252) * np.mean(pr) / np.std(pr)
        stats['{} Skewness'.format(pname)] = ( np.sum(pr - np.mean(pr)) ** 3 ) / ( N_DATES * np.std(pr) ** 3 )
        stats['{} Kurtosis'.format(pname)] = ( np.sum(pr - np.mean(pr)) ** 4 ) / ( N_DATES * np.std(pr) ** 4 )
        if calculate_alphas == 1:
            alpha, beta = get_alpha_beta(pr, mkt.pr) 
            stats['{} alpha'.format(pname)] = alpha
            stats['{} beta'.format(pname)] = beta
        cumPr = np.cumsum(np.log(1 + pr))
        stats['{} drawdowns'.format(pname)] = get_drawdown(cumPr)
    for key, value in stats.items():
        print ('{} = {}'.format(key, value))
    
    
    
    #plt.show()



def get_portfolio_and_plot_results():
    session = quanta_login()

    # find the best performing portfolio A and portfolio B
    portfolioA = session.query(PortfolioReturn) \
            .filter(PortfolioReturn.name == 'Model A') \
            .all()
    allSharpes = [ np.mean(p.pr) / np.std(p.pr) for p in portfolioA ]
    maxInd = np.argmax(allSharpes)
    portfolioA = portfolioA[maxInd]

    portfolioB = session.query(PortfolioReturn) \
            .filter(PortfolioReturn.name == 'Model B') \
            .all()
    allSharpes = [ np.mean(p.pr) / np.std(p.pr) for p in portfolioB ]
    maxInd = np.argmax(allSharpes)
    portfolioB = portfolioB[0]
    
    naive = session.query(PortfolioReturn) \
            .filter(PortfolioReturn.name == 'Market') \
            .one()
    perfect = session.query(PortfolioReturn) \
            .filter(PortfolioReturn.name == 'Perfect Info') \
            .one()
    plot_results((naive, portfolioA), 'modelA')
    plot_results((naive, portfolioA, perfect), 'modelA_perfect')
    plot_results((naive, portfolioA, perfect, portfolioB), 'modelA_B_perfect')
    #plot_results((naive, portfolioA, portfolioB), 'modelA_B')
