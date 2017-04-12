import arrow
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
import pudb
from scipy.optimize import basinhopping
from scipy.optimize import minimize
from sklearn.feature_selection import SelectPercentile, f_regression

# model imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import LassoLarsCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# db imports
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound

# application level imports
from ..models.quanta import Base, Stock, Trade, DailyStat, StatMat, PortfolioReturn
from ..tasks.plot_results import plot_results
from ..utils.quantatools import quanta_login, get_mats, causal_ttsplit 
from ..utils.plottools import plot_histogram



N_VARS = 12
SCALE_FACTOR = 100
RESULTS_DIR = 'results/'



def portfolio_return(wMat, rocMat, indMat):
    ''' Calculate portfolio daily returns, sum of fill * weight * open-close return / absolute value of fill * wt.
        
            r_t = sum_j ( fill_tj * wt_tj * roc_tj ) / ( sum_j ( |fill_tj * wt_tj | ) ), 
        
        where j is stock index, and t is time index.
    '''

    fill =  np.multiply(wMat, indMat)
    fill[fill == 0] = 1.0
    fill = 0.5 * (np.sign(fill) + 1)
    
    numero = np.sum(np.multiply(np.multiply(fill, wMat), rocMat), axis=1)
    denom = np.sum(np.abs(np.multiply (fill, wMat)), axis=1)
    
    result = np.divide(numero,denom)
    result = np.nan_to_num(result)
    return result

def get_XMatrix(allMats, j):
    ''' Goes through a list of matrices and creates a matrix from the j-th column of each.
    '''
    X = [ mat[j] for mat in allMats ]
    X = np.asarray(X).T
    return X




def generate_wts(data, percent=0.8):
    ''' Loop through every stock and run prediction models.
    '''

    trainData, testData = causal_ttsplit(data, percent)

    rocMat = trainData['roc_mat'].T
    # copy data['mats'] and append 'ind_mat'
    allMats = list(trainData['mats'])
    allMats.append(trainData['ind_mat'])
    allMats = [ mat.T for mat in allMats ]

    testRocMat = testData['roc_mat'].T
    testAllMats = list(testData['mats'])
    testAllMats.append(testData['ind_mat'])
    testAllMats = [ mat.T for mat in testAllMats ]

    
    wts = []
    allBestRegressors = []
    allScores = []
    #allCoefs = []
    
    nStocks = rocMat.shape[0] 
    for j in range(nStocks):
        yTrain = rocMat[j] 
        XTrain = get_XMatrix(allMats, j)
        
        yTest = testRocMat[j] 
        XTest = get_XMatrix(testAllMats, j)

        regressors = [
             ("AdaBoostRegressor-exploss",              AdaBoostRegressor(loss='exponential')),
             ("AdaBoostRegressor-estimators=200",       AdaBoostRegressor(n_estimators=200)),
             ("LinearRegression",                       LinearRegression()),
             ("Lasso0.2",                               Lasso(alpha=0.2)),
             ("Lasso0.1",                               Lasso(alpha=0.1)),
             ("Lasso0.8",                               Lasso(alpha=0.8)),
             ("BaggingRegressor",                       BaggingRegressor()),
             ("BaggingRegressor-bootstrapped",          BaggingRegressor(bootstrap_features=True)),
             ("GradientBoostingRegressor",              GradientBoostingRegressor()),
             ("GradientBoostingRegressor-hubertloss",   GradientBoostingRegressor(loss='lad')),
             ("LassoLarsIC",                            LassoLarsIC(criterion='aic')),
             ("LassoLarsCV",                            LassoLarsCV()),
             ("SVR-rbf",                                SVR(kernel='rbf')),
             ("SVR-poly",                               SVR(kernel='poly')),
             ("SVR-sigmoid",                            SVR(kernel='sigmoid')),
             ("DecisionTreeRegressor",                  DecisionTreeRegressor()),
             ("RandomForestRegressor",                  RandomForestRegressor()),
        ]

        maxScore = -100
        bestRegressor = None
        for (name, r) in regressors:
            r.fit(XTrain, yTrain)
            score = r.score(XTest, yTest)

            if score > maxScore:
                maxScore = score
                bestRegressor = (name, r)
        
        allBestRegressors.append(bestRegressor[0])
        allScores.append(maxScore)
        print('Best regressor for stock {} is {}, Score = {}'.format(j, bestRegressor[0], maxScore))
        X = np.concatenate((XTrain, XTest), axis=0)
        predWt = bestRegressor[1].predict(X)
        #allCoefs.append(bestRegressor[1].coef_)
        wts.append(predWt)
    wts = np.asarray(wts).T

    plot_regressor_histogram(allBestRegressors)
    plotData = [ (allScores, r'$R^2$') ]
    plotKwargs = { 'title': r'Histogram of $R^2$ scores',
            'bins': 50,
            'legend' : False,
            'ylabel' : 'Number of stocks',
            'save': 'results/modelB_prediction_hist'}
    plot_histogram(plotData, **plotKwargs)

    return wts

def plot_regressor_histogram(regressors):
    bestRegressors = dict((x, regressors.count(x)) for x in regressors)
    fig1 = plt.figure()
    plt.bar(range(len(bestRegressors)), bestRegressors.values(), align='center')
    plt.title('Best Regression Models')
    plt.ylabel('Number of stocks')
    plt.xticks(range(len(bestRegressors)), bestRegressors.keys(), rotation=90)
    fig1.savefig('{}{}.svg'.format(RESULTS_DIR, 'modelB_regressors'), format='svg', dpi=1200, bbox_inches='tight')
    fig1.savefig('{}{}.eps'.format(RESULTS_DIR, 'modelB_regressors'), format='eps', dpi=1200, bbox_inches='tight')

def optimize_part4():
    ''' Part 4 question: calculate the optimal weights to maximize sharpe ratio. The
    strategy is to predict ROC = f(MATS)
    '''
    session = quanta_login()
    #session = login()
    
    # ***************************************
    # Insert any strategy here
    percent = 0.8 # for test-train split
    matNames = [ 'rcc', 'roo', 'roc', 'rco',
            'tvlrcc', 'tvlroo', 'tvlroc', 'tvlrcc', 
            'rvprcc', 'rvproo', 'rvproc', 'rvprco' ]
    data = get_mats(session, matNames) # data is dict with 'roc_mat', 'ind_mat', 'mats'
    wt = generate_wts(data, percent)
    # ***************************************
    
    pr = portfolio_return(wt, data['roc_mat'], data['ind_mat'])
    currentTime = arrow.utcnow().datetime
    portfolioReturn = {'name': 'Model B',
            'date': currentTime,
            'pr': pr,
            'wt': wt
            }
    newPortfolioEntry = PortfolioReturn(**portfolioReturn)
    session.add(newPortfolioEntry)
    session.commit()

    # Get naive portfolio
    try:
        naivePortfolio = session.query(PortfolioReturn) \
                .filter(PortfolioReturn.name == 'Market') \
                .one()
    except NoResultFound:
        # create naive model
        naiveWt = np.ones([ wt.shape[0], wt.shape[1] ])
        fill =  np.multiply(naiveWt, data['ind_mat'])
        fill[fill == 0] = 1.0
        fill = 0.5 * (np.sign(fill) + 1)
    
        numero = np.sum(np.multiply(np.multiply(fill, naiveWt), data['roc_mat']), axis=1)
        denom = np.sum(np.abs(np.multiply (fill, naiveWt)), axis=1)
        naivePr = np.divide(numero,denom)
        naivePr = np.nan_to_num(naivePr)

        naivePortfolioReturn = {'name': 'Market',
                'date': currentTime,
                'pr': naivePr,
                'wt': naiveWt
                }
        naivePortfolio = PortfolioReturn(**naivePortfolioReturn)
        session.add(naivePortfolio)
        session.commit()
    except MultipleResultsFound:
        print ('Multiple naive portfolios found.')
        sys.exit()
    
    # Get perfect prediction portfolio
    try:
        perfectPortfolio = session.query(PortfolioReturn) \
                .filter(PortfolioReturn.name == 'Perfect Info') \
                .one()
    except NoResultFound:
        # create perfect model
        perfectWt = data['roc_mat']
        fill =  np.multiply(perfectWt, data['ind_mat'])
        fill[fill == 0] = 1.0
        fill = 0.5 * (np.sign(fill) + 1)
    
        numero = np.sum(np.multiply(np.multiply(fill, perfectWt), data['roc_mat']), axis=1)
        denom = np.sum(np.abs(np.multiply (fill, perfectWt)), axis=1)
        perfectPr = np.divide(numero,denom)
        perfectPr = np.nan_to_num(perfectPr)

        perfectPortfolioReturn = {'name': 'Perfect Info',
                'date': currentTime,
                'pr': perfectPr,
                'wt': perfectWt
                }
        perfectPortfolio = PortfolioReturn(**perfectPortfolioReturn)
        session.add(perfectPortfolio)
        session.commit()
    except MultipleResultsFound:
        print ('Multiple naive portfolios found.')
        sys.exit()
    
    plot_results((naivePortfolio, perfectPortfolio, newPortfolioEntry), 'modelB')


