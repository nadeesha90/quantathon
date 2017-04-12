import arrow
import numpy as np
import pandas as pd
import pudb
from scipy.optimize import basinhopping
from scipy.optimize import minimize

# db imports
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound

# application level imports
from ..models.quanta import Base, Stock, Trade, DailyStat, StatMat, PortfolioReturn
from ..tasks.plot_results import plot_results
from ..utils.quantatools import quanta_login 

N_VARS = 12
SCALE_FACTOR = 100

#def login():
#    engine = create_engine('sqlite:///application/quanta.db')
#    Base.metadata.bind = engine
#    DBSession = sessionmaker(bind=engine)
#    session = DBSession()
#    return session


def w2 (ayys, mats):
    ''' Calculate the weight function.
    '''
    #result = np.sum([ayys[i] * mats[i] for i in range(len(ayys))], axis=0)
    result = np.sum([ayy * mat for (ayy, mat) in zip(ayys, mats) ], axis=0)
    return result
def portfolio_return(ayys, rocMat, indMat, mats):
    ''' Calculate portfolio daily returns, sum of fill * weight * open-close return / absolute value of fill * wt.
        
            r_t = sum_j ( fill_tj * wt_tj * roc_tj ) / ( sum_j ( |fill_tj * wt_tj | ) ), 
        
        where j is stock index, and t is time index.
    '''
    wMat = w2(ayys, mats)

    fill =  np.multiply(wMat, indMat)
    fill[fill == 0] = 1.0
    fill = 0.5 * (np.sign(fill) + 1)
    
    numero = np.sum(np.multiply(np.multiply(fill, wMat), rocMat), axis=1)
    denom = np.sum(np.abs(np.multiply (fill, wMat)), axis=1)
    
    result = np.divide(numero,denom)
    result = np.nan_to_num(result)
    return result
def sharpe_ratio(ayys, *args):
    ''' Calculate Sharpe ratio.
            
            S = mean(r)/std(r), where r = portfolio_return
    '''
    #global end_train_index
    rocMat = args[0]
    indMat = args[1]
    mats = args[2]
    result = portfolio_return(ayys, rocMat, indMat, mats)
    result = np.mean(result)/np.std(result)
    return -1*result

def optimize (opt, ayys, data, func_name, optimizer='bhopping'):
    '''
    Args:
        opt (dict): { method=scipy.optimize.minimize method parameter }
        ayys (np.float): init random numbers 
        data (dict): { roc_mat, ind_mat, mats }
        func (function): sharpe_ratio
    '''
    mini = {'args': (data['roc_mat'], data['ind_mat'], data['mats']), 'method':opt['method']}
    func = globals()[func_name]
    if optimizer == 'bhopping':
        opt['temp'] = 10
        opt['niter'] = 10
        opt['stepsize'] = 1
        opt['niter_success'] = 10
        final_opt_ayys = basinhopping (func, ayys, minimizer_kwargs=mini, niter=opt['niter'], T=opt['temp'], stepsize=opt['stepsize'], disp=opt['disp'], niter_success=opt['niter_success'])
    
    elif optimizer == 'minimize':
        final_opt_ayys = minimize (func, ayys, args=mini['args'], options={'maxiter':opt['maxiter'], 'disp':True}, method=opt['method'], jac=False)
    return final_opt_ayys


def get_mats(session):
    ''' Generate the list of matrices.
    '''
    data = {}
    mats = []
    try:
        rocQuery = session.query(StatMat) \
                .filter(StatMat.name == 'rproc') \
                .one()
        data['roc_mat'] = rocQuery.mat

        indQuery = session.query(StatMat) \
                .filter(StatMat.name == 'ind') \
                .one()
        data['ind_mat'] = indQuery.mat

        matNames = [ 'rcc', 'roo', 'roc', 'rco',
                'tvlrcc', 'tvlroo', 'tvlroc', 'tvlrcc', 
                'rvprcc', 'rvproo', 'rvproc', 'rvprco' ]
        for matName in matNames:
            matQuery = session.query(StatMat) \
                    .filter(StatMat.name == matName) \
                    .one()
            mats.append(matQuery.mat)
        data['mats'] = mats

    except NoResultFound:
        print ('Missing a matrix.')
        sys.exit()
    except MultipleResultsFound:
        print ('Multiple matrices found.')
        sys.exit()
    return data        


def ttsplit(data, percent):
    train_data = {}
    test_data = {}
    middle_index = int(np.floor(data['roc_mat'].shape[0] * percent))

    train_data['roc_mat'] = data['roc_mat'][:middle_index]
    test_data['roc_mat'] = data['roc_mat'][middle_index:]
    train_data['ind_mat'] = data['ind_mat'][:middle_index]
    test_data['ind_mat'] = data['ind_mat'][middle_index:]
    train_data['mats'] = []
    test_data['mats'] = []
    for mat in data['mats']:
        train_data['mats'].append(mat[:middle_index])
        test_data['mats'].append(mat[middle_index:])

    return train_data, test_data




def optimize_part2():
    ''' Part 2 question: calculate the optimal weights to maximize sharpe ratio.
    '''
    session = quanta_login()
    #session = login()
    percent = 0.8
    data = get_mats(session)
    trainData, testData = ttsplit(data, percent)
    
    # ***************************************
    # Insert any strategy here
    opt = { 'method': 'BFGS', 'disp': True }
    ayys = np.random.rand(N_VARS) * SCALE_FACTOR
    func_name = 'sharpe_ratio'
    res = optimize (opt, ayys, trainData, func_name)
    optAyys = res.x
    # ***************************************
    
    wt = w2(optAyys, data['mats'])
    pr = portfolio_return(optAyys, data['roc_mat'], data['ind_mat'], data['mats'])
    currentTime = arrow.utcnow().datetime
    portfolioReturn = {'name': 'Model A',
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
    
    plot_results((naivePortfolio, newPortfolioEntry), 'modelA')


