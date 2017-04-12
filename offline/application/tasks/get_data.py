import arrow
import numpy as np
import pandas as pd
import pudb
from toolz import pipe
from progress.bar import IncrementalBar

# db imports
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# application level imports
from ..models.quanta import Base, Stock, Trade, DailyStat, StatMat
from ..utils.quantatools import quanta_login 

# Constants
DATA_SRC = 'application/csv/in_data.csv'
N_STOCKS = 100
N_DATES = 1003

#def login():
#    engine = create_engine('sqlite:///application/quanta.db')
#    Base.metadata.bind = engine
#    DBSession = sessionmaker(bind=engine)
#    session = DBSession()
#    return session


def populate_stocks():
    session = quanta_login()
    df = pd.read_csv(DATA_SRC, header=None)
    allDates = df[0]
    del allDates[0]
    colnames = ['so', 'sh', 'sl', 'sc', 'tvl', 'ind']

    stockBar = IncrementalBar('Populating stocks.', max=N_STOCKS)
    for stock in range(N_STOCKS):
        ind = pipe(range(1,7),
                list,
                lambda rng: [ 6 * stock + i for i in rng ]
                )
        trades = df[ind]
        trades.columns = colnames
        
        newStock = Stock()
        session.add(newStock)
        session.commit()

        tradesDict = trades.to_dict(orient='records')
        scLast = tradesDict[0]['sc']
        soLast = tradesDict[0]['so']
        tvlVec = [ tradesDict[0]['tvl'] ] 
        rvpVec = [ 1 ]
        del tradesDict[0]
        for trade, dateStr in zip(tradesDict, allDates):
            date = arrow.get(str(dateStr), 'YYYYMMDD')
            trade['date'] = date.date()
            # stats to calculate
            trade['rcc'] = trade['sc'] / scLast - 1
            trade['rco'] = trade['so'] / scLast - 1
            trade['roo'] = trade['so'] / soLast - 1
            trade['roc'] = trade['sc'] / trade['so'] - 1
            trade['rvp'] = ( 1 / (4*np.log(2)) ) * ( np.log(trade['sh']) - np.log(trade['so']) ) ** 2

            if len(tvlVec) >= 200:
                del tvlVec[0]
                del rvpVec[0]
            trade['avtvl'] = np.mean(tvlVec)
            trade['avrvp'] = np.mean(rvpVec)
            trade['stock'] = newStock
            newTrade = Trade(**trade)
            session.add(newTrade)

            scLast = trade['sc']
            soLast = trade['so']
            tvlVec.append(trade['tvl'])
            rvpVec.append(trade['rvp'])
        session.commit()
        stockBar.next()
    stockBar.finish()


def populate_dailystats():
    session = quanta_login()
    df = pd.read_csv(DATA_SRC, header=None)
    allDates = df[0].tolist()
    del allDates[0]

    statBar = IncrementalBar('Populating dailystats.', max=len(allDates))
    for dateStr in allDates:
        date = arrow.get(str(dateStr), 'YYYYMMDD')
        
        dailyStat = {}
        dailyStat['date'] = date.date()
        allTrades = session.query(Trade) \
                .filter(Trade.date == date.datetime) \
                .all()
        dailyStat['avrcc'] = np.mean([ trade.rcc for trade in allTrades ])
        dailyStat['avroo'] = np.mean([ trade.roo for trade in allTrades ])
        dailyStat['avroc'] = np.mean([ trade.roc for trade in allTrades ])
        dailyStat['avrco'] = np.mean([ trade.rco for trade in allTrades ])
        newDS = DailyStat(**dailyStat)
        session.add(newDS)
        statBar.next()
    statBar.finish()
    session.commit()
    



def populate_w2_mats():
    session = quanta_login()
    df = pd.read_csv(DATA_SRC, header=None)
    allDates = df[0].tolist()
    del allDates[0]
    allDates = map(lambda x: arrow.get(str(x), 'YYYYMMDD'), allDates)
    allDates = list(allDates)

    rccMat = np.zeros([N_DATES, N_STOCKS])
    rooMat = np.zeros([N_DATES, N_STOCKS])
    rocMat = np.zeros([N_DATES, N_STOCKS])
    rcoMat = np.zeros([N_DATES, N_STOCKS])
    
    tvlRccMat = np.zeros([N_DATES, N_STOCKS])
    tvlRooMat = np.zeros([N_DATES, N_STOCKS])
    tvlRocMat = np.zeros([N_DATES, N_STOCKS])
    tvlRcoMat = np.zeros([N_DATES, N_STOCKS])
    
    rvpRccMat = np.zeros([N_DATES, N_STOCKS])
    rvpRooMat = np.zeros([N_DATES, N_STOCKS])
    rvpRocMat = np.zeros([N_DATES, N_STOCKS])
    rvpRcoMat = np.zeros([N_DATES, N_STOCKS])

    indMat = np.zeros([N_DATES, N_STOCKS])
    rpRoc = np.zeros([N_DATES, N_STOCKS])
    matBar = IncrementalBar('Populating W2 matrices', max=N_DATES)
    for t in range(1, len(allDates)):
        date = allDates[t]
        prevDate = allDates[t-1]
        try:
            dateStat = session.query(DailyStat) \
                    .filter(DailyStat.date == date.datetime) \
                    .one()
        except NoResultFound:
            print ('No daily stats for {}'.format(date))
            sys.exit()
        except MultipleResultsFound:
            print ('Multiple daily stats for {}'.format(date))
        prevDateStat = session.query(DailyStat) \
                    .filter(DailyStat.date == prevDate.datetime) \
                    .one()

        dateTrades = session.query(Trade) \
                .filter(Trade.date == date.datetime) \
                .all()
        prevDateTrades = session.query(Trade) \
                .filter(Trade.date == prevDate.datetime) \
                .all()
        assert len(prevDateTrades) == N_STOCKS, 'Missing stock data for {}'.format(prevDate)
        
        for j in range(N_STOCKS):
            dateTrade = dateTrades[j]
            prevDateTrade = prevDateTrades[j]
            rccMat[t][j] = ( prevDateTrade.rcc - prevDateStat.avrcc ) / N_STOCKS
            rooMat[t][j] = ( dateTrade.roo - dateStat.avroo ) / N_STOCKS
            rocMat[t][j] = ( prevDateTrade.roc - prevDateStat.avroc ) / N_STOCKS
            rcoMat[t][j] = ( dateTrade.rco - dateStat.avrco ) / N_STOCKS

            tvlRccMat[t][j] = ( prevDateTrade.tvl / prevDateTrade.avtvl ) * ( rccMat[t][j] )
            tvlRooMat[t][j] = ( prevDateTrade.tvl / prevDateTrade.avtvl ) * ( rooMat[t][j] )
            tvlRocMat[t][j] = ( prevDateTrade.tvl / prevDateTrade.avtvl ) * ( rocMat[t][j] )
            tvlRcoMat[t][j] = ( prevDateTrade.tvl / prevDateTrade.avtvl ) * ( rcoMat[t][j] )

            rvpRccMat[t][j] = ( prevDateTrade.rvp / prevDateTrade.avrvp ) * ( rccMat[t][j] )
            rvpRooMat[t][j] = ( prevDateTrade.rvp / prevDateTrade.avrvp ) * ( rooMat[t][j] )
            rvpRocMat[t][j] = ( prevDateTrade.rvp / prevDateTrade.avrvp ) * ( rocMat[t][j] )
            rvpRcoMat[t][j] = ( prevDateTrade.rvp / prevDateTrade.avrvp ) * ( rcoMat[t][j] )

            indMat[t][j] = dateTrade.ind
            rpRoc[t][j] = dateTrade.roc
        matBar.next()
    matBar.finish()

    allMats = [
            { 'name': 'rcc', 'mat': rccMat },
            { 'name': 'roo', 'mat': rooMat },
            { 'name': 'roc', 'mat': rocMat },
            { 'name': 'rco', 'mat': rcoMat },
            { 'name': 'tvlrcc', 'mat': tvlRccMat },
            { 'name': 'tvlroo', 'mat': tvlRooMat },
            { 'name': 'tvlroc', 'mat': tvlRocMat },
            { 'name': 'tvlrco', 'mat': tvlRcoMat },
            { 'name': 'rvprcc', 'mat': rvpRccMat },
            { 'name': 'rvproo', 'mat': rvpRooMat },
            { 'name': 'rvproc', 'mat': rvpRocMat },
            { 'name': 'rvprco', 'mat': rvpRcoMat },
            { 'name': 'ind', 'mat': indMat },
            { 'name': 'rproc', 'mat': rpRoc }
            ]

    res = map(lambda a: StatMat(**a), allMats)
    statMats = list(res)
    
    [ session.add(statMat) for statMat in statMats ]
    session.commit()

