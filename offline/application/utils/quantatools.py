import numpy as np

# db imports
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound

# application level imports
from ..models.quanta import Base, StatMat


def quanta_login():
    ''' Login to the database. '''
    engine = create_engine('sqlite:///application/quanta.db')
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    return session



def get_mats(session, matNames):
    ''' Generate the list of matrices.
        Args:
            session : sqlalchemy db session instance
            matNames (list): names of matrices in statmats table.
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


def causal_ttsplit(data, percent):
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


