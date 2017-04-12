from sqlalchemy import Column, Integer, Float, String, PickleType, DateTime, ForeignKey, create_engine
from sqlalchemy.orm import relationship,backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
class Stock(Base):
    __tablename__ = 'stock'
    id = Column(Integer, primary_key=True)


class Trade(Base):
    __tablename__ = 'trade'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime)
    
    # given stats
    so = Column(Float)
    sh = Column(Float)
    sl = Column(Float)
    sc = Column(Float)
    tvl = Column(Float)
    ind = Column(Integer)

    # stats to calculate
    rcc = Column(Float)
    rco = Column(Float)
    roc = Column(Float)
    roo = Column(Float)
    rvp = Column(Float)
    avtvl = Column(Float)
    avrvp = Column(Float)

    stock_id = Column(Integer, ForeignKey('stock.id'))
    stock = relationship('Stock', backref='trades')


class DailyStat(Base):
    __tablename__ = 'dailystat'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime)

    avrcc = Column(Float)
    avroo = Column(Float)
    avroc = Column(Float)
    avrco = Column(Float)

class StatMat(Base):
    __tablename__ = 'statmat'
    id = Column(Integer, primary_key=True)
    name = Column(String(80), unique=True)
    mat = Column(PickleType)

class PortfolioReturn(Base):
    __tablename__ = 'portfolioreturn'
    id = Column(Integer, primary_key=True)
    name = Column(String(80))
    date = Column(DateTime)
    pr = Column(PickleType)
    wt = Column(PickleType)
    




def create_db():
    db = create_engine('sqlite:///../quanta.db', echo=True)
    Base.metadata.create_all(db)
    print('Created db.')

if __name__ == "__main__":
    create_db()


