"""
Database models for NEPSE stock data storage
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Date
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class Company(Base):
    """Company information table"""
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    sector = Column(String(100))
    market_cap = Column(Float)
    listed_date = Column(Date)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Company(symbol='{self.symbol}', name='{self.name}')>"


class DailyPrice(Base):
    """Daily OHLCV (Open, High, Low, Close, Volume) data"""
    __tablename__ = "daily_prices"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    
    # Price data
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    
    # Volume data
    traded_shares = Column(Integer)
    traded_amount = Column(Float)
    
    # Change indicators
    previous_close = Column(Float)
    price_change = Column(Float)
    percent_change = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<DailyPrice(symbol='{self.symbol}', date='{self.date}', close={self.close_price})>"


class MarketIndex(Base):
    """NEPSE Index daily data"""
    __tablename__ = "market_indices"

    id = Column(Integer, primary_key=True, index=True)
    index_name = Column(String(50), nullable=False)  # e.g., NEPSE, SENSITIVE
    date = Column(Date, index=True, nullable=False)
    
    # Index values
    open_value = Column(Float, nullable=False)
    high_value = Column(Float, nullable=False)
    low_value = Column(Float, nullable=False)
    close_value = Column(Float, nullable=False)
    
    # Market data
    total_traded_shares = Column(Integer)
    total_traded_amount = Column(Float)
    
    # Change indicators
    previous_close = Column(Float)
    index_change = Column(Float)
    percent_change = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<MarketIndex(index='{self.index_name}', date='{self.date}', close={self.close_value})>"


class DataFetchLog(Base):
    """Log of data fetch operations"""
    __tablename__ = "data_fetch_logs"

    id = Column(Integer, primary_key=True, index=True)
    fetch_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    source = Column(String(50), nullable=False)  # e.g., 'NEPSE_API', 'WEB_SCRAPER'
    status = Column(String(20), nullable=False)  # 'SUCCESS', 'FAILED', 'PARTIAL'
    records_fetched = Column(Integer, default=0)
    error_message = Column(String(500))
    duration_seconds = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<DataFetchLog(source='{self.source}', status='{self.status}', timestamp='{self.fetch_timestamp}')>"
