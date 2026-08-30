"""
Data cleaning and preprocessing module for NEPSE stock data
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
from sqlalchemy.orm import Session
from database.models import DailyPrice, Company
from database.connection import SessionLocal

logger = logging.getLogger(__name__)


class DataCleaner:
    """Data cleaning and validation module"""
    
    def __init__(self, db_session: Optional[Session] = None):
        self.db = db_session or SessionLocal()
        self.cleaning_log = []

    def load_stock_data(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """
        Load stock data from database
        
        Args:
            symbol: Stock symbol
            days: Number of days to load
            
        Returns:
            DataFrame with OHLCV data or None if no data found
        """
        try:
            start_date = datetime.now().date() - timedelta(days=days)
            
            query = self.db.query(DailyPrice).filter(
                DailyPrice.symbol == symbol,
                DailyPrice.date >= start_date
            ).order_by(DailyPrice.date.asc())
            
            records = query.all()
            
            if not records:
                logger.warning(f"No data found for symbol: {symbol}")
                return None
            
            # Convert to DataFrame
            data = [{
                'date': record.date,
                'open': record.open_price,
                'high': record.high_price,
                'low': record.low_price,
                'close': record.close_price,
                'volume': record.traded_shares,
                'amount': record.traded_amount,
                'previous_close': record.previous_close,
                'price_change': record.price_change,
                'percent_change': record.percent_change
            } for record in records]
            
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading stock data for {symbol}: {str(e)}")
            return None

    def check_missing_values(self, df: pd.DataFrame) -> dict:
        """
        Check for missing values in dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with missing value statistics
        """
        missing_stats = {
            'total_missing': df.isnull().sum().sum(),
            'columns_with_missing': df.columns[df.isnull().any()].tolist(),
            'missing_per_column': df.isnull().sum().to_dict()
        }
        
        logger.info(f"Missing values: {missing_stats['total_missing']}")
        return missing_stats

    def handle_missing_values(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values
        
        Args:
            df: Input DataFrame
            method: 'forward_fill', 'backward_fill', or 'interpolate'
            
        Returns:
            DataFrame with missing values handled
        """
        try:
            df_cleaned = df.copy()
            
            if method == 'forward_fill':
                df_cleaned = df_cleaned.fillna(method='ffill')
                df_cleaned = df_cleaned.fillna(method='bfill')
                
            elif method == 'backward_fill':
                df_cleaned = df_cleaned.fillna(method='bfill')
                df_cleaned = df_cleaned.fillna(method='ffill')
                
            elif method == 'interpolate':
                df_cleaned = df_cleaned.interpolate(method='linear')
                df_cleaned = df_cleaned.fillna(method='bfill')
            
            logger.info(f"Missing values handled using {method}")
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            return df

    def detect_outliers(self, df: pd.DataFrame, column: str, threshold: float = 3.0) -> list:
        """
        Detect outliers using Z-score method
        
        Args:
            df: Input DataFrame
            column: Column to check
            threshold: Z-score threshold (default: 3.0)
            
        Returns:
            List of outlier indices
        """
        try:
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outlier_indices = np.where(z_scores > threshold)[0].tolist()
            
            logger.info(f"Detected {len(outlier_indices)} outliers in {column}")
            return outlier_indices
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            return []

    def remove_outliers(self, df: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from multiple columns
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            threshold: Z-score threshold
            
        Returns:
            DataFrame with outliers removed
        """
        try:
            df_cleaned = df.copy()
            
            for column in columns:
                z_scores = np.abs((df_cleaned[column] - df_cleaned[column].mean()) / df_cleaned[column].std())
                df_cleaned = df_cleaned[z_scores <= threshold]
            
            logger.info(f"Removed {len(df) - len(df_cleaned)} rows containing outliers")
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            return df

    def validate_data_quality(self, df: pd.DataFrame) -> dict:
        """
        Validate data quality
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'missing_values': df.isnull().sum().sum(),
            'duplicate_records': df.duplicated().sum(),
            'price_consistency': {
                'high_gte_low': (df['high'] >= df['low']).sum() == len(df),
                'close_between_hl': ((df['close'] >= df['low']) & (df['close'] <= df['high'])).sum() == len(df)
            }
        }
        
        logger.info(f"Data quality check: {quality_metrics}")
        return quality_metrics

    def clean_stock_data(self, df: pd.DataFrame, remove_outliers_flag: bool = True) -> pd.DataFrame:
        """
        Complete cleaning pipeline
        
        Args:
            df: Input DataFrame
            remove_outliers_flag: Whether to remove outliers
            
        Returns:
            Cleaned DataFrame
        """
        try:
            df_cleaned = df.copy()
            
            # Remove duplicates
            df_cleaned = df_cleaned.drop_duplicates(subset=['date'])
            logger.info(f"Removed {len(df) - len(df_cleaned)} duplicate records")
            
            # Handle missing values
            df_cleaned = self.handle_missing_values(df_cleaned)
            
            # Remove outliers
            if remove_outliers_flag:
                df_cleaned = self.remove_outliers(
                    df_cleaned,
                    columns=['close', 'high', 'low', 'volume']
                )
            
            # Validate
            self.validate_data_quality(df_cleaned)
            
            logger.info("Data cleaning completed successfully")
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Error in cleaning pipeline: {str(e)}")
            return df

    def close(self):
        """Close database session"""
        self.db.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    cleaner = DataCleaner()
    
    # Example usage
    df = cleaner.load_stock_data('NABIL', days=365)
    if df is not None:
        cleaned_df = cleaner.clean_stock_data(df)
        print(cleaned_df.head())
    
    cleaner.close()
