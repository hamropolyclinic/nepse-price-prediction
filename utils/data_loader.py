"""
Data Loader for fetching and managing stock data
"""
import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from preprocessing.cleaner import DataCleaner

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and manages stock data from various sources"""
    
    def __init__(self):
        self.data = {}
        self.cleaner = DataCleaner()

    def fetch_from_yfinance(self, symbol: str, start_date: Optional[str] = None,
                           end_date: Optional[str] = None, interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1wk', '1mo')
            
        Returns:
            DataFrame with stock data or None
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")
            
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            # keep date column lowercase for consistency
            data.rename(columns={'Date': 'date'}, inplace=True)

            # Ensure expected column names are present (Open, High, Low, Close, Adj Close, Volume)
            # yfinance usually returns these with capitalized names so nothing else required here
            
            self.data[symbol] = data
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance: {str(e)}")
            return None

    def fetch_from_csv(self, filepath: str, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load stock data from CSV file
        
        Args:
            filepath: Path to CSV file
            symbol: Stock symbol
            
        Returns:
            DataFrame with stock data or None
        """
        try:
            logger.info(f"Loading data from {filepath}")
            
            data = pd.read_csv(filepath)

            # Trim whitespace from column names
            data.columns = [c.strip() for c in data.columns]

            # Build a mapping to canonical column names to match yfinance output
            col_map = {}
            for col in data.columns:
                lc = col.lower().replace(' ', '').replace('_', '')
                if lc in ('date', 'timestamp'):
                    col_map[col] = 'date'
                elif lc == 'open':
                    col_map[col] = 'Open'
                elif lc == 'high':
                    col_map[col] = 'High'
                elif lc == 'low':
                    col_map[col] = 'Low'
                elif lc in ('close',) and 'adj' not in lc:
                    # prefer raw Close if present
                    col_map[col] = 'Close'
                elif 'adj' in lc and 'close' in lc:
                    col_map[col] = 'Adj Close'
                elif lc == 'volume':
                    col_map[col] = 'Volume'
                else:
                    # leave unknown columns as-is (they'll be kept)
                    col_map[col] = col

            data.rename(columns=col_map, inplace=True)

            # Ensure date column exists
            if 'date' not in data.columns:
                logger.error("CSV must contain a date/timestamp column")
                return None

            # Convert date to datetime
            data['date'] = pd.to_datetime(data['date'])
            data.sort_values('date', inplace=True)

            self.data[symbol] = data
            logger.info(f"Successfully loaded {len(data)} records from {filepath}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            return None

    def load_from_database(self, symbol: str, start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load stock data from database
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with stock data or None
        """
        try:
            # This is a placeholder - implement based on your database
            logger.info(f"Loading {symbol} data from database")
            # Implementation would depend on your database setup
            logger.warning("Database loading not implemented yet")
            return None
            
        except Exception as e:
            logger.error(f"Error loading from database: {str(e)}")
            return None

    def get_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get cached data for symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame or None
        """
        return self.data.get(symbol)

    def clean_data(self, symbol: str, remove_outliers: bool = True,
                  fill_method: str = 'forward') -> Optional[pd.DataFrame]:
        """
        Clean loaded data
        
        Args:
            symbol: Stock symbol
            remove_outliers: Whether to remove outliers
            fill_method: Method for filling missing values
            
        Returns:
            Cleaned DataFrame or None
        """
        try:
            data = self.get_data(symbol)
            if data is None:
                logger.error(f"No data found for {symbol}")
                return None
            
            logger.info(f"Cleaning data for {symbol}")
            
            # Remove duplicates
            data = data.drop_duplicates(subset=['date'], keep='first')
            
            # Sort by date
            data.sort_values('date', inplace=True)
            
            # Handle missing values
            if fill_method == 'forward':
                data.fillna(method='ffill', inplace=True)
            elif fill_method == 'backward':
                data.fillna(method='bfill', inplace=True)
            elif fill_method == 'linear':
                data.interpolate(method='linear', inplace=True)
            
            # Remove outliers if requested
            if remove_outliers:
                data = self._remove_outliers(data)
            
            # Drop any remaining NaN values
            data.dropna(inplace=True)
            
            self.data[symbol] = data
            logger.info(f"Data cleaning completed for {symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return None

    def _remove_outliers(self, data: pd.DataFrame, columns: List[str] = None,
                        std_factor: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers using standard deviation method
        
        Args:
            data: DataFrame with stock data
            columns: Columns to check for outliers
            std_factor: Number of standard deviations
            
        Returns:
            DataFrame with outliers removed
        """
        if columns is None:
            columns = ['Close', 'Volume']
        
        for col in columns:
            if col in data.columns:
                mean = data[col].mean()
                std = data[col].std()
                lower_bound = mean - (std_factor * std)
                upper_bound = mean + (std_factor * std)
                
                # Mark outliers
                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                num_outliers = outliers.sum()
                
                if num_outliers > 0:
                    logger.info(f"Removed {num_outliers} outliers from {col}")
                    data = data[~outliers]
        
        return data

    def get_price_column(self, symbol: str, column: str = 'Close') -> Optional[np.ndarray]:
        """
        Get specific price column as numpy array
        
        Args:
            symbol: Stock symbol
            column: Column name ('Close', 'Open', 'High', 'Low')
            
        Returns:
            NumPy array or None
        """
        try:
            data = self.get_data(symbol)
            if data is None:
                logger.error(f"No data found for {symbol}")
                return None
            
            if column not in data.columns:
                logger.error(f"Column '{column}' not found in data")
                return None
            
            return data[column].values
            
        except Exception as e:
            logger.error(f"Error getting price column: {str(e)}")
            return None

    def get_latest_price(self, symbol: str, column: str = 'Close') -> Optional[float]:
        """
        Get latest price for symbol
        
        Args:
            symbol: Stock symbol
            column: Column name
            
        Returns:
            Latest price or None
        """
        try:
            data = self.get_data(symbol)
            if data is None or len(data) == 0:
                logger.error(f"No data found for {symbol}")
                return None
            
            return float(data[column].iloc[-1])
            
        except Exception as e:
            logger.error(f"Error getting latest price: {str(e)}")
            return None

    def get_date_range(self, symbol: str) -> Optional[Tuple[str, str]]:
        """
        Get date range of available data
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (start_date, end_date) or None
        """
        try:
            data = self.get_data(symbol)
            if data is None or len(data) == 0:
                return None
            
            start_date = data['date'].min().strftime('%Y-%m-%d')
            end_date = data['date'].max().strftime('%Y-%m-%d')
            
            return (start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error getting date range: {str(e)}")
            return None

    def get_recent_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get recent data for symbol
        
        Args:
            symbol: Stock symbol
            days: Number of recent days
            
        Returns:
            DataFrame or None
        """
        try:
            data = self.get_data(symbol)
            if data is None or len(data) == 0:
                return None
            
            return data.tail(days)
            
        except Exception as e:
            logger.error(f"Error getting recent data: {str(e)}")
            return None

    def save_to_csv(self, symbol: str, filepath: str) -> bool:
        """
        Save data to CSV file
        
        Args:
            symbol: Stock symbol
            filepath: Path to save file
            
        Returns:
            True if successful
        """
        try:
            data = self.get_data(symbol)
            if data is None:
                logger.error(f"No data found for {symbol}")
                return False
            
            data.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
            return False

    def get_statistics(self, symbol: str) -> Optional[dict]:
        """
        Get basic statistics for stock data
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with statistics or None
        """
        try:
            data = self.get_data(symbol)
            if data is None or len(data) == 0:
                return None
            
            close_prices = data['Close'].values
            
            stats = {
                'symbol': symbol,
                'total_records': len(data),
                'start_date': data['date'].min().strftime('%Y-%m-%d'),
                'end_date': data['date'].max().strftime('%Y-%m-%d'),
                'price_min': float(close_prices.min()),
                'price_max': float(close_prices.max()),
                'price_mean': float(close_prices.mean()),
                'price_std': float(close_prices.std()),
                'volume_total': float(data['Volume'].sum()),
                'volume_mean': float(data['Volume'].mean()),
            }
            
            logger.info(f"Statistics for {symbol}: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    loader = DataLoader()
    logger.info("Data Loader initialized")
