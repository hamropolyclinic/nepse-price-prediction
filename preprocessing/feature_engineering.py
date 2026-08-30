"""
Feature engineering module for NEPSE stock data
"""
import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering and technical indicators"""
    
    def __init__(self):
        self.scaler_minmax = MinMaxScaler()
        self.scaler_standard = StandardScaler()

    # ============ TREND INDICATORS ============
    
    @staticmethod
    def moving_average(df: pd.DataFrame, column: str, window: int) -> pd.Series:
        """
        Simple Moving Average (SMA)
        
        Args:
            df: DataFrame with price data
            column: Column to calculate MA on
            window: Window size
            
        Returns:
            Series with moving averages
        """
        return df[column].rolling(window=window).mean()

    @staticmethod
    def exponential_moving_average(df: pd.DataFrame, column: str, span: int) -> pd.Series:
        """
        Exponential Moving Average (EMA)
        
        Args:
            df: DataFrame with price data
            column: Column to calculate EMA on
            span: Span size
            
        Returns:
            Series with exponential moving averages
        """
        return df[column].ewm(span=span, adjust=False).mean()

    # ============ MOMENTUM INDICATORS ============
    
    @staticmethod
    def rsi(df: pd.DataFrame, column: str = 'close', period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI)
        
        Args:
            df: DataFrame with price data
            column: Price column to calculate RSI on
            period: Period for RSI calculation
            
        Returns:
            Series with RSI values (0-100)
        """
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    @staticmethod
    def macd(df: pd.DataFrame, column: str = 'close', fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence)
        
        Args:
            df: DataFrame with price data
            column: Price column
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (MACD, Signal line, Histogram)
        """
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    @staticmethod
    def stochastic_oscillator(df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator
        
        Args:
            df: DataFrame with OHLC data
            period: Period for calculation
            smooth_k: Smoothing period for %K
            smooth_d: Smoothing period for %D
            
        Returns:
            Tuple of (%K, %D)
        """
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        k_smooth = k.rolling(window=smooth_k).mean()
        d = k_smooth.rolling(window=smooth_d).mean()
        
        return k_smooth, d

    # ============ VOLATILITY INDICATORS ============
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR)
        
        Args:
            df: DataFrame with OHLC data
            period: Period for calculation
            
        Returns:
            Series with ATR values
        """
        df_copy = df.copy()
        
        # True Range
        df_copy['tr1'] = df_copy['high'] - df_copy['low']
        df_copy['tr2'] = abs(df_copy['high'] - df_copy['close'].shift())
        df_copy['tr3'] = abs(df_copy['low'] - df_copy['close'].shift())
        
        df_copy['tr'] = df_copy[['tr1', 'tr2', 'tr3']].max(axis=1)
        atr = df_copy['tr'].rolling(window=period).mean()
        
        return atr

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, column: str = 'close', period: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        
        Args:
            df: DataFrame with price data
            column: Price column
            period: Period for SMA
            num_std: Number of standard deviations
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        sma = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()
        
        upper_band = sma + (num_std * std)
        lower_band = sma - (num_std * std)
        
        return upper_band, sma, lower_band

    # ============ VOLUME INDICATORS ============
    
    @staticmethod
    def volume_sma(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Volume Simple Moving Average"""
        return df['volume'].rolling(window=window).mean()

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """
        On-Balance Volume (OBV)
        
        Args:
            df: DataFrame with price and volume data
            
        Returns:
            Series with OBV values
        """
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv

    # ============ PRICE-BASED FEATURES ============
    
    @staticmethod
    def returns(df: pd.DataFrame, column: str = 'close', periods: int = 1) -> pd.Series:
        """
        Calculate returns (simple and log returns)
        
        Args:
            df: DataFrame with price data
            column: Price column
            periods: Number of periods
            
        Returns:
            Series with returns
        """
        return df[column].pct_change(periods=periods)

    @staticmethod
    def log_returns(df: pd.DataFrame, column: str = 'close', periods: int = 1) -> pd.Series:
        """
        Calculate log returns
        
        Args:
            df: DataFrame with price data
            column: Price column
            periods: Number of periods
            
        Returns:
            Series with log returns
        """
        return np.log(df[column] / df[column].shift(periods))

    @staticmethod
    def volatility(df: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.Series:
        """
        Calculate historical volatility
        
        Args:
            df: DataFrame with price data
            column: Price column
            window: Window for std calculation
            
        Returns:
            Series with volatility values
        """
        returns = df[column].pct_change()
        volatility = returns.rolling(window=window).std()
        
        return volatility

    @staticmethod
    def high_low_ratio(df: pd.DataFrame) -> pd.Series:
        """High-Low ratio"""
        return (df['high'] - df['close']) / (df['high'] - df['low'])

    @staticmethod
    def close_location_value(df: pd.DataFrame) -> pd.Series:
        """Close Location Value (position of close between high and low)"""
        return (df['close'] - df['low']) / (df['high'] - df['low'])

    # ============ FEATURE CREATION ============
    
    def create_features(self, df: pd.DataFrame, include_indicators: List[str] = None) -> pd.DataFrame:
        """
        Create all technical indicators as features
        
        Args:
            df: Input DataFrame with OHLCV data
            include_indicators: List of indicators to include
            
        Returns:
            DataFrame with all features
        """
        df_features = df.copy()
        
        if include_indicators is None:
            include_indicators = [
                'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'rsi', 'macd', 'stochastic', 'atr', 'bollinger',
                'obv', 'returns', 'volatility'
            ]
        
        try:
            # Trend indicators
            if any(ind in include_indicators for ind in ['sma_10', 'sma_20', 'sma_50']):
                df_features['sma_10'] = self.moving_average(df, 'close', 10)
                df_features['sma_20'] = self.moving_average(df, 'close', 20)
                df_features['sma_50'] = self.moving_average(df, 'close', 50)
            
            if any(ind in include_indicators for ind in ['ema_12', 'ema_26']):
                df_features['ema_12'] = self.exponential_moving_average(df, 'close', 12)
                df_features['ema_26'] = self.exponential_moving_average(df, 'close', 26)
            
            # Momentum indicators
            if 'rsi' in include_indicators:
                df_features['rsi'] = self.rsi(df)
            
            if 'macd' in include_indicators:
                macd_line, signal_line, histogram = self.macd(df)
                df_features['macd'] = macd_line
                df_features['macd_signal'] = signal_line
                df_features['macd_histogram'] = histogram
            
            if 'stochastic' in include_indicators:
                k, d = self.stochastic_oscillator(df)
                df_features['stoch_k'] = k
                df_features['stoch_d'] = d
            
            # Volatility indicators
            if 'atr' in include_indicators:
                df_features['atr'] = self.atr(df)
            
            if 'bollinger' in include_indicators:
                upper, middle, lower = self.bollinger_bands(df)
                df_features['bb_upper'] = upper
                df_features['bb_middle'] = middle
                df_features['bb_lower'] = lower
            
            # Volume indicators
            if 'obv' in include_indicators:
                df_features['obv'] = self.obv(df)
            
            # Price-based features
            if 'returns' in include_indicators:
                df_features['returns'] = self.returns(df)
                df_features['log_returns'] = self.log_returns(df)
            
            if 'volatility' in include_indicators:
                df_features['volatility'] = self.volatility(df)
            
            # Additional features
            df_features['hl_ratio'] = self.high_low_ratio(df)
            df_features['clv'] = self.close_location_value(df)
            
            logger.info(f"Created {len(df_features.columns) - len(df.columns)} features")
            return df_features
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            return df

    def normalize_features(self, df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize feature values
        
        Args:
            df: DataFrame with features
            method: 'minmax' or 'standard'
            
        Returns:
            Normalized DataFrame
        """
        df_normalized = df.copy()
        
        try:
            if method == 'minmax':
                df_normalized = pd.DataFrame(
                    self.scaler_minmax.fit_transform(df),
                    columns=df.columns,
                    index=df.index
                )
            elif method == 'standard':
                df_normalized = pd.DataFrame(
                    self.scaler_standard.fit_transform(df),
                    columns=df.columns,
                    index=df.index
                )
            
            logger.info(f"Features normalized using {method} scaling")
            return df_normalized
            
        except Exception as e:
            logger.error(f"Error normalizing features: {str(e)}")
            return df

    def create_lagged_features(self, df: pd.DataFrame, column: str, lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features for time series
        
        Args:
            df: DataFrame with price data
            column: Column to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        df_lagged = df.copy()
        
        for lag in lags:
            df_lagged[f'{column}_lag_{lag}'] = df[column].shift(lag)
        
        logger.info(f"Created {len(lags)} lagged features for {column}")
        return df_lagged

    def create_rolling_features(self, df: pd.DataFrame, column: str, windows: List[int]) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: DataFrame with price data
            column: Column to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        df_rolling = df.copy()
        
        for window in windows:
            df_rolling[f'{column}_rolling_mean_{window}'] = df[column].rolling(window).mean()
            df_rolling[f'{column}_rolling_std_{window}'] = df[column].rolling(window).std()
            df_rolling[f'{column}_rolling_min_{window}'] = df[column].rolling(window).min()
            df_rolling[f'{column}_rolling_max_{window}'] = df[column].rolling(window).max()
        
        logger.info(f"Created rolling features for {len(windows)} windows")
        return df_rolling


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    import pandas as pd
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'close': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    
    engineer = FeatureEngineer()
    features_df = engineer.create_features(sample_data)
    
    print(features_df.head())
    print(f"\nTotal features created: {len(features_df.columns)}")
