"""
Main data collector module for NEPSE stock data
"""
import logging
import time
from datetime import datetime
from typing import Optional, List
from sqlalchemy.orm import Session
from data_collection.nepse_api_client import NepseAPIClient
from database.models import DailyPrice, Company, MarketIndex, DataFetchLog
from database.connection import SessionLocal

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_collector.log'),
        logging.StreamHandler()
    ]
)


class NEPSEDataCollector:
    """Main data collector for NEPSE stock data"""
    
    def __init__(self, db_session: Optional[Session] = None):
        self.client = NepseAPIClient()
        self.db = db_session or SessionLocal()
        self.fetch_start_time = None

    def _log_fetch(self, source: str, status: str, records_fetched: int, error_msg: Optional[str] = None):
        """Log data fetch operation"""
        duration = time.time() - self.fetch_start_time if self.fetch_start_time else 0
        
        log_entry = DataFetchLog(
            source=source,
            status=status,
            records_fetched=records_fetched,
            error_message=error_msg,
            duration_seconds=duration
        )
        self.db.add(log_entry)
        self.db.commit()
        
        logger.info(
            f"Fetch Log - Source: {source}, Status: {status}, "
            f"Records: {records_fetched}, Duration: {duration:.2f}s"
        )

    def collect_today_prices(self) -> bool:
        """
        Collect today's price data for all listed companies
        
        Returns:
            True if successful, False otherwise
        """
        self.fetch_start_time = time.time()
        
        try:
            logger.info("Starting today's price data collection...")
            data = self.client.get_today_price()
            
            if not data:
                self._log_fetch("NEPSE_API", "FAILED", 0, "No data returned from API")
                return False
            
            inserted_count = 0
            
            for item in data:
                try:
                    # Extract required fields
                    symbol = item.get('symbol')
                    if not symbol:
                        logger.warning("Skipping entry without symbol")
                        continue
                    
                    # Create or update company info
                    company = self.db.query(Company).filter(Company.symbol == symbol).first()
                    if not company:
                        company = Company(
                            symbol=symbol,
                            name=item.get('companyName', symbol)
                        )
                        self.db.add(company)
                    
                    # Create daily price record
                    daily_price = DailyPrice(
                        symbol=symbol,
                        date=datetime.now().date(),
                        open_price=float(item.get('openPrice', 0)),
                        high_price=float(item.get('highPrice', 0)),
                        low_price=float(item.get('lowPrice', 0)),
                        close_price=float(item.get('closingPrice', 0)),
                        traded_shares=int(item.get('tradedShares', 0)),
                        traded_amount=float(item.get('amount', 0)),
                        previous_close=float(item.get('previousClosing', 0)),
                        price_change=float(item.get('difference', 0))
                    )
                    
                    # Calculate percent change
                    if daily_price.previous_close > 0:
                        daily_price.percent_change = (daily_price.price_change / daily_price.previous_close) * 100
                    
                    self.db.add(daily_price)
                    inserted_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing stock {symbol}: {str(e)}")
                    continue
            
            self.db.commit()
            self._log_fetch("NEPSE_API", "SUCCESS", inserted_count)
            logger.info(f"Successfully collected {inserted_count} stock records")
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error collecting today's prices: {error_msg}")
            self._log_fetch("NEPSE_API", "FAILED", 0, error_msg)
            self.db.rollback()
            return False

    def collect_nepse_index(self) -> bool:
        """
        Collect NEPSE index data
        
        Returns:
            True if successful, False otherwise
        """
        self.fetch_start_time = time.time()
        
        try:
            logger.info("Starting NEPSE index data collection...")
            data = self.client.get_nepse_index()
            
            if not data:
                self._log_fetch("NEPSE_INDEX", "FAILED", 0, "No index data returned")
                return False
            
            # Parse index data
            index_data = MarketIndex(
                index_name="NEPSE",
                date=datetime.now().date(),
                open_value=float(data.get('openValue', 0)),
                high_value=float(data.get('highValue', 0)),
                low_value=float(data.get('lowValue', 0)),
                close_value=float(data.get('closeValue', 0)),
                total_traded_shares=int(data.get('totalTradedShares', 0)),
                total_traded_amount=float(data.get('totalTradedAmount', 0)),
                previous_close=float(data.get('previousClose', 0)),
                index_change=float(data.get('change', 0))
            )
            
            if index_data.previous_close > 0:
                index_data.percent_change = (index_data.index_change / index_data.previous_close) * 100
            
            self.db.add(index_data)
            self.db.commit()
            
            self._log_fetch("NEPSE_INDEX", "SUCCESS", 1)
            logger.info("Successfully collected NEPSE index data")
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error collecting NEPSE index: {error_msg}")
            self._log_fetch("NEPSE_INDEX", "FAILED", 0, error_msg)
            self.db.rollback()
            return False

    def collect_all(self) -> bool:
        """
        Collect all available data
        
        Returns:
            True if all collections successful, False otherwise
        """
        logger.info("Starting complete data collection cycle...")
        
        results = {
            'prices': self.collect_today_prices(),
            'index': self.collect_nepse_index()
        }
        
        if all(results.values()):
            logger.info("Data collection cycle completed successfully")
            return True
        else:
            logger.warning(f"Data collection cycle completed with issues: {results}")
            return False

    def continuous_collection(self, interval_seconds: int = 300):
        """
        Run continuous data collection at specified intervals
        
        Args:
            interval_seconds: Time between collections in seconds (default 5 minutes)
        """
        logger.info(f"Starting continuous data collection (interval: {interval_seconds}s)")
        
        try:
            while True:
                self.collect_all()
                logger.info(f"Waiting {interval_seconds}s before next collection...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Continuous collection stopped by user")
        except Exception as e:
            logger.error(f"Error in continuous collection: {str(e)}")
        finally:
            self.close()

    def close(self):
        """Close database and API client"""
        self.db.close()
        self.client.close()
        logger.info("Data collector closed")


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NEPSE Data Collector")
    parser.add_argument(
        "--mode",
        choices=["once", "continuous"],
        default="once",
        help="Collection mode: once or continuous"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Interval between collections in seconds (default: 300)"
    )
    
    args = parser.parse_args()
    
    collector = NEPSEDataCollector()
    
    try:
        if args.mode == "once":
            collector.collect_all()
        else:
            collector.continuous_collection(args.interval)
    except KeyboardInterrupt:
        logger.info("Collector interrupted")
    finally:
        collector.close()
