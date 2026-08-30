"""
NEPSE API client for fetching real-time stock data
"""
import requests
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class NepseAPIClient:
    """Client for NEPSE API interactions"""
    
    def __init__(self, base_url: str = "https://nepalstock.com/api"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json, text/plain, */*',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'en-US,en;q=0.9',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.retry_attempts = 3
        self.retry_delay = 5

    def _make_request(self, endpoint: str, method: str = "POST", data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make HTTP request to NEPSE API with retry logic
        
        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST)
            data: Request body data
            
        Returns:
            Response JSON data or None if failed
        """
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.retry_attempts):
            try:
                if method == "POST":
                    response = self.session.post(
                        url,
                        json=data or {},
                        timeout=10
                    )
                else:
                    response = self.session.get(url, timeout=10)
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{self.retry_attempts} failed for {endpoint}: {str(e)}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to fetch {endpoint} after {self.retry_attempts} attempts")
                    return None
        
        return None

    def get_today_price(self) -> Optional[List[Dict]]:
        """
        Fetch today's price data for all listed companies
        
        Returns:
            List of stock data dictionaries or None if failed
        """
        endpoint = "nots/nepse-data/today-price"
        data = self.session.post(
            f"{self.base_url}/{endpoint}",
            json={},
            timeout=10
        ).json()
        
        if data:
            logger.info(f"Fetched today's price data for {len(data)} stocks")
            return data
        return None

    def get_company_info(self, symbol: str) -> Optional[Dict]:
        """
        Fetch detailed information for a specific company
        
        Args:
            symbol: Stock symbol (e.g., 'NABIL')
            
        Returns:
            Company information dictionary or None if failed
        """
        endpoint = f"nots/company/{symbol}"
        return self._make_request(endpoint, method="GET")

    def get_nepse_index(self) -> Optional[Dict]:
        """
        Fetch NEPSE index data
        
        Returns:
            Index data dictionary or None if failed
        """
        endpoint = "nots/nepse-data/nepse-index"
        return self._make_request(endpoint)

    def get_top_gainers(self, limit: int = 10) -> Optional[List[Dict]]:
        """
        Fetch top gaining stocks
        
        Args:
            limit: Number of top gainers to fetch
            
        Returns:
            List of top gainer stocks or None if failed
        """
        endpoint = "nots/nepse-data/top-gainers"
        data = self._make_request(endpoint)
        
        if data:
            return data[:limit]
        return None

    def get_top_losers(self, limit: int = 10) -> Optional[List[Dict]]:
        """
        Fetch top losing stocks
        
        Args:
            limit: Number of top losers to fetch
            
        Returns:
            List of top loser stocks or None if failed
        """
        endpoint = "nots/nepse-data/top-losers"
        data = self._make_request(endpoint)
        
        if data:
            return data[:limit]
        return None

    def get_market_status(self) -> Optional[Dict]:
        """
        Fetch current market status
        
        Returns:
            Market status dictionary or None if failed
        """
        endpoint = "nots/market-open"
        return self._make_request(endpoint)

    def get_summary(self) -> Optional[Dict]:
        """
        Fetch market summary data
        
        Returns:
            Market summary dictionary or None if failed
        """
        endpoint = "nots/nepse-data/summary"
        return self._make_request(endpoint)

    def close(self):
        """Close the session"""
        self.session.close()
        logger.info("API client session closed")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    client = NepseAPIClient()
    
    # Fetch today's prices
    prices = client.get_today_price()
    if prices:
        print(f"Sample data: {json.dumps(prices[0], indent=2)}")
    
    # Fetch market status
    status = client.get_market_status()
    if status:
        print(f"Market Status: {json.dumps(status, indent=2)}")
    
    client.close()
