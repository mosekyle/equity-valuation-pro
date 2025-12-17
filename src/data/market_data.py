"""
Market Data Module for Equity Valuation Dashboard
Handles real-time and historical market data from multiple sources
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataProvider:
    """
    Comprehensive market data provider with multiple data sources
    """
    
    def __init__(self, cache_duration: int = 300):
        """
        Initialize market data provider
        
        Args:
            cache_duration (int): Cache duration in seconds (default: 5 minutes)
        """
        self.cache_duration = cache_duration
        self._cache = {}
        self._cache_timestamps = {}
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._cache_timestamps:
            return False
        
        cache_time = self._cache_timestamps[key]
        current_time = time.time()
        return (current_time - cache_time) < self.cache_duration
    
    def get_stock_data(self, 
                      symbol: str, 
                      period: str = "5y", 
                      interval: str = "1d") -> pd.DataFrame:
        """
        Get historical stock price data
        
        Args:
            symbol (str): Stock ticker symbol (e.g., 'AAPL')
            period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            pd.DataFrame: Historical price data with OHLCV columns
        """
        cache_key = f"stock_data_{symbol}_{period}_{interval}"
        
        if self._is_cache_valid(cache_key):
            logger.info(f"Returning cached data for {symbol}")
            return self._cache[cache_key]
        
        try:
            logger.info(f"Fetching stock data for {symbol}")
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
            
            # Clean and standardize the data
            data.index = pd.to_datetime(data.index)
            data.columns = [col.replace(' ', '_').lower() for col in data.columns]
            
            # Cache the data
            self._cache[cache_key] = data
            self._cache_timestamps[cache_key] = time.time()
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
            raise
    
    def get_company_info(self, symbol: str) -> Dict:
        """
        Get comprehensive company information
        
        Args:
            symbol (str): Stock ticker symbol
            
        Returns:
            Dict: Company information including financials, ratios, and metadata
        """
        cache_key = f"company_info_{symbol}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            logger.info(f"Fetching company info for {symbol}")
            stock = yf.Ticker(symbol)
            
            # Get all available information
            info = stock.info
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
            
            # Compile comprehensive company data
            company_data = {
                'basic_info': {
                    'symbol': symbol,
                    'company_name': info.get('longName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap', 0),
                    'enterprise_value': info.get('enterpriseValue', 0),
                    'employees': info.get('fullTimeEmployees', 0),
                    'country': info.get('country', 'N/A'),
                    'website': info.get('website', 'N/A')
                },
                'current_metrics': {
                    'current_price': info.get('currentPrice', 0),
                    'previous_close': info.get('previousClose', 0),
                    'open': info.get('open', 0),
                    'day_high': info.get('dayHigh', 0),
                    'day_low': info.get('dayLow', 0),
                    'volume': info.get('volume', 0),
                    'avg_volume': info.get('averageVolume', 0),
                    'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                    'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0)
                },
                'financial_metrics': {
                    'pe_ratio': info.get('trailingPE', 0),
                    'forward_pe': info.get('forwardPE', 0),
                    'peg_ratio': info.get('pegRatio', 0),
                    'price_to_book': info.get('priceToBook', 0),
                    'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                    'ev_to_ebitda': info.get('enterpriseToEbitda', 0),
                    'ev_to_revenue': info.get('enterpriseToRevenue', 0),
                    'profit_margin': info.get('profitMargins', 0),
                    'operating_margin': info.get('operatingMargins', 0),
                    'return_on_assets': info.get('returnOnAssets', 0),
                    'return_on_equity': info.get('returnOnEquity', 0),
                    'debt_to_equity': info.get('debtToEquity', 0),
                    'current_ratio': info.get('currentRatio', 0),
                    'quick_ratio': info.get('quickRatio', 0)
                },
                'growth_metrics': {
                    'revenue_growth': info.get('revenueGrowth', 0),
                    'earnings_growth': info.get('earningsGrowth', 0),
                    'revenue_per_share': info.get('revenuePerShare', 0),
                    'book_value': info.get('bookValue', 0),
                    'earnings_per_share': info.get('trailingEps', 0),
                    'forward_eps': info.get('forwardEps', 0)
                },
                'dividend_info': {
                    'dividend_rate': info.get('dividendRate', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'payout_ratio': info.get('payoutRatio', 0),
                    'ex_dividend_date': info.get('exDividendDate', None),
                    'last_dividend_value': info.get('lastDividendValue', 0)
                },
                'financial_statements': {
                    'income_statement': financials.to_dict() if not financials.empty else {},
                    'balance_sheet': balance_sheet.to_dict() if not balance_sheet.empty else {},
                    'cash_flow': cashflow.to_dict() if not cashflow.empty else {}
                }
            }
            
            # Cache the data
            self._cache[cache_key] = company_data
            self._cache_timestamps[cache_key] = time.time()
            
            return company_data
            
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {str(e)}")
            raise
    
    def get_peers(self, symbol: str, sector: str = None) -> List[str]:
        """
        Get peer companies for comparative analysis
        
        Args:
            symbol (str): Target company symbol
            sector (str): Company sector (optional, will fetch if not provided)
            
        Returns:
            List[str]: List of peer company symbols
        """
        # This is a simplified implementation
        # In production, you'd use more sophisticated peer selection algorithms
        
        # Common peer mappings (you can expand this)
        peer_mapping = {
            'AAPL': ['MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA'],
            'MSFT': ['AAPL', 'GOOGL', 'AMZN', 'META', 'ORCL'],
            'GOOGL': ['AAPL', 'MSFT', 'META', 'AMZN', 'NFLX'],
            'AMZN': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NFLX'],
            'TSLA': ['F', 'GM', 'NIO', 'RIVN', 'LCID'],
            'JPM': ['BAC', 'WFC', 'C', 'GS', 'MS'],
            'JNJ': ['PFE', 'UNH', 'MRK', 'ABBV', 'TMO']
        }
        
        return peer_mapping.get(symbol.upper(), [])
    
    def get_multiple_stocks_data(self, 
                               symbols: List[str], 
                               period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks efficiently
        
        Args:
            symbols (List[str]): List of stock symbols
            period (str): Data period
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
        """
        results = {}
        
        for symbol in symbols:
            try:
                results[symbol] = self.get_stock_data(symbol, period)
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {str(e)}")
                continue
        
        return results
    
    def calculate_returns(self, 
                         data: pd.DataFrame, 
                         periods: List[str] = ['1D', '1W', '1M', '3M', '6M', '1Y']) -> Dict[str, float]:
        """
        Calculate returns for different time periods
        
        Args:
            data (pd.DataFrame): Stock price data
            periods (List[str]): Time periods to calculate returns for
            
        Returns:
            Dict[str, float]: Dictionary of returns for each period
        """
        if data.empty or 'close' not in data.columns:
            return {period: 0.0 for period in periods}
        
        current_price = data['close'].iloc[-1]
        returns = {}
        
        for period in periods:
            try:
                if period == '1D':
                    past_price = data['close'].iloc[-2] if len(data) > 1 else current_price
                elif period == '1W':
                    past_price = data['close'].iloc[-6] if len(data) > 5 else current_price
                elif period == '1M':
                    past_price = data['close'].iloc[-22] if len(data) > 21 else current_price
                elif period == '3M':
                    past_price = data['close'].iloc[-66] if len(data) > 65 else current_price
                elif period == '6M':
                    past_price = data['close'].iloc[-132] if len(data) > 131 else current_price
                elif period == '1Y':
                    past_price = data['close'].iloc[-252] if len(data) > 251 else current_price
                else:
                    past_price = current_price
                
                returns[period] = ((current_price - past_price) / past_price) * 100
                
            except (IndexError, ZeroDivisionError):
                returns[period] = 0.0
        
        return returns


# Convenience functions for easy usage
def get_stock_price(symbol: str) -> float:
    """Get current stock price"""
    provider = MarketDataProvider()
    try:
        data = provider.get_stock_data(symbol, period="1d")
        return float(data['close'].iloc[-1])
    except:
        return 0.0


def get_company_overview(symbol: str) -> Dict:
    """Get basic company overview"""
    provider = MarketDataProvider()
    try:
        info = provider.get_company_info(symbol)
        return {
            'name': info['basic_info']['company_name'],
            'sector': info['basic_info']['sector'],
            'current_price': info['current_metrics']['current_price'],
            'market_cap': info['basic_info']['market_cap'],
            'pe_ratio': info['financial_metrics']['pe_ratio']
        }
    except:
        return {}
