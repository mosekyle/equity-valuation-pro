import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import requests

class MarketDataProvider:
    """
    Centralized market data provider using Yahoo Finance and other sources.
    """
    
    def __init__(self):
        self.cache = {}
        self.last_update = {}
    
    def get_stock_info(self, ticker: str, force_refresh: bool = False) -> Dict:
        """Get comprehensive stock information."""
        
        cache_key = f"info_{ticker}"
        
        # Check cache
        if not force_refresh and cache_key in self.cache:
            last_update = self.last_update.get(cache_key)
            if last_update and (datetime.now() - last_update).seconds < 300:  # 5 minutes cache
                return self.cache[cache_key]
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Clean and standardize the data
            cleaned_info = {
                'symbol': ticker,
                'longName': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'country': info.get('country', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                
                # Price data
                'currentPrice': info.get('currentPrice', 0),
                'previousClose': info.get('previousClose', 0),
                'open': info.get('open', 0),
                'dayLow': info.get('dayLow', 0),
                'dayHigh': info.get('dayHigh', 0),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 0),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
                
                # Volume
                'volume': info.get('volume', 0),
                'averageVolume': info.get('averageVolume', 0),
                'averageVolume10days': info.get('averageVolume10days', 0),
                
                # Market data
                'marketCap': info.get('marketCap', 0),
                'sharesOutstanding': info.get('sharesOutstanding', 0),
                'floatShares': info.get('floatShares', 0),
                'beta': info.get('beta', 1.0),
                
                # Valuation metrics
                'trailingPE': info.get('trailingPE', 0),
                'forwardPE': info.get('forwardPE', 0),
                'priceToBook': info.get('priceToBook', 0),
                'priceToSalesTrailing12Months': info.get('priceToSalesTrailing12Months', 0),
                'enterpriseValue': info.get('enterpriseValue', 0),
                'enterpriseToRevenue': info.get('enterpriseToRevenue', 0),
                'enterpriseToEbitda': info.get('enterpriseToEbitda', 0),
                'pegRatio': info.get('pegRatio', 0),
                
                # Financial health
                'totalCash': info.get('totalCash', 0),
                'totalDebt': info.get('totalDebt', 0),
                'totalRevenue': info.get('totalRevenue', 0),
                'revenuePerShare': info.get('revenuePerShare', 0),
                'returnOnAssets': info.get('returnOnAssets', 0),
                'returnOnEquity': info.get('returnOnEquity', 0),
                'grossMargins': info.get('grossMargins', 0),
                'operatingMargins': info.get('operatingMargins', 0),
                'profitMargins': info.get('profitMargins', 0),
                
                # Growth
                'revenueGrowth': info.get('revenueGrowth', 0),
                'earningsGrowth': info.get('earningsGrowth', 0),
                
                # Dividends
                'dividendRate': info.get('dividendRate', 0),
                'dividendYield': info.get('dividendYield', 0),
                'payoutRatio': info.get('payoutRatio', 0),
                'exDividendDate': info.get('exDividendDate', None),
                
                # Analyst data
                'targetHighPrice': info.get('targetHighPrice', 0),
                'targetLowPrice': info.get('targetLowPrice', 0),
                'targetMeanPrice': info.get('targetMeanPrice', 0),
                'recommendationMean': info.get('recommendationMean', 0),
                'recommendationKey': info.get('recommendationKey', 'none'),
                'numberOfAnalystOpinions': info.get('numberOfAnalystOpinions', 0),
                
                # Additional metrics
                'ebitda': info.get('ebitda', 0),
                'debtToEquity': info.get('debtToEquity', 0),
                'currentRatio': info.get('currentRatio', 0),
                'quickRatio': info.get('quickRatio', 0),
                
                'lastUpdate': datetime.now().isoformat()
            }
            
            # Cache the result
            self.cache[cache_key] = cleaned_info
            self.last_update[cache_key] = datetime.now()
            
            return cleaned_info
            
        except Exception as e:
            return {
                'symbol': ticker,
                'error': str(e),
                'lastUpdate': datetime.now().isoformat()
            }
    
    def get_historical_prices(self, 
                            ticker: str, 
                            period: str = "1y", 
                            interval: str = "1d") -> pd.DataFrame:
        """Get historical price data."""
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            return hist
        except Exception as e:
            return pd.DataFrame()
    
    def get_financial_statements(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """Get financial statements (income statement, balance sheet, cash flow)."""
        
        try:
            stock = yf.Ticker(ticker)
            
            return {
                'income_statement': stock.financials,
                'balance_sheet': stock.balance_sheet,
                'cash_flow': stock.cashflow,
                'quarterly_income': stock.quarterly_financials,
                'quarterly_balance': stock.quarterly_balance_sheet,
                'quarterly_cashflow': stock.quarterly_cashflow
            }
        except Exception as e:
            return {
                'income_statement': pd.DataFrame(),
                'balance_sheet': pd.DataFrame(),
                'cash_flow': pd.DataFrame(),
                'quarterly_income': pd.DataFrame(),
                'quarterly_balance': pd.DataFrame(),
                'quarterly_cashflow': pd.DataFrame(),
                'error': str(e)
            }
    
    def search_tickers(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for ticker symbols based on company name or symbol."""
        
        try:
            # Use yfinance's Ticker to validate
            results = []
            
            # Simple search by trying the query as a ticker
            try:
                test_ticker = yf.Ticker(query.upper())
                info = test_ticker.info
                if info.get('symbol'):
                    results.append({
                        'symbol': info.get('symbol', query.upper()),
                        'name': info.get('longName', query.upper()),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown')
                    })
            except:
                pass
            
            return results[:limit]
            
        except Exception as e:
            return []
    
    def get_sector_companies(self, sector: str, limit: int = 50) -> List[str]:
        """Get list of companies in a specific sector."""
        
        # Predefined lists of major companies by sector
        sector_companies = {
            'Technology': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
                'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'PYPL', 'CSCO', 'AVGO',
                'TXN', 'QCOM', 'NOW', 'INTU', 'MU', 'AMAT', 'ADI', 'LRCX'
            ],
            'Healthcare': [
                'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY',
                'MRK', 'MDT', 'GILD', 'AMGN', 'CVS', 'CI', 'HUM', 'ANTM',
                'SYK', 'BSX', 'VRTX', 'REGN', 'ZTS', 'EW', 'ILMN', 'BIIB'
            ],
            'Financial Services': [
                'BRK-B', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP',
                'USB', 'TFC', 'PNC', 'COF', 'BLK', 'SCHW', 'CB', 'ICE',
                'CME', 'SPGI', 'MCO', 'MMC', 'AJG', 'TRV', 'ALL', 'PGR'
            ],
            'Consumer Cyclical': [
                'AMZN', 'TSLA', 'HD', 'MCD', 'DIS', 'NKE', 'LOW', 'SBUX',
                'TJX', 'BKNG', 'CMG', 'ORLY', 'AZO', 'RCL', 'CCL', 'NCLH',
                'MAR', 'HLT', 'MGM', 'WYNN', 'LVS', 'GRMN', 'POOL', 'WHR'
            ],
            'Consumer Defensive': [
                'PG', 'KO', 'PEP', 'WMT', 'COST', 'MDLZ', 'KMB', 'GIS',
                'K', 'HSY', 'TSN', 'CPB', 'CAG', 'SJM', 'HRL', 'MKC',
                'CHD', 'CL', 'CLX', 'SYY', 'KR', 'TGT', 'DG', 'DLTR'
            ],
            'Energy': [
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC',
                'KMI', 'OKE', 'WMB', 'EPD', 'ET', 'MPLX', 'BKR', 'HAL',
                'DVN', 'FANG', 'APA', 'EQT', 'CNX', 'AR', 'MRO', 'OVV'
            ],
            'Industrials': [
                'BA', 'CAT', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM',
                'FDX', 'NSC', 'CSX', 'UNP', 'DE', 'EMR', 'ETN', 'ITW',
                'PH', 'CMI', 'GD', 'NOC', 'LHX', 'TDG', 'CARR', 'OTIS'
            ]
        }
        
        return sector_companies.get(sector, [])[:limit]
    
    def get_market_indices(self) -> Dict[str, Dict]:
        """Get major market indices data."""
        
        indices = {
            'S&P 500': '^GSPC',
            'Dow Jones': '^DJI',
            'NASDAQ': '^IXIC',
            'Russell 2000': '^RUT',
            'VIX': '^VIX'
        }
        
        market_data = {}
        
        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="5d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                    
                    market_data[name] = {
                        'symbol': symbol,
                        'price': current_price,
                        'change': change,
                        'change_percent': change_pct,
                        'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
                    }
            except:
                market_data[name] = {
                    'symbol': symbol,
                    'price': 0,
                    'change': 0,
                    'change_percent': 0,
                    'volume': 0
                }
        
        return market_data
    
    def calculate_returns(self, ticker: str, periods: List[str] = None) -> Dict:
        """Calculate returns over various periods."""
        
        if periods is None:
            periods = ['1d', '1wk', '1mo', '3mo', '6mo', '1y', '2y', '5y']
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5y", interval="1d")
            
            if hist.empty:
                return {}
            
            current_price = hist['Close'].iloc[-1]
            returns = {}
            
            for period in periods:
                try:
                    if period == '1d':
                        past_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    elif period == '1wk':
                        past_price = hist['Close'].iloc[-6] if len(hist) > 5 else current_price
                    elif period == '1mo':
                        past_price = hist['Close'].iloc[-22] if len(hist) > 21 else current_price
                    elif period == '3mo':
                        past_price = hist['Close'].iloc[-66] if len(hist) > 65 else current_price
                    elif period == '6mo':
                        past_price = hist['Close'].iloc[-132] if len(hist) > 131 else current_price
                    elif period == '1y':
                        past_price = hist['Close'].iloc[-252] if len(hist) > 251 else current_price
                    elif period == '2y':
                        past_price = hist['Close'].iloc[-504] if len(hist) > 503 else current_price
                    elif period == '5y':
                        past_price = hist['Close'].iloc[0] if len(hist) > 0 else current_price
                    else:
                        continue
                    
                    if past_price != 0:
                        return_pct = ((current_price / past_price) - 1) * 100
                        returns[period] = return_pct
                except:
                    returns[period] = 0
            
            return returns
            
        except Exception as e:
            return {}

# Global instance
market_data = MarketDataProvider()
