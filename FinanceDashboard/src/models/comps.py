import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import yfinance as yf
from datetime import datetime, timedelta

class ComparableAnalysis:
    """
    Comparable company analysis for equity valuation.
    """
    
    def __init__(self, target_ticker: str, peer_tickers: List[str]):
        self.target_ticker = target_ticker
        self.peer_tickers = peer_tickers
        self.all_tickers = [target_ticker] + peer_tickers
        self.company_data = {}
        
    def fetch_company_data(self) -> Dict:
        """Fetch financial data for all companies."""
        
        for ticker in self.all_tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Key metrics
                self.company_data[ticker] = {
                    'name': info.get('longName', ticker),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'enterprise_value': info.get('enterpriseValue', 0),
                    'current_price': info.get('currentPrice', 0),
                    'shares_outstanding': info.get('sharesOutstanding', 0),
                    
                    # Valuation metrics
                    'pe_ratio': info.get('forwardPE', info.get('trailingPE', 0)),
                    'pb_ratio': info.get('priceToBook', 0),
                    'ps_ratio': info.get('priceToSalesTrailing12Months', 0),
                    'ev_ebitda': info.get('enterpriseToEbitda', 0),
                    'ev_revenue': info.get('enterpriseToRevenue', 0),
                    'peg_ratio': info.get('pegRatio', 0),
                    
                    # Financial metrics
                    'revenue_ttm': info.get('totalRevenue', 0),
                    'ebitda_ttm': info.get('ebitda', 0),
                    'net_income_ttm': info.get('netIncomeToCommon', 0),
                    'total_debt': info.get('totalDebt', 0),
                    'total_cash': info.get('totalCash', 0),
                    
                    # Profitability metrics
                    'roe': info.get('returnOnEquity', 0),
                    'roa': info.get('returnOnAssets', 0),
                    'roic': info.get('returnOnCapital', 0),
                    'gross_margin': info.get('grossMargins', 0),
                    'operating_margin': info.get('operatingMargins', 0),
                    'profit_margin': info.get('profitMargins', 0),
                    
                    # Growth metrics
                    'revenue_growth': info.get('revenueGrowth', 0),
                    'earnings_growth': info.get('earningsGrowth', 0),
                    
                    # Risk metrics
                    'beta': info.get('beta', 1.0),
                    'debt_to_equity': info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0,
                    
                    # Dividend metrics
                    'dividend_yield': info.get('dividendYield', 0),
                    'payout_ratio': info.get('payoutRatio', 0),
                }
                
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                self.company_data[ticker] = self._get_empty_data_dict(ticker)
        
        return self.company_data
    
    def _get_empty_data_dict(self, ticker: str) -> Dict:
        """Return empty data dictionary for failed ticker."""
        return {
            'name': ticker,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 0,
            'enterprise_value': 0,
            'current_price': 0,
            'shares_outstanding': 0,
            'pe_ratio': 0,
            'pb_ratio': 0,
            'ps_ratio': 0,
            'ev_ebitda': 0,
            'ev_revenue': 0,
            'peg_ratio': 0,
            'revenue_ttm': 0,
            'ebitda_ttm': 0,
            'net_income_ttm': 0,
            'total_debt': 0,
            'total_cash': 0,
            'roe': 0,
            'roa': 0,
            'roic': 0,
            'gross_margin': 0,
            'operating_margin': 0,
            'profit_margin': 0,
            'revenue_growth': 0,
            'earnings_growth': 0,
            'beta': 1.0,
            'debt_to_equity': 0,
            'dividend_yield': 0,
            'payout_ratio': 0,
            'error': True
        }
    
    def create_comp_table(self) -> pd.DataFrame:
        """Create comparable company analysis table."""
        
        if not self.company_data:
            self.fetch_company_data()
        
        # Define metrics to include in comp table
        metrics = [
            'market_cap', 'enterprise_value', 'revenue_ttm', 'ebitda_ttm',
            'pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda', 'ev_revenue',
            'roe', 'roa', 'gross_margin', 'operating_margin', 'profit_margin',
            'revenue_growth', 'earnings_growth', 'beta', 'debt_to_equity',
            'dividend_yield'
        ]
        
        # Create DataFrame
        comp_data = {}
        for ticker in self.all_tickers:
            if ticker in self.company_data:
                comp_data[ticker] = {
                    metric: self.company_data[ticker].get(metric, 0)
                    for metric in metrics
                }
        
        comp_df = pd.DataFrame(comp_data).T
        
        # Add company names as index
        comp_df.index = [
            self.company_data.get(ticker, {}).get('name', ticker)
            for ticker in comp_df.index
        ]
        
        return comp_df
    
    def calculate_peer_statistics(self) -> Dict:
        """Calculate peer group statistics excluding target company."""
        
        if not self.company_data:
            self.fetch_company_data()
        
        # Get peer data (excluding target)
        peer_data = {
            ticker: data for ticker, data in self.company_data.items()
            if ticker != self.target_ticker and not data.get('error', False)
        }
        
        if not peer_data:
            return {}
        
        # Metrics to analyze
        metrics = [
            'pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda', 'ev_revenue',
            'roe', 'roa', 'gross_margin', 'operating_margin', 'profit_margin',
            'revenue_growth', 'earnings_growth', 'beta', 'debt_to_equity'
        ]
        
        statistics = {}
        for metric in metrics:
            values = [
                data[metric] for data in peer_data.values()
                if data[metric] and data[metric] > 0
            ]
            
            if values:
                statistics[metric] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'count': len(values)
                }
            else:
                statistics[metric] = {
                    'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0, 'count': 0
                }
        
        return statistics
    
    def relative_valuation(self) -> Dict:
        """Calculate relative valuation based on peer multiples."""
        
        if not self.company_data:
            self.fetch_company_data()
        
        target_data = self.company_data.get(self.target_ticker, {})
        peer_stats = self.calculate_peer_statistics()
        
        if not target_data or not peer_stats:
            return {}
        
        # Calculate implied values based on peer multiples
        implied_values = {}
        
        # P/E multiple valuation
        if peer_stats.get('pe_ratio', {}).get('median', 0) > 0:
            target_eps = target_data['net_income_ttm'] / target_data['shares_outstanding'] if target_data['shares_outstanding'] > 0 else 0
            implied_pe_value = target_eps * peer_stats['pe_ratio']['median']
            implied_values['pe_valuation'] = implied_pe_value
        
        # P/B multiple valuation
        if peer_stats.get('pb_ratio', {}).get('median', 0) > 0:
            target_bvps = (target_data['market_cap'] / target_data['pb_ratio']) / target_data['shares_outstanding'] if target_data['pb_ratio'] > 0 and target_data['shares_outstanding'] > 0 else 0
            implied_pb_value = target_bvps * peer_stats['pb_ratio']['median']
            implied_values['pb_valuation'] = implied_pb_value
        
        # P/S multiple valuation
        if peer_stats.get('ps_ratio', {}).get('median', 0) > 0:
            target_sps = target_data['revenue_ttm'] / target_data['shares_outstanding'] if target_data['shares_outstanding'] > 0 else 0
            implied_ps_value = target_sps * peer_stats['ps_ratio']['median']
            implied_values['ps_valuation'] = implied_ps_value
        
        # EV/EBITDA multiple valuation
        if peer_stats.get('ev_ebitda', {}).get('median', 0) > 0:
            target_ebitda = target_data['ebitda_ttm']
            implied_ev = target_ebitda * peer_stats['ev_ebitda']['median']
            implied_equity_value = implied_ev - target_data['total_debt'] + target_data['total_cash']
            implied_ev_ebitda_value = implied_equity_value / target_data['shares_outstanding'] if target_data['shares_outstanding'] > 0 else 0
            implied_values['ev_ebitda_valuation'] = implied_ev_ebitda_value
        
        # Calculate summary statistics
        valid_values = [v for v in implied_values.values() if v > 0]
        if valid_values:
            implied_values['average_implied_value'] = np.mean(valid_values)
            implied_values['median_implied_value'] = np.median(valid_values)
            implied_values['min_implied_value'] = np.min(valid_values)
            implied_values['max_implied_value'] = np.max(valid_values)
        
        # Compare to current price
        current_price = target_data['current_price']
        if current_price > 0 and valid_values:
            implied_values['upside_downside_avg'] = (implied_values['average_implied_value'] / current_price - 1) * 100
            implied_values['upside_downside_median'] = (implied_values['median_implied_value'] / current_price - 1) * 100
        
        return implied_values
    
    def peer_ranking(self) -> pd.DataFrame:
        """Rank companies across key metrics."""
        
        comp_table = self.create_comp_table()
        
        # Metrics for ranking (higher is better)
        positive_metrics = ['roe', 'roa', 'gross_margin', 'operating_margin', 'profit_margin', 'revenue_growth', 'earnings_growth']
        
        # Metrics for ranking (lower is better)
        negative_metrics = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda', 'ev_revenue', 'beta', 'debt_to_equity']
        
        ranking_df = pd.DataFrame(index=comp_table.index)
        
        # Rank positive metrics (1 = best, higher rank = worse)
        for metric in positive_metrics:
            if metric in comp_table.columns:
                ranking_df[f'{metric}_rank'] = comp_table[metric].rank(ascending=False, method='min')
        
        # Rank negative metrics (1 = best, higher rank = worse)
        for metric in negative_metrics:
            if metric in comp_table.columns:
                ranking_df[f'{metric}_rank'] = comp_table[metric].rank(ascending=True, method='min')
        
        # Calculate overall rank (average of all ranks)
        rank_columns = [col for col in ranking_df.columns if col.endswith('_rank')]
        ranking_df['overall_rank'] = ranking_df[rank_columns].mean(axis=1)
        ranking_df['overall_rank_position'] = ranking_df['overall_rank'].rank(method='min')
        
        return ranking_df
