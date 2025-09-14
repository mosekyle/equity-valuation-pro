"""
Comparable Company Analysis Module for Equity Valuation Dashboard
Professional-grade peer analysis and multiple valuation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ..data.market_data import MarketDataProvider
from ..utils.calculations import FinancialCalculations, ComparableAnalysis

logger = logging.getLogger(__name__)


@dataclass
class CompanyProfile:
    """Data class for company profile information"""
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    enterprise_value: float
    revenue: float
    ebitda: float
    net_income: float
    total_debt: float
    cash: float
    shares_outstanding: int
    current_price: float


class ComparableAnalyzer:
    """
    Professional comparable company analysis engine
    """
    
    def __init__(self, target_symbol: str):
        """
        Initialize comparable analyzer
        
        Args:
            target_symbol (str): Target company symbol for analysis
        """
        self.target_symbol = target_symbol.upper()
        self.market_provider = MarketDataProvider()
        self.target_data = None
        self.peer_data = {}
        self.peer_profiles = {}
        self.multiples_analysis = None
        
    def load_target_company(self) -> CompanyProfile:
        """
        Load target company data
        
        Returns:
            CompanyProfile: Target company profile
        """
        try:
            logger.info(f"Loading target company data for {self.target_symbol}")
            company_info = self.market_provider.get_company_info(self.target_symbol)
            
            # Extract key metrics
            basic_info = company_info.get('basic_info', {})
            financial_metrics = company_info.get('financial_metrics', {})
            current_metrics = company_info.get('current_metrics', {})
            
            # Create company profile
            profile = CompanyProfile(
                symbol=self.target_symbol,
                name=basic_info.get('company_name', 'N/A'),
                sector=basic_info.get('sector', 'N/A'),
                industry=basic_info.get('industry', 'N/A'),
                market_cap=basic_info.get('market_cap', 0),
                enterprise_value=basic_info.get('enterprise_value', 0),
                revenue=self._extract_revenue(company_info),
                ebitda=self._extract_ebitda(company_info),
                net_income=self._extract_net_income(company_info),
                total_debt=self._extract_total_debt(company_info),
                cash=self._extract_cash(company_info),
                shares_outstanding=basic_info.get('shares_outstanding', 0),
                current_price=current_metrics.get('current_price', 0)
            )
            
            self.target_data = profile
            return profile
            
        except Exception as e:
            logger.error(f"Error loading target company {self.target_symbol}: {str(e)}")
            raise
    
    def find_peer_companies(self, 
                          custom_peers: List[str] = None,
                          auto_select: bool = True,
                          min_market_cap: float = 1e9) -> List[str]:
        """
        Find peer companies using multiple methods
        
        Args:
            custom_peers (List[str]): Custom list of peer symbols
            auto_select (bool): Whether to use automatic peer selection
            min_market_cap (float): Minimum market cap for peers
            
        Returns:
            List[str]: List of peer company symbols
        """
        peer_symbols = set()
        
        # Method 1: Use custom peer list if provided
        if custom_peers:
            peer_symbols.update([p.upper() for p in custom_peers])
        
        # Method 2: Use pre-defined peer mappings (industry knowledge)
        if auto_select:
            predefined_peers = self._get_predefined_peers(self.target_symbol)
            peer_symbols.update(predefined_peers)
        
        # Method 3: Sector-based selection (simplified)
        if auto_select and self.target_data:
            sector_peers = self._get_sector_peers(
                self.target_data.sector, 
                self.target_data.market_cap,
                min_market_cap
            )
            peer_symbols.update(sector_peers)
        
        # Remove target company from peers
        peer_symbols.discard(self.target_symbol)
        
        # Limit to reasonable number of peers (5-15 is typical)
        peer_list = list(peer_symbols)[:12]
        
        logger.info(f"Selected {len(peer_list)} peer companies: {peer_list}")
        return peer_list
    
    def load_peer_data(self, peer_symbols: List[str]) -> Dict[str, CompanyProfile]:
        """
        Load data for all peer companies
        
        Args:
            peer_symbols (List[str]): List of peer symbols
            
        Returns:
            Dict[str, CompanyProfile]: Dictionary of peer profiles
        """
        peer_profiles = {}
        
        for symbol in peer_symbols:
            try:
                logger.info(f"Loading peer data for {symbol}")
                company_info = self.market_provider.get_company_info(symbol)
                
                basic_info = company_info.get('basic_info', {})
                current_metrics = company_info.get('current_metrics', {})
                
                profile = CompanyProfile(
                    symbol=symbol,
                    name=basic_info.get('company_name', symbol),
                    sector=basic_info.get('sector', 'N/A'),
                    industry=basic_info.get('industry', 'N/A'),
                    market_cap=basic_info.get('market_cap', 0),
                    enterprise_value=basic_info.get('enterprise_value', 0),
                    revenue=self._extract_revenue(company_info),
                    ebitda=self._extract_ebitda(company_info),
                    net_income=self._extract_net_income(company_info),
                    total_debt=self._extract_total_debt(company_info),
                    cash=self._extract_cash(company_info),
                    shares_outstanding=basic_info.get('shares_outstanding', 1),
                    current_price=current_metrics.get('current_price', 0)
                )
                
                # Only include peers with reasonable data
                if profile.market_cap > 0 and profile.revenue > 0:
                    peer_profiles[symbol] = profile
                else:
                    logger.warning(f"Insufficient data for {symbol}, excluding from analysis")
                    
            except Exception as e:
                logger.warning(f"Error loading data for peer {symbol}: {str(e)}")
                continue
        
        self.peer_profiles = peer_profiles
        logger.info(f"Successfully loaded data for {len(peer_profiles)} peers")
        return peer_profiles
    
    def calculate_multiples(self) -> pd.DataFrame:
        """
        Calculate valuation multiples for all companies
        
        Returns:
            pd.DataFrame: Multiples analysis table
        """
        if not self.target_data or not self.peer_profiles:
            raise ValueError("Target and peer data must be loaded first")
        
        multiples_data = []
        
        # Add target company
        target_multiples = self._calculate_company_multiples(self.target_data)
        target_multiples['company_type'] = 'Target'
        multiples_data.append(target_multiples)
        
        # Add peer companies
        for symbol, profile in self.peer_profiles.items():
            peer_multiples = self._calculate_company_multiples(profile)
            peer_multiples['company_type'] = 'Peer'
            multiples_data.append(peer_multiples)
        
        # Create DataFrame
        multiples_df = pd.DataFrame(multiples_data)
        
        # Calculate peer statistics
        peer_stats = self._calculate_peer_statistics(multiples_df)
        
        # Store results
        self.multiples_analysis = {
            'multiples_df': multiples_df,
            'peer_stats': peer_stats
        }
        
        return multiples_df
    
    def calculate_implied_valuation(self) -> Dict[str, Dict]:
        """
        Calculate implied valuations using peer multiples
        
        Returns:
            Dict[str, Dict]: Implied valuations using different methods
        """
        if not self.multiples_analysis:
            self.calculate_multiples()
        
        multiples_df = self.multiples_analysis['multiples_df']
        peer_stats = self.multiples_analysis['peer_stats']
        
        # Get target company financials
        target = self.target_data
        
        implied_valuations = {}
        
        # Revenue-based valuation
        if 'ev_revenue' in peer_stats.columns:
            median_ev_rev = peer_stats.loc['50%', 'ev_revenue']
            mean_ev_rev = peer_stats.loc['mean', 'ev_revenue']
            
            implied_ev_median = target.revenue * median_ev_rev
            implied_ev_mean = target.revenue * mean_ev_rev
            
            # Convert to equity value (EV - Net Debt)
            net_debt = target.total_debt - target.cash
            equity_value_median = implied_ev_median - net_debt
            equity_value_mean = implied_ev_mean - net_debt
            
[O            implied_valuations['revenue_multiple'] = {
                'median_multiple': median_ev_rev,
                'mean_multiple': mean_ev_rev,
                'implied_ev_median': implied_ev_median,
                'implied_ev_mean': implied_ev_mean,
                'implied_price_median': equity_value_median / target.shares_outstanding,
                'implied_price_mean': equity_value_mean / target.shares_outstanding,
                'current_price': target.current_price
            }
        
        # EBITDA-based valuation
        if 'ev_ebitda' in peer_stats.columns and target.ebitda > 0:
            median_ev_ebitda = peer_stats.loc['50%', 'ev_ebitda']
            mean_ev_ebitda = peer_stats.loc['mean', 'ev_ebitda']
            
            implied_ev_median = target.ebitda * median_ev_ebitda
            implied_ev_mean = target.ebitda * mean_ev_ebitda
            
            net_debt = target.total_debt - target.cash
            equity_value_median = implied_ev_median - net_debt
            equity_value_mean = implied_ev_mean - net_debt
            
            implied_valuations['ebitda_multiple'] = {
                'median_multiple': median_ev_ebitda,
                'mean_multiple': mean_ev_ebitda,
                'implied_ev_median': implied_ev_median,
                'implied_ev_mean': implied_ev_mean,
                'implied_price_median': equity_value_median / target.shares_outstanding,
                'implied_price_mean': equity_value_mean / target.shares_outstanding,
                'current_price': target.current_price
            }
        
        # P/E based valuation
        if 'pe_ratio' in peer_stats.columns and target.net_income > 0:
            median_pe = peer_stats.loc['50%', 'pe_ratio']
            mean_pe = peer_stats.loc['mean', 'pe_ratio']
            
            implied_market_cap_median = target.net_income * median_pe
            implied_market_cap_mean = target.net_income * mean_pe
            
            implied_valuations['earnings_multiple'] = {
                'median_multiple': median_pe,
                'mean_multiple': mean_pe,
                'implied_market_cap_median': implied_market_cap_median,
                'implied_market_cap_mean': implied_market_cap_mean,
                'implied_price_median': implied_market_cap_median / target.shares_outstanding,
                'implied_price_mean': implied_market_cap_mean / target.shares_outstanding,
                'current_price': target.current_price
            }
        
        return implied_valuations
    
    def generate_football_field(self) -> Dict[str, Tuple[float, float]]:
        """
        Generate football field chart data (valuation ranges)
        
        Returns:
            Dict[str, Tuple[float, float]]: Valuation ranges for each method
        """
        implied_vals = self.calculate_implied_valuation()
        football_field = {}
        
        for method, data in implied_vals.items():
            if 'implied_price_median' in data and 'implied_price_mean' in data:
                median_price = data['implied_price_median']
                mean_price = data['implied_price_mean']
                
                # Create range (use 25th and 75th percentiles if available)
                # For simplicity, use mean +/- 15%
                low_price = min(median_price, mean_price) * 0.85
                high_price = max(median_price, mean_price) * 1.15
                
                football_field[method.replace('_', ' ').title()] = (low_price, high_price)
        
        return football_field
    
    def _calculate_company_multiples(self, profile: CompanyProfile) -> Dict:
        """Calculate multiples for a single company"""
        multiples = {
            'symbol': profile.symbol,
            'company_name': profile.name,
            'market_cap': profile.market_cap,
            'enterprise_value': profile.enterprise_value,
            'revenue': profile.revenue,
            'ebitda': profile.ebitda,
            'net_income': profile.net_income,
            'current_price': profile.current_price
        }
        
        # Price multiples
        if profile.net_income > 0:
            multiples['pe_ratio'] = profile.market_cap / profile.net_income
        else:
            multiples['pe_ratio'] = np.nan
            
        # Enterprise multiples
        if profile.revenue > 0:
            multiples['ev_revenue'] = profile.enterprise_value / profile.revenue
        else:
            multiples['ev_revenue'] = np.nan
            
        if profile.ebitda > 0:
            multiples['ev_ebitda'] = profile.enterprise_value / profile.ebitda
        else:
            multiples['ev_ebitda'] = np.nan
        
        # Other ratios
        if profile.revenue > 0:
            multiples['price_sales'] = profile.market_cap / profile.revenue
        else:
            multiples['price_sales'] = np.nan
        
        return multiples
    
    def _calculate_peer_statistics(self, multiples_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical measures for peer multiples"""
        # Filter to peer companies only
        peer_df = multiples_df[multiples_df['company_type'] == 'Peer']
        
        # Select numeric columns for statistics
        numeric_cols = ['pe_ratio', 'ev_revenue', 'ev_ebitda', 'price_sales']
        peer_numeric = peer_df[numeric_cols]
        
        # Calculate statistics
        stats_dict = {}
        for col in numeric_cols:
            valid_data = peer_numeric[col].dropna()
            if len(valid_data) > 0:
                stats_dict[col] = {
                    'count': len(valid_data),
                    'mean': valid_data.mean(),
                    'median': valid_data.median(),
                    'std': valid_data.std(),
                    'min': valid_data.min(),
                    'max': valid_data.max(),
                    '25%': valid_data.quantile(0.25),
                    '75%': valid_data.quantile(0.75)
                }
            else:
                stats_dict[col] = {k: np.nan for k in ['count', 'mean', 'median', 'std', 'min', 'max', '25%', '75%']}
        
        # Convert to DataFrame
        peer_stats = pd.DataFrame(stats_dict).T
        
        return peer_stats
    
    def _extract_revenue(self, company_info: Dict) -> float:
        """Extract revenue from company info (TTM)"""
        # Try multiple sources for revenue
        financial_statements = company_info.get('financial_statements', {})
        income_statement = financial_statements.get('income_statement', {})
        
        # Look for revenue in different formats
        revenue_keys = ['Total Revenue', 'Revenue', 'Net Sales', 'Total Revenues']
        
        for key in revenue_keys:
            if key in income_statement:
                revenue_data = income_statement[key]
                if isinstance(revenue_data, dict) and len(revenue_data) > 0:
                    # Get most recent year's data
                    latest_date = max(revenue_data.keys())
                    return abs(revenue_data[latest_date])  # Ensure positive
        
        # Fallback: try basic info
        basic_info = company_info.get('basic_info', {})
        return basic_info.get('revenue', 0)
    
    def _extract_ebitda(self, company_info: Dict) -> float:
        """Extract EBITDA from company info"""
        financial_statements = company_info.get('financial_statements', {})
        income_statement = financial_statements.get('income_statement', {})
        
        # Try to find EBITDA or calculate it
        ebitda_keys = ['EBITDA', 'Normalized EBITDA']
        
        for key in ebitda_keys:
            if key in income_statement:
                ebitda_data = income_statement[key]
                if isinstance(ebitda_data, dict) and len(ebitda_data) > 0:
                    latest_date = max(ebitda_data.keys())
                    return abs(ebitda_data[latest_date])
        
        # Calculate EBITDA if not available directly
        # EBITDA = Operating Income + Depreciation & Amortization
        operating_income = self._extract_operating_income(company_info)
        depreciation = self._extract_depreciation(company_info)
        
        if operating_income > 0:
            return operating_income + depreciation
        
        return 0
    
    def _extract_net_income(self, company_info: Dict) -> float:
        """Extract net income from company info"""
        financial_statements = company_info.get('financial_statements', {})
        income_statement = financial_statements.get('income_statement', {})
        
        income_keys = ['Net Income', 'Net Income Common Stockholders', 'Net Income Applicable To Common Shares']
        
        for key in income_keys:
            if key in income_statement:
                income_data = income_statement[key]
                if isinstance(income_data, dict) and len(income_data) > 0:
                    latest_date = max(income_data.keys())
                    return income_data[latest_date]  # Can be negative
        
        return 0
    
    def _extract_operating_income(self, company_info: Dict) -> float:
        """Extract operating income"""
        financial_statements = company_info.get('financial_statements', {})
        income_statement = financial_statements.get('income_statement', {})
        
        operating_keys = ['Operating Income', 'Operating Revenue', 'Income From Operations']
        
        for key in operating_keys:
            if key in income_statement:
                operating_data = income_statement[key]
                if isinstance(operating_data, dict) and len(operating_data) > 0:
                    latest_date = max(operating_data.keys())
                    return abs(operating_data[latest_date])
        
        return 0
    
    def _extract_depreciation(self, company_info: Dict) -> float:
        """Extract depreciation and amortization"""
        financial_statements = company_info.get('financial_statements', {})
        cashflow = financial_statements.get('cash_flow', {})
        
        depreciation_keys = ['Depreciation And Amortization', 'Depreciation', 'Amortization']
        
        for key in depreciation_keys:
            if key in cashflow:
                depreciation_data = cashflow[key]
                if isinstance(depreciation_data, dict) and len(depreciation_data) > 0:
                    latest_date = max(depreciation_data.keys())
                    return abs(depreciation_data[latest_date])
        
        return 0
    
    def _extract_total_debt(self, company_info: Dict) -> float:
        """Extract total debt"""
        financial_statements = company_info.get('financial_statements', {})
        balance_sheet = financial_statements.get('balance_sheet', {})
        
        debt_keys = ['Total Debt', 'Long Term Debt', 'Total Liabilities']
        
        for key in debt_keys:
            if key in balance_sheet:
                debt_data = balance_sheet[key]
                if isinstance(debt_data, dict) and len(debt_data) > 0:
                    latest_date = max(debt_data.keys())
                    return abs(debt_data[latest_date])
        
        return 0
    
    def _extract_cash(self, company_info: Dict) -> float:
        """Extract cash and cash equivalents"""
        financial_statements = company_info.get('financial_statements', {})
        balance_sheet = financial_statements.get('balance_sheet', {})
        
        cash_keys = ['Cash And Cash Equivalents', 'Cash', 'Cash And Short Term Investments']
        
        for key in cash_keys:
            if key in balance_sheet:
                cash_data = balance_sheet[key]
                if isinstance(cash_data, dict) and len(cash_data) > 0:
                    latest_date = max(cash_data.keys())
                    return abs(cash_data[latest_date])
        
        return 0
    
    def _get_predefined_peers(self, symbol: str) -> List[str]:
        """Get predefined peer companies based on industry knowledge"""
        peer_mapping = {
            # Technology
            'AAPL': ['MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ORCL', 'CRM', 'ADBE'],
            'MSFT': ['AAPL', 'GOOGL', 'AMZN', 'META', 'ORCL', 'CRM', 'ADBE', 'IBM', 'INTC'],
            'GOOGL': ['AAPL', 'MSFT', 'META', 'AMZN', 'NFLX', 'ADBE', 'CRM', 'SNAP', 'TWTR'],
            'META': ['GOOGL', 'SNAP', 'PINS', 'TWTR', 'NFLX', 'ROKU', 'SPOT', 'AAPL', 'MSFT'],
            'AMZN': ['GOOGL', 'MSFT', 'AAPL', 'WMT', 'TGT', 'COST', 'SHOP', 'BABA', 'JD'],
            'TSLA': ['F', 'GM', 'NIO', 'RIVN', 'LCID', 'XPEV', 'LI', 'NKLA', 'RIDE'],
            'NVDA': ['AMD', 'INTC', 'QCOM', 'AVGO', 'MU', 'AMAT', 'LRCX', 'KLAC', 'ADI'],
            
            # Banking
            'JPM': ['BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF'],
            'BAC': ['JPM', 'WFC', 'C', 'USB', 'PNC', 'TFC', 'COF', 'KEY', 'RF'],
            'WFC': ['JPM', 'BAC', 'C', 'USB', 'PNC', 'TFC', 'COF', 'KEY', 'RF'],
            
            # Healthcare
            'JNJ': ['PFE', 'UNH', 'MRK', 'ABBV', 'TMO', 'DHR', 'ABT', 'BMY', 'LLY'],
            'PFE': ['JNJ', 'MRK', 'ABBV', 'BMY', 'LLY', 'GILD', 'BIIB', 'AMGN', 'GSK'],
            
            # Retail
            'WMT': ['TGT', 'COST', 'HD', 'LOW', 'AMZN', 'DG', 'DLTR', 'KR', 'SYY'],
            'TGT': ['WMT', 'COST', 'HD', 'LOW', 'DG', 'DLTR', 'KR', 'BBBY', 'JWN'],
            
            # Energy
            'XOM': ['CVX', 'COP', 'SLB', 'EOG', 'PSX', 'VLO', 'MPC', 'OXY', 'DVN'],
            'CVX': ['XOM', 'COP', 'SLB', 'EOG', 'PSX', 'VLO', 'MPC', 'OXY', 'DVN']
        }
        
        return peer_mapping.get(symbol, [])
    
    def _get_sector_peers(self, sector: str, target_market_cap: float, min_market_cap: float) -> List[str]:
        """Get peers based on sector (simplified implementation)"""
        # This is a simplified implementation
        # In a production system, you would query a database of companies by sector
        
        sector_mapping = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'ORCL', 'CRM', 'ADBE', 'INTC'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'TMO', 'DHR', 'ABT', 'BMY', 'LLY'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'LOW', 'NKE', 'SBUX', 'TJX', 'F', 'GM'],
            'Consumer Defensive': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'TGT', 'CL', 'KMB', 'GIS'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'VLO', 'MPC', 'OXY'],
            'Industrials': ['BA', 'CAT', 'DE', 'UNP', 'UPS', 'FDX', 'RTX', 'LMT', 'GE'],
            'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'CHTR'],
            'Utilities': ['NEE', 'D', 'SO', 'DUK', 'EXC', 'AEP', 'PEG', 'SRE', 'XEL'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'WELL', 'SPG', 'PSA', 'O', 'CBRE']
        }
        
        return sector_mapping.get(sector, [])


def create_sample_comparable_analysis(target_symbol: str = "AAPL") -> ComparableAnalyzer:
    """
    Create a sample comparable analysis for demonstration
    
    Args:
        target_symbol (str): Target company symbol
        
    Returns:
        ComparableAnalyzer: Configured comparable analyzer
    """
    analyzer = ComparableAnalyzer(target_symbol)
    
    try:
        # Load target company
        target_profile = analyzer.load_target_company()
        
        # Find peers
        peer_symbols = analyzer.find_peer_companies(auto_select=True)
        
        # Load peer data
        peer_profiles = analyzer.load_peer_data(peer_symbols)
        
        return analyzer
        
    except Exception as e:
        logger.error(f"Error creating sample analysis: {str(e)}")
        raise
