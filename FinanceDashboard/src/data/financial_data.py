import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FinancialDataProcessor:
    """
    Process and clean financial statement data for valuation models.
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.financial_data = {}
        
    def fetch_all_statements(self) -> Dict:
        """Fetch all financial statements."""
        
        try:
            self.financial_data = {
                'income_statement': self.stock.financials,
                'balance_sheet': self.stock.balance_sheet,
                'cash_flow': self.stock.cashflow,
                'quarterly_income': self.stock.quarterly_financials,
                'quarterly_balance': self.stock.quarterly_balance_sheet,
                'quarterly_cashflow': self.stock.quarterly_cashflow,
                'info': self.stock.info
            }
            return self.financial_data
        except Exception as e:
            return {'error': str(e)}
    
    def get_revenue_data(self, quarterly: bool = False) -> pd.Series:
        """Extract revenue data."""
        
        if not self.financial_data:
            self.fetch_all_statements()
        
        try:
            if quarterly:
                stmt = self.financial_data.get('quarterly_income', pd.DataFrame())
            else:
                stmt = self.financial_data.get('income_statement', pd.DataFrame())
            
            if stmt.empty:
                return pd.Series()
            
            # Try different revenue line items
            revenue_items = ['Total Revenue', 'Revenue', 'Net Sales', 'Sales']
            
            for item in revenue_items:
                if item in stmt.index:
                    return stmt.loc[item].dropna()
            
            return pd.Series()
            
        except Exception as e:
            return pd.Series()
    
    def get_profitability_metrics(self) -> Dict:
        """Calculate key profitability metrics."""
        
        if not self.financial_data:
            self.fetch_all_statements()
        
        try:
            income_stmt = self.financial_data.get('income_statement', pd.DataFrame())
            
            if income_stmt.empty:
                return {}
            
            # Get latest year data
            latest_data = income_stmt.iloc[:, 0] if not income_stmt.empty else pd.Series()
            
            metrics = {}
            
            # Revenue
            revenue = self._get_line_item(latest_data, ['Total Revenue', 'Revenue', 'Net Sales'])
            metrics['revenue'] = revenue
            
            # Gross Profit and Margin
            gross_profit = self._get_line_item(latest_data, ['Gross Profit'])
            if gross_profit and revenue:
                metrics['gross_profit'] = gross_profit
                metrics['gross_margin'] = gross_profit / revenue
            
            # Operating Income and Margin
            operating_income = self._get_line_item(latest_data, ['Operating Income', 'EBIT'])
            if operating_income and revenue:
                metrics['operating_income'] = operating_income
                metrics['operating_margin'] = operating_income / revenue
            
            # EBITDA
            ebitda = self._get_line_item(latest_data, ['EBITDA'])
            if not ebitda:
                # Calculate EBITDA if not directly available
                depreciation = self._get_line_item(latest_data, ['Depreciation And Amortization'])
                if operating_income and depreciation:
                    ebitda = operating_income + depreciation
            
            if ebitda and revenue:
                metrics['ebitda'] = ebitda
                metrics['ebitda_margin'] = ebitda / revenue
            
            # Net Income and Margin
            net_income = self._get_line_item(latest_data, ['Net Income', 'Net Income Common Stockholders'])
            if net_income and revenue:
                metrics['net_income'] = net_income
                metrics['net_margin'] = net_income / revenue
            
            return metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_balance_sheet_metrics(self) -> Dict:
        """Extract key balance sheet metrics."""
        
        if not self.financial_data:
            self.fetch_all_statements()
        
        try:
            balance_sheet = self.financial_data.get('balance_sheet', pd.DataFrame())
            
            if balance_sheet.empty:
                return {}
            
            latest_data = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
            
            metrics = {}
            
            # Assets
            total_assets = self._get_line_item(latest_data, ['Total Assets'])
            current_assets = self._get_line_item(latest_data, ['Current Assets'])
            cash = self._get_line_item(latest_data, ['Cash And Cash Equivalents', 'Cash'])
            
            metrics.update({
                'total_assets': total_assets,
                'current_assets': current_assets,
                'cash_and_equivalents': cash
            })
            
            # Liabilities
            total_liabilities = self._get_line_item(latest_data, ['Total Liabilities Net Minority Interest', 'Total Liabilities'])
            current_liabilities = self._get_line_item(latest_data, ['Current Liabilities'])
            total_debt = self._get_line_item(latest_data, ['Total Debt'])
            
            metrics.update({
                'total_liabilities': total_liabilities,
                'current_liabilities': current_liabilities,
                'total_debt': total_debt
            })
            
            # Equity
            shareholders_equity = self._get_line_item(latest_data, ['Total Equity Gross Minority Interest', 'Stockholders Equity'])
            
            metrics['shareholders_equity'] = shareholders_equity
            
            # Calculate ratios
            if current_assets and current_liabilities:
                metrics['current_ratio'] = current_assets / current_liabilities
            
            if cash and current_liabilities:
                metrics['quick_ratio'] = cash / current_liabilities
            
            if total_debt and shareholders_equity:
                metrics['debt_to_equity'] = total_debt / shareholders_equity
            
            if total_debt and total_assets:
                metrics['debt_to_assets'] = total_debt / total_assets
            
            return metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_cash_flow_metrics(self) -> Dict:
        """Extract cash flow metrics."""
        
        if not self.financial_data:
            self.fetch_all_statements()
        
        try:
            cash_flow = self.financial_data.get('cash_flow', pd.DataFrame())
            
            if cash_flow.empty:
                return {}
            
            latest_data = cash_flow.iloc[:, 0] if not cash_flow.empty else pd.Series()
            
            metrics = {}
            
            # Operating Cash Flow
            operating_cf = self._get_line_item(latest_data, ['Operating Cash Flow', 'Cash Flow From Operations'])
            metrics['operating_cash_flow'] = operating_cf
            
            # Investing Cash Flow
            investing_cf = self._get_line_item(latest_data, ['Investing Cash Flow'])
            metrics['investing_cash_flow'] = investing_cf
            
            # Financing Cash Flow
            financing_cf = self._get_line_item(latest_data, ['Financing Cash Flow'])
            metrics['financing_cash_flow'] = financing_cf
            
            # Free Cash Flow
            capex = self._get_line_item(latest_data, ['Capital Expenditure', 'Capital Expenditures'])
            if operating_cf and capex:
                # CapEx is usually negative, so we add it (subtract the absolute value)
                free_cash_flow = operating_cf + capex  # capex is negative
                metrics['free_cash_flow'] = free_cash_flow
            
            # Dividends
            dividends = self._get_line_item(latest_data, ['Common Stock Dividend Paid', 'Dividends Paid'])
            metrics['dividends_paid'] = dividends
            
            return metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_growth_rates(self, periods: int = 3) -> Dict:
        """Calculate historical growth rates."""
        
        if not self.financial_data:
            self.fetch_all_statements()
        
        growth_rates = {}
        
        # Revenue growth
        revenue_data = self.get_revenue_data()
        if len(revenue_data) >= periods:
            revenue_growth = self._calculate_cagr(revenue_data, periods)
            growth_rates['revenue_cagr'] = revenue_growth
        
        # Income statement growth rates
        try:
            income_stmt = self.financial_data.get('income_statement', pd.DataFrame())
            
            if not income_stmt.empty and income_stmt.shape[1] >= periods:
                # EBITDA growth
                ebitda_row = None
                for item in ['EBITDA']:
                    if item in income_stmt.index:
                        ebitda_row = income_stmt.loc[item]
                        break
                
                if ebitda_row is not None and len(ebitda_row.dropna()) >= periods:
                    ebitda_growth = self._calculate_cagr(ebitda_row, periods)
                    growth_rates['ebitda_cagr'] = ebitda_growth
                
                # Net income growth
                ni_row = None
                for item in ['Net Income', 'Net Income Common Stockholders']:
                    if item in income_stmt.index:
                        ni_row = income_stmt.loc[item]
                        break
                
                if ni_row is not None and len(ni_row.dropna()) >= periods:
                    ni_growth = self._calculate_cagr(ni_row, periods)
                    growth_rates['net_income_cagr'] = ni_growth
        
        except Exception as e:
            growth_rates['error'] = str(e)
        
        return growth_rates
    
    def _get_line_item(self, data: pd.Series, possible_names: List[str]) -> float:
        """Get line item value from financial statement."""
        
        for name in possible_names:
            if name in data.index and pd.notna(data[name]):
                return float(data[name])
        return 0.0
    
    def _calculate_cagr(self, data_series: pd.Series, periods: int) -> float:
        """Calculate Compound Annual Growth Rate."""
        
        try:
            clean_data = data_series.dropna()
            if len(clean_data) < periods:
                return 0.0
            
            # Get the most recent values
            latest_values = clean_data.head(periods)
            
            if len(latest_values) < 2:
                return 0.0
            
            ending_value = latest_values.iloc[0]  # Most recent
            beginning_value = latest_values.iloc[-1]  # Oldest in the period
            
            if beginning_value <= 0 or ending_value <= 0:
                return 0.0
            
            years = periods - 1
            if years <= 0:
                return 0.0
            
            cagr = (ending_value / beginning_value) ** (1/years) - 1
            return cagr
            
        except Exception as e:
            return 0.0
    
    def get_financial_summary(self) -> Dict:
        """Get comprehensive financial summary."""
        
        summary = {
            'ticker': self.ticker,
            'last_updated': datetime.now().isoformat()
        }
        
        # Get all metrics
        profitability = self.get_profitability_metrics()
        balance_sheet = self.get_balance_sheet_metrics()
        cash_flow = self.get_cash_flow_metrics()
        growth_rates = self.calculate_growth_rates()
        
        # Combine all metrics
        summary.update(profitability)
        summary.update(balance_sheet)
        summary.update(cash_flow)
        summary.update(growth_rates)
        
        # Add market data from info
        if 'info' in self.financial_data:
            info = self.financial_data['info']
            summary.update({
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'current_price': info.get('currentPrice', 0),
                'beta': info.get('beta', 1.0)
            })
        
        return summary
