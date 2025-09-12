import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import yfinance as yf

class DCFModel:
    """
    Discounted Cash Flow valuation model implementing institutional-grade methodology.
    """
    
    def __init__(self, ticker: str, company_data: Dict = None):
        self.ticker = ticker
        self.company_data = company_data or {}
        self.projections = {}
        self.assumptions = {}
        
    def set_assumptions(self, 
                       revenue_growth: List[float],
                       ebitda_margin: List[float],
                       tax_rate: float = 0.25,
                       capex_pct_revenue: List[float] = None,
                       depreciation_pct_revenue: List[float] = None,
                       working_capital_pct_revenue: float = 0.02,
                       terminal_growth: float = 0.025,
                       discount_rate: float = 0.09) -> None:
        """Set valuation assumptions for DCF model."""
        
        self.assumptions = {
            'revenue_growth': revenue_growth,
            'ebitda_margin': ebitda_margin,
            'tax_rate': tax_rate,
            'capex_pct_revenue': capex_pct_revenue or [0.03] * len(revenue_growth),
            'depreciation_pct_revenue': depreciation_pct_revenue or [0.025] * len(revenue_growth),
            'working_capital_pct_revenue': working_capital_pct_revenue,
            'terminal_growth': terminal_growth,
            'discount_rate': discount_rate
        }
    
    def get_historical_data(self) -> Dict:
        """Fetch historical financial data for the company."""
        try:
            stock = yf.Ticker(self.ticker)
            
            # Get financial statements
            income_stmt = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            info = stock.info
            
            # Extract base year data
            if not income_stmt.empty:
                base_revenue = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else 0
                base_ebitda = income_stmt.loc['EBITDA'].iloc[0] if 'EBITDA' in income_stmt.index else 0
            else:
                base_revenue = 0
                base_ebitda = 0
            
            # Get shares outstanding and current price
            shares_outstanding = info.get('sharesOutstanding', 0)
            current_price = info.get('currentPrice', 0)
            
            return {
                'base_revenue': base_revenue,
                'base_ebitda': base_ebitda,
                'shares_outstanding': shares_outstanding,
                'current_price': current_price,
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'total_debt': info.get('totalDebt', 0),
                'total_cash': info.get('totalCash', 0)
            }
        except Exception as e:
            return {
                'base_revenue': 0,
                'base_ebitda': 0,
                'shares_outstanding': 0,
                'current_price': 0,
                'market_cap': 0,
                'enterprise_value': 0,
                'total_debt': 0,
                'total_cash': 0,
                'error': str(e)
            }
    
    def build_projections(self) -> pd.DataFrame:
        """Build financial projections based on assumptions."""
        
        if not self.assumptions:
            raise ValueError("Assumptions must be set before building projections")
        
        historical = self.get_historical_data()
        base_revenue = historical['base_revenue']
        
        if base_revenue <= 0:
            raise ValueError(f"Invalid base revenue data for {self.ticker}")
        
        years = len(self.assumptions['revenue_growth'])
        projection_years = list(range(1, years + 1))
        
        # Initialize projections dataframe
        projections = pd.DataFrame(index=[
            'Revenue', 'Revenue Growth %', 'EBITDA', 'EBITDA Margin %',
            'Depreciation', 'EBIT', 'Taxes', 'NOPAT', 'CapEx',
            'Change in NWC', 'Unlevered FCF', 'PV Factor', 'PV of FCF'
        ], columns=[f'Year {i}' for i in projection_years])
        
        # Revenue projections
        current_revenue = base_revenue
        for i, year in enumerate(projection_years):
            growth = self.assumptions['revenue_growth'][i]
            current_revenue = current_revenue * (1 + growth)
            projections.loc['Revenue', f'Year {year}'] = current_revenue
            projections.loc['Revenue Growth %', f'Year {year}'] = growth * 100
        
        # EBITDA projections
        for i, year in enumerate(projection_years):
            revenue = projections.loc['Revenue', f'Year {year}']
            ebitda_margin = self.assumptions['ebitda_margin'][i]
            ebitda = revenue * ebitda_margin
            projections.loc['EBITDA', f'Year {year}'] = ebitda
            projections.loc['EBITDA Margin %', f'Year {year}'] = ebitda_margin * 100
        
        # Depreciation, EBIT, and taxes
        for i, year in enumerate(projection_years):
            revenue = projections.loc['Revenue', f'Year {year}']
            ebitda = projections.loc['EBITDA', f'Year {year}']
            
            depreciation = revenue * self.assumptions['depreciation_pct_revenue'][i]
            ebit = ebitda - depreciation
            taxes = ebit * self.assumptions['tax_rate']
            nopat = ebit - taxes
            
            projections.loc['Depreciation', f'Year {year}'] = depreciation
            projections.loc['EBIT', f'Year {year}'] = ebit
            projections.loc['Taxes', f'Year {year}'] = taxes
            projections.loc['NOPAT', f'Year {year}'] = nopat
        
        # CapEx and Working Capital
        for i, year in enumerate(projection_years):
            revenue = projections.loc['Revenue', f'Year {year}']
            capex = revenue * self.assumptions['capex_pct_revenue'][i]
            
            # Working capital change (simplified)
            if i == 0:
                prev_revenue = base_revenue
            else:
                prev_revenue = projections.loc['Revenue', f'Year {i}']
            
            nwc_change = (revenue - prev_revenue) * self.assumptions['working_capital_pct_revenue']
            
            projections.loc['CapEx', f'Year {year}'] = capex
            projections.loc['Change in NWC', f'Year {year}'] = nwc_change
        
        # Unlevered Free Cash Flow
        for i, year in enumerate(projection_years):
            nopat = projections.loc['NOPAT', f'Year {year}']
            depreciation = projections.loc['Depreciation', f'Year {year}']
            capex = projections.loc['CapEx', f'Year {year}']
            nwc_change = projections.loc['Change in NWC', f'Year {year}']
            
            ufcf = nopat + depreciation - capex - nwc_change
            projections.loc['Unlevered FCF', f'Year {year}'] = ufcf
        
        # Present value calculations
        discount_rate = self.assumptions['discount_rate']
        for i, year in enumerate(projection_years):
            pv_factor = 1 / ((1 + discount_rate) ** year)
            ufcf = projections.loc['Unlevered FCF', f'Year {year}']
            pv_fcf = ufcf * pv_factor
            
            projections.loc['PV Factor', f'Year {year}'] = pv_factor
            projections.loc['PV of FCF', f'Year {year}'] = pv_fcf
        
        self.projections = projections
        return projections
    
    def calculate_terminal_value(self) -> Tuple[float, float]:
        """Calculate terminal value using Gordon Growth Model."""
        
        if self.projections.empty:
            raise ValueError("Projections must be built before calculating terminal value")
        
        # Get final year FCF
        final_year = f'Year {len(self.assumptions["revenue_growth"])}'
        final_fcf = self.projections.loc['Unlevered FCF', final_year]
        
        # Terminal FCF (grow by terminal growth rate)
        terminal_growth = self.assumptions['terminal_growth']
        discount_rate = self.assumptions['discount_rate']
        terminal_fcf = final_fcf * (1 + terminal_growth)
        
        # Terminal value
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        
        # Present value of terminal value
        years = len(self.assumptions['revenue_growth'])
        pv_terminal_value = terminal_value / ((1 + discount_rate) ** years)
        
        return terminal_value, pv_terminal_value
    
    def calculate_fair_value(self) -> Dict:
        """Calculate fair value per share."""
        
        # Build projections
        projections = self.build_projections()
        
        # Sum of PV of projected FCF
        pv_projection_period = projections.loc['PV of FCF'].sum()
        
        # Terminal value
        terminal_value, pv_terminal_value = self.calculate_terminal_value()
        
        # Enterprise value
        enterprise_value = pv_projection_period + pv_terminal_value
        
        # Get company data
        historical = self.get_historical_data()
        total_debt = historical['total_debt']
        total_cash = historical['total_cash']
        shares_outstanding = historical['shares_outstanding']
        
        # Equity value
        equity_value = enterprise_value - total_debt + total_cash
        
        # Fair value per share
        if shares_outstanding > 0:
            fair_value_per_share = equity_value / shares_outstanding
        else:
            fair_value_per_share = 0
        
        return {
            'enterprise_value': enterprise_value,
            'equity_value': equity_value,
            'fair_value_per_share': fair_value_per_share,
            'current_price': historical['current_price'],
            'upside_downside': (fair_value_per_share / historical['current_price'] - 1) * 100 if historical['current_price'] > 0 else 0,
            'pv_projection_period': pv_projection_period,
            'pv_terminal_value': pv_terminal_value,
            'terminal_value': terminal_value,
            'total_debt': total_debt,
            'total_cash': total_cash,
            'shares_outstanding': shares_outstanding
        }
    
    def sensitivity_analysis(self, 
                           discount_rate_range: List[float], 
                           terminal_growth_range: List[float]) -> pd.DataFrame:
        """Perform sensitivity analysis on key variables."""
        
        sensitivity_matrix = pd.DataFrame(
            index=discount_rate_range,
            columns=terminal_growth_range
        )
        
        # Store original assumptions
        original_discount_rate = self.assumptions['discount_rate']
        original_terminal_growth = self.assumptions['terminal_growth']
        
        for discount_rate in discount_rate_range:
            for terminal_growth in terminal_growth_range:
                # Update assumptions
                self.assumptions['discount_rate'] = discount_rate
                self.assumptions['terminal_growth'] = terminal_growth
                
                try:
                    # Calculate fair value
                    valuation = self.calculate_fair_value()
                    fair_value = valuation['fair_value_per_share']
                    sensitivity_matrix.loc[discount_rate, terminal_growth] = fair_value
                except:
                    sensitivity_matrix.loc[discount_rate, terminal_growth] = np.nan
        
        # Restore original assumptions
        self.assumptions['discount_rate'] = original_discount_rate
        self.assumptions['terminal_growth'] = original_terminal_growth
        
        return sensitivity_matrix
