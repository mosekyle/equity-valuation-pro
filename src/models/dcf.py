"""
DCF (Discounted Cash Flow) Model for Equity Valuation Dashboard
Professional-grade DCF implementation with scenario analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from ..utils.calculations import FinancialCalculations

logger = logging.getLogger(__name__)


@dataclass
class DCFAssumptions:
    """Data class to store DCF model assumptions"""
    
    # Revenue assumptions
    base_revenue: float
    revenue_growth_rates: List[float]  # Year-by-year growth rates
    terminal_growth_rate: float = 0.025
    
    # Profitability assumptions
    operating_margin: float
    tax_rate: float = 0.25
    
    # Investment assumptions
    capex_percent_of_revenue: float = 0.03
    depreciation_percent_of_revenue: float = 0.025
    working_capital_percent_of_revenue: float = 0.02
    
    # Cost of capital assumptions
    risk_free_rate: float = 0.03
    market_risk_premium: float = 0.06
    beta: float = 1.0
    debt_to_equity_ratio: float = 0.3
    cost_of_debt: float = 0.04
    
    # Other assumptions
    shares_outstanding: int = 1000000000
    net_debt: float = 0  # Net debt (debt - cash)
    
    def calculate_wacc(self) -> float:
        """Calculate WACC based on assumptions"""
        debt_ratio = self.debt_to_equity_ratio / (1 + self.debt_to_equity_ratio)
        equity_ratio = 1 - debt_ratio
        
        cost_of_equity = self.risk_free_rate + self.beta * self.market_risk_premium
        wacc = (equity_ratio * cost_of_equity) + (debt_ratio * self.cost_of_debt * (1 - self.tax_rate))
        
        return wacc


class DCFModel:
    """
    Professional DCF Model with comprehensive valuation capabilities
    """
    
    def __init__(self, company_symbol: str = ""):
        """
        Initialize DCF Model
        
        Args:
            company_symbol (str): Stock symbol for the company being valued
        """
        self.symbol = company_symbol
        self.assumptions = None
        self.projections = None
        self.valuation_results = None
        
    def set_assumptions(self, assumptions: DCFAssumptions):
        """Set DCF assumptions"""
        self.assumptions = assumptions
        
    def build_projections(self) -> pd.DataFrame:
        """
        Build financial projections based on assumptions
        
        Returns:
            pd.DataFrame: Detailed financial projections
        """
        if not self.assumptions:
            raise ValueError("DCF assumptions must be set before building projections")
        
        num_years = len(self.assumptions.revenue_growth_rates)
        projections = []
        
        # Initialize base year values
        current_revenue = self.assumptions.base_revenue
        
        for year in range(num_years):
            year_num = year + 1
            growth_rate = self.assumptions.revenue_growth_rates[year]
            
            # Revenue projection
            if year == 0:
                revenue = current_revenue * (1 + growth_rate)
            else:
                revenue = projections[year-1]['revenue'] * (1 + growth_rate)
            
            # Operating income
            operating_income = revenue * self.assumptions.operating_margin
            
            # Taxes
            taxes = operating_income * self.assumptions.tax_rate
            nopat = operating_income - taxes  # Net Operating Profit After Tax
            
            # Depreciation & Amortization
            depreciation = revenue * self.assumptions.depreciation_percent_of_revenue
            
            # Capital Expenditure
            capex = revenue * self.assumptions.capex_percent_of_revenue
            
            # Working Capital Change
            if year == 0:
                working_capital = revenue * self.assumptions.working_capital_percent_of_revenue
                wc_change = working_capital  # Assume starting from zero
            else:
                prev_wc = projections[year-1]['working_capital']
                working_capital = revenue * self.assumptions.working_capital_percent_of_revenue
                wc_change = working_capital - prev_wc
            
            # Free Cash Flow calculation
            fcf = nopat + depreciation - capex - wc_change
            
            # Store year projections
            year_projection = {
                'year': year_num,
                'revenue': revenue,
                'revenue_growth': growth_rate,
                'operating_income': operating_income,
                'operating_margin': operating_income / revenue,
                'taxes': taxes,
                'nopat': nopat,
                'depreciation': depreciation,
                'capex': capex,
                'working_capital': working_capital,
                'wc_change': wc_change,
                'free_cash_flow': fcf
            }
            
            projections.append(year_projection)
        
        self.projections = pd.DataFrame(projections)
        return self.projections
    
    def calculate_terminal_value(self) -> Dict[str, float]:
        """
        Calculate terminal value using multiple methods
        
        Returns:
            Dict[str, float]: Terminal value calculations
        """
        if self.projections is None or self.projections.empty:
            raise ValueError("Projections must be built before calculating terminal value")
        
        wacc = self.assumptions.calculate_wacc()
        final_year_fcf = self.projections['free_cash_flow'].iloc[-1]
        projection_years = len(self.projections)
        terminal_growth = self.assumptions.terminal_growth_rate
        
        # Method 1: Gordon Growth Model
        terminal_year_fcf = final_year_fcf * (1 + terminal_growth)
        terminal_value_gordon = terminal_year_fcf / (wacc - terminal_growth)
        pv_terminal_gordon = terminal_value_gordon / ((1 + wacc) ** projection_years)
        
        # Method 2: Exit Multiple Method (using conservative EV/EBITDA multiple)
        final_year_operating_income = self.projections['operating_income'].iloc[-1]
        final_year_depreciation = self.projections['depreciation'].iloc[-1]
        final_year_ebitda = final_year_operating_income + final_year_depreciation
        
        # Conservative exit multiple (typically 10-15x EBITDA for mature companies)
        exit_multiple = 12.0
        terminal_value_multiple = final_year_ebitda * exit_multiple
        pv_terminal_multiple = terminal_value_multiple / ((1 + wacc) ** projection_years)
        
        # Take average of both methods for conservative estimate
        average_terminal_value = (pv_terminal_gordon + pv_terminal_multiple) / 2
        
        return {
            'gordon_growth_terminal_value': pv_terminal_gordon,
            'exit_multiple_terminal_value': pv_terminal_multiple,
            'average_terminal_value': average_terminal_value,
            'terminal_growth_rate': terminal_growth,
            'exit_multiple_used': exit_multiple,
            'terminal_year_fcf': terminal_year_fcf,
            'terminal_year_ebitda': final_year_ebitda
        }
    
    def calculate_dcf_valuation(self) -> Dict[str, float]:
        """
        Calculate complete DCF valuation
        
        Returns:
            Dict[str, float]: Comprehensive valuation results
        """
        if self.projections is None:
            self.build_projections()
        
        wacc = self.assumptions.calculate_wacc()
        terminal_values = self.calculate_terminal_value()
        
        # Present value of projected FCFs
        pv_fcfs = []
        for i, fcf in enumerate(self.projections['free_cash_flow']):
            pv = fcf / ((1 + wacc) ** (i + 1))
            pv_fcfs.append(pv)
        
        total_pv_fcf = sum(pv_fcfs)
        
        # Enterprise Value
        enterprise_value = total_pv_fcf + terminal_values['average_terminal_value']
        
        # Equity Value (Enterprise Value - Net Debt)
        equity_value = enterprise_value - self.assumptions.net_debt
        
        # Value per Share
        value_per_share = equity_value / self.assumptions.shares_outstanding
        
        # Store comprehensive results
        self.valuation_results = {
            'enterprise_value': enterprise_value,
            'equity_value': equity_value,
            'value_per_share': value_per_share,
            'pv_projection_period': total_pv_fcf,
            'pv_terminal_value': terminal_values['average_terminal_value'],
            'wacc': wacc,
            'terminal_value_gordon': terminal_values['gordon_growth_terminal_value'],
            'terminal_value_multiple': terminal_values['exit_multiple_terminal_value'],
            'net_debt': self.assumptions.net_debt,
            'shares_outstanding': self.assumptions.shares_outstanding,
            'pv_fcf_breakdown': pv_fcfs
        }
        
        return self.valuation_results
    
    def sensitivity_analysis(self, 
                           wacc_range: List[float] = None,
                           terminal_growth_range: List[float] = None) -> pd.DataFrame:
        """
        Perform sensitivity analysis on key variables
        
        Args:
            wacc_range (List[float]): Range of WACC values to test
            terminal_growth_range (List[float]): Range of terminal growth rates to test
            
        Returns:
            pd.DataFrame: Sensitivity analysis matrix
        """
        if wacc_range is None:
            base_wacc = self.assumptions.calculate_wacc()
            wacc_range = np.arange(base_wacc - 0.02, base_wacc + 0.025, 0.005)
        
        if terminal_growth_range is None:
            base_terminal = self.assumptions.terminal_growth_rate
            terminal_growth_range = np.arange(base_terminal - 0.01, base_terminal + 0.015, 0.005)
        
        sensitivity_matrix = []
        
        # Store original assumptions
        original_terminal_growth = self.assumptions.terminal_growth_rate
        original_risk_free_rate = self.assumptions.risk_free_rate
        original_beta = self.assumptions.beta
        original_mrp = self.assumptions.market_risk_premium
        
        for terminal_growth in terminal_growth_range:
            row = []
            for wacc in wacc_range:
                # Temporarily modify assumptions
                self.assumptions.terminal_growth_rate = terminal_growth
                
                # Adjust risk-free rate to achieve target WACC (simplified)
                # This is an approximation - in practice you'd adjust specific components
                target_cost_of_equity = wacc  # Simplified assumption
                implied_risk_free = target_cost_of_equity - (self.assumptions.beta * self.assumptions.market_risk_premium)
                self.assumptions.risk_free_rate = max(0.01, implied_risk_free)
                
                # Calculate valuation with modified assumptions
                try:
                    # Rebuild projections and recalculate
                    self.build_projections()
                    valuation = self.calculate_dcf_valuation()
                    value_per_share = valuation['value_per_share']
                    row.append(value_per_share)
                except:
                    row.append(0)
            
            sensitivity_matrix.append(row)
        
        # Restore original assumptions
        self.assumptions.terminal_growth_rate = original_terminal_growth
        self.assumptions.risk_free_rate = original_risk_free_rate
        
        # Create DataFrame
        sensitivity_df = pd.DataFrame(
            sensitivity_matrix,
            index=[f"{tg:.1%}" for tg in terminal_growth_range],
            columns=[f"{w:.1%}" for w in wacc_range]
        )
        
        return sensitivity_df
    
    def scenario_analysis(self) -> Dict[str, Dict]:
        """
        Perform scenario analysis (Bear, Base, Bull cases)
        
        Returns:
            Dict[str, Dict]: Results for each scenario
        """
        base_assumptions = self.assumptions
        scenarios = {}
        
        # Base Case
        scenarios['Base'] = self.calculate_dcf_valuation()
        
        # Bear Case (pessimistic assumptions)
        bear_assumptions = DCFAssumptions(
            base_revenue=base_assumptions.base_revenue,
            revenue_growth_rates=[max(0, rate - 0.03) for rate in base_assumptions.revenue_growth_rates],
            terminal_growth_rate=max(0.01, base_assumptions.terminal_growth_rate - 0.01),
            operating_margin=max(0.05, base_assumptions.operating_margin - 0.02),
            tax_rate=base_assumptions.tax_rate + 0.05,
            capex_percent_of_revenue=base_assumptions.capex_percent_of_revenue + 0.01,
            risk_free_rate=base_assumptions.risk_free_rate,
            market_risk_premium=base_assumptions.market_risk_premium + 0.01,
            beta=base_assumptions.beta + 0.2,
            debt_to_equity_ratio=base_assumptions.debt_to_equity_ratio,
            cost_of_debt=base_assumptions.cost_of_debt + 0.01,
            shares_outstanding=base_assumptions.shares_outstanding,
            net_debt=base_assumptions.net_debt
        )
        
        bear_model = DCFModel(self.symbol)
        bear_model.set_assumptions(bear_assumptions)
        scenarios['Bear'] = bear_model.calculate_dcf_valuation()
        
        # Bull Case (optimistic assumptions)
        bull_assumptions = DCFAssumptions(
            base_revenue=base_assumptions.base_revenue,
            revenue_growth_rates=[rate + 0.03 for rate in base_assumptions.revenue_growth_rates],
            terminal_growth_rate=min(0.05, base_assumptions.terminal_growth_rate + 0.01),
            operating_margin=min(0.40, base_assumptions.operating_margin + 0.02),
            tax_rate=max(0.15, base_assumptions.tax_rate - 0.05),
            capex_percent_of_revenue=max(0.01, base_assumptions.capex_percent_of_revenue - 0.005),
            risk_free_rate=base_assumptions.risk_free_rate,
            market_risk_premium=max(0.04, base_assumptions.market_risk_premium - 0.01),
            beta=max(0.5, base_assumptions.beta - 0.2),
            debt_to_equity_ratio=base_assumptions.debt_to_equity_ratio,
            cost_of_debt=max(0.02, base_assumptions.cost_of_debt - 0.01),
            shares_outstanding=base_assumptions.shares_outstanding,
            net_debt=base_assumptions.net_debt
        )
        
        bull_model = DCFModel(self.symbol)
        bull_model.set_assumptions(bull_assumptions)
        scenarios['Bull'] = bull_model.calculate_dcf_valuation()
        
        return scenarios
    
    def get_summary_table(self) -> pd.DataFrame:
        """
        Get summary table of key valuation metrics
        
        Returns:
            pd.DataFrame: Summary of valuation results
        """
        if not self.valuation_results:
            self.calculate_dcf_valuation()
        
        summary_data = {
            'Metric': [
                'Enterprise Value ($M)',
                'Equity Value ($M)',
                'Value per Share ($)',
                'PV of Projection Period ($M)',
                'PV of Terminal Value ($M)',
                'Terminal Value % of Total',
                'WACC',
                'Terminal Growth Rate',
                'Shares Outstanding (M)'
            ],
            'Value': [
                f"${self.valuation_results['enterprise_value']/1e6:,.0f}",
                f"${self.valuation_results['equity_value']/1e6:,.0f}",
                f"${self.valuation_results['value_per_share']:.2f}",
                f"${self.valuation_results['pv_projection_period']/1e6:,.0f}",
                f"${self.valuation_results['pv_terminal_value']/1e6:,.0f}",
                f"{(self.valuation_results['pv_terminal_value']/self.valuation_results['enterprise_value']*100):.1f}%",
                f"{self.valuation_results['wacc']:.1%}",
                f"{self.assumptions.terminal_growth_rate:.1%}",
                f"{self.assumptions.shares_outstanding/1e6:,.0f}"
            ]
        }
        
        return pd.DataFrame(summary_data)


def create_sample_dcf_model(symbol: str = "AAPL") -> DCFModel:
    """
    Create a sample DCF model with reasonable assumptions
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        DCFModel: Configured DCF model
    """
    # Sample assumptions for demonstration
    assumptions = DCFAssumptions(
        base_revenue=394_328_000_000,  # Apple's 2023 revenue (~$394B)
        revenue_growth_rates=[0.05, 0.04, 0.03, 0.03, 0.02],  # 5-year declining growth
        terminal_growth_rate=0.025,
        operating_margin=0.30,  # Apple's operating margin is typically ~30%
        tax_rate=0.25,
        capex_percent_of_revenue=0.025,
        depreciation_percent_of_revenue=0.030,
        working_capital_percent_of_revenue=0.01,
        risk_free_rate=0.035,
        market_risk_premium=0.065,
        beta=1.2,
        debt_to_equity_ratio=0.15,
        cost_of_debt=0.03,
        shares_outstanding=15_550_000_000,  # Apple shares outstanding
        net_debt=-60_000_000_000  # Apple has net cash position
    )
    
    model = DCFModel(symbol)
    model.set_assumptions(assumptions)
    
    return model
