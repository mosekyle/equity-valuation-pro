"""
Financial Calculations Module for Equity Valuation Dashboard
Contains all core financial calculation functions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class FinancialCalculations:
    """
    Core financial calculations for equity valuation
    """
    
    @staticmethod
    def calculate_wacc(risk_free_rate: float,
                      market_return: float,
                      beta: float,
                      tax_rate: float,
                      debt_ratio: float,
                      cost_of_debt: float) -> float:
        """
        Calculate Weighted Average Cost of Capital (WACC)
        
        Args:
            risk_free_rate (float): Risk-free rate (e.g., 10-year treasury)
            market_return (float): Expected market return
            beta (float): Stock beta
            tax_rate (float): Corporate tax rate
            debt_ratio (float): Debt / (Debt + Equity)
            cost_of_debt (float): Cost of debt
            
        Returns:
            float: WACC as decimal (e.g., 0.10 for 10%)
        """
        cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
        equity_ratio = 1 - debt_ratio
        
        wacc = (equity_ratio * cost_of_equity) + (debt_ratio * cost_of_debt * (1 - tax_rate))
        return wacc
    
    @staticmethod
    def calculate_terminal_value(final_fcf: float,
                               terminal_growth_rate: float,
                               discount_rate: float,
                               years: int) -> float:
        """
        Calculate terminal value using Gordon Growth Model
        
        Args:
            final_fcf (float): Final year free cash flow
            terminal_growth_rate (float): Long-term growth rate
            discount_rate (float): Discount rate (WACC)
            years (int): Number of projection years
            
        Returns:
            float: Present value of terminal value
        """
        terminal_fcf = final_fcf * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
        present_value_terminal = terminal_value / ((1 + discount_rate) ** years)
        
        return present_value_terminal
    
    @staticmethod
    def calculate_dcf_value(fcf_projections: List[float],
                          terminal_value: float,
                          discount_rate: float,
                          shares_outstanding: int) -> Dict[str, float]:
        """
        Calculate DCF valuation
        
        Args:
            fcf_projections (List[float]): Projected free cash flows
            terminal_value (float): Terminal value
            discount_rate (float): Discount rate
            shares_outstanding (int): Number of shares outstanding
            
        Returns:
            Dict[str, float]: DCF valuation results
        """
        # Calculate present value of projected FCFs
        pv_fcfs = []
        for i, fcf in enumerate(fcf_projections, 1):
            pv = fcf / ((1 + discount_rate) ** i)
            pv_fcfs.append(pv)
        
        total_pv_fcf = sum(pv_fcfs)
        enterprise_value = total_pv_fcf + terminal_value
        
        # Calculate per-share value (simplified - assumes no net debt)
        value_per_share = enterprise_value / shares_outstanding
        
        return {
            'enterprise_value': enterprise_value,
            'pv_projection_period': total_pv_fcf,
            'pv_terminal_value': terminal_value,
            'value_per_share': value_per_share,
            'fcf_breakdown': pv_fcfs
        }
    
    @staticmethod
    def calculate_free_cash_flow(revenue: float,
                               operating_margin: float,
                               tax_rate: float,
                               capex_percent: float,
                               working_capital_change: float,
                               depreciation: float) -> float:
        """
        Calculate Free Cash Flow
        
        Args:
            revenue (float): Revenue
            operating_margin (float): Operating margin as decimal
            tax_rate (float): Tax rate as decimal
            capex_percent (float): CapEx as % of revenue
            working_capital_change (float): Change in working capital
            depreciation (float): Depreciation & Amortization
            
        Returns:
            float: Free Cash Flow
        """
        operating_income = revenue * operating_margin
        nopat = operating_income * (1 - tax_rate)  # Net Operating Profit After Tax
        capex = revenue * capex_percent
        
        fcf = nopat + depreciation - capex - working_capital_change
        return fcf
    
    @staticmethod
    def calculate_multiples(stock_data: Dict, 
                          financial_data: Dict) -> Dict[str, float]:
        """
        Calculate key valuation multiples
        
        Args:
            stock_data (Dict): Current stock market data
            financial_data (Dict): Financial statement data
            
        Returns:
            Dict[str, float]: Dictionary of calculated multiples
        """
        market_cap = stock_data.get('market_cap', 0)
        current_price = stock_data.get('current_price', 0)
        
        # Get financial metrics
        revenue = financial_data.get('revenue', 0)
        ebitda = financial_data.get('ebitda', 0)
        net_income = financial_data.get('net_income', 0)
        book_value = financial_data.get('book_value', 0)
        shares = financial_data.get('shares_outstanding', 1)
        enterprise_value = financial_data.get('enterprise_value', market_cap)
        
        # Calculate multiples
        multiples = {}
        
        # Price multiples
        multiples['pe_ratio'] = current_price / (net_income / shares) if net_income > 0 else 0
        multiples['price_to_book'] = current_price / (book_value / shares) if book_value > 0 else 0
        multiples['price_to_sales'] = market_cap / revenue if revenue > 0 else 0
        
        # Enterprise multiples
        multiples['ev_to_revenue'] = enterprise_value / revenue if revenue > 0 else 0
        multiples['ev_to_ebitda'] = enterprise_value / ebitda if ebitda > 0 else 0
        
        return multiples
    
    @staticmethod
    def calculate_growth_rates(historical_data: pd.DataFrame, 
                             metric: str,
                             periods: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """
        Calculate compound annual growth rates (CAGR)
        
        Args:
            historical_data (pd.DataFrame): Historical financial data
            metric (str): Column name to calculate growth for
            periods (List[int]): Periods to calculate CAGR for (in years)
            
        Returns:
            Dict[str, float]: CAGR for each period
        """
        if metric not in historical_data.columns or len(historical_data) < 2:
            return {f'{period}y_cagr': 0.0 for period in periods}
        
        growth_rates = {}
        current_value = historical_data[metric].iloc[-1]
        
        for period in periods:
            try:
                if len(historical_data) > period:
                    past_value = historical_data[metric].iloc[-(period+1)]
                    if past_value > 0:
                        cagr = ((current_value / past_value) ** (1/period)) - 1
                        growth_rates[f'{period}y_cagr'] = cagr
                    else:
                        growth_rates[f'{period}y_cagr'] = 0.0
                else:
                    growth_rates[f'{period}y_cagr'] = 0.0
            except (IndexError, ZeroDivisionError):
                growth_rates[f'{period}y_cagr'] = 0.0
        
        return growth_rates
    
    @staticmethod
    def calculate_financial_ratios(financial_data: Dict) -> Dict[str, float]:
        """
        Calculate comprehensive financial ratios
        
        Args:
            financial_data (Dict): Financial statement data
            
        Returns:
            Dict[str, float]: Calculated financial ratios
        """
        ratios = {}
        
        # Profitability Ratios
        revenue = financial_data.get('revenue', 0)
        net_income = financial_data.get('net_income', 0)
        operating_income = financial_data.get('operating_income', 0)
        total_assets = financial_data.get('total_assets', 0)
        shareholders_equity = financial_data.get('shareholders_equity', 0)
        
        ratios['net_profit_margin'] = net_income / revenue if revenue > 0 else 0
        ratios['operating_margin'] = operating_income / revenue if revenue > 0 else 0
        ratios['roa'] = net_income / total_assets if total_assets > 0 else 0
        ratios['roe'] = net_income / shareholders_equity if shareholders_equity > 0 else 0
        
        # Liquidity Ratios
        current_assets = financial_data.get('current_assets', 0)
        current_liabilities = financial_data.get('current_liabilities', 0)
        cash = financial_data.get('cash', 0)
        inventory = financial_data.get('inventory', 0)
        
        ratios['current_ratio'] = current_assets / current_liabilities if current_liabilities > 0 else 0
        ratios['quick_ratio'] = (current_assets - inventory) / current_liabilities if current_liabilities > 0 else 0
        ratios['cash_ratio'] = cash / current_liabilities if current_liabilities > 0 else 0
        
        # Leverage Ratios
        total_debt = financial_data.get('total_debt', 0)
        
        ratios['debt_to_equity'] = total_debt / shareholders_equity if shareholders_equity > 0 else 0
        ratios['debt_to_assets'] = total_debt / total_assets if total_assets > 0 else 0
        ratios['equity_ratio'] = shareholders_equity / total_assets if total_assets > 0 else 0
        
        # Efficiency Ratios
        ratios['asset_turnover'] = revenue / total_assets if total_assets > 0 else 0
        
        return ratios
    
    @staticmethod
    def monte_carlo_simulation(base_assumptions: Dict,
                             volatility_ranges: Dict,
                             num_simulations: int = 1000) -> Dict[str, List[float]]:
        """
        Perform Monte Carlo simulation for valuation
        
        Args:
            base_assumptions (Dict): Base case assumptions
            volatility_ranges (Dict): Standard deviations for each variable
            num_simulations (int): Number of simulations to run
            
        Returns:
            Dict[str, List[float]]: Simulation results
        """
        np.random.seed(42)  # For
