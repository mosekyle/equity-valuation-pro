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
        np.random.seed(42)  # For reproducibility
        
        results = {
            'valuations': [],
            'revenue_growth': [],
            'operating_margin': [],
            'discount_rate': [],
            'terminal_growth': []
        }
        
        for _ in range(num_simulations):
            # Generate random variations for each assumption
            simulation_assumptions = {}
            
            for key, base_value in base_assumptions.items():
                if key in volatility_ranges:
                    std_dev = volatility_ranges[key]
                    # Use normal distribution with base value as mean
                    simulated_value = np.random.normal(base_value, std_dev)
                    
                    # Apply reasonable bounds
                    if key == 'terminal_growth':
                        simulated_value = max(0.005, min(0.05, simulated_value))  # 0.5% to 5%
                    elif key == 'discount_rate':
                        simulated_value = max(0.05, min(0.20, simulated_value))   # 5% to 20%
                    elif key == 'operating_margin':
                        simulated_value = max(0.01, min(0.50, simulated_value))   # 1% to 50%
                    elif key == 'revenue_growth':
                        simulated_value = max(-0.20, min(0.50, simulated_value))  # -20% to 50%
                    
                    simulation_assumptions[key] = simulated_value
                else:
                    simulation_assumptions[key] = base_value
            
            # Store simulation parameters
            results['revenue_growth'].append(simulation_assumptions.get('revenue_growth', 0))
            results['operating_margin'].append(simulation_assumptions.get('operating_margin', 0))
            results['discount_rate'].append(simulation_assumptions.get('discount_rate', 0))
            results['terminal_growth'].append(simulation_assumptions.get('terminal_growth', 0))
            
            # Calculate valuation for this simulation (simplified DCF)
            # This would typically call your full DCF model with these assumptions
            try:
                # Simplified valuation calculation
                base_revenue = simulation_assumptions.get('base_revenue', 100000)
                growth = simulation_assumptions.get('revenue_growth', 0.05)
                margin = simulation_assumptions.get('operating_margin', 0.15)
                discount_rate = simulation_assumptions.get('discount_rate', 0.10)
                terminal_growth = simulation_assumptions.get('terminal_growth', 0.025)
                
                # 5-year projection
                revenues = []
                for year in range(5):
                    revenue = base_revenue * ((1 + growth) ** year)
                    revenues.append(revenue)
                
                # Calculate FCFs (simplified)
                fcfs = [rev * margin * 0.7 for rev in revenues]  # Simplified FCF calculation
                
                # Terminal value
                terminal_fcf = fcfs[-1] * (1 + terminal_growth)
                terminal_value = terminal_fcf / (discount_rate - terminal_growth)
                pv_terminal = terminal_value / ((1 + discount_rate) ** 5)
                
                # Present value of FCFs
                pv_fcfs = sum([fcf / ((1 + discount_rate) ** (i+1)) for i, fcf in enumerate(fcfs)])
                
                enterprise_value = pv_fcfs + pv_terminal
                valuation_per_share = enterprise_value / simulation_assumptions.get('shares_outstanding', 1000)
                
                results['valuations'].append(valuation_per_share)
                
            except (ZeroDivisionError, ValueError):
                # Handle edge cases where calculation fails
                results['valuations'].append(0)
        
        return results
    
    @staticmethod
    def calculate_sensitivity_analysis(base_valuation_func,
                                     variable_ranges: Dict[str, List[float]],
                                     base_assumptions: Dict) -> pd.DataFrame:
        """
        Perform sensitivity analysis on key variables
        
        Args:
            base_valuation_func: Function that calculates valuation
            variable_ranges (Dict): Ranges for each variable to test
            base_assumptions (Dict): Base case assumptions
            
        Returns:
            pd.DataFrame: Sensitivity analysis results
        """
        sensitivity_results = []
        
        for var_name, var_range in variable_ranges.items():
            for var_value in var_range:
                # Create modified assumptions
                modified_assumptions = base_assumptions.copy()
                modified_assumptions[var_name] = var_value
                
                # Calculate valuation with modified assumption
                try:
                    valuation = base_valuation_func(modified_assumptions)
                    base_valuation = base_valuation_func(base_assumptions)
                    
                    change_percent = ((valuation - base_valuation) / base_valuation) * 100
                    
                    sensitivity_results.append({
                        'variable': var_name,
                        'value': var_value,
                        'valuation': valuation,
                        'change_percent': change_percent
                    })
                except:
                    continue
        
        return pd.DataFrame(sensitivity_results)
    
    @staticmethod
    def calculate_beta(stock_returns: pd.Series, 
                      market_returns: pd.Series,
                      period_days: int = 252) -> float:
        """
        Calculate stock beta vs market
        
        Args:
            stock_returns (pd.Series): Daily stock returns
            market_returns (pd.Series): Daily market returns
            period_days (int): Period for calculation (default: 1 year = 252 days)
            
        Returns:
            float: Beta coefficient
        """
        # Align the series and take last period_days
        aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
        
        if len(aligned_data) < 50:  # Minimum data requirement
            return 1.0  # Default beta
        
        # Take last period_days of data
        recent_data = aligned_data.tail(period_days)
        stock_rets = recent_data.iloc[:, 0]
        market_rets = recent_data.iloc[:, 1]
        
        # Calculate beta using covariance method
        covariance = np.cov(stock_rets, market_rets)[0][1]
        market_variance = np.var(market_rets)
        
        if market_variance == 0:
            return 1.0
        
        beta = covariance / market_variance
        return beta
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, 
                             risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns (pd.Series): Return series
            risk_free_rate (float): Risk-free rate (annualized)
            
        Returns:
            float: Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
        
        # Convert to daily risk-free rate
        daily_rf_rate = risk_free_rate / 252
        
        excess_returns = returns - daily_rf_rate
        
        if excess_returns.std() == 0:
            return 0.0
        
        # Annualize
        sharpe = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
        return sharpe
    
    @staticmethod
    def calculate_var(returns: pd.Series, 
                     confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns (pd.Series): Return series
            confidence_level (float): Confidence level (default: 5% for 95% VaR)
            
        Returns:
            float: VaR as a positive number
        """
        if len(returns) == 0:
            return 0.0
        
        var = np.percentile(returns, confidence_level * 100)
        return abs(var)


class ComparableAnalysis:
    """
    Handles comparable company analysis calculations
    """
    
    @staticmethod
    def calculate_peer_multiples(peer_data: Dict[str, Dict]) -> pd.DataFrame:
        """
        Calculate multiples for peer companies
        
        Args:
            peer_data (Dict): Dictionary of peer company data
            
        Returns:
            pd.DataFrame: Peer multiples comparison
        """
        multiples_data = []
        
        for symbol, data in peer_data.items():
            try:
                financials = data.get('financial_metrics', {})
                basic_info = data.get('basic_info', {})
                
                multiples_data.append({
                    'symbol': symbol,
                    'company_name': basic_info.get('company_name', symbol),
                    'market_cap': basic_info.get('market_cap', 0),
                    'pe_ratio': financials.get('pe_ratio', 0),
                    'forward_pe': financials.get('forward_pe', 0),
                    'ev_to_ebitda': financials.get('ev_to_ebitda', 0),
                    'price_to_book': financials.get('price_to_book', 0),
                    'price_to_sales': financials.get('price_to_sales', 0),
                    'debt_to_equity': financials.get('debt_to_equity', 0),
                    'roe': financials.get('return_on_equity', 0),
                    'profit_margin': financials.get('profit_margin', 0)
                })
            except Exception as e:
                continue
        
        df = pd.DataFrame(multiples_data)
        
        # Add statistical measures
        if not df.empty:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            summary_stats = df[numeric_columns].describe()
            
            return df, summary_stats
        
        return df, pd.DataFrame()
    
    @staticmethod
    def calculate_implied_valuation(target_multiples: Dict,
                                  peer_stats: pd.DataFrame,
                                  target_financials: Dict) -> Dict[str, float]:
        """
        Calculate implied valuation based on peer multiples
        
        Args:
            target_multiples (Dict): Target company current multiples
            peer_stats (pd.DataFrame): Peer companies statistics
            target_financials (Dict): Target company financials
            
        Returns:
            Dict[str, float]: Implied valuations using different multiples
        """
        implied_values = {}
        
        # Revenue-based valuation
        if 'price_to_sales' in peer_stats.index:
            median_ps = peer_stats.loc['50%', 'price_to_sales']
            revenue = target_financials.get('revenue', 0)
            if revenue > 0:
                implied_values['ps_valuation'] = median_ps * revenue
        
        # Earnings-based valuation
        if 'pe_ratio' in peer_stats.index:
            median_pe = peer_stats.loc['50%', 'pe_ratio']
            net_income = target_financials.get('net_income', 0)
            if net_income > 0:
                implied_values['pe_valuation'] = median_pe * net_income
        
        # Book value-based valuation
        if 'price_to_book' in peer_stats.index:
            median_pb = peer_stats.loc['50%', 'price_to_book']
            book_value = target_financials.get('book_value', 0)
            if book_value > 0:
                implied_values['pb_valuation'] = median_pb * book_value
        
        return implied_values
