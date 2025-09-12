import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize_scalar

class FinancialCalculations:
    """
    Collection of financial calculation utilities for valuation and analysis.
    """
    
    @staticmethod
    def present_value(future_value: float, discount_rate: float, periods: int) -> float:
        """Calculate present value of a future cash flow."""
        return future_value / ((1 + discount_rate) ** periods)
    
    @staticmethod
    def future_value(present_value: float, growth_rate: float, periods: int) -> float:
        """Calculate future value with compound growth."""
        return present_value * ((1 + growth_rate) ** periods)
    
    @staticmethod
    def npv(cash_flows: List[float], discount_rate: float) -> float:
        """Calculate Net Present Value of cash flows."""
        npv = 0
        for i, cf in enumerate(cash_flows):
            npv += cf / ((1 + discount_rate) ** i)
        return npv
    
    @staticmethod
    def irr(cash_flows: List[float], guess: float = 0.1) -> float:
        """Calculate Internal Rate of Return using optimization."""
        
        def npv_func(rate):
            return abs(FinancialCalculations.npv(cash_flows, rate))
        
        try:
            result = minimize_scalar(npv_func, bounds=(0.001, 0.5), method='bounded')
            return result.x
        except:
            return np.nan
    
    @staticmethod
    def cagr(beginning_value: float, ending_value: float, periods: int) -> float:
        """Calculate Compound Annual Growth Rate."""
        if beginning_value <= 0 or ending_value <= 0 or periods <= 0:
            return 0.0
        return (ending_value / beginning_value) ** (1 / periods) - 1
    
    @staticmethod
    def gordon_growth_model(next_dividend: float, growth_rate: float, discount_rate: float) -> float:
        """Calculate value using Gordon Growth Model."""
        if discount_rate <= growth_rate:
            return np.inf
        return next_dividend / (discount_rate - growth_rate)
    
    @staticmethod
    def dividend_discount_model(dividends: List[float], growth_rates: List[float], 
                               discount_rate: float, terminal_growth: float) -> float:
        """Multi-stage dividend discount model."""
        
        pv_dividends = 0
        current_dividend = dividends[0] if dividends else 0
        
        # Present value of growth stage dividends
        for i, growth_rate in enumerate(growth_rates):
            current_dividend *= (1 + growth_rate)
            pv_dividends += FinancialCalculations.present_value(current_dividend, discount_rate, i + 1)
        
        # Terminal value
        terminal_dividend = current_dividend * (1 + terminal_growth)
        terminal_value = FinancialCalculations.gordon_growth_model(
            terminal_dividend, terminal_growth, discount_rate
        )
        pv_terminal = FinancialCalculations.present_value(terminal_value, discount_rate, len(growth_rates))
        
        return pv_dividends + pv_terminal
    
    @staticmethod
    def beta_calculation(stock_returns: List[float], market_returns: List[float]) -> float:
        """Calculate beta using regression analysis."""
        
        if len(stock_returns) != len(market_returns) or len(stock_returns) < 2:
            return 1.0
        
        try:
            # Convert to numpy arrays
            stock_returns = np.array(stock_returns)
            market_returns = np.array(market_returns)
            
            # Calculate covariance and variance
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            return beta
            
        except:
            return 1.0
    
    @staticmethod
    def sharpe_ratio(returns: List[float], risk_free_rate: float) -> float:
        """Calculate Sharpe ratio."""
        
        if not returns or len(returns) < 2:
            return 0.0
        
        excess_returns = [r - risk_free_rate for r in returns]
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns)
        
        if std_excess_return == 0:
            return 0.0
        
        return mean_excess_return / std_excess_return
    
    @staticmethod
    def value_at_risk(returns: List[float], confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)."""
        
        if not returns or len(returns) < 2:
            return 0.0
        
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def expected_shortfall(returns: List[float], confidence_level: float = 0.05) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        
        if not returns or len(returns) < 2:
            return 0.0
        
        var = FinancialCalculations.value_at_risk(returns, confidence_level)
        tail_returns = [r for r in returns if r <= var]
        
        if not tail_returns:
            return var
        
        return np.mean(tail_returns)
    
    @staticmethod
    def correlation_matrix(returns_dict: Dict[str, List[float]]) -> pd.DataFrame:
        """Calculate correlation matrix for multiple assets."""
        
        # Convert to DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # Calculate correlation matrix
        return returns_df.corr()
    
    @staticmethod
    def portfolio_metrics(weights: List[float], returns: List[List[float]], 
                         covariance_matrix: pd.DataFrame) -> Dict:
        """Calculate portfolio return, risk, and Sharpe ratio."""
        
        weights = np.array(weights)
        
        # Portfolio return
        mean_returns = [np.mean(asset_returns) for asset_returns in returns]
        portfolio_return = np.dot(weights, mean_returns)
        
        # Portfolio risk (standard deviation)
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        
        return {
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        }
    
    @staticmethod
    def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes call option pricing."""
        
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        return call_price
    
    @staticmethod
    def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes put option pricing."""
        
        if T <= 0 or sigma <= 0:
            return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        return put_price
    
    @staticmethod
    def implied_volatility(option_price: float, S: float, K: float, T: float, r: float, 
                          option_type: str = 'call') -> float:
        """Calculate implied volatility using numerical methods."""
        
        def objective_function(sigma):
            if option_type.lower() == 'call':
                theoretical_price = FinancialCalculations.black_scholes_call(S, K, T, r, sigma)
            else:
                theoretical_price = FinancialCalculations.black_scholes_put(S, K, T, r, sigma)
            return abs(theoretical_price - option_price)
        
        try:
            result = minimize_scalar(objective_function, bounds=(0.01, 5.0), method='bounded')
            return result.x
        except:
            return 0.0
    
    @staticmethod
    def monte_carlo_simulation(initial_value: float, drift: float, volatility: float,
                             time_horizon: float, num_simulations: int = 10000,
                             num_steps: int = 252) -> List[float]:
        """Monte Carlo simulation for asset price paths."""
        
        dt = time_horizon / num_steps
        final_values = []
        
        for _ in range(num_simulations):
            price = initial_value
            for _ in range(num_steps):
                random_shock = np.random.normal(0, 1)
                price *= np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * random_shock)
            final_values.append(price)
        
        return final_values
    
    @staticmethod
    def bond_price(face_value: float, coupon_rate: float, yield_rate: float, 
                   years_to_maturity: float, payments_per_year: int = 2) -> float:
        """Calculate bond price using present value of cash flows."""
        
        coupon_payment = (coupon_rate / payments_per_year) * face_value
        discount_rate = yield_rate / payments_per_year
        num_payments = int(years_to_maturity * payments_per_year)
        
        # Present value of coupon payments
        pv_coupons = 0
        for i in range(1, num_payments + 1):
            pv_coupons += coupon_payment / ((1 + discount_rate) ** i)
        
        # Present value of face value
        pv_face_value = face_value / ((1 + discount_rate) ** num_payments)
        
        return pv_coupons + pv_face_value
    
    @staticmethod
    def bond_duration(face_value: float, coupon_rate: float, yield_rate: float,
                     years_to_maturity: float, payments_per_year: int = 2) -> float:
        """Calculate Macaulay duration of a bond."""
        
        coupon_payment = (coupon_rate / payments_per_year) * face_value
        discount_rate = yield_rate / payments_per_year
        num_payments = int(years_to_maturity * payments_per_year)
        
        bond_price = FinancialCalculations.bond_price(face_value, coupon_rate, yield_rate, years_to_maturity, payments_per_year)
        
        weighted_time = 0
        for i in range(1, num_payments + 1):
            pv_payment = coupon_payment / ((1 + discount_rate) ** i)
            weighted_time += (i / payments_per_year) * pv_payment
        
        # Add face value payment
        pv_face_value = face_value / ((1 + discount_rate) ** num_payments)
        weighted_time += years_to_maturity * pv_face_value
        
        return weighted_time / bond_price if bond_price > 0 else 0
    
    @staticmethod
    def wacc(market_value_equity: float, market_value_debt: float, 
             cost_of_equity: float, cost_of_debt: float, tax_rate: float) -> float:
        """Calculate Weighted Average Cost of Capital (WACC)."""
        
        total_value = market_value_equity + market_value_debt
        
        if total_value == 0:
            return cost_of_equity
        
        weight_equity = market_value_equity / total_value
        weight_debt = market_value_debt / total_value
        
        wacc = weight_equity * cost_of_equity + weight_debt * cost_of_debt * (1 - tax_rate)
        return wacc
    
    @staticmethod
    def capm(risk_free_rate: float, beta: float, market_return: float) -> float:
        """Calculate expected return using Capital Asset Pricing Model (CAPM)."""
        return risk_free_rate + beta * (market_return - risk_free_rate)
    
    @staticmethod
    def operating_leverage(ebit_1: float, ebit_2: float, revenue_1: float, revenue_2: float) -> float:
        """Calculate degree of operating leverage."""
        
        ebit_change = (ebit_2 - ebit_1) / ebit_1 if ebit_1 != 0 else 0
        revenue_change = (revenue_2 - revenue_1) / revenue_1 if revenue_1 != 0 else 0
        
        if revenue_change == 0:
            return 0
        
        return ebit_change / revenue_change
    
    @staticmethod
    def financial_leverage(eps_1: float, eps_2: float, ebit_1: float, ebit_2: float) -> float:
        """Calculate degree of financial leverage."""
        
        eps_change = (eps_2 - eps_1) / eps_1 if eps_1 != 0 else 0
        ebit_change = (ebit_2 - ebit_1) / ebit_1 if ebit_1 != 0 else 0
        
        if ebit_change == 0:
            return 0
        
        return eps_change / ebit_change
    
    @staticmethod
    def calculate_financial_ratios(financial_data: Dict) -> Dict:
        """Calculate comprehensive financial ratios from financial data."""
        
        ratios = {}
        
        # Liquidity ratios
        current_assets = financial_data.get('current_assets', 0)
        current_liabilities = financial_data.get('current_liabilities', 0)
        cash = financial_data.get('cash_and_equivalents', 0)
        inventory = financial_data.get('inventory', 0)
        
        if current_liabilities > 0:
            ratios['current_ratio'] = current_assets / current_liabilities
            ratios['quick_ratio'] = (current_assets - inventory) / current_liabilities
            ratios['cash_ratio'] = cash / current_liabilities
        
        # Profitability ratios
        revenue = financial_data.get('revenue', 0)
        net_income = financial_data.get('net_income', 0)
        total_assets = financial_data.get('total_assets', 0)
        shareholders_equity = financial_data.get('shareholders_equity', 0)
        
        if revenue > 0:
            ratios['net_profit_margin'] = net_income / revenue
        
        if total_assets > 0:
            ratios['return_on_assets'] = net_income / total_assets
            ratios['asset_turnover'] = revenue / total_assets
        
        if shareholders_equity > 0:
            ratios['return_on_equity'] = net_income / shareholders_equity
        
        # Leverage ratios
        total_debt = financial_data.get('total_debt', 0)
        total_liabilities = financial_data.get('total_liabilities', 0)
        
        if total_assets > 0:
            ratios['debt_to_assets'] = total_debt / total_assets
        
        if shareholders_equity > 0:
            ratios['debt_to_equity'] = total_debt / shareholders_equity
            ratios['equity_multiplier'] = total_assets / shareholders_equity
        
        # Efficiency ratios
        accounts_receivable = financial_data.get('accounts_receivable', 0)
        accounts_payable = financial_data.get('accounts_payable', 0)
        
        if accounts_receivable > 0 and revenue > 0:
            ratios['receivables_turnover'] = revenue / accounts_receivable
            ratios['days_sales_outstanding'] = 365 / ratios['receivables_turnover']
        
        if accounts_payable > 0 and revenue > 0:
            ratios['payables_turnover'] = revenue / accounts_payable
            ratios['days_payable_outstanding'] = 365 / ratios['payables_turnover']
        
        if inventory > 0 and revenue > 0:
            ratios['inventory_turnover'] = revenue / inventory
            ratios['days_inventory_outstanding'] = 365 / ratios['inventory_turnover']
        
        return ratios
    
    @staticmethod
    def z_score_altman(working_capital: float, retained_earnings: float, ebit: float,
                      market_value_equity: float, sales: float, total_assets: float,
                      total_liabilities: float) -> float:
        """Calculate Altman Z-Score for bankruptcy prediction."""
        
        if total_assets == 0:
            return 0
        
        # Altman Z-Score components
        z1 = 1.2 * (working_capital / total_assets)
        z2 = 1.4 * (retained_earnings / total_assets)
        z3 = 3.3 * (ebit / total_assets)
        z4 = 0.6 * (market_value_equity / total_liabilities)
        z5 = 1.0 * (sales / total_assets)
        
        z_score = z1 + z2 + z3 + z4 + z5
        return z_score
    
    @staticmethod
    def piotroski_f_score(financial_data: Dict) -> int:
        """Calculate Piotroski F-Score for fundamental strength."""
        
        score = 0
        
        # Profitability (4 points)
        if financial_data.get('net_income', 0) > 0:
            score += 1  # Positive net income
        
        if financial_data.get('operating_cash_flow', 0) > 0:
            score += 1  # Positive operating cash flow
        
        if financial_data.get('return_on_assets_current', 0) > financial_data.get('return_on_assets_previous', 0):
            score += 1  # Increasing ROA
        
        if financial_data.get('operating_cash_flow', 0) > financial_data.get('net_income', 0):
            score += 1  # Operating CF > Net Income
        
        # Leverage, Liquidity, and Source of Funds (3 points)
        if financial_data.get('debt_to_assets_current', 1) < financial_data.get('debt_to_assets_previous', 1):
            score += 1  # Decreasing debt ratio
        
        if financial_data.get('current_ratio_current', 0) > financial_data.get('current_ratio_previous', 0):
            score += 1  # Increasing current ratio
        
        if financial_data.get('shares_outstanding_current', 1) <= financial_data.get('shares_outstanding_previous', 1):
            score += 1  # No dilution
        
        # Operating Efficiency (2 points)
        if financial_data.get('gross_margin_current', 0) > financial_data.get('gross_margin_previous', 0):
            score += 1  # Increasing gross margin
        
        if financial_data.get('asset_turnover_current', 0) > financial_data.get('asset_turnover_previous', 0):
            score += 1  # Increasing asset turnover
        
        return score

# Utility functions for common calculations
def calculate_percentile_rank(value: float, data_series: List[float]) -> float:
    """Calculate percentile rank of a value in a dataset."""
    if not data_series:
        return 50.0
    
    sorted_data = sorted(data_series)
    n = len(sorted_data)
    
    # Find position of value
    for i, data_point in enumerate(sorted_data):
        if value <= data_point:
            return (i / n) * 100
    
    return 100.0

def normalize_metrics(data_dict: Dict[str, float], method: str = 'z_score') -> Dict[str, float]:
    """Normalize financial metrics using specified method."""
    
    if not data_dict:
        return {}
    
    values = list(data_dict.values())
    
    if method == 'z_score':
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val == 0:
            return {k: 0 for k in data_dict.keys()}
        return {k: (v - mean_val) / std_val for k, v in data_dict.items()}
    
    elif method == 'min_max':
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return {k: 0 for k in data_dict.keys()}
        return {k: (v - min_val) / (max_val - min_val) for k, v in data_dict.items()}
    
    else:
        return data_dict

def calculate_composite_score(metrics: Dict[str, float], weights: Dict[str, float] = None) -> float:
    """Calculate weighted composite score from multiple metrics."""
    
    if not metrics:
        return 0.0
    
    if weights is None:
        weights = {k: 1.0 for k in metrics.keys()}
    
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    
    weighted_score = sum(metrics.get(k, 0) * weights.get(k, 0) for k in metrics.keys())
    return weighted_score / total_weight
