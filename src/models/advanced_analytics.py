"""
Advanced Analytics Module for Equity Valuation Dashboard
Sophisticated features for institutional-grade analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats, optimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..data.market_data import MarketDataProvider
from ..utils.calculations import FinancialCalculations


class LeveragedBuyoutModel:
    """
    LBO Model for Private Equity Analysis
    """
    
    def __init__(self, target_symbol: str):
        self.target_symbol = target_symbol
        self.market_provider = MarketDataProvider()
        
    def calculate_lbo_returns(self,
                            purchase_price: float,
                            debt_financing: float,
                            equity_contribution: float,
                            exit_multiple: float,
                            hold_period: int = 5) -> Dict[str, float]:
        """
        Calculate LBO returns and IRR
        
        Args:
            purchase_price (float): Total acquisition price
            debt_financing (float): Debt amount
            equity_contribution (float): Equity invested
            exit_multiple (float): Exit EV/EBITDA multiple
            hold_period (int): Investment period in years
            
        Returns:
            Dict[str, float]: LBO analysis results
        """
        # Get company data
        company_info = self.market_provider.get_company_info(self.target_symbol)
        current_ebitda = self._extract_ebitda(company_info)
        
        if current_ebitda <= 0:
            raise ValueError("Invalid EBITDA for LBO analysis")
        
        # Project EBITDA growth (conservative 5% annually)
        ebitda_growth = 0.05
        exit_ebitda = current_ebitda * ((1 + ebitda_growth) ** hold_period)
        
        # Calculate exit enterprise value
        exit_enterprise_value = exit_ebitda * exit_multiple
        
        # Assume debt paydown (50% of original debt)
        remaining_debt = debt_financing * 0.5
        exit_equity_value = exit_enterprise_value - remaining_debt
        
        # Calculate returns
        total_return = exit_equity_value / equity_contribution
        irr = (total_return ** (1/hold_period)) - 1
        
        return {
            'purchase_price': purchase_price,
            'debt_financing': debt_financing,
            'equity_contribution': equity_contribution,
            'current_ebitda': current_ebitda,
            'exit_ebitda': exit_ebitda,
            'exit_enterprise_value': exit_enterprise_value,
            'exit_equity_value': exit_equity_value,
            'total_return_multiple': total_return,
            'irr': irr,
            'hold_period': hold_period
        }
    
    def _extract_ebitda(self, company_info: Dict) -> float:
        """Extract EBITDA from company info"""
        financial_statements = company_info.get('financial_statements', {})
        income_statement = financial_statements.get('income_statement', {})
        
        # Try to find EBITDA
        ebitda_keys = ['EBITDA', 'Normalized EBITDA']
        
        for key in ebitda_keys:
            if key in income_statement:
                ebitda_data = income_statement[key]
                if isinstance(ebitda_data, dict) and len(ebitda_data) > 0:
                    latest_date = max(ebitda_data.keys())
                    return abs(ebitda_data[latest_date])
        
        # Calculate EBITDA if not available
        operating_income = self._extract_operating_income(company_info)
        depreciation = self._extract_depreciation(company_info)
        
        return operating_income + depreciation
    
    def _extract_operating_income(self, company_info: Dict) -> float:
        """Extract operating income"""
        financial_statements = company_info.get('financial_statements', {})
        income_statement = financial_statements.get('income_statement', {})
        
        operating_keys = ['Operating Income', 'Income From Operations']
        
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
        
        depreciation_keys = ['Depreciation And Amortization', 'Depreciation']
        
        for key in depreciation_keys:
            if key in cashflow:
                depreciation_data = cashflow[key]
                if isinstance(depreciation_data, dict) and len(depreciation_data) > 0:
                    latest_date = max(depreciation_data.keys())
                    return abs(depreciation_data[latest_date])
        
        return 0


class RealOptionsValuation:
    """
    Real Options Valuation using Black-Scholes framework
    """
    
    @staticmethod
    def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Black-Scholes call option pricing
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration
            r (float): Risk-free rate
            sigma (float): Volatility
            
        Returns:
            float: Call option value
        """
        from scipy.stats import norm
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        call_value = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return call_value
    
    def calculate_expansion_option(self,
                                 project_npv: float,
                                 expansion_cost: float,
                                 volatility: float,
                                 time_to_expand: float,
                                 risk_free_rate: float = 0.03) -> Dict[str, float]:
        """
        Calculate expansion option value
        
        Args:
            project_npv (float): Current project NPV
            expansion_cost (float): Cost to expand
            volatility (float): Project volatility
            time_to_expand (float): Time window for expansion
            risk_free_rate (float): Risk-free rate
            
        Returns:
            Dict[str, float]: Option valuation results
        """
        option_value = self.black_scholes_call(
            S=project_npv,
            K=expansion_cost,
            T=time_to_expand,
            r=risk_free_rate,
            sigma=volatility
        )
        
        return {
            'project_npv': project_npv,
            'expansion_cost': expansion_cost,
            'option_value': option_value,
            'total_project_value': project_npv + option_value,
            'option_premium': (option_value / project_npv) * 100 if project_npv > 0 else 0
        }


class MachineLearningValuation:
    """
    ML-based valuation models and predictions
    """
    
    def __init__(self):
        self.market_provider = MarketDataProvider()
        self.model = None
        self.feature_importance = None
        
    def build_valuation_model(self, symbols: List[str]) -> Dict[str, float]:
        """
        Build ML model to predict stock valuations
        
        Args:
            symbols (List[str]): Training symbols
            
        Returns:
            Dict[str, float]: Model performance metrics
        """
        # Collect training data
        training_data = []
        
        for symbol in symbols:
            try:
                company_info = self.market_provider.get_company_info(symbol)
                stock_data = self.market_provider.get_stock_data(symbol, period='1y')
                
                features = self._extract_features(company_info, stock_data)
                if features:
                    training_data.append(features)
                    
            except Exception as e:
                continue
        
        if len(training_data) < 10:
            raise ValueError("Insufficient data for ML model training")
        
        # Prepare features and target
        df = pd.DataFrame(training_data)
        
        # Define features and target
        feature_columns = [
            'pe_ratio', 'ev_ebitda', 'price_to_sales', 'debt_to_equity',
            'roe', 'profit_margin', 'revenue_growth', 'market_cap_log'
        ]
        
        # Clean data
        df = df.dropna(subset=feature_columns + ['current_price'])
        
        X = df[feature_columns]
        y = df['current_price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        self.feature_importance = dict(zip(feature_columns, self.model.feature_importances_))
        
        return {
            'model_trained': True,
            'training_samples': len(training_data),
            'mse': mse,
            'r2_score': r2,
            'features_used': feature_columns
        }
    
    def predict_valuation(self, target_symbol: str) -> Dict[str, float]:
        """
        Predict stock valuation using trained ML model
        
        Args:
            target_symbol (str): Symbol to predict
            
        Returns:
            Dict[str, float]: Prediction results
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Get target company data
        company_info = self.market_provider.get_company_info(target_symbol)
        stock_data = self.market_provider.get_stock_data(target_symbol, period='1y')
        
        features = self._extract_features(company_info, stock_data)
        
        if not features:
            raise ValueError("Could not extract features for prediction")
        
        # Prepare features
        feature_columns = [
            'pe_ratio', 'ev_ebitda', 'price_to_sales', 'debt_to_equity',
            'roe', 'profit_margin', 'revenue_growth', 'market_cap_log'
        ]
        
        feature_vector = [features.get(col, 0) for col in feature_columns]
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        # Make prediction
        predicted_price = self.model.predict(feature_array)[0]
        current_price = features['current_price']
        
        # Calculate confidence intervals (simplified)
        predictions = []
        for estimator in self.model.estimators_:
            pred = estimator.predict(feature_array)[0]
            predictions.append(pred)
        
        prediction_std = np.std(predictions)
        confidence_low = predicted_price - 1.96 * prediction_std
        confidence_high = predicted_price + 1.96 * prediction_std
        
        return {
            'symbol': target_symbol,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'upside_downside': ((predicted_price - current_price) / current_price) * 100,
            'confidence_interval_low': confidence_low,
            'confidence_interval_high': confidence_high,
            'prediction_std': prediction_std
        }
    
    def _extract_features(self, company_info: Dict, stock_data: pd.DataFrame) -> Optional[Dict]:
        """Extract features for ML model"""
        try:
            basic_info = company_info.get('basic_info', {})
            financial_metrics = company_info.get('financial_metrics', {})
            current_metrics = company_info.get('current_metrics', {})
            growth_metrics = company_info.get('growth_metrics', {})
            
            # Calculate additional features
            returns = FinancialCalculations.calculate_returns(stock_data)
            
            features = {
                'symbol': basic_info.get('symbol', ''),
                'current_price': current_metrics.get('current_price', 0),
                'market_cap': basic_info.get('market_cap', 0),
                'market_cap_log': np.log(basic_info.get('market_cap', 1)),
                'pe_ratio': financial_metrics.get('pe_ratio', 0),
                'ev_ebitda': financial_metrics.get('ev_to_ebitda', 0),
                'price_to_sales': financial_metrics.get('price_to_sales', 0),
                'debt_to_equity': financial_metrics.get('debt_to_equity', 0),
                'roe': financial_metrics.get('return_on_equity', 0),
                'profit_margin': financial_metrics.get('profit_margin', 0),
                'revenue_growth': growth_metrics.get('revenue_growth', 0),
                'sector': basic_info.get('sector', ''),
                'volatility': stock_data['close'].pct_change().std() * np.sqrt(252) if not stock_data.empty else 0
            }
            
            # Clean infinite values
            for key, value in features.items():
                if isinstance(value, (int, float)) and (np.isinf(value) or np.isnan(value)):
                    features[key] = 0
            
            return features
            
        except Exception as e:
            return None


class SentimentAnalysis:
    """
    News sentiment analysis for investment decisions
    """
    
    def __init__(self):
        self.market_provider = MarketDataProvider()
    
    def analyze_news_sentiment(self, symbol: str, days_back: int = 30) -> Dict[str, float]:
        """
        Analyze news sentiment for a stock
        
        Args:
            symbol (str): Stock symbol
            days_back (int): Days of news to analyze
            
        Returns:
            Dict[str, float]: Sentiment analysis results
        """
        try:
            # This is a simplified implementation
            # In production, you would integrate with news APIs and NLP libraries
            
            # Mock sentiment scores for demonstration
            # Positive sentiment: 0.6-1.0, Neutral: 0.4-0.6, Negative: 0.0-0.4
            
            # Generate realistic but random sentiment based on stock performance
            stock_data = self.market_provider.get_stock_data(symbol, period='1mo')
            
            if stock_data.empty:
                return {'sentiment_score': 0.5, 'confidence': 0.0, 'articles_analyzed': 0}
            
            # Calculate recent performance
            recent_return = ((stock_data['close'].iloc[-1] - stock_data['close'].iloc[-10]) / 
                           stock_data['close'].iloc[-10]) if len(stock_data) >= 10 else 0
            
            # Base sentiment on performance (simplified)
            base_sentiment = 0.5 + (recent_return * 2)  # Scale return to sentiment
            base_sentiment = max(0.1, min(0.9, base_sentiment))  # Bound between 0.1-0.9
            
            # Add some noise for realism
            sentiment_score = base_sentiment + np.random.normal(0, 0.1)
            sentiment_score = max(0.0, min(1.0, sentiment_score))
            
            # Mock other metrics
            confidence = np.random.uniform(0.7, 0.95)
            articles_analyzed = np.random.randint(10, 50)
            
            return {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'sentiment_label': self._get_sentiment_label(sentiment_score),
                'confidence': confidence,
                'articles_analyzed': articles_analyzed,
                'period_days': days_back,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'sentiment_score': 0.5,
                'sentiment_label': 'Neutral',
                'confidence': 0.0,
                'articles_analyzed': 0,
                'error': str(e)
            }
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score >= 0.7:
            return 'Very Positive'
        elif score >= 0.6:
            return 'Positive'
        elif score >= 0.4:
            return 'Neutral'
        elif score >= 0.3:
            return 'Negative'
        else:
            return 'Very Negative'


class RiskAnalytics:
    """
    Advanced risk analytics and stress testing
    """
    
    def __init__(self):
        self.market_provider = MarketDataProvider()
    
    def calculate_var_cvar(self, 
                          symbols: List[str], 
                          weights: List[float], 
                          confidence_level: float = 0.05,
                          time_horizon: int = 252) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR for portfolio
        
        Args:
            symbols (List[str]): Portfolio symbols
            weights (List[float]): Portfolio weights
            confidence_level (float): Confidence level (5% for 95% VaR)
            time_horizon (int): Time horizon in days
            
        Returns:
            Dict[str, float]: Risk metrics
        """
        # Get historical data for all symbols
        returns_data = {}
        
        for symbol in symbols:
            try:
                stock_data = self.market_provider.get_stock_data(symbol, period='2y')
                if not stock_data.empty:
                    returns = stock_data['close'].pct_change().dropna()
                    returns_data[symbol] = returns
            except:
                continue
        
        if len(returns_data) == 0:
            return {'var_95': 0, 'cvar_95': 0, 'portfolio_volatility': 0}
        
        # Align returns data
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if returns_df.empty:
            return {'var_95': 0, 'cvar_95': 0, 'portfolio_volatility': 0}
        
        # Calculate portfolio returns
        weights_array = np.array(weights[:len(returns_df.columns)])
        weights_array = weights_array / weights_array.sum()  # Normalize weights
        
        portfolio_returns = (returns_df * weights_array).sum(axis=1)
        
        # Calculate VaR
        var_95 = np.percentile(portfolio_returns, confidence_level * 100)
        
        # Calculate Conditional VaR (Expected Shortfall)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Portfolio volatility
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Annualize VaR and CVaR
        var_95_annual = var_95 * np.sqrt(252)
        cvar_95_annual = cvar_95 * np.sqrt(252)
        
        return {
            'var_95': abs(var_95_annual),
            'cvar_95': abs(cvar_95_annual),
            'portfolio_volatility': portfolio_volatility,
            'max_daily_loss': abs(portfolio_returns.min()),
            'best_daily_gain': portfolio_returns.max(),
            'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
        }
    
    def stress_test_scenarios(self, 
                            symbol: str, 
                            scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Perform stress testing under various market scenarios
        
        Args:
            symbol (str): Stock symbol
            scenarios (Dict): Stress test scenarios
            
        Returns:
            Dict[str, Dict[str, float]]: Stress test results
        """
        # Get current stock data
        company_info = self.market_provider.get_company_info(symbol)
        current_price = company_info.get('current_metrics', {}).get('current_price', 0)
        
        stress_results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            # Apply scenario shocks
            market_shock = scenario_params.get('market_decline', 0)  # e.g., -0.3 for 30% decline
            sector_shock = scenario_params.get('sector_decline', 0)
            interest_rate_shock = scenario_params.get('interest_rate_change', 0)
            
            # Calculate stressed price (simplified model)
            # In practice, you'd use more sophisticated factor models
            stressed_price = current_price * (1 + market_shock) * (1 + sector_shock)
            
            # Account for interest rate sensitivity (duration-like effect)
            if interest_rate_shock != 0:
                # Higher rates typically hurt growth stocks more
                pe_ratio = company_info.get('financial_metrics', {}).get('pe_ratio', 20)
                rate_sensitivity = min(pe_ratio / 20, 2.0)  # Cap sensitivity
                rate_impact = interest_rate_shock * rate_sensitivity * -0.5
                stressed_price *= (1 + rate_impact)
            
            # Ensure price doesn't go negative
            stressed_price = max(stressed_price, current_price * 0.1)
            
            price_change = ((stressed_price - current_price) / current_price) * 100
            
            stress_results[scenario_name] = {
                'current_price': current_price,
                'stressed_price': stressed_price,
                'price_change_percent': price_change,
                'scenario_parameters': scenario_params
            }
        
        return stress_results


# Predefined stress test scenarios
DEFAULT_STRESS_SCENARIOS = {
    'Financial Crisis': {
        'market_decline': -0.40,
        'sector_decline': -0.20,
        'interest_rate_change': 0.02,
        'description': '2008-style financial crisis'
    },
    'COVID-19 Style Shock': {
        'market_decline': -0.35,
        'sector_decline': -0.10,
        'interest_rate_change': -0.015,
        'description': 'Pandemic-style market disruption'
    },
    'Inflation Spike': {
        'market_decline': -0.15,
        'sector_decline': -0.05,
        'interest_rate_change': 0.03,
        'description': '1970s-style inflation surge'
    },
    'Tech Bubble Burst': {
        'market_decline': -0.25,
        'sector_decline': -0.45,
        'interest_rate_change': 0.01,
        'description': 'Dot-com bubble collapse'
    },
    'Mild Recession': {
        'market_decline': -0.20,
        'sector_decline': -0.10,
        'interest_rate_change': -0.01,
        'description': 'Standard recession scenario'
    }
}
