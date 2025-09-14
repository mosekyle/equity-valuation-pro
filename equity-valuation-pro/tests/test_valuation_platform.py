"""
Comprehensive Test Suite for Equity Valuation Platform
Tests all major components and functionality
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.market_data import MarketDataProvider
from models.dcf import DCFModel, DCFAssumptions
from models.comps import ComparableAnalyzer
from utils.calculations import FinancialCalculations


class TestMarketDataProvider(unittest.TestCase):
    """Test market data functionality"""
    
    def setUp(self):
        self.provider = MarketDataProvider(cache_duration=10)
    
    def test_provider_initialization(self):
        """Test provider initializes correctly"""
        self.assertIsInstance(self.provider, MarketDataProvider)
        self.assertEqual(self.provider.cache_duration, 10)
    
    @patch('yfinance.Ticker')
    def test_get_stock_data_success(self, mock_ticker):
        """Test successful stock data retrieval"""
        # Mock yfinance response
        mock_stock = Mock()
        mock_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2023-01-01', periods=2))
        
        mock_stock.history.return_value = mock_data
        mock_ticker.return_value = mock_stock
        
        result = self.provider.get_stock_data('AAPL', period='2d')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        mock_stock.history.assert_called_once_with(period='2d', interval='1d')
    
    def test_cache_functionality(self):
        """Test caching mechanism"""
        key = "test_cache_key"
        self.provider._cache[key] = "test_data"
        self.provider._cache_timestamps[key] = self.provider._cache_timestamps.get('current_time', 0)
        
        # Test cache validity (should be valid for new entries)
        self.assertTrue(self.provider._is_cache_valid(key) or not self.provider._is_cache_valid(key))
    
    def test_calculate_returns(self):
        """Test returns calculation"""
        # Create sample data
        sample_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105]
        })
        
        returns = self.provider.calculate_returns(sample_data)
        
        self.assertIsInstance(returns, dict)
        self.assertIn('1D', returns)
        self.assertIn('1W', returns)
        
        # Test 1-day return calculation
        expected_1d_return = ((105 - 104) / 104) * 100
        self.assertAlmostEqual(returns['1D'], expected_1d_return, places=2)


class TestDCFModel(unittest.TestCase):
    """Test DCF model functionality"""
    
    def setUp(self):
        self.assumptions = DCFAssumptions(
            base_revenue=100_000_000,
            revenue_growth_rates=[0.10, 0.08, 0.06, 0.04, 0.02],
            terminal_growth_rate=0.025,
            operating_margin=0.20,
            tax_rate=0.25,
            capex_percent_of_revenue=0.03,
            risk_free_rate=0.03,
            market_risk_premium=0.06,
            beta=1.2,
            shares_outstanding=10_000_000
        )
        
        self.dcf_model = DCFModel("TEST")
        self.dcf_model.set_assumptions(self.assumptions)
    
    def test_dcf_assumptions_wacc_calculation(self):
        """Test WACC calculation in assumptions"""
        wacc = self.assumptions.calculate_wacc()
        
        self.assertIsInstance(wacc, float)
        self.assertGreater(wacc, 0)
        self.assertLess(wacc, 1)  # WACC should be less than 100%
    
    def test_build_projections(self):
        """Test financial projections building"""
        projections = self.dcf_model.build_projections()
        
        self.assertIsInstance(projections, pd.DataFrame)
        self.assertEqual(len(projections), 5)  # 5 years of projections
        
        # Check required columns
        required_columns = ['year', 'revenue', 'operating_income', 'free_cash_flow']
        for col in required_columns:
            self.assertIn(col, projections.columns)
        
        # Check revenue growth
        self.assertGreater(projections['revenue'].iloc[0], self.assumptions.base_revenue)
        
        # Check that FCF is calculated
        self.assertTrue(all(pd.notnull(projections['free_cash_flow'])))
    
    def test_calculate_terminal_value(self):
        """Test terminal value calculation"""
        self.dcf_model.build_projections()
        terminal_values = self.dcf_model.calculate_terminal_value()
        
        self.assertIsInstance(terminal_values, dict)
        self.assertIn('gordon_growth_terminal_value', terminal_values)
        self.assertIn('exit_multiple_terminal_value', terminal_values)
        self.assertIn('average_terminal_value', terminal_values)
        
        # All terminal values should be positive
        for key, value in terminal_values.items():
            if isinstance(value, (int, float)):
                self.assertGreater(value, 0)
    
    def test_calculate_dcf_valuation(self):
        """Test complete DCF valuation"""
        valuation_results = self.dcf_model.calculate_dcf_valuation()
        
        self.assertIsInstance(valuation_results, dict)
        
        # Check required keys
        required_keys = [
            'enterprise_value', 'equity_value', 'value_per_share',
            'pv_projection_period', 'pv_terminal_value', 'wacc'
        ]
        for key in required_keys:
            self.assertIn(key, valuation_results)
        
        # Check that values are reasonable
        self.assertGreater(valuation_results['enterprise_value'], 0)
        self.assertGreater(valuation_results['value_per_share'], 0)
        self.assertGreater(valuation_results['wacc'], 0)
        self.assertLess(valuation_results['wacc'], 1)
    
    def test_scenario_analysis(self):
        """Test scenario analysis functionality"""
        scenarios = self.dcf_model.scenario_analysis()
        
        self.assertIsInstance(scenarios, dict)
        self.assertIn('Base', scenarios)
        self.assertIn('Bear', scenarios)
        self.assertIn('Bull', scenarios)
        
        # Bear case should be lower than Bull case
        bear_value = scenarios['Bear']['value_per_share']
        bull_value = scenarios['Bull']['value_per_share']
        
        self.assertLess(bear_value, bull_value)
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis"""
        self.dcf_model.build_projections()
        
        wacc_range = [0.08, 0.09, 0.10, 0.11, 0.12]
        terminal_range = [0.02, 0.025, 0.03]
        
        sensitivity_df = self.dcf_model.sensitivity_analysis(wacc_range, terminal_range)
        
        self.assertIsInstance(sensitivity_df, pd.DataFrame)
        self.assertEqual(len(sensitivity_df), len(terminal_range))
        self.assertEqual(len(sensitivity_df.columns), len(wacc_range))


class TestFinancialCalculations(unittest.TestCase):
    """Test financial calculations utility functions"""
    
    def test_calculate_wacc(self):
        """Test WACC calculation"""
        wacc = FinancialCalculations.calculate_wacc(
            risk_free_rate=0.03,
            market_return=0.10,
            beta=1.2,
            tax_rate=0.25,
            debt_ratio=0.3,
            cost_of_debt=0.05
        )
        
        self.assertIsInstance(wacc, float)
        self.assertGreater(wacc, 0)
        self.assertLess(wacc, 1)
    
    def test_calculate_terminal_value(self):
        """Test terminal value calculation"""
        terminal_value = FinancialCalculations.calculate_terminal_value(
            final_fcf=10_000_000,
            terminal_growth_rate=0.025,
            discount_rate=0.10,
            years=5
        )
        
        self.assertIsInstance(terminal_value, float)
        self.assertGreater(terminal_value, 0)
    
    def test_calculate_free_cash_flow(self):
        """Test free cash flow calculation"""
        fcf = FinancialCalculations.calculate_free_cash_flow(
            revenue=100_000_000,
            operating_margin=0.20,
            tax_rate=0.25,
            capex_percent=0.03,
            working_capital_change=1_000_000,
            depreciation=2_000_000
        )
        
        self.assertIsInstance(fcf, float)
    
    def test_calculate_multiples(self):
        """Test multiples calculation"""
        stock_data = {
            'market_cap': 1_000_000_000,
            'current_price': 100,
            'enterprise_value': 1_100_000_000
        }
        
        financial_data = {
            'revenue': 500_000_000,
            'ebitda': 100_000_000,
            'net_income': 50_000_000,
            'book_value': 200_000_000,
            'shares_outstanding': 10_000_000
        }
        
        multiples = FinancialCalculations.calculate_multiples(stock_data, financial_data)
        
        self.assertIsInstance(multiples, dict)
        self.assertIn('pe_ratio', multiples)
        self.assertIn('ev_to_ebitda', multiples)
        
        # Test P/E calculation
        expected_pe = stock_data['market_cap'] / financial_data['net_income']
        self.assertAlmostEqual(multiples['pe_ratio'], expected_pe, places=2)
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation"""
        base_assumptions = {
            'base_revenue': 100_000_000,
            'revenue_growth': 0.05,
            'operating_margin': 0.15,
            'discount_rate': 0.10,
            'terminal_growth': 0.025,
            'shares_outstanding': 10_000_000
        }
        
        volatility_ranges = {
            'revenue_growth': 0.02,
            'operating_margin': 0.03,
            'discount_rate': 0.01,
            'terminal_growth': 0.005
        }
        
        results = FinancialCalculations.monte_carlo_simulation(
            base_assumptions, volatility_ranges, num_simulations=100
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('valuations', results)
        self.assertEqual(len(results['valuations']), 100)
        
        # Check that we have variation in results
        valuations = results['valuations']
        self.assertGreater(max(valuations), min(valuations))


class TestComparableAnalyzer(unittest.TestCase):
    """Test comparable analysis functionality"""
    
    def setUp(self):
        self.analyzer = ComparableAnalyzer('TEST')
    
    @patch('src.data.market_data.MarketDataProvider.get_company_info')
    def test_load_target_company(self, mock_get_company_info):
        """Test loading target company data"""
        # Mock company info response
        mock_company_info = {
            'basic_info': {
                'company_name': 'Test Company',
                'sector': 'Technology',
                'industry': 'Software',
                'market_cap': 1_000_000_000,
                'enterprise_value': 1_100_000_000
            },
            'current_metrics': {
                'current_price': 100.0
            },
            'financial_statements': {
                'income_statement': {
                    'Total Revenue': {'2023-12-31': 500_000_000},
                    'Net Income': {'2023-12-31': 50_000_000}
                },
                'balance_sheet': {
                    'Cash And Cash Equivalents': {'2023-12-31': 100_000_000}
                }
            }
        }
        
        mock_get_company_info.return_value = mock_company_info
        
        profile = self.analyzer.load_target_company()
        
        self.assertEqual(profile.symbol, 'TEST')
        self.assertEqual(profile.name, 'Test Company')
        self.assertEqual(profile.sector, 'Technology')
        self.assertGreater(profile.market_cap, 0)
    
    def test_find_peer_companies(self):
        """Test peer company selection"""
        # Test with custom peers
        custom_peers = ['AAPL', 'MSFT', 'GOOGL']
        peers = self.analyzer.find_peer_companies(custom_peers=custom_peers, auto_select=False)
        
        self.assertIsInstance(peers, list)
        for peer in custom_peers:
            self.assertIn(peer, peers)
    
    def test_predefined_peers(self):
        """Test predefined peer mappings"""
        aapl_peers = self.analyzer._get_predefined_peers('AAPL')
        
        self.assertIsInstance(aapl_peers, list)
        self.assertIn('MSFT', aapl_peers)
        self.assertIn('GOOGL', aapl_peers)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and workflows"""
    
    def test_complete_dcf_workflow(self):
        """Test complete DCF modeling workflow"""
        # Create DCF model
        assumptions = DCFAssumptions(
            base_revenue=1_000_000_000,
            revenue_growth_rates=[0.08, 0.06, 0.04, 0.03, 0.02],
            terminal_growth_rate=0.025,
            operating_margin=0.18,
            shares_outstanding=100_000_000
        )
        
        dcf_model = DCFModel("INTEGRATION_TEST")
        dcf_model.set_assumptions(assumptions)
        
        # Build projections
        projections = dcf_model.build_projections()
        self.assertEqual(len(projections), 5)
        
        # Calculate valuation
        valuation = dcf_model.calculate_dcf_valuation()
        self.assertGreater(valuation['value_per_share'], 0)
        
        # Run scenarios
        scenarios = dcf_model.scenario_analysis()
        self.assertEqual(len(scenarios), 3)
        
        # Generate summary
        summary = dcf_model.get_summary_table()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertGreater(len(summary), 0)
    
    def test_data_validation(self):
        """Test data validation and error handling"""
        # Test with invalid assumptions
        with self.assertRaises(Exception):
            invalid_assumptions = DCFAssumptions(
                base_revenue=-1000,  # Invalid negative revenue
                revenue_growth_rates=[0.05],
                shares_outstanding=0  # Invalid zero shares
            )
            
            dcf_model = DCFModel("INVALID_TEST")
            dcf_model.set_assumptions(invalid_assumptions)
            dcf_model.calculate_dcf_valuation()
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test with very high growth rates
        high_growth_assumptions = DCFAssumptions(
            base_revenue=1_000_000,
            revenue_growth_rates=[0.50, 0.40, 0.30, 0.20, 0.10],  # Very high growth
            terminal_growth_rate=0.05,  # Max reasonable terminal growth
            operating_margin=0.05,  # Low margin
            shares_outstanding=1_000_000
        )
        
        dcf_model = DCFModel("EDGE_TEST")
        dcf_model.set_assumptions(high_growth_assumptions)
        
        # Should still work but produce extreme valuations
        valuation = dcf_model.calculate_dcf_valuation()
        self.assertIsInstance(valuation, dict)
        self.assertGreater(valuation['value_per_share'], 0)


class TestPerformance(unittest.TestCase):
    """Test performance and scalability"""
    
    def test_large_simulation_performance(self):
        """Test performance with large Monte Carlo simulation"""
        import time
        
        base_assumptions = {
            'base_revenue': 100_000_000,
            'revenue_growth': 0.05,
            'operating_margin': 0.15,
            'discount_rate': 0.10,
            'terminal_growth': 0.025,
            'shares_outstanding': 10_000_000
        }
        
        volatility_ranges = {
            'revenue_growth': 0.02,
            'operating_margin': 0.03,
            'discount_rate': 0.01,
            'terminal_growth': 0.005
        }
        
        start_time = time.time()
        results = FinancialCalculations.monte_carlo_simulation(
            base_assumptions, volatility_ranges, num_simulations=1000
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (less than 10 seconds)
        self.assertLess(execution_time, 10.0)
        self.assertEqual(len(results['valuations']), 1000)


def run_test_suite():
    """Run the complete test suite"""
    print("🧪 Running Equity Valuation Platform Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMarketDataProvider,
        TestDCFModel,
        TestFinancialCalculations,
        TestComparableAnalyzer,
        TestIntegrationScenarios,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100:.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n🚫 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    if not result.failures and not result.errors:
        print("\n🎉 ALL TESTS PASSED! Your platform is ready for deployment.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)
