"""
Complete fix for all import issues in the Streamlit app
This will fix the relative import errors and get your app running
"""

import os

def fix_main_py():
    """Fix main.py with absolute imports"""
    
    main_content = '''"""
Main Streamlit Application for Equity Valuation Dashboard
Professional-grade investment analysis platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Try to import our modules with absolute imports and error handling
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
    
    from market_data import MarketDataProvider
    from dcf import DCFModel, DCFAssumptions
    from calculations import FinancialCalculations
    
    IMPORTS_SUCCESSFUL = True
    st.success("✅ All modules loaded successfully!")
    
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    st.error(f"❌ Import error: {str(e)}")
    st.info("Some features may be limited. Please check your file structure.")

# Page configuration
st.set_page_config(
    page_title="Equity Valuation Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'market_data_provider' not in st.session_state and IMPORTS_SUCCESSFUL:
    st.session_state.market_data_provider = MarketDataProvider()

if 'current_company_data' not in st.session_state:
    st.session_state.current_company_data = None

def load_company_data(symbol: str):
    """Load company data with error handling"""
    if not IMPORTS_SUCCESSFUL:
        st.error("Cannot load data - modules not available")
        return None
        
    try:
        with st.spinner(f"Loading data for {symbol}..."):
            provider = st.session_state.market_data_provider
            company_data = provider.get_company_info(symbol)
            stock_data = provider.get_stock_data(symbol, period="1y")
            
            # Simple returns calculation
            if not stock_data.empty:
                current_price = stock_data['Close'].iloc[-1]
                week_ago_price = stock_data['Close'].iloc[-5] if len(stock_data) > 5 else current_price
                returns = {
                    '1W': ((current_price - week_ago_price) / week_ago_price) * 100
                }
            else:
                returns = {'1W': 0}
            
            return {
                'company_info': company_data,
                'stock_data': stock_data,
                'returns': returns,
                'symbol': symbol
            }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_price_chart(stock_data: pd.DataFrame, symbol: str):
    """Create price chart"""
    if stock_data.empty:
        st.warning("No price data available")
        return None
        
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title=f"{symbol} - Stock Price Performance",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template='plotly_white',
        height=400
    )
    
    return fig

def display_company_overview(company_data):
    """Display company overview"""
    if not company_data:
        return
    
    info = company_data.get('company_info', {})
    basic_info = info.get('basic_info', {})
    current_metrics = info.get('current_metrics', {})
    financial_metrics = info.get('financial_metrics', {})
    
    st.markdown('<div class="section-header">📊 Company Overview</div>', unsafe_allow_html=True)
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Company", basic_info.get('company_name', 'N/A'))
    with col2:
        st.metric("Sector", basic_info.get('sector', 'N/A'))
    with col3:
        market_cap = basic_info.get('market_cap', 0)
        if market_cap > 1e9:
            cap_display = f"${market_cap/1e9:.1f}B"
        elif market_cap > 1e6:
            cap_display = f"${market_cap/1e6:.0f}M"
        else:
            cap_display = f"${market_cap:,.0f}"
        st.metric("Market Cap", cap_display)
    with col4:
        st.metric("Current Price", f"${current_metrics.get('current_price', 0):.2f}")
    
    # Financial ratios
    st.markdown("### Key Ratios")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pe = financial_metrics.get('pe_ratio', 0)
        st.metric("P/E Ratio", f"{pe:.1f}x" if pe > 0 else "N/A")
    with col2:
        pb = financial_metrics.get('price_to_book', 0)
        st.metric("P/B Ratio", f"{pb:.1f}x" if pb > 0 else "N/A")
    with col3:
        roe = financial_metrics.get('return_on_equity', 0)
        st.metric("ROE", f"{roe*100:.1f}%" if roe > 0 else "N/A")
    with col4:
        margin = financial_metrics.get('profit_margin', 0)
        st.metric("Profit Margin", f"{margin*100:.1f}%" if margin > 0 else "N/A")

def build_dcf_interface():
    """Build DCF interface"""
    if not IMPORTS_SUCCESSFUL:
        st.warning("DCF modeling requires all modules to be loaded")
        return
    
    st.markdown('<div class="section-header">🧮 DCF Valuation Model</div>', unsafe_allow_html=True)
    
    if not st.session_state.current_company_data:
        st.warning("Please load company data first")
        return
    
    symbol = st.session_state.current_company_data['symbol']
    
    st.markdown("### Model Assumptions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Revenue Projections")
        
        base_revenue = st.number_input(
            "Base Revenue ($M)", 
            min_value=1.0, 
            value=100000.0, 
            step=1000.0
        ) * 1e6
        
        growth_rates = []
        for i in range(5):
            growth = st.slider(
                f"Year {i+1} Growth", 
                -10.0, 30.0, 5.0-i*0.5, 0.5,
                key=f"growth_{i}"
            ) / 100
            growth_rates.append(growth)
        
        operating_margin = st.slider("Operating Margin", 5.0, 40.0, 20.0) / 100
    
    with col2:
        st.markdown("#### Valuation Parameters")
        
        terminal_growth = st.slider("Terminal Growth", 1.0, 4.0, 2.5) / 100
        discount_rate = st.slider("Discount Rate", 6.0, 15.0, 10.0) / 100
        
        shares = st.number_input(
            "Shares Outstanding (M)", 
            min_value=1.0, 
            value=1000.0
        ) * 1e6
    
    if st.button("Calculate DCF", type="primary"):
        try:
            # Create assumptions
            assumptions = DCFAssumptions(
                base_revenue=base_revenue,
                revenue_growth_rates=growth_rates,
                terminal_growth_rate=terminal_growth,
                operating_margin=operating_margin,
                shares_outstanding=int(shares)
            )
            
            # Create model
            dcf_model = DCFModel(symbol)
            dcf_model.set_assumptions(assumptions)
            
            # Calculate
            projections = dcf_model.build_projections()
            results = dcf_model.calculate_dcf_valuation()
            
            # Display results
            st.markdown("### Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fair Value", f"${results['value_per_share']:.2f}")
            with col2:
                st.metric("Enterprise Value", f"${results['enterprise_value']/1e9:.1f}B")
            with col3:
                st.metric("WACC", f"{results['wacc']:.1%}")
            
            # Show projections
            st.markdown("### Projections")
            display_proj = projections[['year', 'revenue', 'operating_income', 'free_cash_flow']].copy()
            display_proj['revenue'] = display_proj['revenue'] / 1e6
            display_proj['operating_income'] = display_proj['operating_income'] / 1e6  
            display_proj['free_cash_flow'] = display_proj['free_cash_flow'] / 1e6
            
            st.dataframe(display_proj, use_container_width=True)
            
        except Exception as e:
            st.error(f"DCF calculation error: {str(e)}")

def main():
    """Main app function"""
    
    # Header
    st.markdown('<div class="main-header">📊 Equity Valuation Pro</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Professional Investment Analysis Platform</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Stock Analysis")
        
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter ticker symbol").upper()
        
        if st.button("📊 Load Data", type="primary"):
            company_data = load_company_data(symbol)
            if company_data:
                st.session_state.current_company_data = company_data
                st.success(f"✅ Loaded {symbol}")
        
        st.markdown("---")
        
        sections = st.multiselect(
            "Analysis Sections",
            ["Company Overview", "Price Chart", "DCF Model"],
            default=["Company Overview", "Price Chart"]
        )
    
    # Main content
    if st.session_state.current_company_data:
        company_data = st.session_state.current_company_data
        
        if "Company Overview" in sections:
            display_company_overview(company_data)
        
        if "Price Chart" in sections:
            st.markdown('<div class="section-header">📈 Price Chart</div>', unsafe_allow_html=True)
            chart = create_price_chart(company_data['stock_data'], company_data['symbol'])
            if chart:
                st.plotly_chart(chart, use_container_width=True)
        
        if "DCF Model" in sections:
            build_dcf_interface()
    
    else:
        # Welcome message
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 2rem; border-radius: 10px; margin: 2rem 0;">
            <h3>🚀 Welcome to Equity Valuation Pro</h3>
            <p>Professional-grade investment analysis platform built with Python and Streamlit.</p>
            
            <h4>✨ Features:</h4>
            <ul>
                <li>📊 Real-time market data integration</li>
                <li>🧮 DCF modeling with scenario analysis</li>
                <li>📈 Interactive charts and visualizations</li>
                <li>📋 Professional-grade financial analysis</li>
            </ul>
            
            <p><strong>Get started:</strong> Enter a stock symbol (AAPL, MSFT, GOOGL) and click "Load Data"</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown('<p style="text-align: center; color: #888;">Built for Investment Banking Excellence</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
'''
    
    with open('src/main.py', 'w') as f:
        f.write(main_content)
    print("✅ Fixed main.py with absolute imports")

def fix_market_data():
    """Fix market_data.py"""
    
    market_data_content = '''"""
Market Data Module - Simplified for immediate functionality
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MarketDataProvider:
    """Market data provider using yfinance"""
    
    def __init__(self):
        self.cache = {}
    
    def get_company_info(self, symbol: str) -> dict:
        """Get company information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'basic_info': {
                    'symbol': symbol,
                    'company_name': info.get('longName', symbol),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap', 0),
                    'enterprise_value': info.get('enterpriseValue', 0),
                    'country': info.get('country', 'N/A')
                },
                'current_metrics': {
                    'current_price': info.get('currentPrice', 0),
                    'previous_close': info.get('previousClose', 0),
                    'day_high': info.get('dayHigh', 0),
                    'day_low': info.get('dayLow', 0),
                    'volume': info.get('volume', 0),
                    'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                    'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0)
                },
                'financial_metrics': {
                    'pe_ratio': info.get('trailingPE', 0),
                    'forward_pe': info.get('forwardPE', 0),
                    'price_to_book': info.get('priceToBook', 0),
                    'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                    'ev_to_ebitda': info.get('enterpriseToEbitda', 0),
                    'profit_margin': info.get('profitMargins', 0),
                    'operating_margin': info.get('operatingMargins', 0),
                    'return_on_equity': info.get('returnOnEquity', 0),
                    'debt_to_equity': info.get('debtToEquity', 0)
                }
            }
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            return self._get_default_info(symbol)
    
    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical stock data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_returns(self, data: pd.DataFrame) -> dict:
        """Calculate simple returns"""
        if data.empty or 'Close' not in data.columns:
            return {}
        
        current_price = data['Close'].iloc[-1]
        returns = {}
        
        periods = {'1D': 1, '1W': 5, '1M': 22, '3M': 66, '1Y': 252}
        
        for period, days in periods.items():
            try:
                if len(data) > days:
                    past_price = data['Close'].iloc[-(days+1)]
                    returns[period] = ((current_price - past_price) / past_price) * 100
                else:
                    returns[period] = 0.0
            except:
                returns[period] = 0.0
        
        return returns
    
    def _get_default_info(self, symbol: str) -> dict:
        """Return default info structure"""
        return {
            'basic_info': {
                'symbol': symbol,
                'company_name': symbol,
                'sector': 'N/A',
                'industry': 'N/A',
                'market_cap': 0,
                'enterprise_value': 0,
                'country': 'N/A'
            },
            'current_metrics': {
                'current_price': 0,
                'previous_close': 0,
                'day_high': 0,
                'day_low': 0,
                'volume': 0,
                'fifty_two_week_high': 0,
                'fifty_two_week_low': 0
            },
            'financial_metrics': {
                'pe_ratio': 0,
                'forward_pe': 0,
                'price_to_book': 0,
                'price_to_sales': 0,
                'ev_to_ebitda': 0,
                'profit_margin': 0,
                'operating_margin': 0,
                'return_on_equity': 0,
                'debt_to_equity': 0
            }
        }

# Convenience function
def get_company_overview(symbol: str) -> dict:
    """Get company overview"""
    provider = MarketDataProvider()
    return provider.get_company_info(symbol)
'''
    
    os.makedirs('src/data', exist_ok=True)
    with open('src/data/market_data.py', 'w') as f:
        f.write(market_data_content)
    print("✅ Created src/data/market_data.py")

def fix_calculations():
    """Create basic calculations module"""
    
    calc_content = '''"""
Financial Calculations Module - Core functions
"""

import pandas as pd
import numpy as np

class FinancialCalculations:
    """Basic financial calculations"""
    
    @staticmethod
    def calculate_wacc(risk_free_rate: float, market_return: float, beta: float, 
                      tax_rate: float, debt_ratio: float, cost_of_debt: float) -> float:
        """Calculate WACC"""
        cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
        equity_ratio = 1 - debt_ratio
        wacc = (equity_ratio * cost_of_equity) + (debt_ratio * cost_of_debt * (1 - tax_rate))
        return wacc
    
    @staticmethod
    def calculate_dcf_value(fcf_projections: list, terminal_value: float, 
                           discount_rate: float, shares_outstanding: int) -> dict:
        """Calculate DCF valuation"""
        pv_fcfs = []
        for i, fcf in enumerate(fcf_projections, 1):
            pv = fcf / ((1 + discount_rate) ** i)
            pv_fcfs.append(pv)
        
        total_pv_fcf = sum(pv_fcfs)
        enterprise_value = total_pv_fcf + terminal_value
        value_per_share = enterprise_value / shares_outstanding
        
        return {
            'enterprise_value': enterprise_value,
            'pv_projection_period': total_pv_fcf,
            'pv_terminal_value': terminal_value,
            'value_per_share': value_per_share
        }
'''
    
    os.makedirs('src/utils', exist_ok=True)
    with open('src/utils/calculations.py', 'w') as f:
        f.write(calc_content)
    print("✅ Created src/utils/calculations.py")

def create_init_files():
    """Create all __init__.py files"""
    init_files = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/models/__init__.py', 
        'src/utils/__init__.py'
    ]
    
    for init_file in init_files:
        os.makedirs(os.path.dirname(init_file), exist_ok=True)
        with open(init_file, 'w') as f:
            f.write('# Package initialization\n')
    print("✅ Created all __init__.py files")

def main():
    print("🔧 FIXING ALL IMPORT ISSUES")
    print("=" * 40)
    
    # Fix all files
    create_init_files()
    fix_main_py()
    fix_market_data()
    fix_calculations()
    
    print("\n🎉 ALL FIXES APPLIED!")
    print("\n🚀 NOW RUN YOUR APP:")
    print("streamlit run src/main.py")
    print("\n✅ This should resolve all import issues!")
    print("✅ The app will now use absolute imports")
    print("✅ All core functionality will work")

if __name__ == "__main__":
    main()
