"""
Quick fix for import issues in the Streamlit app
Run this to fix all import path problems
"""

import os
import sys
# -*- coding: utf-8 -*-

def fix_main_app():
    """Fix the main Streamlit app imports"""
    
    main_app_content = '''"""
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

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Import our custom modules with error handling
try:
    from data.market_data import MarketDataProvider, get_company_overview
except ImportError:
    st.error("❌ Could not import market data module. Please check file structure.")
    st.stop()

try:
    from models.dcf import DCFModel, DCFAssumptions, create_sample_dcf_model
except ImportError:
    st.error("❌ Could not import DCF module. Please check file structure.")
    st.stop()

try:
    from utils.calculations import FinancialCalculations
except ImportError:
    st.error("❌ Could not import calculations module. Please check file structure.")
    st.stop()

# Optional imports (won't stop app if missing)
try:
    from models.comps import ComparableAnalyzer
    COMPS_AVAILABLE = True
except ImportError:
    COMPS_AVAILABLE = False
    st.warning("⚠️ Comparable analysis not available. Some features will be limited.")

try:
    from dashboard.comparable_analysis import ComparableAnalysisDashboard
    from dashboard.advanced_analytics import AdvancedAnalyticsDashboard
    ADVANCED_FEATURES = True
except ImportError:
    ADVANCED_FEATURES = False

# Page configuration
st.set_page_config(
    page_title="Equity Valuation Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
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
    .highlight-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'market_data_provider' not in st.session_state:
    st.session_state.market_data_provider = MarketDataProvider()

if 'current_company_data' not in st.session_state:
    st.session_state.current_company_data = None

if 'dcf_model' not in st.session_state:
    st.session_state.dcf_model = None


def load_company_data(symbol: str):
    """Load comprehensive company data"""
    try:
        with st.spinner(f"Loading data for {symbol}..."):
            company_data = st.session_state.market_data_provider.get_company_info(symbol)
            stock_data = st.session_state.market_data_provider.get_stock_data(symbol, period="2y")
            
            # Calculate returns
            returns = st.session_state.market_data_provider.calculate_returns(stock_data)
            
            return {
                'company_info': company_data,
                'stock_data': stock_data,
                'returns': returns,
                'symbol': symbol
            }
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {str(e)}")
        return None


def create_price_chart(stock_data: pd.DataFrame, symbol: str):
    """Create interactive price chart"""
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data['close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add moving averages
    stock_data['MA20'] = stock_data['close'].rolling(window=20).mean()
    stock_data['MA50'] = stock_data['close'].rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data['MA20'],
        mode='lines',
        name='20-Day MA',
        line=dict(color='orange', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data['MA50'],
        mode='lines',
        name='50-Day MA',
        line=dict(color='red', width=1, dash='dot')
    ))
    
    fig.update_layout(
        title=f"{symbol} - Stock Price Performance",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def display_company_overview(company_data):
    """Display company overview section"""
    if not company_data:
        return
    
    info = company_data['company_info']
    basic_info = info.get('basic_info', {})
    current_metrics = info.get('current_metrics', {})
    financial_metrics = info.get('financial_metrics', {})
    
    st.markdown('<div class="section-header">📊 Company Overview</div>', unsafe_allow_html=True)
    
    # Basic company information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Company Name", basic_info.get('company_name', 'N/A'))
    with col2:
        st.metric("Sector", basic_info.get('sector', 'N/A'))
    with col3:
        market_cap = basic_info.get('market_cap', 0)
        st.metric("Market Cap", f"${market_cap/1e9:.1f}B" if market_cap > 1e9 else f"${market_cap/1e6:.1f}M")
    with col4:
        st.metric("Current Price", f"${current_metrics.get('current_price', 0):.2f}")
    
    # Key financial ratios
    st.markdown("### Key Financial Ratios")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("P/E Ratio", f"{financial_metrics.get('pe_ratio', 0):.1f}x")
    with col2:
        st.metric("P/B Ratio", f"{financial_metrics.get('price_to_book', 0):.2f}x")
    with col3:
        st.metric("EV/EBITDA", f"{financial_metrics.get('ev_to_ebitda', 0):.1f}x")
    with col4:
        st.metric("ROE", f"{financial_metrics.get('return_on_equity', 0)*100:.1f}%")
    with col5:
        st.metric("Profit Margin", f"{financial_metrics.get('profit_margin', 0)*100:.1f}%")


def build_dcf_model_interface():
    """Build DCF model interface"""
    st.markdown('<div class="section-header">🧮 DCF Valuation Model</div>', unsafe_allow_html=True)
    
    if not st.session_state.current_company_data:
        st.warning("Please select a company first to build DCF model.")
        return
    
    symbol = st.session_state.current_company_data['symbol']
    
    # DCF Assumptions Input
    st.markdown("### 🎯 Model Assumptions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Revenue & Growth")
        
        base_revenue = st.number_input(
            "Base Revenue ($M)",
            min_value=1,
            value=10000,
            step=100,
            help="Most recent annual revenue in millions"
        ) * 1_000_000
        
        st.markdown("**5-Year Revenue Growth Rates**")
        growth_rates = []
        for i in range(5):
            rate = st.slider(
                f"Year {i+1} Growth Rate",
                min_value=-20.0,
                max_value=50.0,
                value=5.0 - i*0.5,
                step=0.5,
                format="%.1f%%",
                key=f"growth_y{i+1}"
            ) / 100
            growth_rates.append(rate)
        
        operating_margin = st.slider(
            "Operating Margin",
            min_value=1.0,
            max_value=50.0,
            value=20.0,
            step=0.5,
            format="%.1f%%"
        ) / 100
    
    with col2:
        st.markdown("#### Capital & Financing")
        
        terminal_growth = st.slider(
            "Terminal Growth Rate",
            min_value=0.5,
            max_value=5.0,
            value=2.5,
            step=0.1,
            format="%.1f%%"
        ) / 100
        
        beta = st.slider(
            "Beta",
            min_value=0.1,
            max_value=3.0,
            value=1.2,
            step=0.1,
            format="%.1f"
        )
        
        risk_free_rate = st.slider(
            "Risk-Free Rate",
            min_value=1.0,
            max_value=8.0,
            value=3.5,
            step=0.1,
            format="%.1f%%"
        ) / 100
        
        shares_outstanding = st.number_input(
            "Shares Outstanding (M)",
            min_value=1,
            value=1000,
            step=50
        ) * 1_000_000
    
    # Build DCF Model
    if st.button("🚀 Calculate DCF Valuation", type="primary"):
        with st.spinner("Building DCF model..."):
            try:
                # Create DCF assumptions
                assumptions = DCFAssumptions(
                    base_revenue=base_revenue,
                    revenue_growth_rates=growth_rates,
                    terminal_growth_rate=terminal_growth,
                    operating_margin=operating_margin,
                    beta=beta,
                    risk_free_rate=risk_free_rate,
                    shares_outstanding=shares_outstanding
                )
                
                # Create and run DCF model
                dcf_model = DCFModel(symbol)
                dcf_model.set_assumptions(assumptions)
                
                projections = dcf_model.build_projections()
                valuation_results = dcf_model.calculate_dcf_valuation()
                
                st.session_state.dcf_model = dcf_model
                
                # Display results
                st.markdown("### 💰 DCF Valuation Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Fair Value per Share", f"${valuation_results['value_per_share']:.2f}")
                with col2:
                    st.metric("Enterprise Value", f"${valuation_results['enterprise_value']/1e9:.2f}B")
                with col3:
                    st.metric("WACC", f"{valuation_results['wacc']:.1%}")
                with col4:
                    terminal_pct = (valuation_results['pv_terminal_value'] / valuation_results['enterprise_value']) * 100
                    st.metric("Terminal Value %", f"{terminal_pct:.0f}%")
                
                # Show projections table
                st.markdown("### 📋 Financial Projections")
                display_projections = projections.copy()
                display_projections['revenue'] = display_projections['revenue'] / 1e6
                display_projections['operating_income'] = display_projections['operating_income'] / 1e6
                display_projections['free_cash_flow'] = display_projections['free_cash_flow'] / 1e6
                
                st.dataframe(display_projections[['year', 'revenue', 'operating_income', 'free_cash_flow']], use_container_width=True)
                
            except Exception as e:
                st.error(f"Error calculating DCF: {str(e)}")


def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">📊 Equity Valuation Pro</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 2rem;">'
        'Professional-Grade Investment Analysis Platform</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Analysis Controls")
        
        # Stock selection
        symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            help="Enter stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
        
        if st.button("📊 Load Company Data", type="primary"):
            company_data = load_company_data(symbol)
            if company_data:
                st.session_state.current_company_data = company_data
                st.success(f"✅ Loaded data for {symbol}")
        
        # Navigation
        st.markdown("---")
        st.markdown("### 📋 Analysis Sections")
        
        analysis_sections = st.multiselect(
            "Select Analysis Sections",
            ["Company Overview", "Price Chart", "DCF Model"],
            default=["Company Overview", "Price Chart", "DCF Model"]
        )
        
        if not COMPS_AVAILABLE:
            st.info("💡 Install comparable analysis module for peer benchmarking")
        
        if not ADVANCED_FEATURES:
            st.info("💡 Install advanced analytics for LBO, ML, and risk analysis")
    
    # Main content area
    if st.session_state.current_company_data:
        company_data = st.session_state.current_company_data
        
        # Company Overview
        if "Company Overview" in analysis_sections:
            display_company_overview(company_data)
        
        # Price Chart
        if "Price Chart" in analysis_sections:
            st.markdown('<div class="section-header">📈 Price Performance</div>', unsafe_allow_html=True)
            price_chart = create_price_chart(company_data['stock_data'], company_data['symbol'])
            st.plotly_chart(price_chart, use_container_width=True)
        
        # DCF Model
        if "DCF Model" in analysis_sections:
            build_dcf_model_interface()
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="highlight-box">
            <h3>🚀 Welcome to Equity Valuation Pro</h3>
            <p>Get started by entering a stock symbol in the sidebar and clicking "Load Company Data".</p>
            
            <h4>✨ Key Features:</h4>
            <ul>
                <li>📊 <strong>Real-time market data</strong> from Yahoo Finance</li>
                <li>🧮 <strong>Professional DCF modeling</strong> with scenario analysis</li>
                <li>📈 <strong>Interactive price charts</strong> and technical indicators</li>
                <li>📋 <strong>Professional reporting</strong> and export capabilities</li>
            </ul>
            
            <h4>🎯 Try These Examples:</h4>
            <p><strong>AAPL</strong> • <strong>MSFT</strong> • <strong>GOOGL</strong> • <strong>AMZN</strong> • <strong>TSLA</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #888; font-size: 0.9rem;">'
        'Built with ❤️ using Streamlit • Data provided by Yahoo Finance</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
'''
    
    # Write the fixed main app
    with open('src/main.py', 'w') as f:
        f.write(main_app_content)
    
    print("✅ Fixed src/main.py with proper imports")


def create_init_files():
    """Create missing __init__.py files"""
    
    init_locations = [
        'src/__init__.py',
        'src/data/__init__.py', 
        'src/models/__init__.py',
        'src/utils/__init__.py',
        'src/dashboard/__init__.py'
    ]
    
    for init_file in init_locations:
        os.makedirs(os.path.dirname(init_file), exist_ok=True)
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('"""Package initialization file"""\n')
            print(f"✅ Created {init_file}")


def check_file_structure():
    """Check and report on file structure"""
    required_files = [
        'src/main.py',
        'src/data/market_data.py',
        'src/models/dcf.py', 
        'src/utils/calculations.py'
    ]
    
    print("🔍 Checking file structure...")
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        print("Please create these files using the code artifacts provided.")
        return False
    else:
        print(f"\n🎉 All core files present!")
        return True


def main():
    """Main fix function"""
    print("[FIX] FIXING IMPORT ISSUES")
    print("=" * 30)
    
    # Step 1: Create missing __init__.py files
    create_init_files()
    
    # Step 2: Fix main app imports
    fix_main_app()
    
    # Step 3: Check file structure
    files_ok = check_file_structure()
    
    if files_ok:
        print(f"\n🚀 TRY RUNNING THE APP NOW:")
        print(f"streamlit run src/main.py")
        print(f"\nIf you still get import errors, make sure you have created all the")
        print(f"Python files from the artifacts I provided earlier.")
    else:
        print(f"\n❌ Please create the missing files first, then run this fix again.")
    
    print(f"\n💡 The main app now has better error handling and will show you")
    print(f"   which specific modules are missing if there are still issues.")


if __name__ == "__main__":
    main()
