"""
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
from typing import Dict, List

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our custom modules
try:
    from data.market_data import MarketDataProvider, get_company_overview
    from models.dcf import DCFModel, DCFAssumptions, create_sample_dcf_model
    from utils.calculations import FinancialCalculations, ComparableAnalysis
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

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
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
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


def load_company_data(symbol: str) -> Dict:
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


def create_price_chart(stock_data: pd.DataFrame, symbol: str) -> go.Figure:
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


def display_company_overview(company_data: Dict):
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
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Company Name",
            basic_info.get('company_name', 'N/A')
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Sector",
            basic_info.get('sector', 'N/A')
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        market_cap = basic_info.get('market_cap', 0)
        st.metric(
            "Market Cap",
            f"${market_cap/1e9:.1f}B" if market_cap > 1e9 else f"${market_cap/1e6:.1f}M"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Employees",
            f"{basic_info.get('employees', 0):,}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Current trading metrics
    st.markdown("### Current Trading Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = [
        ("Current Price", f"${current_metrics.get('current_price', 0):.2f}"),
        ("Day Range", f"${current_metrics.get('day_low', 0):.2f} - ${current_metrics.get('day_high', 0):.2f}"),
        ("52W Range", f"${current_metrics.get('fifty_two_week_low', 0):.2f} - ${current_metrics.get('fifty_two_week_high', 0):.2f}"),
        ("Volume", f"{current_metrics.get('volume', 0):,}"),
        ("Avg Volume", f"{current_metrics.get('avg_volume', 0):,}")
    ]
    
    for i, (label, value) in enumerate(metrics):
        with [col1, col2, col3, col4, col5][i]:
            st.metric(label, value)
    
    # Key financial ratios
    st.markdown("### Key Financial Ratios")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    ratios = [
        ("P/E Ratio", f"{financial_metrics.get('pe_ratio', 0):.1f}x"),
        ("P/B Ratio", f"{financial_metrics.get('price_to_book', 0):.2f}x"),
        ("EV/EBITDA", f"{financial_metrics.get('ev_to_ebitda', 0):.1f}x"),
        ("ROE", f"{financial_metrics.get('return_on_equity', 0)*100:.1f}%"),
        ("Profit Margin", f"{financial_metrics.get('profit_margin', 0)*100:.1f}%")
    ]
    
    for i, (label, value) in enumerate(ratios):
        with [col1, col2, col3, col4, col5][i]:
            st.metric(label, value)


def display_returns_analysis(returns_data: Dict):
    """Display returns analysis"""
    st.markdown('<div class="section-header">📈 Performance Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    periods = ['1D', '1W', '1M', '3M', '6M', '1Y']
    colors = ['red' if returns_data.get(period, 0) < 0 else 'green' for period in periods]
    
    for i, period in enumerate(periods):
        with [col1, col2, col3, col4, col5, col6][i]:
            return_val = returns_data.get(period, 0)
            st.metric(
                f"{period} Return",
                f"{return_val:+.2f}%",
                delta=None
            )


def build_dcf_model_interface():
    """Build DCF model interface"""
    st.markdown('<div class="section-header">🧮 DCF Valuation Model</div>', unsafe_allow_html=True)
    
    if not st.session_state.current_company_data:
        st.warning("Please select a company first to build DCF model.")
        return
    
    company_info = st.session_state.current_company_data['company_info']
    symbol = st.session_state.current_company_data['symbol']
    
    # DCF Assumptions Input
    st.markdown("### 🎯 Model Assumptions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Revenue & Growth")
        
        # Extract current revenue (simplified - you'd want to get actual financial data)
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
                value=5.0 - i*0.5,  # Declining growth assumption
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
        
        tax_rate = st.slider(
            "Tax Rate",
            min_value=10.0,
            max_value=40.0,
            value=25.0,
            step=1.0,
            format="%.0f%%"
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
        
        capex_percent = st.slider(
            "CapEx (% of Revenue)",
            min_value=0.5,
            max_value=10.0,
            value=3.0,
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
        
        market_risk_premium = st.slider(
            "Market Risk Premium",
            min_value=3.0,
            max_value=10.0,
            value=6.5,
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
            # Create DCF assumptions
            assumptions = DCFAssumptions(
                base_revenue=base_revenue,
                revenue_growth_rates=growth_rates,
                terminal_growth_rate=terminal_growth,
                operating_margin=operating_margin,
                tax_rate=tax_rate,
                capex_percent_of_revenue=capex_percent,
                beta=beta,
                risk_free_rate=risk_free_rate,
                market_risk_premium=market_risk_premium,
                shares_outstanding=shares_outstanding
            )
            
            # Create and run DCF model
            dcf_model = DCFModel(symbol)
            dcf_model.set_assumptions(assumptions)
            
            try:
                # Calculate valuation
                projections = dcf_model.build_projections()
                valuation_results = dcf_model.calculate_dcf_valuation()
                
                st.session_state.dcf_model = dcf_model
                
                # Display results
                display_dcf_results(dcf_model, valuation_results, projections)
                
            except Exception as e:
                st.error(f"Error calculating DCF: {str(e)}")


def display_dcf_results(dcf_model: DCFModel, valuation_results: Dict, projections: pd.DataFrame):
    """Display DCF results"""
    st.markdown('<div class="section-header">💰 DCF Valuation Results</div>', unsafe_allow_html=True)
    
    # Key valuation metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Fair Value per Share",
            f"${valuation_results['value_per_share']:.2f}",
            help="DCF-derived fair value per share"
        )
    
    with col2:
        st.metric(
            "Enterprise Value",
            f"${valuation_results['enterprise_value']/1e9:.2f}B",
            help="Total enterprise value"
        )
    
    with col3:
        st.metric(
            "WACC",
            f"{valuation_results['wacc']:.1%}",
            help="Weighted Average Cost of Capital"
        )
    
    with col4:
        terminal_pct = (valuation_results['pv_terminal_value'] / valuation_results['enterprise_value']) * 100
        st.metric(
            "Terminal Value %",
            f"{terminal_pct:.0f}%",
            help="Terminal value as % of enterprise value"
        )
    
    # Value breakdown chart
    st.markdown("### 📊 Value Composition")
    
    fig_breakdown = go.Figure(data=[
        go.Pie(
            labels=['Projection Period', 'Terminal Value'],
            values=[
                valuation_results['pv_projection_period'],
                valuation_results['pv_terminal_value']
            ],
            hole=0.4,
            marker_colors=['#1f77b4', '#ff7f0e']
        )
    ])
    
    fig_breakdown.update_traces(textposition='inside', textinfo='percent+label')
    fig_breakdown.update_layout(
        title="Enterprise Value Breakdown",
        height=400,
        template='plotly_white'
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # Cash flow projections
    with col2:
        fig_fcf = go.Figure()
        fig_fcf.add_trace(go.Bar(
            x=[f"Year {i+1}" for i in range(len(projections))],
            y=projections['free_cash_flow'] / 1e6,
            name='Free Cash Flow',
            marker_color='#2ca02c'
        ))
        
        fig_fcf.update_layout(
            title="Free Cash Flow Projections",
            xaxis_title="Year",
            yaxis_title="FCF ($M)",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_fcf, use_container_width=True)
    
    # Detailed projections table
    st.markdown("### 📋 Detailed Financial Projections")
    
    # Format projections for display
    display_projections = projections.copy()
    display_projections['revenue'] = display_projections['revenue'] / 1e6
    display_projections['operating_income'] = display_projections['operating_income'] / 1e6
    display_projections['free_cash_flow'] = display_projections['free_cash_flow'] / 1e6
    display_projections['capex'] = display_projections['capex'] / 1e6
    
    # Rename columns for display
    display_projections = display_projections.rename(columns={
        'year': 'Year',
        'revenue': 'Revenue ($M)',
        'revenue_growth': 'Revenue Growth (%)',
        'operating_income': 'Operating Income ($M)',
        'operating_margin': 'Operating Margin (%)',
        'free_cash_flow': 'Free Cash Flow ($M)',
        'capex': 'CapEx ($M)'
    })
    
    # Format percentages
    display_projections['Revenue Growth (%)'] = display_projections['Revenue Growth (%)'].apply(lambda x: f"{x:.1%}")
    display_projections['Operating Margin (%)'] = display_projections['Operating Margin (%)'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(
        display_projections[['Year', 'Revenue ($M)', 'Revenue Growth (%)', 
                           'Operating Income ($M)', 'Operating Margin (%)', 
                           'Free Cash Flow ($M)', 'CapEx ($M)']],
        use_container_width=True
    )
    
    # Scenario Analysis
    if st.button("🎯 Run Scenario Analysis"):
        with st.spinner("Running scenario analysis..."):
            scenarios = dcf_model.scenario_analysis()
            display_scenario_analysis(scenarios)


def display_scenario_analysis(scenarios: Dict):
    """Display scenario analysis results"""
    st.markdown('<div class="section-header">🎭 Scenario Analysis</div>', unsafe_allow_html=True)
    
    # Scenario comparison
    scenario_df = pd.DataFrame({
        scenario_name: {
            'Value per Share ($)': f"${results['value_per_share']:.2f}",
            'Enterprise Value ($B)': f"${results['enterprise_value']/1e9:.2f}",
            'WACC': f"{results['wacc']:.1%}",
        }
        for scenario_name, results in scenarios.items()
    }).T
    
    st.dataframe(scenario_df, use_container_width=True)
    
    # Scenario chart
    scenario_values = [scenarios[scenario]['value_per_share'] for scenario in ['Bear', 'Base', 'Bull']]
    
    fig_scenario = go.Figure()
    fig_scenario.add_trace(go.Bar(
        x=['Bear Case', 'Base Case', 'Bull Case'],
        y=scenario_values,
        marker_color=['#d62728', '#1f77b4', '#2ca02c'],
        text=[f"${val:.2f}" for val in scenario_values],
        textposition='auto'
    ))
    
    fig_scenario.update_layout(
        title="Scenario Analysis - Value per Share",
        yaxis_title="Value per Share ($)",
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_scenario, use_container_width=True)


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
        st.markdown('<div class="sidebar-header">🎯 Analysis Controls</div>', unsafe_allow_html=True)
        
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
        st.markdown('<div class="sidebar-header">📋 Analysis Sections</div>', unsafe_allow_html=True)
        
        analysis_sections = st.multiselect(
            "Select Analysis Sections",
            ["Company Overview", "Price Chart", "DCF Model", "Comparable Analysis"],
            default=["Company Overview", "Price Chart", "DCF Model"]
        )
    
    # Main content area
    if st.session_state.current_company_data:
        company_data = st.session_state.current_company_data
        
        # Company Overview
        if "Company Overview" in analysis_sections:
            display_company_overview(company_data)
            display_returns_analysis(company_data['returns'])
        
        # Price Chart
        if "Price Chart" in analysis_sections:
            st.markdown('<div class="section-header">📈 Price Performance</div>', unsafe_allow_html=True)
            price_chart = create_price_chart(company_data['stock_data'], company_data['symbol'])
            st.plotly_chart(price_chart, use_container_width=True)
        
        # DCF Model
        if "DCF Model" in analysis_sections:
            build_dcf_model_interface()
        
        # Comparable Analysis
        if "Comparable Analysis" in analysis_sections:
            st.session_state.comps_dashboard.render_main_interface(company_data['symbol'])
    
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
                <li>🔍 <strong>Comparable company analysis</strong> (coming soon)</li>
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
