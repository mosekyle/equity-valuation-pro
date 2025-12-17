"""
Equity Valuation Pro - Main Application
Professional Investment Analysis Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go 
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import with error handling
try:
    import yfinance as yf
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Equity Valuation Pro",
    page_icon="📈",
    layout="wide"
)

# Custom CSS - Fixed version
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert > div {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class SimpleMarketData:
    """Simple market data provider using yfinance"""
    
    def get_company_info(self, symbol):
        if not DATA_AVAILABLE:
            return self._dummy_data(symbol)
            
        try:
            ticker = yf.Ticker(symbol)
            B
            info = ticker.info
            
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'price_to_book': info.get('priceToBook', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'profit_margin': info.get('profitMargins', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'debt_to_equity': info.get('debtToEquity', 0)
            }
        except Exception as e:
            st.warning(f"Error loading {symbol}: {str(e)}")
            return self._dummy_data(symbol)
    
    def get_stock_data(self, symbol, period="1y"):
        if not DATA_AVAILABLE:
            # Generate dummy data for demo
            dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
            base_price = 150
            returns = np.random.normal(0.001, 0.02, 252)
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            return pd.DataFrame({
                'Open': [p * 0.99 for p in prices[:252]],
                'High': [p * 1.02 for p in prices[:252]],
                'Low': [p * 0.98 for p in prices[:252]],
                'Close': prices[:252],
                'Volume': [np.random.randint(1000000, 10000000) for _ in range(252)]
            }, index=dates)
            
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                st.warning(f"No data found for {symbol}")
                
            return data
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def _dummy_data(self, symbol):
        """Return dummy data for demonstration"""
        return {
            'name': f'{symbol} Corporation',
            'sector': 'Technology',
            'industry': 'Software',
            'market_cap': 1000000000000,  # 1T
            'current_price': 150.0,
            'previous_close': 148.5,
            'day_high': 152.0,
            'day_low': 147.0,
            'pe_ratio': 25.0,
            'forward_pe': 22.0,
            'price_to_book': 3.5,
            'price_to_sales': 8.0,
            'profit_margin': 0.25,
            'return_on_equity': 0.30,
            'debt_to_equity': 0.15
        }

class SimpleDCF:
    """Simple DCF calculator for demonstrations"""
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.base_revenue = 0
        self.growth_rates = []
        self.operating_margin = 0.2
        self.tax_rate = 0.25
        self.capex_rate = 0.03
        self.discount_rate = 0.1
        self.terminal_growth = 0.025
        self.shares = 1000000000
    
    def set_assumptions(self, base_revenue, growth_rates, operating_margin, 
                       discount_rate, terminal_growth, shares):
        self.base_revenue = base_revenue
        self.growth_rates = growth_rates
        self.operating_margin = operating_margin
        self.discount_rate = discount_rate
        self.terminal_growth = terminal_growth
        self.shares = shares
    
    def calculate_dcf(self):
        """Calculate DCF valuation"""
        try:
            # Build projections
            projections = []
            current_revenue = self.base_revenue
            
            for i, growth in enumerate(self.growth_rates):
                year = i + 1
                current_revenue *= (1 + growth)
                
                # Income statement items
                operating_income = current_revenue * self.operating_margin
                taxes = operating_income * self.tax_rate
                nopat = operating_income - taxes
                
                # Cash flow items  
                capex = current_revenue * self.capex_rate
                depreciation = current_revenue * 0.025  # 2.5% of revenue
                
                # Free cash flow
                free_cash_flow = nopat + depreciation - capex
                
                projections.append({
                    'year': year,
                    'revenue': current_revenue,
                    'operating_income': operating_income,
                    'free_cash_flow': free_cash_flow,
                    'growth_rate': growth
                })
            
            # Calculate present values
            pv_fcfs = []
            for proj in projections:
                pv = proj['free_cash_flow'] / ((1 + self.discount_rate) ** proj['year'])
                pv_fcfs.append(pv)
            
            # Terminal value
            final_fcf = projections[-1]['free_cash_flow']
            terminal_fcf = final_fcf * (1 + self.terminal_growth)
            terminal_value = terminal_fcf / (self.discount_rate - self.terminal_growth)
            pv_terminal = terminal_value / ((1 + self.discount_rate) ** len(projections))
            
            # Enterprise and equity value
            enterprise_value = sum(pv_fcfs) + pv_terminal
            equity_value = enterprise_value  # Assuming no net debt
            value_per_share = equity_value / self.shares
            
            return {
                'projections': projections,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'value_per_share': value_per_share,
                'pv_fcfs': sum(pv_fcfs),
                'pv_terminal': pv_terminal,
                'terminal_value_pct': (pv_terminal / enterprise_value) * 100,
                'wacc': self.discount_rate
            }
            
        except Exception as e:
            st.error(f"DCF calculation error: {str(e)}")
            return None

# Initialize session state
if 'data_provider' not in st.session_state:
    st.session_state.data_provider = SimpleMarketData()

if 'company_data' not in st.session_state:
    st.session_state.company_data = None

def display_welcome_screen():
    """Display welcome screen using proper Streamlit components"""
    
    # Welcome container
    with st.container():
        st.success("🚀 Welcome to Equity Valuation Pro")
        st.write("**Professional investment analysis platform built with Python and Streamlit.**")
        
        st.subheader("✨ Key Features")
        st.write("📊 **Real-time Market Data** - Live stock prices and company information")
        st.write("🧮 **DCF Modeling** - Build comprehensive discounted cash flow models")
        st.write("📈 **Interactive Charts** - Professional visualizations and analysis")
        st.write("🎯 **Scenario Analysis** - Test different assumptions and outcomes")
        
        st.subheader("🚀 Getting Started")
        st.write("1️⃣ Enter a stock symbol in the sidebar (try AAPL, MSFT, GOOGL, AMZN, TSLA)")
        st.write("2️⃣ Click 'Load Company Data' to fetch real-time information")
        st.write("3️⃣ Explore company overview, price charts, and build DCF models")
        
        st.info("💡 This platform demonstrates professional-grade financial modeling capabilities perfect for investment banking applications and interviews.")

def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">📊 Equity Valuation Pro</div>', unsafe_allow_html=True)
    st.markdown("### Professional Investment Analysis Platform")
    
    if not DATA_AVAILABLE:
        st.warning("⚠️ yfinance not installed. Install it with: pip install yfinance")
        st.info("💡 App will work with demo data for testing purposes.")
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Stock Analysis")
        
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter ticker symbol").upper()
        
        if st.button("📊 Load Company Data", type="primary"):
            if symbol:
                with st.spinner(f"Loading data for {symbol}..."):
                    try:
                        company_info = st.session_state.data_provider.get_company_info(symbol)
                        stock_data = st.session_state.data_provider.get_stock_data(symbol)
                        
                        st.session_state.company_data = {
                            'symbol': symbol,
                            'info': company_info,
                            'stock_data': stock_data
                        }
                        st.success(f"✅ Successfully loaded {symbol}")
                    except Exception as e:
                        st.error(f"❌ Error loading {symbol}: {str(e)}")
            else:
                st.warning("Please enter a stock symbol")
        
        st.markdown("---")
        
        # Analysis sections
        if st.session_state.company_data:
            st.subheader("📋 Analysis Sections")
            sections = st.multiselect(
                "Select sections to display:",
                ["Company Overview", "Stock Chart", "DCF Valuation"],
                default=["Company Overview", "Stock Chart"]
            )
        else:
            sections = []
    
    # Main content area
    if st.session_state.company_data:
        data = st.session_state.company_data
        symbol = data['symbol']
        info = data['info']
        stock_data = data['stock_data']
        
        # Company Overview Section
        if "Company Overview" in sections:
            st.header("🏢 Company Overview")
            
            # Basic info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Company Name", info['name'])
            with col2:
                st.metric("Sector", info['sector'])
            with col3:
                market_cap = info['market_cap']
                if market_cap > 1e12:
                    cap_display = f"${market_cap/1e12:.2f}T"
                elif market_cap > 1e9:
                    cap_display = f"${market_cap/1e9:.1f}B"
                elif market_cap > 1e6:
                    cap_display = f"${market_cap/1e6:.0f}M"
                else:
                    cap_display = f"${market_cap:,.0f}"
                st.metric("Market Cap", cap_display)
            with col4:
                st.metric("Current Price", f"${info['current_price']:.2f}")
            
            # Price info
            st.subheader("📈 Trading Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Previous Close", f"${info['previous_close']:.2f}")
            with col2:
                st.metric("Day High", f"${info['day_high']:.2f}")
            with col3:
                st.metric("Day Low", f"${info['day_low']:.2f}")
            
            # Financial ratios
            st.subheader("📊 Key Financial Ratios")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                pe = info['pe_ratio']
                st.metric("P/E Ratio", f"{pe:.1f}x" if pe > 0 else "N/A")
            with col2:
                pb = info['price_to_book']
                st.metric("P/B Ratio", f"{pb:.1f}x" if pb > 0 else "N/A")
            with col3:
                roe = info['return_on_equity']
                st.metric("ROE", f"{roe*100:.1f}%" if roe > 0 else "N/A")
            with col4:
                margin = info['profit_margin']
                st.metric("Profit Margin", f"{margin*100:.1f}%" if margin > 0 else "N/A")
        
        # Stock Chart Section
        if "Stock Chart" in sections:
            st.header("📈 Stock Price Performance")
            
            if not stock_data.empty:
                fig = go.Figure()
                
                # Main price line
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Add moving averages
                if len(stock_data) > 20:
                    stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['MA20'],
                        mode='lines',
                        name='20-Day MA',
                        line=dict(color='orange', width=1, dash='dash')
                    ))
                
                if len(stock_data) > 50:
                    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['MA50'],
                        mode='lines',
                        name='50-Day MA',
                        line=dict(color='red', width=1, dash='dot')
                    ))
                
                fig.update_layout(
                    title=f"{symbol} Stock Price Performance",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Price statistics
                if len(stock_data) > 1:
                    current_price = stock_data['Close'].iloc[-1]
                    start_price = stock_data['Close'].iloc[0]
                    total_return = ((current_price - start_price) / start_price) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Period Return", f"{total_return:+.1f}%")
                    with col2:
                        volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
                        st.metric("Annualized Volatility", f"{volatility:.1f}%")
                    with col3:
                        avg_volume = stock_data['Volume'].mean()
                        st.metric("Average Volume", f"{avg_volume:,.0f}")
            else:
                st.warning("No price data available for charting")
        
        # DCF Valuation Section
        if "DCF Valuation" in sections:
            st.header("🧮 DCF Valuation Model")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📋 Model Assumptions")
                
                # Revenue assumptions
                base_revenue = st.number_input(
                    "Base Revenue ($M)",
                    min_value=1.0,
                    max_value=1000000.0,
                    value=100000.0,
                    step=1000.0,
                    help="Most recent annual revenue in millions"
                ) * 1_000_000
                
                st.write("**📈 5-Year Revenue Growth Rates**")
                growth_rates = []
                for i in range(5):
                    growth = st.slider(
                        f"Year {i+1} Growth Rate (%)",
                        min_value=-20.0,
                        max_value=50.0,
                        value=float(8.0 - i * 1.0),  # Declining growth
                        step=0.5,
                        key=f"growth_rate_{i}"
                    ) / 100
                    growth_rates.append(growth)
                
                operating_margin = st.slider(
                    "Operating Margin (%)",
                    min_value=1.0,
                    max_value=50.0,
                    value=20.0,
                    step=0.5
                ) / 100
                
                # Financial assumptions
                discount_rate = st.slider(
                    "Discount Rate/WACC (%)",
                    min_value=5.0,
                    max_value=20.0,
                    value=10.0,
                    step=0.1
                ) / 100
                
                terminal_growth = st.slider(
                    "Terminal Growth Rate (%)",
                    min_value=0.5,
                    max_value=5.0,
                    value=2.5,
                    step=0.1
                ) / 100
                
                shares = st.number_input(
                    "Shares Outstanding (Millions)",
                    min_value=1.0,
                    max_value=50000.0,
                    value=1000.0,
                    step=50.0
                ) * 1_000_000
            
            with col2:
                st.subheader("💰 Valuation Results")
                
                if st.button("🚀 Calculate DCF Valuation", type="primary"):
                    dcf = SimpleDCF(symbol)
                    dcf.set_assumptions(
                        base_revenue, growth_rates, operating_margin,
                        discount_rate, terminal_growth, int(shares)
                    )
                    
                    results = dcf.calculate_dcf()
                    
                    if results:
                        # Key metrics
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric(
                                "Fair Value per Share",
                                f"${results['value_per_share']:.2f}",
                                help="DCF-derived intrinsic value"
                            )
                            
                            st.metric(
                                "Enterprise Value",
                                f"${results['enterprise_value']/1e9:.2f}B",
                                help="Total firm value"
                            )
                        
                        with col_b:
                            current_price = info['current_price']
                            if current_price > 0:
                                upside = ((results['value_per_share'] - current_price) / current_price) * 100
                                st.metric(
                                    "Upside/Downside",
                                    f"{upside:+.1f}%",
                                    help="Difference vs current price"
                                )
                            
                            st.metric(
                                "Terminal Value %",
                                f"{results['terminal_value_pct']:.0f}%",
                                help="Terminal value as % of total"
                            )
                        
                        # Value breakdown chart
                        st.subheader("📊 Value Composition")
                        
                        breakdown_fig = go.Figure(data=[
                            go.Pie(
                                labels=['5-Year FCF', 'Terminal Value'],
                                values=[results['pv_fcfs'], results['pv_terminal']],
                                hole=0.4,
                                marker_colors=['#1f77b4', '#ff7f0e']
                            )
                        ])
                        
                        breakdown_fig.update_traces(textposition='inside', textinfo='percent+label')
                        breakdown_fig.update_layout(
                            title="Enterprise Value Breakdown",
                            height=300,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(breakdown_fig, use_container_width=True)
                        
                        # Projections table
                        st.subheader("📋 Financial Projections")
                        
                        proj_data = []
                        for proj in results['projections']:
                            proj_data.append({
                                'Year': proj['year'],
                                'Revenue ($M)': f"{proj['revenue']/1e6:.0f}",
                                'Growth (%)': f"{proj['growth_rate']:.1%}",
                                'Operating Income ($M)': f"{proj['operating_income']/1e6:.0f}",
                                'Free Cash Flow ($M)': f"{proj['free_cash_flow']/1e6:.0f}"
                            })
                        
                        proj_df = pd.DataFrame(proj_data)
                        st.dataframe(proj_df, use_container_width=True)
    
    else:
        # Welcome screen when no company is loaded
        display_welcome_screen()
        
        # Quick start examples
        st.subheader("🎯 Popular Stocks to Analyze")
        st.write("Click on any stock below to get started:")
        
        example_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        cols = st.columns(len(example_stocks))
        
        for i, symbol in enumerate(example_stocks):
            with cols[i]:
                if st.button(f"📊 {symbol}", key=f"example_{symbol}"):
                    with st.spinner(f"Loading {symbol}..."):
                        company_info = st.session_state.data_provider.get_company_info(symbol)
                        stock_data = st.session_state.data_provider.get_stock_data(symbol)
                        
                        st.session_state.company_data = {
                            'symbol': symbol,
                            'info': company_info,
                            'stock_data': stock_data
                        }
                        st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption("📊 Equity Valuation Pro | Built for Investment Banking Excellence | Data provided by Yahoo Finance")

if __name__ == "__main__":
    main()
