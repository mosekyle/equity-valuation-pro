import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from data.market_data import market_data
from utils.visualizations import create_market_overview_chart, create_performance_chart

def show():
    """Display the dashboard overview page."""
    
    st.header("📊 Market Overview")
    
    # Market indices
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Major Market Indices")
        display_market_indices()
    
    with col2:
        st.subheader("Market Statistics")
        display_market_stats()
    
    st.markdown("---")
    
    # Portfolio overview section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Quick Analysis")
        display_quick_analysis_form()
    
    with col2:
        st.subheader("📈 Recent Performance")
        display_recent_performance()

def display_market_indices():
    """Display major market indices."""
    
    try:
        indices_data = market_data.get_market_indices()
        
        if not indices_data:
            st.error("Unable to fetch market data. Please try again later.")
            return
        
        # Create metrics display
        cols = st.columns(len(indices_data))
        
        for i, (name, data) in enumerate(indices_data.items()):
            with cols[i % len(cols)]:
                price = data.get('price', 0)
                change = data.get('change', 0)
                change_pct = data.get('change_percent', 0)
                
                # Format the display
                price_str = f"{price:,.2f}" if price > 0 else "N/A"
                change_str = f"{change:+.2f}" if change != 0 else "0.00"
                change_pct_str = f"({change_pct:+.2f}%)" if change_pct != 0 else "(0.00%)"
                
                # Color based on performance
                delta_color = "normal" if change >= 0 else "inverse"
                
                st.metric(
                    label=name,
                    value=price_str,
                    delta=f"{change_str} {change_pct_str}",
                    delta_color=delta_color
                )
        
        # Create chart
        if indices_data:
            chart_data = []
            for name, data in indices_data.items():
                if data.get('price', 0) > 0:
                    chart_data.append({
                        'Index': name,
                        'Price': data['price'],
                        'Change %': data['change_percent']
                    })
            
            if chart_data:
                df = pd.DataFrame(chart_data)
                
                fig = px.bar(
                    df, 
                    x='Index', 
                    y='Change %',
                    title="Daily Performance (%)",
                    color='Change %',
                    color_continuous_scale=['red', 'white', 'green'],
                    color_continuous_midpoint=0
                )
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="Change (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error displaying market indices: {str(e)}")

def display_market_stats():
    """Display market statistics."""
    
    st.info("""
    **Market Insights**
    
    📊 Real-time data from Yahoo Finance
    
    🔄 Updates every 5 minutes during market hours
    
    📈 Historical data available for analysis
    """)
    
    # Current market status
    now = datetime.now()
    
    # Simple market hours check (approximate)
    is_market_hours = (now.weekday() < 5 and  # Monday to Friday
                      9 <= now.hour < 16)  # 9 AM to 4 PM (approximate)
    
    if is_market_hours:
        st.success("🟢 Market is currently **OPEN**")
    else:
        st.warning("🔴 Market is currently **CLOSED**")
    
    st.markdown(f"**Last Updated:** {now.strftime('%Y-%m-%d %H:%M:%S')}")

def display_quick_analysis_form():
    """Display quick analysis form."""
    
    with st.form("quick_analysis"):
        st.markdown("Enter a ticker symbol for quick valuation:")
        
        ticker = st.text_input(
            "Ticker Symbol",
            placeholder="e.g., AAPL, MSFT, GOOGL",
            help="Enter a valid stock ticker symbol"
        )
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Quick Overview", "DCF Analysis", "Comparable Analysis"],
            help="Select the type of analysis to perform"
        )
        
        submitted = st.form_submit_button("🔍 Analyze")
        
        if submitted and ticker:
            if analysis_type == "Quick Overview":
                perform_quick_overview(ticker.upper())
            else:
                st.info(f"Navigate to the {analysis_type} page for detailed analysis of {ticker.upper()}")

def perform_quick_overview(ticker: str):
    """Perform quick overview analysis."""
    
    try:
        with st.spinner(f"Fetching data for {ticker}..."):
            stock_info = market_data.get_stock_info(ticker)
            
            if 'error' in stock_info:
                st.error(f"Error fetching data for {ticker}: {stock_info['error']}")
                return
            
            # Display key metrics
            st.subheader(f"📊 {stock_info.get('longName', ticker)} ({ticker})")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${stock_info.get('currentPrice', 0):.2f}",
                    f"{stock_info.get('currentPrice', 0) - stock_info.get('previousClose', 0):.2f}"
                )
            
            with col2:
                st.metric(
                    "Market Cap",
                    f"${stock_info.get('marketCap', 0) / 1e9:.1f}B" if stock_info.get('marketCap', 0) > 0 else "N/A"
                )
            
            with col3:
                st.metric(
                    "P/E Ratio",
                    f"{stock_info.get('trailingPE', 0):.1f}" if stock_info.get('trailingPE', 0) > 0 else "N/A"
                )
            
            with col4:
                st.metric(
                    "Beta",
                    f"{stock_info.get('beta', 1.0):.2f}"
                )
            
            # Additional info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Company Info:**")
                st.write(f"• **Sector:** {stock_info.get('sector', 'Unknown')}")
                st.write(f"• **Industry:** {stock_info.get('industry', 'Unknown')}")
                st.write(f"• **Country:** {stock_info.get('country', 'Unknown')}")
            
            with col2:
                st.markdown("**Valuation Metrics:**")
                st.write(f"• **P/B Ratio:** {stock_info.get('priceToBook', 0):.2f}")
                st.write(f"• **EV/EBITDA:** {stock_info.get('enterpriseToEbitda', 0):.2f}")
                st.write(f"• **Dividend Yield:** {stock_info.get('dividendYield', 0)*100:.2f}%" if stock_info.get('dividendYield') else "• **Dividend Yield:** N/A")
    
    except Exception as e:
        st.error(f"Error performing quick analysis: {str(e)}")

def display_recent_performance():
    """Display recent performance of popular stocks."""
    
    # Popular stocks to track
    popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
    
    try:
        performance_data = []
        
        for ticker in popular_stocks:
            try:
                returns = market_data.calculate_returns(ticker, ['1d', '1wk', '1mo'])
                if returns:
                    performance_data.append({
                        'Ticker': ticker,
                        '1 Day': returns.get('1d', 0),
                        '1 Week': returns.get('1wk', 0),
                        '1 Month': returns.get('1mo', 0)
                    })
            except:
                continue
        
        if performance_data:
            df = pd.DataFrame(performance_data)
            
            # Display as table with color coding
            styled_df = df.style.format({
                '1 Day': '{:.2f}%',
                '1 Week': '{:.2f}%',
                '1 Month': '{:.2f}%'
            }).background_gradient(
                subset=['1 Day', '1 Week', '1 Month'],
                cmap='RdYlGn',
                vmin=-5,
                vmax=5
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Create performance chart
            fig = go.Figure()
            
            for period in ['1 Day', '1 Week', '1 Month']:
                fig.add_trace(go.Bar(
                    name=period,
                    x=df['Ticker'],
                    y=df[period],
                    text=[f"{val:.1f}%" for val in df[period]],
                    textposition='auto',
                ))
            
            fig.update_layout(
                title="Popular Stocks Performance",
                xaxis_title="Ticker",
                yaxis_title="Return (%)",
                barmode='group',
                height=300,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Unable to fetch performance data for popular stocks.")
    
    except Exception as e:
        st.error(f"Error displaying recent performance: {str(e)}")
