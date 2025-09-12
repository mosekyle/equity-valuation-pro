import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from data.market_data import market_data
from models.comps import ComparableAnalysis
from utils.visualizations import create_sector_performance_chart, create_valuation_comparison_chart

def show():
    """Display the sector comparison page."""
    
    st.header("📊 Sector Comparison & Analysis")
    
    # Sector selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        available_sectors = [
            'Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
            'Consumer Defensive', 'Energy', 'Industrials', 'Communication Services',
            'Utilities', 'Real Estate', 'Basic Materials'
        ]
        
        selected_sector = st.selectbox(
            "Select Sector for Analysis",
            available_sectors,
            help="Choose a sector to analyze and compare companies"
        )
    
    with col2:
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Quick Overview", "Detailed Analysis", "Custom Selection"],
            help="Select the depth of sector analysis"
        )
    
    if selected_sector:
        if analysis_depth == "Custom Selection":
            display_custom_selection(selected_sector)
        else:
            display_sector_analysis(selected_sector, analysis_depth == "Detailed Analysis")

def display_sector_analysis(sector: str, detailed: bool = False):
    """Display sector analysis."""
    
    st.markdown("---")
    st.subheader(f"📈 {sector} Sector Analysis")
    
    # Get sector companies
    with st.spinner(f"Fetching {sector} sector data..."):
        sector_companies = market_data.get_sector_companies(sector, 20 if detailed else 10)
        
        if not sector_companies:
            st.warning(f"No companies found for {sector} sector.")
            return
        
        # Fetch data for all companies
        sector_data = {}
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(sector_companies):
            try:
                company_info = market_data.get_stock_info(ticker)
                if 'error' not in company_info:
                    sector_data[ticker] = company_info
                progress_bar.progress((i + 1) / len(sector_companies))
            except:
                continue
        
        progress_bar.empty()
    
    if not sector_data:
        st.error("Unable to fetch sector data. Please try again later.")
        return
    
    # Display sector overview
    display_sector_overview(sector, sector_data)
    
    # Performance comparison
    display_sector_performance(sector_data)
    
    # Valuation metrics comparison
    display_valuation_comparison(sector_data)
    
    if detailed:
        # Financial health comparison
        display_financial_health_comparison(sector_data)
        
        # Sector leaders and laggards
        display_sector_rankings(sector_data)

def display_sector_overview(sector: str, sector_data: Dict):
    """Display sector overview metrics."""
    
    st.markdown("### 🎯 Sector Overview")
    
    # Calculate sector aggregates
    total_market_cap = sum(data.get('marketCap', 0) for data in sector_data.values())
    avg_pe_ratio = np.mean([data.get('trailingPE', 0) for data in sector_data.values() if data.get('trailingPE', 0) > 0])
    avg_beta = np.mean([data.get('beta', 1.0) for data in sector_data.values() if data.get('beta')])
    avg_div_yield = np.mean([data.get('dividendYield', 0) for data in sector_data.values() if data.get('dividendYield', 0) > 0])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Market Cap",
            f"${total_market_cap/1e12:.2f}T" if total_market_cap >= 1e12 else f"${total_market_cap/1e9:.1f}B"
        )
    
    with col2:
        st.metric(
            "Average P/E Ratio",
            f"{avg_pe_ratio:.1f}x" if avg_pe_ratio > 0 else "N/A"
        )
    
    with col3:
        st.metric(
            "Average Beta",
            f"{avg_beta:.2f}"
        )
    
    with col4:
        st.metric(
            "Average Dividend Yield",
            f"{avg_div_yield*100:.2f}%" if avg_div_yield > 0 else "N/A"
        )
    
    # Sector composition
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Market Cap Distribution")
        
        # Create market cap distribution chart
        market_cap_data = []
        for ticker, data in sector_data.items():
            market_cap = data.get('marketCap', 0)
            if market_cap > 0:
                market_cap_data.append({
                    'Company': data.get('longName', ticker)[:20] + "..." if len(data.get('longName', ticker)) > 20 else data.get('longName', ticker),
                    'Ticker': ticker,
                    'Market Cap': market_cap
                })
        
        if market_cap_data:
            market_cap_df = pd.DataFrame(market_cap_data).sort_values('Market Cap', ascending=False)
            
            fig = px.treemap(
                market_cap_df.head(10),
                path=['Company'],
                values='Market Cap',
                title="Market Cap Distribution (Top 10)"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Performance Distribution")
        
        # Calculate recent performance
        performance_data = []
        for ticker, data in sector_data.items():
            try:
                returns = market_data.calculate_returns(ticker, ['1mo'])
                if returns and '1mo' in returns:
                    performance_data.append({
                        'Company': data.get('longName', ticker)[:15] + "..." if len(data.get('longName', ticker)) > 15 else data.get('longName', ticker),
                        'Ticker': ticker,
                        '1 Month Return': returns['1mo']
                    })
            except:
                continue
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            
            fig = px.histogram(
                perf_df,
                x='1 Month Return',
                nbins=10,
                title="1-Month Return Distribution",
                color_discrete_sequence=['lightblue']
            )
            fig.update_layout(
                xaxis_title="1-Month Return (%)",
                yaxis_title="Number of Companies",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

def display_sector_performance(sector_data: Dict):
    """Display sector performance comparison."""
    
    st.markdown("### 📈 Performance Comparison")
    
    # Create performance comparison table
    performance_data = []
    
    for ticker, data in sector_data.items():
        try:
            returns = market_data.calculate_returns(ticker, ['1d', '1wk', '1mo', '3mo', '1y'])
            
            performance_data.append({
                'Company': data.get('longName', ticker),
                'Ticker': ticker,
                'Current Price': data.get('currentPrice', 0),
                '1 Day': returns.get('1d', 0),
                '1 Week': returns.get('1wk', 0),
                '1 Month': returns.get('1mo', 0),
                '3 Months': returns.get('3mo', 0),
                '1 Year': returns.get('1y', 0),
                'Beta': data.get('beta', 1.0)
            })
        except:
            performance_data.append({
                'Company': data.get('longName', ticker),
                'Ticker': ticker,
                'Current Price': data.get('currentPrice', 0),
                '1 Day': 0, '1 Week': 0, '1 Month': 0, '3 Months': 0, '1 Year': 0,
                'Beta': data.get('beta', 1.0)
            })
    
    if performance_data:
        perf_df = pd.DataFrame(performance_data)
        
        # Sort by 1-year performance
        perf_df = perf_df.sort_values('1 Year', ascending=False)
        
        # Format the display
        display_columns = ['Company', 'Ticker', 'Current Price', '1 Day', '1 Week', '1 Month', '3 Months', '1 Year', 'Beta']
        display_df = perf_df[display_columns].copy()
        
        # Style the dataframe
        def color_performance(val):
            if isinstance(val, (int, float)) and val != 0:
                if val > 5:
                    return 'background-color: lightgreen'
                elif val > 0:
                    return 'background-color: lightblue'
                elif val > -5:
                    return 'background-color: lightyellow'
                else:
                    return 'background-color: lightcoral'
            return ''
        
        styled_df = display_df.style.format({
            'Current Price': '${:.2f}',
            '1 Day': '{:.2f}%',
            '1 Week': '{:.2f}%',
            '1 Month': '{:.2f}%',
            '3 Months': '{:.2f}%',
            '1 Year': '{:.2f}%',
            'Beta': '{:.2f}'
        }).applymap(color_performance, subset=['1 Day', '1 Week', '1 Month', '3 Months', '1 Year'])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Create performance chart
        fig = go.Figure()
        
        periods = ['1 Day', '1 Week', '1 Month', '3 Months', '1 Year']
        for period in periods:
            fig.add_trace(go.Box(
                y=perf_df[period],
                name=period,
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title="Performance Distribution Across Time Periods",
            yaxis_title="Return (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_valuation_comparison(sector_data: Dict):
    """Display valuation metrics comparison."""
    
    st.markdown("### 💰 Valuation Metrics Comparison")
    
    # Create valuation comparison table
    valuation_data = []
    
    for ticker, data in sector_data.items():
        valuation_data.append({
            'Company': data.get('longName', ticker),
            'Ticker': ticker,
            'Market Cap': data.get('marketCap', 0),
            'P/E Ratio': data.get('trailingPE', 0),
            'Forward P/E': data.get('forwardPE', 0),
            'P/B Ratio': data.get('priceToBook', 0),
            'P/S Ratio': data.get('priceToSalesTrailing12Months', 0),
            'EV/EBITDA': data.get('enterpriseToEbitda', 0),
            'EV/Revenue': data.get('enterpriseToRevenue', 0),
            'PEG Ratio': data.get('pegRatio', 0)
        })
    
    if valuation_data:
        val_df = pd.DataFrame(valuation_data)
        
        # Create summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Valuation Summary Statistics")
            
            summary_stats = []
            valuation_metrics = ['P/E Ratio', 'P/B Ratio', 'P/S Ratio', 'EV/EBITDA']
            
            for metric in valuation_metrics:
                values = val_df[metric][val_df[metric] > 0]
                if len(values) > 0:
                    summary_stats.append({
                        'Metric': metric,
                        'Median': f"{values.median():.1f}x",
                        'Mean': f"{values.mean():.1f}x",
                        'Min': f"{values.min():.1f}x",
                        'Max': f"{values.max():.1f}x"
                    })
            
            if summary_stats:
                summary_df = pd.DataFrame(summary_stats)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### Valuation Distribution")
            
            # Create valuation scatter plot
            fig = px.scatter(
                val_df[val_df['P/E Ratio'] > 0],
                x='P/E Ratio',
                y='P/B Ratio',
                size='Market Cap',
                hover_name='Company',
                title="P/E vs P/B Ratio (Size = Market Cap)",
                color='EV/EBITDA',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Full valuation table
        st.markdown("#### Complete Valuation Metrics")
        
        # Format and display the full table
        display_val_df = val_df.copy()
        
        # Format market cap
        display_val_df['Market Cap'] = display_val_df['Market Cap'].apply(
            lambda x: f"${x/1e9:.1f}B" if x >= 1e9 else f"${x/1e6:.1f}M" if x >= 1e6 else f"${x:,.0f}"
        )
        
        # Format ratios
        ratio_cols = ['P/E Ratio', 'Forward P/E', 'P/B Ratio', 'P/S Ratio', 'EV/EBITDA', 'EV/Revenue', 'PEG Ratio']
        for col in ratio_cols:
            display_val_df[col] = display_val_df[col].apply(
                lambda x: f"{x:.1f}x" if x > 0 else "N/A"
            )
        
        st.dataframe(display_val_df, use_container_width=True, hide_index=True)

def display_financial_health_comparison(sector_data: Dict):
    """Display financial health metrics comparison."""
    
    st.markdown("### 🏥 Financial Health Comparison")
    
    # Create financial health table
    health_data = []
    
    for ticker, data in sector_data.items():
        health_data.append({
            'Company': data.get('longName', ticker),
            'Ticker': ticker,
            'ROE': data.get('returnOnEquity', 0),
            'ROA': data.get('returnOnAssets', 0),
            'Debt/Equity': data.get('debtToEquity', 0) / 100 if data.get('debtToEquity') else 0,
            'Current Ratio': data.get('currentRatio', 0),
            'Quick Ratio': data.get('quickRatio', 0),
            'Gross Margin': data.get('grossMargins', 0),
            'Operating Margin': data.get('operatingMargins', 0),
            'Profit Margin': data.get('profitMargins', 0),
            'Revenue Growth': data.get('revenueGrowth', 0),
            'Earnings Growth': data.get('earningsGrowth', 0)
        })
    
    if health_data:
        health_df = pd.DataFrame(health_data)
        
        # Create health score
        def calculate_health_score(row):
            score = 0
            # ROE score (higher is better)
            if row['ROE'] > 0.20: score += 2
            elif row['ROE'] > 0.15: score += 1
            elif row['ROE'] > 0.10: score += 0.5
            
            # Debt/Equity score (lower is better)
            if row['Debt/Equity'] < 0.3: score += 2
            elif row['Debt/Equity'] < 0.5: score += 1
            elif row['Debt/Equity'] < 1.0: score += 0.5
            
            # Profit Margin score (higher is better)
            if row['Profit Margin'] > 0.20: score += 2
            elif row['Profit Margin'] > 0.10: score += 1
            elif row['Profit Margin'] > 0.05: score += 0.5
            
            # Revenue Growth score (higher is better)
            if row['Revenue Growth'] > 0.20: score += 2
            elif row['Revenue Growth'] > 0.10: score += 1
            elif row['Revenue Growth'] > 0.05: score += 0.5
            
            return score
        
        health_df['Health Score'] = health_df.apply(calculate_health_score, axis=1)
        health_df = health_df.sort_values('Health Score', ascending=False)
        
        # Display top performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Top Financial Performers")
            top_performers = health_df.head(5)[['Company', 'Ticker', 'ROE', 'Profit Margin', 'Revenue Growth', 'Health Score']].copy()
            
            # Format percentages
            top_performers['ROE'] = top_performers['ROE'].apply(lambda x: f"{x*100:.1f}%" if x != 0 else "N/A")
            top_performers['Profit Margin'] = top_performers['Profit Margin'].apply(lambda x: f"{x*100:.1f}%" if x != 0 else "N/A")
            top_performers['Revenue Growth'] = top_performers['Revenue Growth'].apply(lambda x: f"{x*100:.1f}%" if x != 0 else "N/A")
            top_performers['Health Score'] = top_performers['Health Score'].apply(lambda x: f"{x:.1f}")
            
            st.dataframe(top_performers, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 📊 Health Score Distribution")
            
            fig = px.histogram(
                health_df,
                x='Health Score',
                nbins=10,
                title="Financial Health Score Distribution"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Full health metrics table
        st.markdown("#### Complete Financial Health Metrics")
        
        # Format the display
        display_health_df = health_df.copy()
        
        percentage_cols = ['ROE', 'ROA', 'Gross Margin', 'Operating Margin', 'Profit Margin', 'Revenue Growth', 'Earnings Growth']
        for col in percentage_cols:
            display_health_df[col] = display_health_df[col].apply(
                lambda x: f"{x*100:.1f}%" if x != 0 else "N/A"
            )
        
        ratio_cols = ['Debt/Equity', 'Current Ratio', 'Quick Ratio']
        for col in ratio_cols:
            display_health_df[col] = display_health_df[col].apply(
                lambda x: f"{x:.2f}" if x > 0 else "N/A"
            )
        
        display_health_df['Health Score'] = display_health_df['Health Score'].apply(lambda x: f"{x:.1f}")
        
        st.dataframe(display_health_df, use_container_width=True, hide_index=True)

def display_sector_rankings(sector_data: Dict):
    """Display sector company rankings."""
    
    st.markdown("### 🏅 Sector Rankings")
    
    # Create comprehensive ranking
    ranking_data = []
    
    for ticker, data in sector_data.items():
        ranking_data.append({
            'Company': data.get('longName', ticker),
            'Ticker': ticker,
            'Market Cap': data.get('marketCap', 0),
            'Revenue Growth': data.get('revenueGrowth', 0),
            'ROE': data.get('returnOnEquity', 0),
            'Profit Margin': data.get('profitMargins', 0),
            'P/E Ratio': data.get('trailingPE', 0),
            'Beta': data.get('beta', 1.0),
            'Dividend Yield': data.get('dividendYield', 0)
        })
    
    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        
        # Create different ranking categories
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📈 Growth Leaders")
            growth_leaders = ranking_df[ranking_df['Revenue Growth'] > 0].nlargest(5, 'Revenue Growth')[
                ['Company', 'Ticker', 'Revenue Growth']
            ].copy()
            growth_leaders['Revenue Growth'] = growth_leaders['Revenue Growth'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(growth_leaders, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 💰 Value Plays")
            value_plays = ranking_df[ranking_df['P/E Ratio'] > 0].nsmallest(5, 'P/E Ratio')[
                ['Company', 'Ticker', 'P/E Ratio']
            ].copy()
            value_plays['P/E Ratio'] = value_plays['P/E Ratio'].apply(lambda x: f"{x:.1f}x")
            st.dataframe(value_plays, use_container_width=True, hide_index=True)
        
        with col3:
            st.markdown("#### 💎 Dividend Champions")
            dividend_champions = ranking_df[ranking_df['Dividend Yield'] > 0].nlargest(5, 'Dividend Yield')[
                ['Company', 'Ticker', 'Dividend Yield']
            ].copy()
            dividend_champions['Dividend Yield'] = dividend_champions['Dividend Yield'].apply(lambda x: f"{x*100:.2f}%")
            st.dataframe(dividend_champions, use_container_width=True, hide_index=True)

def display_custom_selection(sector: str):
    """Display custom company selection for sector analysis."""
    
    st.markdown("---")
    st.subheader(f"🎯 Custom {sector} Analysis")
    
    # Get available companies
    available_companies = market_data.get_sector_companies(sector, 50)
    
    if not available_companies:
        st.warning(f"No companies found for {sector} sector.")
        return
    
    # Company selection
    st.markdown("#### Select Companies for Analysis")
    
    selected_companies = st.multiselect(
        "Choose companies to analyze",
        available_companies,
        default=available_companies[:5],
        help="Select 3-15 companies for detailed comparison"
    )
    
    if len(selected_companies) < 3:
        st.warning("Please select at least 3 companies for meaningful comparison.")
        return
    
    if len(selected_companies) > 15:
        st.warning("Please select no more than 15 companies to ensure optimal performance.")
        selected_companies = selected_companies[:15]
    
    # Analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_metrics = st.multiselect(
            "Select Analysis Metrics",
            [
                "Valuation Multiples", "Financial Health", "Growth Metrics",
                "Profitability", "Risk Metrics", "Dividend Analysis"
            ],
            default=["Valuation Multiples", "Financial Health", "Growth Metrics"]
        )
    
    with col2:
        comparison_base = st.selectbox(
            "Comparison Base",
            ["Absolute Values", "Percentile Rankings", "Z-Score Normalization"],
            help="How to display the comparison metrics"
        )
    
    if st.button("🚀 Run Custom Analysis", type="primary"):
        with st.spinner("Running custom sector analysis..."):
            run_custom_analysis(selected_companies, analysis_metrics, comparison_base)

def run_custom_analysis(companies: List[str], metrics: List[str], comparison_base: str):
    """Run custom sector analysis with selected parameters."""
    
    # Fetch data for selected companies
    company_data = {}
    progress_bar = st.progress(0)
    
    for i, ticker in enumerate(companies):
        try:
            company_info = market_data.get_stock_info(ticker)
            if 'error' not in company_info:
                company_data[ticker] = company_info
            progress_bar.progress((i + 1) / len(companies))
        except:
            continue
    
    progress_bar.empty()
    
    if not company_data:
        st.error("Unable to fetch data for selected companies.")
        return
    
    st.success(f"Successfully loaded data for {len(company_data)} companies.")
    
    # Display analysis based on selected metrics
    if "Valuation Multiples" in metrics:
        st.markdown("### 💰 Valuation Multiples Analysis")
        display_custom_valuation_analysis(company_data, comparison_base)
    
    if "Financial Health" in metrics:
        st.markdown("### 🏥 Financial Health Analysis")
        display_custom_health_analysis(company_data, comparison_base)
    
    if "Growth Metrics" in metrics:
        st.markdown("### 📈 Growth Metrics Analysis")
        display_custom_growth_analysis(company_data, comparison_base)
    
    if "Profitability" in metrics:
        st.markdown("### 💎 Profitability Analysis")
        display_custom_profitability_analysis(company_data, comparison_base)
    
    if "Risk Metrics" in metrics:
        st.markdown("### ⚠️ Risk Metrics Analysis")
        display_custom_risk_analysis(company_data, comparison_base)
    
    if "Dividend Analysis" in metrics:
        st.markdown("### 💰 Dividend Analysis")
        display_custom_dividend_analysis(company_data, comparison_base)

def display_custom_valuation_analysis(company_data: Dict, comparison_base: str):
    """Display custom valuation analysis."""
    
    # Create valuation data
    val_metrics = ['trailingPE', 'forwardPE', 'priceToBook', 'priceToSalesTrailing12Months', 'enterpriseToEbitda']
    val_names = ['P/E Ratio', 'Forward P/E', 'P/B Ratio', 'P/S Ratio', 'EV/EBITDA']
    
    val_data = []
    for ticker, data in company_data.items():
        row = {'Company': data.get('longName', ticker), 'Ticker': ticker}
        for metric, name in zip(val_metrics, val_names):
            row[name] = data.get(metric, 0)
        val_data.append(row)
    
    val_df = pd.DataFrame(val_data)
    
    if comparison_base == "Percentile Rankings":
        # Convert to percentile rankings
        for name in val_names:
            val_df[f"{name} Rank"] = val_df[name].rank(pct=True) * 100
        
        # Display rankings
        rank_cols = [col for col in val_df.columns if 'Rank' in col]
        display_df = val_df[['Company', 'Ticker'] + rank_cols].copy()
        
        # Format rankings
        for col in rank_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}%" if x > 0 else "N/A")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    elif comparison_base == "Z-Score Normalization":
        # Convert to z-scores
        for name in val_names:
            values = val_df[name][val_df[name] > 0]
            if len(values) > 1:
                mean_val = values.mean()
                std_val = values.std()
                val_df[f"{name} Z-Score"] = (val_df[name] - mean_val) / std_val
        
        # Display z-scores
        zscore_cols = [col for col in val_df.columns if 'Z-Score' in col]
        display_df = val_df[['Company', 'Ticker'] + zscore_cols].copy()
        
        # Format z-scores
        for col in zscore_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    else:  # Absolute Values
        # Format absolute values
        display_df = val_df.copy()
        for name in val_names:
            display_df[name] = display_df[name].apply(lambda x: f"{x:.1f}x" if x > 0 else "N/A")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Create visualization
    if len(val_df) > 2:
        fig = px.scatter_matrix(
            val_df,
            dimensions=val_names[:4],  # Use first 4 metrics
            hover_name='Company',
            title="Valuation Metrics Correlation Matrix"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

def display_custom_health_analysis(company_data: Dict, comparison_base: str):
    """Display custom financial health analysis."""
    
    # Create health metrics data
    health_metrics = ['returnOnEquity', 'returnOnAssets', 'grossMargins', 'operatingMargins', 'profitMargins']
    health_names = ['ROE', 'ROA', 'Gross Margin', 'Operating Margin', 'Profit Margin']
    
    health_data = []
    for ticker, data in company_data.items():
        row = {'Company': data.get('longName', ticker), 'Ticker': ticker}
        for metric, name in zip(health_metrics, health_names):
            row[name] = data.get(metric, 0)
        health_data.append(row)
    
    health_df = pd.DataFrame(health_data)
    
    # Format and display based on comparison base
    if comparison_base == "Absolute Values":
        display_df = health_df.copy()
        for name in health_names:
            display_df[name] = display_df[name].apply(lambda x: f"{x*100:.2f}%" if x != 0 else "N/A")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Create health score radar chart for top 5 companies
    top_companies = health_df.nlargest(5, 'ROE')
    
    if len(top_companies) > 0:
        fig = go.Figure()
        
        for _, company in top_companies.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[company[name]*100 for name in health_names],
                theta=health_names,
                fill='toself',
                name=company['Company'][:20] + "..." if len(company['Company']) > 20 else company['Company']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max([health_df[name].max()*100 for name in health_names])]
                )),
            showlegend=True,
            title="Financial Health Comparison (Top 5 by ROE)"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_custom_growth_analysis(company_data: Dict, comparison_base: str):
    """Display custom growth analysis."""
    
    growth_data = []
    for ticker, data in company_data.items():
        growth_data.append({
            'Company': data.get('longName', ticker),
            'Ticker': ticker,
            'Revenue Growth': data.get('revenueGrowth', 0),
            'Earnings Growth': data.get('earningsGrowth', 0)
        })
    
    growth_df = pd.DataFrame(growth_data)
    
    # Create growth comparison chart
    fig = px.scatter(
        growth_df,
        x='Revenue Growth',
        y='Earnings Growth',
        hover_name='Company',
        title="Revenue Growth vs Earnings Growth",
        labels={'Revenue Growth': 'Revenue Growth (%)', 'Earnings Growth': 'Earnings Growth (%)'}
    )
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Format and display table
    display_df = growth_df.copy()
    display_df['Revenue Growth'] = display_df['Revenue Growth'].apply(lambda x: f"{x*100:.2f}%" if x != 0 else "N/A")
    display_df['Earnings Growth'] = display_df['Earnings Growth'].apply(lambda x: f"{x*100:.2f}%" if x != 0 else "N/A")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def display_custom_profitability_analysis(company_data: Dict, comparison_base: str):
    """Display custom profitability analysis."""
    
    prof_data = []
    for ticker, data in company_data.items():
        prof_data.append({
            'Company': data.get('longName', ticker),
            'Ticker': ticker,
            'Gross Margin': data.get('grossMargins', 0),
            'Operating Margin': data.get('operatingMargins', 0),
            'Profit Margin': data.get('profitMargins', 0),
            'ROE': data.get('returnOnEquity', 0),
            'ROA': data.get('returnOnAssets', 0)
        })
    
    prof_df = pd.DataFrame(prof_data)
    
    # Create profitability waterfall chart
    fig = go.Figure()
    
    for _, company in prof_df.head(10).iterrows():  # Show top 10
        fig.add_trace(go.Bar(
            name=company['Company'][:15] + "..." if len(company['Company']) > 15 else company['Company'],
            x=['Gross Margin', 'Operating Margin', 'Profit Margin'],
            y=[company['Gross Margin']*100, company['Operating Margin']*100, company['Profit Margin']*100]
        ))
    
    fig.update_layout(
        title="Profitability Margins Comparison",
        xaxis_title="Margin Type",
        yaxis_title="Margin (%)",
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_custom_risk_analysis(company_data: Dict, comparison_base: str):
    """Display custom risk analysis."""
    
    risk_data = []
    for ticker, data in company_data.items():
        risk_data.append({
            'Company': data.get('longName', ticker),
            'Ticker': ticker,
            'Beta': data.get('beta', 1.0),
            'Debt/Equity': data.get('debtToEquity', 0) / 100 if data.get('debtToEquity') else 0
        })
    
    risk_df = pd.DataFrame(risk_data)
    
    # Create risk matrix
    fig = px.scatter(
        risk_df,
        x='Beta',
        y='Debt/Equity',
        hover_name='Company',
        title="Risk Matrix: Beta vs Debt/Equity Ratio",
        labels={'Beta': 'Beta (Market Risk)', 'Debt/Equity': 'Debt/Equity Ratio'}
    )
    
    # Add risk quadrant lines
    fig.add_hline(y=risk_df['Debt/Equity'].median(), line_dash="dash", line_color="gray")
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

def display_custom_dividend_analysis(company_data: Dict, comparison_base: str):
    """Display custom dividend analysis."""
    
    div_data = []
    for ticker, data in company_data.items():
        div_yield = data.get('dividendYield', 0)
        payout_ratio = data.get('payoutRatio', 0)
        
        if div_yield > 0:  # Only include dividend-paying companies
            div_data.append({
                'Company': data.get('longName', ticker),
                'Ticker': ticker,
                'Dividend Yield': div_yield,
                'Payout Ratio': payout_ratio,
                'Dividend Rate': data.get('dividendRate', 0)
            })
    
    if div_data:
        div_df = pd.DataFrame(div_data)
        
        # Create dividend yield vs payout ratio chart
        fig = px.scatter(
            div_df,
            x='Dividend Yield',
            y='Payout Ratio',
            size='Dividend Rate',
            hover_name='Company',
            title="Dividend Yield vs Payout Ratio",
            labels={'Dividend Yield': 'Dividend Yield (%)', 'Payout Ratio': 'Payout Ratio (%)'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Format and display table
        display_df = div_df.copy()
        display_df['Dividend Yield'] = display_df['Dividend Yield'].apply(lambda x: f"{x*100:.2f}%")
        display_df['Payout Ratio'] = display_df['Payout Ratio'].apply(lambda x: f"{x*100:.2f}%" if x > 0 else "N/A")
        display_df['Dividend Rate'] = display_df['Dividend Rate'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No dividend-paying companies found in the selected group.")
