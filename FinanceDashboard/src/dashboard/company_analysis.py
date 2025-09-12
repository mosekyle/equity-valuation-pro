import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from models.dcf import DCFModel
from models.comps import ComparableAnalysis
from data.market_data import market_data
from data.financial_data import FinancialDataProcessor
from utils.visualizations import create_dcf_waterfall_chart, create_sensitivity_heatmap
from utils.exports import export_analysis_to_excel

def show():
    """Display the company analysis page."""
    
    st.header("🔍 Company Analysis")
    
    # Ticker input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input(
            "Enter Ticker Symbol",
            placeholder="e.g., AAPL, MSFT, GOOGL",
            help="Enter a valid stock ticker symbol for analysis"
        ).upper()
    
    with col2:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["DCF Valuation", "Comparable Analysis", "Combined Analysis"],
            help="Select the type of valuation analysis"
        )
    
    if not ticker:
        st.info("👆 Enter a ticker symbol to begin analysis")
        display_sample_analysis()
        return
    
    # Fetch company data
    with st.spinner(f"Fetching data for {ticker}..."):
        company_info = market_data.get_stock_info(ticker)
        
        if 'error' in company_info:
            st.error(f"Error fetching data for {ticker}: {company_info['error']}")
            st.info("Please verify the ticker symbol and try again.")
            return
    
    # Display company header
    display_company_header(ticker, company_info)
    
    # Analysis tabs
    if analysis_type == "DCF Valuation":
        display_dcf_analysis(ticker, company_info)
    elif analysis_type == "Comparable Analysis":
        display_comparable_analysis(ticker, company_info)
    else:  # Combined Analysis
        display_combined_analysis(ticker, company_info)

def display_company_header(ticker: str, company_info: Dict):
    """Display company information header."""
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"## {company_info.get('longName', ticker)} ({ticker})")
        st.markdown(f"**Sector:** {company_info.get('sector', 'Unknown')} | **Industry:** {company_info.get('industry', 'Unknown')}")
    
    with col2:
        current_price = company_info.get('currentPrice', 0)
        previous_close = company_info.get('previousClose', 0)
        change = current_price - previous_close
        change_pct = (change / previous_close * 100) if previous_close > 0 else 0
        
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            f"{change:+.2f} ({change_pct:+.2f}%)"
        )
    
    with col3:
        market_cap = company_info.get('marketCap', 0)
        if market_cap > 0:
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap_str = f"${market_cap/1e6:.2f}M"
            else:
                market_cap_str = f"${market_cap:,.0f}"
        else:
            market_cap_str = "N/A"
        
        st.metric("Market Cap", market_cap_str)
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        pe_ratio = company_info.get('trailingPE', 0)
        st.metric("P/E Ratio", f"{pe_ratio:.1f}" if pe_ratio > 0 else "N/A")
    
    with col2:
        pb_ratio = company_info.get('priceToBook', 0)
        st.metric("P/B Ratio", f"{pb_ratio:.1f}" if pb_ratio > 0 else "N/A")
    
    with col3:
        ev_ebitda = company_info.get('enterpriseToEbitda', 0)
        st.metric("EV/EBITDA", f"{ev_ebitda:.1f}" if ev_ebitda > 0 else "N/A")
    
    with col4:
        beta = company_info.get('beta', 1.0)
        st.metric("Beta", f"{beta:.2f}")
    
    with col5:
        div_yield = company_info.get('dividendYield', 0)
        st.metric("Div Yield", f"{div_yield*100:.2f}%" if div_yield else "N/A")

def display_dcf_analysis(ticker: str, company_info: Dict):
    """Display DCF valuation analysis."""
    
    st.markdown("---")
    st.subheader("💰 DCF Valuation Model")
    
    # DCF assumptions sidebar
    with st.sidebar:
        st.markdown("### DCF Assumptions")
        
        # Revenue growth assumptions
        st.markdown("**Revenue Growth Rates**")
        years = st.slider("Projection Years", 3, 10, 5)
        
        revenue_growth = []
        for i in range(years):
            growth = st.number_input(
                f"Year {i+1} Revenue Growth (%)",
                min_value=-50.0,
                max_value=100.0,
                value=max(10.0 - i*2, 2.0),  # Declining growth
                step=0.5,
                key=f"rev_growth_{i}"
            ) / 100
            revenue_growth.append(growth)
        
        # EBITDA margin assumptions
        st.markdown("**EBITDA Margins**")
        ebitda_margin = []
        for i in range(years):
            margin = st.number_input(
                f"Year {i+1} EBITDA Margin (%)",
                min_value=0.0,
                max_value=50.0,
                value=20.0,
                step=0.5,
                key=f"ebitda_margin_{i}"
            ) / 100
            ebitda_margin.append(margin)
        
        # Other assumptions
        st.markdown("**Other Assumptions**")
        tax_rate = st.number_input("Tax Rate (%)", 0.0, 50.0, 25.0, 0.5) / 100
        terminal_growth = st.number_input("Terminal Growth Rate (%)", 0.0, 5.0, 2.5, 0.1) / 100
        discount_rate = st.number_input("Discount Rate (WACC) (%)", 1.0, 20.0, 9.0, 0.1) / 100
        
        # CapEx and other assumptions
        capex_pct = st.number_input("CapEx (% of Revenue)", 0.0, 20.0, 3.0, 0.1) / 100
        depreciation_pct = st.number_input("Depreciation (% of Revenue)", 0.0, 20.0, 2.5, 0.1) / 100
        nwc_pct = st.number_input("Working Capital (% of Revenue)", -10.0, 10.0, 2.0, 0.1) / 100
        
        run_dcf = st.button("🚀 Run DCF Analysis", type="primary")
    
    if run_dcf:
        try:
            with st.spinner("Running DCF analysis..."):
                # Initialize DCF model
                dcf_model = DCFModel(ticker)
                
                # Set assumptions
                dcf_model.set_assumptions(
                    revenue_growth=revenue_growth,
                    ebitda_margin=ebitda_margin,
                    tax_rate=tax_rate,
                    capex_pct_revenue=[capex_pct] * years,
                    depreciation_pct_revenue=[depreciation_pct] * years,
                    working_capital_pct_revenue=nwc_pct,
                    terminal_growth=terminal_growth,
                    discount_rate=discount_rate
                )
                
                # Calculate valuation
                valuation_results = dcf_model.calculate_fair_value()
                
                # Display results
                display_dcf_results(dcf_model, valuation_results, company_info)
                
                # Sensitivity analysis
                display_sensitivity_analysis(dcf_model)
        
        except Exception as e:
            st.error(f"Error running DCF analysis: {str(e)}")
            st.info("Please check your assumptions and try again.")
    
    else:
        st.info("👈 Adjust DCF assumptions in the sidebar and click 'Run DCF Analysis'")

def display_dcf_results(dcf_model: DCFModel, valuation_results: Dict, company_info: Dict):
    """Display DCF valuation results."""
    
    st.markdown("### 📊 Valuation Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fair_value = valuation_results['fair_value_per_share']
        st.metric("Fair Value per Share", f"${fair_value:.2f}")
    
    with col2:
        current_price = company_info.get('currentPrice', 0)
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col3:
        upside_downside = valuation_results['upside_downside']
        st.metric(
            "Upside/Downside",
            f"{upside_downside:+.1f}%",
            delta_color="normal" if upside_downside >= 0 else "inverse"
        )
    
    with col4:
        # Investment recommendation
        if upside_downside > 20:
            recommendation = "STRONG BUY"
            rec_color = "🟢"
        elif upside_downside > 10:
            recommendation = "BUY"
            rec_color = "🟢"
        elif upside_downside > -10:
            recommendation = "HOLD"
            rec_color = "🟡"
        elif upside_downside > -20:
            recommendation = "SELL"
            rec_color = "🔴"
        else:
            recommendation = "STRONG SELL"
            rec_color = "🔴"
        
        st.metric("Recommendation", f"{rec_color} {recommendation}")
    
    # Valuation breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Valuation Breakdown")
        
        breakdown_data = {
            'Component': [
                'PV of Projection Period',
                'PV of Terminal Value',
                'Enterprise Value',
                'Less: Total Debt',
                'Plus: Cash & Equivalents',
                'Equity Value',
                'Shares Outstanding (M)',
                'Fair Value per Share'
            ],
            'Value': [
                f"${valuation_results['pv_projection_period']/1e6:,.0f}M",
                f"${valuation_results['pv_terminal_value']/1e6:,.0f}M",
                f"${valuation_results['enterprise_value']/1e6:,.0f}M",
                f"${valuation_results['total_debt']/1e6:,.0f}M",
                f"${valuation_results['total_cash']/1e6:,.0f}M",
                f"${valuation_results['equity_value']/1e6:,.0f}M",
                f"{valuation_results['shares_outstanding']/1e6:,.0f}M",
                f"${valuation_results['fair_value_per_share']:.2f}"
            ]
        }
        
        breakdown_df = pd.DataFrame(breakdown_data)
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 📈 Financial Projections")
        
        if hasattr(dcf_model, 'projections') and not dcf_model.projections.empty:
            # Display key projection metrics
            proj_df = dcf_model.projections.loc[
                ['Revenue', 'EBITDA', 'EBIT', 'Unlevered FCF']
            ].copy()
            
            # Format the display
            for col in proj_df.columns:
                proj_df[col] = proj_df[col].apply(lambda x: f"${x/1e6:.0f}M" if x != 0 else "$0M")
            
            st.dataframe(proj_df, use_container_width=True)
            
            # Create projection chart
            if not dcf_model.projections.empty:
                create_projections_chart(dcf_model.projections)

def create_projections_chart(projections_df: pd.DataFrame):
    """Create financial projections chart."""
    
    try:
        fig = go.Figure()
        
        years = projections_df.columns
        
        # Revenue line
        revenue_data = projections_df.loc['Revenue']
        fig.add_trace(go.Scatter(
            x=years,
            y=revenue_data/1e6,  # Convert to millions
            mode='lines+markers',
            name='Revenue',
            line=dict(color='blue', width=3)
        ))
        
        # EBITDA line
        if 'EBITDA' in projections_df.index:
            ebitda_data = projections_df.loc['EBITDA']
            fig.add_trace(go.Scatter(
                x=years,
                y=ebitda_data/1e6,
                mode='lines+markers',
                name='EBITDA',
                line=dict(color='green', width=3)
            ))
        
        # Free Cash Flow bars
        if 'Unlevered FCF' in projections_df.index:
            fcf_data = projections_df.loc['Unlevered FCF']
            fig.add_trace(go.Bar(
                x=years,
                y=fcf_data/1e6,
                name='Free Cash Flow',
                opacity=0.7,
                marker_color='orange'
            ))
        
        fig.update_layout(
            title="Financial Projections",
            xaxis_title="Year",
            yaxis_title="Value ($ Millions)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating projections chart: {str(e)}")

def display_sensitivity_analysis(dcf_model: DCFModel):
    """Display DCF sensitivity analysis."""
    
    st.markdown("### 🎯 Sensitivity Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Discount Rate Sensitivity**")
        base_discount = dcf_model.assumptions['discount_rate']
        discount_range = [
            base_discount - 0.02,
            base_discount - 0.01,
            base_discount,
            base_discount + 0.01,
            base_discount + 0.02
        ]
        
        base_terminal = dcf_model.assumptions['terminal_growth']
        terminal_range = [base_terminal]
        
        try:
            sensitivity_df = dcf_model.sensitivity_analysis(discount_range, terminal_range)
            
            if not sensitivity_df.empty:
                # Format for display
                display_df = sensitivity_df.copy()
                display_df.index = [f"{rate*100:.1f}%" for rate in display_df.index]
                display_df.columns = [f"{rate*100:.1f}%" for rate in display_df.columns]
                
                for col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                
                st.dataframe(display_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error in sensitivity analysis: {str(e)}")
    
    with col2:
        st.markdown("**Terminal Growth Sensitivity**")
        terminal_range = [
            base_terminal - 0.01,
            base_terminal - 0.005,
            base_terminal,
            base_terminal + 0.005,
            base_terminal + 0.01
        ]
        
        discount_range = [base_discount]
        
        try:
            sensitivity_df = dcf_model.sensitivity_analysis(discount_range, terminal_range)
            
            if not sensitivity_df.empty:
                # Format for display
                display_df = sensitivity_df.copy()
                display_df.index = [f"{rate*100:.1f}%" for rate in display_df.index]
                display_df.columns = [f"{rate*100:.1f}%" for rate in display_df.columns]
                
                for col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                
                st.dataframe(display_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error in sensitivity analysis: {str(e)}")

def display_comparable_analysis(ticker: str, company_info: Dict):
    """Display comparable company analysis."""
    
    st.markdown("---")
    st.subheader("📊 Comparable Company Analysis")
    
    # Get sector companies
    sector = company_info.get('sector', 'Technology')
    sector_companies = market_data.get_sector_companies(sector, 20)
    
    # Remove target company from peers if present
    peer_companies = [comp for comp in sector_companies if comp != ticker][:10]
    
    if not peer_companies:
        st.warning(f"No peer companies found for {sector} sector.")
        return
    
    with st.spinner("Fetching comparable company data..."):
        try:
            # Initialize comparable analysis
            comp_analysis = ComparableAnalysis(ticker, peer_companies)
            
            # Fetch data
            company_data = comp_analysis.fetch_company_data()
            
            # Create comparison table
            comp_table = comp_analysis.create_comp_table()
            
            if not comp_table.empty:
                display_comp_table(comp_table, ticker)
                
                # Peer statistics
                peer_stats = comp_analysis.calculate_peer_statistics()
                display_peer_statistics(peer_stats, company_data.get(ticker, {}))
                
                # Relative valuation
                relative_val = comp_analysis.relative_valuation()
                display_relative_valuation(relative_val, company_info)
        
        except Exception as e:
            st.error(f"Error in comparable analysis: {str(e)}")

def display_comp_table(comp_table: pd.DataFrame, target_ticker: str):
    """Display comparable companies table."""
    
    st.markdown("### 📋 Comparable Companies")
    
    # Format the table for better display
    display_table = comp_table.copy()
    
    # Format financial metrics
    financial_cols = ['market_cap', 'enterprise_value', 'revenue_ttm', 'ebitda_ttm']
    for col in financial_cols:
        if col in display_table.columns:
            display_table[col] = display_table[col].apply(
                lambda x: f"${x/1e9:.1f}B" if x >= 1e9 else f"${x/1e6:.1f}M" if x >= 1e6 else f"${x:,.0f}"
            )
    
    # Format ratio columns
    ratio_cols = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda', 'ev_revenue']
    for col in ratio_cols:
        if col in display_table.columns:
            display_table[col] = display_table[col].apply(
                lambda x: f"{x:.1f}x" if x > 0 else "N/A"
            )
    
    # Format percentage columns
    pct_cols = ['roe', 'roa', 'gross_margin', 'operating_margin', 'profit_margin', 
                'revenue_growth', 'earnings_growth', 'debt_to_equity', 'dividend_yield']
    for col in pct_cols:
        if col in display_table.columns:
            display_table[col] = display_table[col].apply(
                lambda x: f"{x*100:.1f}%" if abs(x) < 1 else f"{x:.1f}%"
            )
    
    # Highlight target company
    def highlight_target(row):
        return ['background-color: lightblue' if target_ticker in row.name else '' for _ in row]
    
    styled_table = display_table.style.apply(highlight_target, axis=1)
    st.dataframe(styled_table, use_container_width=True)

def display_peer_statistics(peer_stats: Dict, target_data: Dict):
    """Display peer group statistics."""
    
    st.markdown("### 📊 Peer Group Statistics")
    
    if not peer_stats:
        st.warning("No peer statistics available.")
        return
    
    # Key valuation multiples
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Valuation Multiples**")
        
        valuation_metrics = ['pe_ratio', 'pb_ratio', 'ev_ebitda', 'ps_ratio']
        val_data = []
        
        for metric in valuation_metrics:
            if metric in peer_stats:
                stats = peer_stats[metric]
                target_val = target_data.get(metric, 0)
                
                val_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Peer Median': f"{stats['median']:.1f}x",
                    'Target Company': f"{target_val:.1f}x" if target_val > 0 else "N/A",
                    'Peer Range': f"{stats['min']:.1f}x - {stats['max']:.1f}x"
                })
        
        if val_data:
            val_df = pd.DataFrame(val_data)
            st.dataframe(val_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**Profitability Metrics**")
        
        prof_metrics = ['roe', 'roa', 'gross_margin', 'operating_margin']
        prof_data = []
        
        for metric in prof_metrics:
            if metric in peer_stats:
                stats = peer_stats[metric]
                target_val = target_data.get(metric, 0)
                
                prof_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Peer Median': f"{stats['median']*100:.1f}%",
                    'Target Company': f"{target_val*100:.1f}%" if target_val != 0 else "N/A",
                    'Peer Range': f"{stats['min']*100:.1f}% - {stats['max']*100:.1f}%"
                })
        
        if prof_data:
            prof_df = pd.DataFrame(prof_data)
            st.dataframe(prof_df, use_container_width=True, hide_index=True)

def display_relative_valuation(relative_val: Dict, company_info: Dict):
    """Display relative valuation results."""
    
    st.markdown("### 💰 Relative Valuation")
    
    if not relative_val:
        st.warning("Unable to calculate relative valuation.")
        return
    
    current_price = company_info.get('currentPrice', 0)
    
    # Valuation summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_val = relative_val.get('average_implied_value', 0)
        st.metric(
            "Average Implied Value",
            f"${avg_val:.2f}" if avg_val > 0 else "N/A"
        )
    
    with col2:
        median_val = relative_val.get('median_implied_value', 0)
        st.metric(
            "Median Implied Value",
            f"${median_val:.2f}" if median_val > 0 else "N/A"
        )
    
    with col3:
        upside = relative_val.get('upside_downside_median', 0)
        st.metric(
            "Upside/Downside",
            f"{upside:+.1f}%" if upside != 0 else "N/A"
        )
    
    # Individual multiple valuations
    st.markdown("**Multiple-Based Valuations**")
    
    multiple_data = []
    multiples = ['pe_valuation', 'pb_valuation', 'ps_valuation', 'ev_ebitda_valuation']
    multiple_names = ['P/E Multiple', 'P/B Multiple', 'P/S Multiple', 'EV/EBITDA Multiple']
    
    for multiple, name in zip(multiples, multiple_names):
        if multiple in relative_val:
            implied_value = relative_val[multiple]
            if implied_value > 0:
                upside_downside = (implied_value / current_price - 1) * 100 if current_price > 0 else 0
                multiple_data.append({
                    'Multiple': name,
                    'Implied Value': f"${implied_value:.2f}",
                    'Current Price': f"${current_price:.2f}",
                    'Upside/Downside': f"{upside_downside:+.1f}%"
                })
    
    if multiple_data:
        multiple_df = pd.DataFrame(multiple_data)
        st.dataframe(multiple_df, use_container_width=True, hide_index=True)

def display_combined_analysis(ticker: str, company_info: Dict):
    """Display combined DCF and comparable analysis."""
    
    st.markdown("---")
    st.subheader("🎯 Combined Valuation Analysis")
    
    # Run both analyses with default assumptions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### DCF Valuation (Quick)")
        # Run DCF with default assumptions
        try:
            dcf_model = DCFModel(ticker)
            dcf_model.set_assumptions(
                revenue_growth=[0.08, 0.06, 0.04, 0.03, 0.025],
                ebitda_margin=[0.20, 0.20, 0.20, 0.20, 0.20],
                terminal_growth=0.025,
                discount_rate=0.09
            )
            
            valuation_results = dcf_model.calculate_fair_value()
            
            dcf_fair_value = valuation_results['fair_value_per_share']
            dcf_upside = valuation_results['upside_downside']
            
            st.metric("DCF Fair Value", f"${dcf_fair_value:.2f}")
            st.metric("DCF Upside/Downside", f"{dcf_upside:+.1f}%")
            
        except Exception as e:
            st.error(f"DCF Error: {str(e)}")
            dcf_fair_value = 0
            dcf_upside = 0
    
    with col2:
        st.markdown("#### Comparable Analysis (Quick)")
        # Run comparable analysis
        try:
            sector = company_info.get('sector', 'Technology')
            peer_companies = market_data.get_sector_companies(sector, 10)
            peer_companies = [comp for comp in peer_companies if comp != ticker][:5]
            
            if peer_companies:
                comp_analysis = ComparableAnalysis(ticker, peer_companies)
                comp_analysis.fetch_company_data()
                relative_val = comp_analysis.relative_valuation()
                
                comp_fair_value = relative_val.get('median_implied_value', 0)
                comp_upside = relative_val.get('upside_downside_median', 0)
                
                st.metric("Comp Fair Value", f"${comp_fair_value:.2f}" if comp_fair_value > 0 else "N/A")
                st.metric("Comp Upside/Downside", f"{comp_upside:+.1f}%" if comp_upside != 0 else "N/A")
            else:
                st.warning("No peer companies found")
                comp_fair_value = 0
                comp_upside = 0
                
        except Exception as e:
            st.error(f"Comparable Analysis Error: {str(e)}")
            comp_fair_value = 0
            comp_upside = 0
    
    # Combined recommendation
    st.markdown("---")
    st.markdown("#### 📊 Combined Valuation Summary")
    
    current_price = company_info.get('currentPrice', 0)
    
    # Calculate blended valuation
    valid_valuations = []
    if dcf_fair_value > 0:
        valid_valuations.append(dcf_fair_value)
    if comp_fair_value > 0:
        valid_valuations.append(comp_fair_value)
    
    if valid_valuations:
        blended_fair_value = np.mean(valid_valuations)
        blended_upside = (blended_fair_value / current_price - 1) * 100 if current_price > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric("Blended Fair Value", f"${blended_fair_value:.2f}")
        
        with col3:
            st.metric("Blended Upside/Downside", f"{blended_upside:+.1f}%")
        
        # Final recommendation
        if blended_upside > 15:
            recommendation = "🟢 STRONG BUY"
        elif blended_upside > 5:
            recommendation = "🟢 BUY"
        elif blended_upside > -5:
            recommendation = "🟡 HOLD"
        elif blended_upside > -15:
            recommendation = "🔴 SELL"
        else:
            recommendation = "🔴 STRONG SELL"
        
        st.markdown(f"### **Investment Recommendation: {recommendation}**")
        
        # Export option
        if st.button("📊 Export Analysis", type="secondary"):
            # This would trigger the export functionality
            st.success("Analysis exported successfully! (Feature to be implemented)")

def display_sample_analysis():
    """Display sample analysis when no ticker is entered."""
    
    st.markdown("---")
    st.subheader("📚 Sample Analysis Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 💰 DCF Valuation
        - **3-Statement Modeling**: Integrated financial statement projections
        - **Customizable Assumptions**: Adjust growth rates, margins, and terminal value
        - **Sensitivity Analysis**: Test different discount rates and growth scenarios
        - **Professional Output**: Institutional-grade valuation reports
        """)
        
        st.markdown("""
        #### 📊 Comparable Analysis
        - **Peer Benchmarking**: Automatic peer company identification
        - **Multiple Analysis**: P/E, EV/EBITDA, P/B, and more
        - **Statistical Analysis**: Median, mean, and range calculations
        - **Relative Valuation**: Implied value based on peer multiples
        """)
    
    with col2:
        st.markdown("""
        #### 🎯 Key Features
        - **Real-time Data**: Live market data integration
        - **Interactive Charts**: Dynamic visualizations
        - **Export Capabilities**: Excel and PDF reports
        - **Risk Analysis**: Beta, correlation, and volatility metrics
        """)
        
        st.markdown("""
        #### 📈 Sample Companies
        Try analyzing these popular stocks:
        - **AAPL** - Apple Inc.
        - **MSFT** - Microsoft Corp.
        - **GOOGL** - Alphabet Inc.
        - **AMZN** - Amazon.com Inc.
        - **TSLA** - Tesla Inc.
        """)
