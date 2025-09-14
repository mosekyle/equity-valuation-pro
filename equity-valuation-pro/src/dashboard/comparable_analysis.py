"""
Comparable Analysis Dashboard Component
Interactive interface for peer analysis and multiple valuation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.comps import ComparableAnalyzer, CompanyProfile


class ComparableAnalysisDashboard:
    """Dashboard component for comparable analysis"""
    
    def __init__(self):
        self.analyzer = None
        
    def render_main_interface(self, target_symbol: str):
        """Render the main comparable analysis interface"""
        
        st.markdown('<div class="section-header">🔍 Comparable Company Analysis</div>', unsafe_allow_html=True)
        
        # Initialize analyzer
        if 'comps_analyzer' not in st.session_state or st.session_state.get('comps_target_symbol') != target_symbol:
            with st.spinner("Initializing comparable analysis..."):
                try:
                    self.analyzer = ComparableAnalyzer(target_symbol)
                    target_profile = self.analyzer.load_target_company()
                    st.session_state.comps_analyzer = self.analyzer
                    st.session_state.comps_target_symbol = target_symbol
                    st.session_state.comps_target_profile = target_profile
                except Exception as e:
                    st.error(f"Error initializing comparable analysis: {str(e)}")
                    return
        else:
            self.analyzer = st.session_state.comps_analyzer
        
        # Target company overview
        self._render_target_overview()
        
        # Peer selection interface
        self._render_peer_selection()
        
        # Analysis results
        if hasattr(st.session_state, 'comps_peer_profiles') and st.session_state.comps_peer_profiles:
            self._render_multiples_analysis()
            self._render_implied_valuation()
            self._render_football_field_chart()
    
    def _render_target_overview(self):
        """Render target company overview"""
        if 'comps_target_profile' not in st.session_state:
            return
        
        target = st.session_state.comps_target_profile
        
        st.markdown("### 🎯 Target Company Profile")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Company",
                target.name,
                help="Target company for comparable analysis"
            )
        
        with col2:
            st.metric(
                "Sector",
                target.sector,
                help="Industry sector classification"
            )
        
        with col3:
            market_cap_display = f"${target.market_cap/1e9:.1f}B" if target.market_cap > 1e9 else f"${target.market_cap/1e6:.1f}M"
            st.metric(
                "Market Cap",
                market_cap_display,
                help="Current market capitalization"
            )
        
        with col4:
            st.metric(
                "Current Price",
                f"${target.current_price:.2f}",
                help="Current stock price"
            )
    
    def _render_peer_selection(self):
        """Render peer company selection interface"""
        st.markdown("### 🏢 Peer Company Selection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Auto peer selection
            auto_select = st.checkbox(
                "Auto-select peer companies",
                value=True,
                help="Automatically select peers based on sector and size"
            )
            
            # Custom peer input
            custom_peers_input = st.text_input(
                "Custom Peer Symbols (comma-separated)",
                placeholder="MSFT, GOOGL, META, AMZN",
                help="Enter custom peer company symbols separated by commas"
            )
            
            # Parse custom peers
            custom_peers = []
            if custom_peers_input:
                custom_peers = [p.strip().upper() for p in custom_peers_input.split(',') if p.strip()]
        
        with col2:
            # Analysis controls
            min_market_cap = st.selectbox(
                "Minimum Market Cap",
                options=[1e9, 5e9, 10e9, 50e9, 100e9],
                index=0,
                format_func=lambda x: f"${x/1e9:.0f}B"
            )
            
            max_peers = st.slider(
                "Maximum Peers",
                min_value=5,
                max_value=20,
                value=10,
                help="Maximum number of peer companies to analyze"
            )
        
        # Load peer data button
        if st.button("🔄 Load Peer Analysis", type="primary"):
            with st.spinner("Loading peer company data..."):
                try:
                    # Find peer companies
                    peer_symbols = self.analyzer.find_peer_companies(
                        custom_peers=custom_peers if custom_peers else None,
                        auto_select=auto_select,
                        min_market_cap=min_market_cap
                    )
                    
                    # Limit to max_peers
                    peer_symbols = peer_symbols[:max_peers]
                    
                    # Load peer data
                    peer_profiles = self.analyzer.load_peer_data(peer_symbols)
                    
                    if peer_profiles:
                        st.session_state.comps_peer_profiles = peer_profiles
                        st.session_state.comps_peer_symbols = list(peer_profiles.keys())
                        st.success(f"✅ Loaded data for {len(peer_profiles)} peer companies")
                    else:
                        st.warning("⚠️ No peer companies found with sufficient data")
                        
                except Exception as e:
                    st.error(f"Error loading peer data: {str(e)}")
        
        # Display selected peers
        if 'comps_peer_symbols' in st.session_state:
            st.markdown("**Selected Peer Companies:**")
            peers_display = " • ".join(st.session_state.comps_peer_symbols)
            st.info(f"📊 {peers_display}")
    
    def _render_multiples_analysis(self):
        """Render multiples analysis results"""
        st.markdown("### 📊 Multiples Analysis")
        
        try:
            # Calculate multiples
            multiples_df = self.analyzer.calculate_multiples()
            
            # Display multiples table
            self._display_multiples_table(multiples_df)
            
            # Multiples visualization
            self._create_multiples_charts(multiples_df)
            
            # Statistical analysis
            self._display_peer_statistics()
            
        except Exception as e:
            st.error(f"Error in multiples analysis: {str(e)}")
    
    def _display_multiples_table(self, multiples_df: pd.DataFrame):
        """Display the multiples comparison table"""
        st.markdown("#### 📋 Multiples Comparison Table")
        
        # Format the display DataFrame
        display_df = multiples_df.copy()
        
        # Format financial figures
        for col in ['market_cap', 'enterprise_value', 'revenue', 'ebitda']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"${x/1e9:.2f}B" if pd.notnull(x) and x > 1e9 
                    else f"${x/1e6:.0f}M" if pd.notnull(x) and x > 1e6 
                    else f"${x:.0f}" if pd.notnull(x) else "N/A"
                )
        
        # Format multiples
        for col in ['pe_ratio', 'ev_revenue', 'ev_ebitda', 'price_sales']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.1f}x" if pd.notnull(x) else "N/A"
                )
        
        # Format price
        if 'current_price' in display_df.columns:
            display_df['current_price'] = display_df['current_price'].apply(
                lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A"
            )
        
        # Rename columns for display
        display_columns = {
            'symbol': 'Symbol',
            'company_name': 'Company',
            'market_cap': 'Market Cap',
            'enterprise_value': 'Enterprise Value',
            'current_price': 'Stock Price',
            'pe_ratio': 'P/E',
            'ev_revenue': 'EV/Revenue',
            'ev_ebitda': 'EV/EBITDA',
            'price_sales': 'P/S',
            'company_type': 'Type'
        }
        
        # Select and rename columns
        available_cols = [col for col in display_columns.keys() if col in display_df.columns]
        display_df = display_df[available_cols].rename(columns=display_columns)
        
        # Highlight target company
        def highlight_target(row):
            return ['background-color: #e3f2fd' if row['Type'] == 'Target' else '' for _ in row]
        
        styled_df = display_df.style.apply(highlight_target, axis=1)
        
        st.dataframe(styled_df, use_container_width=True, height=400)
    
    def _create_multiples_charts(self, multiples_df: pd.DataFrame):
        """Create visualization charts for multiples"""
        st.markdown("#### 📈 Multiples Visualization")
        
        # Prepare data for charts
        peer_df = multiples_df[multiples_df['company_type'] == 'Peer'].copy()
        target_df = multiples_df[multiples_df['company_type'] == 'Target'].copy()
        
        # Create subplot with multiple charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['P/E Ratio', 'EV/Revenue', 'EV/EBITDA', 'P/S Ratio'],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        multiples_to_plot = [
            ('pe_ratio', 1, 1),
            ('ev_revenue', 1, 2),
            ('ev_ebitda', 2, 1),
            ('price_sales', 2, 2)
        ]
        
        for multiple, row, col in multiples_to_plot:
            if multiple in peer_df.columns:
                # Peer companies (scatter)
                peer_data = peer_df[multiple].dropna()
                if not peer_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(peer_data))),
                            y=peer_data.values,
                            mode='markers',
                            name=f'Peers - {multiple}',
                            marker=dict(color='lightblue', size=8),
                            text=peer_df[peer_df[multiple].notna()]['symbol'].values,
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                
                # Target company (highlighted)
                if not target_df.empty and multiple in target_df.columns:
                    target_value = target_df[multiple].iloc[0]
                    if pd.notnull(target_value):
                        fig.add_trace(
                            go.Scatter(
                                x=[len(peer_data)],
                                y=[target_value],
                                mode='markers',
                                name=f'Target - {multiple}',
                                marker=dict(color='red', size=12, symbol='star'),
                                text=[target_df['symbol'].iloc[0]],
                                showlegend=False
                            ),
                            row=row, col=col
                        )
                
                # Add median line
                if not peer_data.empty:
                    median_value = peer_data.median()
                    fig.add_hline(
                        y=median_value,
                        line_dash="dash",
                        line_color="green",
                        annotation_text=f"Median: {median_value:.1f}x",
                        row=row, col=col
                    )
        
        fig.update_layout(
            height=600,
            title_text="Valuation Multiples Comparison",
            showlegend=False,
            template='plotly_white'
        )
        
[O        # Update y-axis labels
        fig.update_yaxes(title_text="Multiple (x)", row=1, col=1)
        fig.update_yaxes(title_text="Multiple (x)", row=1, col=2)
        fig.update_yaxes(title_text="Multiple (x)", row=2, col=1)
        fig.update_yaxes(title_text="Multiple (x)", row=2, col=2)
[I        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_peer_statistics(self):
        """Display peer group statistics"""
        st.markdown("#### 📊 Peer Group Statistics")
        
        if not hasattr(self.analyzer, 'multiples_analysis') or not self.analyzer.multiples_analysis:
            return
        
        peer_stats = self.analyzer.multiples_analysis['peer_stats']
        
        # Format statistics for display
        display_stats = peer_stats.copy()
        
        # Round values
        for col in display_stats.columns:
            display_stats[col] = display_stats[col].apply(
                lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
            )
        
        # Rename index
        display_stats.index = ['P/E Ratio', 'EV/Revenue', 'EV/EBITDA', 'P/S Ratio']
        
        # Display as formatted table
        st.dataframe(display_stats, use_container_width=True)
    
    def _render_implied_valuation(self):
        """Render implied valuation analysis"""
        st.markdown("### 💰 Implied Valuation Analysis")
[O        
        try:
            implied_valuations = self.analyzer.calculate_implied_valuation()
            
            if not implied_valuations:
                st.warning("⚠️ Insufficient data for implied valuation calculation")
                return
            
            # Display implied valuations
            self._display_implied_valuations(implied_valuations)
            
            # Create valuation summary chart
            self._create_valuation_summary_chart(implied_valuations)
            
        except Exception as e:
            st.error(f"Error calculating implied valuations: {str(e)}")
    
    def _display_implied_valuations(self, implied_valuations: Dict):
        """Display implied valuation results"""
        st.markdown("#### 🎯 Valuation Summary")
        
        # Get current price for comparison
        current_price = st.session_state.comps_target_profile.current_price
        
        # Create summary table
        summary_data = []
        
        for method, data in implied_valuations.items():
            method_name = method.replace('_', ' ').title()
            
            if 'implied_price_median' in data:
                median_price = data['implied_price_median']
                mean_price = data['implied_price_mean']
                
                median_upside = ((median_price - current_price) / current_price) * 100
                mean_upside = ((mean_price - current_price) / current_price) * 100
                
                summary_data.append({
                    'Valuation Method': method_name,
                    'Median Multiple': f"{data.get('median_multiple', 0):.2f}x",
                    'Mean Multiple': f"{data.get('mean_multiple', 0):.2f}x",
                    'Implied Price (Median)': f"${median_price:.2f}",
                    'Implied Price (Mean)': f"${mean_price:.2f}",
                    'Upside/Downside (Median)': f"{median_upside:+.1f}%",
                    'Upside/Downside (Mean)': f"{mean_upside:+.1f}%"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Style the dataframe
            def color_upside(val):
                if isinstance(val, str) and '%' in val:
                    try:
                        num_val = float(val.replace('%', '').replace('+', ''))
                        if num_val > 0:
                            return 'color: green'
                        elif num_val < 0:
                            return 'color: red'
                    except:
                        pass
                return ''
            
            styled_df = summary_df.style.applymap(
                color_upside, 
                subset=['Upside/Downside (Median)', 'Upside/Downside (Mean)']
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Add current price reference
            col1, col2, col3 = st.columns(3)
            with col2:
                st.metric(
                    "Current Stock Price",
                    f"${current_price:.2f}",
                    help="Current market price for comparison"
                )
    
    def _create_valuation_summary_chart(self, implied_valuations: Dict):
        """Create valuation summary chart"""
        st.markdown("#### 📊 Price Target Analysis")
        
        current_price = st.session_state.comps_target_profile.current_price
        
        # Prepare data for chart
        methods = []
        median_prices = []
        mean_prices = []
        
        for method, data in implied_valuations.items():
            if 'implied_price_median' in data:
                methods.append(method.replace('_', ' ').title())
                median_prices.append(data['implied_price_median'])
                mean_prices.append(data['implied_price_mean'])
        
        if methods:
            fig = go.Figure()
            
            # Add median prices
            fig.add_trace(go.Bar(
                name='Median Multiple',
                x=methods,
                y=median_prices,
                marker_color='lightblue',
                text=[f"${price:.2f}" for price in median_prices],
                textposition='auto'
            ))
            
            # Add mean prices
            fig.add_trace(go.Bar(
                name='Mean Multiple',
                x=methods,
                y=mean_prices,
                marker_color='darkblue',
                text=[f"${price:.2f}" for price in mean_prices],
                textposition='auto'
            ))
            
            # Add current price line
            fig.add_hline(
                y=current_price,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Current Price: ${current_price:.2f}"
            )
            
            fig.update_layout(
                title="Implied Valuations vs Current Price",
                xaxis_title="Valuation Method",
                yaxis_title="Stock Price ($)",
                barmode='group',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_football_field_chart(self):
        """Render football field valuation chart"""
        st.markdown("### 🏈 Football Field Chart")
        
        try:
            football_field_data = self.analyzer.generate_football_field()
            
            if not football_field_data:
                st.warning("⚠️ Insufficient data for football field chart")
                return
            
            current_price = st.session_state.comps_target_profile.current_price
            
            # Create football field chart
            fig = go.Figure()
            
            methods = list(football_field_data.keys())
            low_values = [data[0] for data in football_field_data.values()]
            high_values = [data[1] for data in football_field_data.values()]
            
            # Add ranges as horizontal bars
            for i, method in enumerate(methods):
                low_val, high_val = football_field_data[method]
                
                fig.add_trace(go.Scatter(
                    x=[low_val, high_val],
                    y=[method, method],
                    mode='lines+markers',
                    line=dict(width=10, color='lightblue'),
                    marker=dict(size=8, color=['blue', 'blue']),
                    name=method,
                    showlegend=False
                ))
                
                # Add range text
                fig.add_annotation(
                    x=(low_val + high_val) / 2,
                    y=method,
                    text=f"${low_val:.0f} - ${high_val:.0f}",
                    showarrow=False,
                    font=dict(color='white', size=10)
                )
            
            # Add current price line
            fig.add_vline(
                x=current_price,
                line_dash="dash",
                line_color="red",
                line_width=3,
                annotation_text=f"Current: ${current_price:.2f}"
            )
            
            # Calculate overall range for better display
            all_values = low_values + high_values + [current_price]
            x_min = min(all_values) * 0.9
            x_max = max(all_values) * 1.1
            
            fig.update_layout(
                title="Valuation Range Analysis (Football Field Chart)",
                xaxis_title="Stock Price ($)",
                xaxis=dict(range=[x_min, x_max]),
                yaxis_title="Valuation Method",
                template='plotly_white',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_low = np.mean(low_values)
                st.metric("Average Low", f"${avg_low:.2f}")
            
            with col2:
                avg_high = np.mean(high_values)
                st.metric("Average High", f"${avg_high:.2f}")
            
            with col3:
                avg_mid = (avg_low + avg_high) / 2
                upside = ((avg_mid - current_price) / current_price) * 100
                st.metric("Average Target", f"${avg_mid:.2f}", f"{upside:+.1f}%")
                
        except Exception as e:
            st.error(f"Error creating football field chart: {str(e)}")
