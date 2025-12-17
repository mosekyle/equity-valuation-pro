"""
Advanced Analytics Dashboard Component
Sophisticated features for institutional-grade analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.advanced_analytics import (
    LeveragedBuyoutModel, RealOptionsValuation, MachineLearningValuation,
    SentimentAnalysis, RiskAnalytics, DEFAULT_STRESS_SCENARIOS
)


class AdvancedAnalyticsDashboard:
    """Advanced analytics dashboard component"""
    
    def __init__(self):
        self.lbo_model = None
        self.ml_model = MachineLearningValuation()
        self.sentiment_analyzer = SentimentAnalysis()
        self.risk_analyzer = RiskAnalytics()
    
    def render_main_interface(self, target_symbol: str):
        """Render the advanced analytics interface"""
        
        st.markdown('<div class="section-header">🎯 Advanced Analytics Suite</div>', unsafe_allow_html=True)
        
        # Navigation tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏢 LBO Analysis", 
            "🤖 ML Valuation", 
            "📰 Sentiment Analysis", 
            "⚡ Real Options", 
            "⚠️ Risk Analytics"
        ])
        
        with tab1:
            self._render_lbo_analysis(target_symbol)
        
        with tab2:
            self._render_ml_valuation(target_symbol)
        
        with tab3:
            self._render_sentiment_analysis(target_symbol)
        
        with tab4:
            self._render_real_options(target_symbol)
        
        with tab5:
            self._render_risk_analytics(target_symbol)
    
    def _render_lbo_analysis(self, target_symbol: str):
        """Render LBO analysis interface"""
        st.markdown("### 🏢 Leveraged Buyout Analysis")
        st.markdown("*Analyze potential private equity acquisition scenarios*")
        
        # Initialize LBO model
        if not self.lbo_model or getattr(self.lbo_model, 'target_symbol', None) != target_symbol:
            self.lbo_model = LeveragedBuyoutModel(target_symbol)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Deal Structure")
            
            # Get current market data for defaults
            try:
                from data.market_data import MarketDataProvider
                provider = MarketDataProvider()
                company_info = provider.get_company_info(target_symbol)
                current_market_cap = company_info.get('basic_info', {}).get('market_cap', 1e9)
            except:
                current_market_cap = 1e9
            
            # Deal parameters
            purchase_multiple = st.slider(
                "Purchase Multiple (EV/EBITDA)",
                min_value=8.0,
                max_value=20.0,
                value=12.0,
                step=0.5,
                help="Enterprise value multiple for acquisition"
            )
            
            # Estimate purchase price based on multiple
            try:
                company_info = provider.get_company_info(target_symbol)
                current_ebitda = self.lbo_model._extract_ebitda(company_info)
                estimated_purchase_price = current_ebitda * purchase_multiple
            except:
                estimated_purchase_price = current_market_cap * 1.3  # 30% premium
            
            purchase_price = st.number_input(
                "Purchase Price ($M)",
                min_value=100.0,
                value=estimated_purchase_price / 1e6,
                step=100.0,
                format="%.0f"
            ) * 1e6
            
            debt_percentage = st.slider(
                "Debt Financing (%)",
                min_value=30.0,
                max_value=80.0,
                value=65.0,
                step=5.0,
                help="Percentage of purchase price financed with debt"
            )
            
            debt_financing = purchase_price * (debt_percentage / 100)
            equity_contribution = purchase_price - debt_financing
            
            hold_period = st.selectbox(
                "Hold Period (Years)",
                options=[3, 4, 5, 6, 7],
                index=2,
                help="Expected investment period"
            )
            
            exit_multiple = st.slider(
                "Exit Multiple (EV/EBITDA)",
                min_value=8.0,
                max_value=18.0,
                value=11.0,
                step=0.5,
                help="Expected exit multiple"
            )
        
        with col2:
            st.markdown("#### Deal Summary")
            
            # Display deal structure
            deal_summary = pd.DataFrame({
                'Component': [
                    'Purchase Price',
                    'Debt Financing',
                    'Equity Contribution',
                    'Debt/Equity Ratio',
                    'Purchase Multiple',
                    'Exit Multiple'
                ],
                'Value': [
                    f"${purchase_price/1e6:.0f}M",
                    f"${debt_financing/1e6:.0f}M ({debt_percentage:.0f}%)",
                    f"${equity_contribution/1e6:.0f}M ({100-debt_percentage:.0f}%)",
                    f"{debt_financing/equity_contribution:.1f}x",
                    f"{purchase_multiple:.1f}x",
                    f"{exit_multiple:.1f}x"
                ]
            })
            
            st.dataframe(deal_summary, use_container_width=True, hide_index=True)
        
        # Run LBO Analysis
        if st.button("🚀 Calculate LBO Returns", type="primary"):
            with st.spinner("Calculating LBO returns..."):
                try:
                    lbo_results = self.lbo_model.calculate_lbo_returns(
                        purchase_price=purchase_price,
                        debt_financing=debt_financing,
                        equity_contribution=equity_contribution,
                        exit_multiple=exit_multiple,
                        hold_period=hold_period
                    )
                    
                    st.session_state.lbo_results = lbo_results
                    
                    # Display results
                    self._display_lbo_results(lbo_results)
                    
                except Exception as e:
                    st.error(f"Error calculating LBO returns: {str(e)}")
        
        # Display cached results if available
        if 'lbo_results' in st.session_state:
            self._display_lbo_results(st.session_state.lbo_results)
    
    def _display_lbo_results(self, lbo_results: Dict):
        """Display LBO analysis results"""
        st.markdown("#### 💰 LBO Returns Analysis")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return Multiple",
                f"{lbo_results['total_return_multiple']:.1f}x",
                help="Total cash-on-cash return multiple"
            )
        
        with col2:
            irr_color = "normal" if lbo_results['irr'] < 0.15 else "inverse"
            st.metric(
                "IRR",
                f"{lbo_results['irr']:.1%}",
                help="Internal Rate of Return"
            )
        
        with col3:
            st.metric(
                "Exit Equity Value",
                f"${lbo_results['exit_equity_value']/1e6:.0f}M",
                help="Projected equity value at exit"
            )
        
        with col4:
            st.metric(
                "Hold Period",
                f"{lbo_results['hold_period']} years",
                help="Investment holding period"
            )
        
        # Returns waterfall chart
        fig_waterfall = go.Figure()
        
        # Create waterfall data
        categories = ['Initial Investment', 'EBITDA Growth', 'Multiple Expansion', 'Debt Paydown', 'Final Value']
        values = [
            -lbo_results['equity_contribution']/1e6,
            (lbo_results['exit_ebitda'] - lbo_results['current_ebitda']) * lbo_results.get('exit_multiple', 11) / 1e6,
            0,  # Placeholder for multiple expansion
            (lbo_results['debt_financing'] * 0.5) / 1e6,  # Debt paydown benefit
            lbo_results['exit_equity_value']/1e6
        ]
        
        fig_waterfall.add_trace(go.Waterfall(
            name="LBO Returns",
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "total"],
            x=categories,
            y=values,
            text=[f"${v:.0f}M" for v in values],
            textposition="outside",
            connector={"line":{"color":"rgb(63, 63, 63)"}},
        ))
        
        fig_waterfall.update_layout(
            title="LBO Value Creation Waterfall",
            xaxis_title="Value Drivers",
            yaxis_title="Value ($M)",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_waterfall, use_container_width=True)
    
    def _render_ml_valuation(self, target_symbol: str):
        """Render ML valuation interface"""
        st.markdown("### 🤖 Machine Learning Valuation")
        st.markdown("*AI-powered stock price prediction using fundamental factors*")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Model Training")
            
            # Training universe selection
            training_universe = st.multiselect(
                "Training Universe (Select Stocks)",
                options=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'ADBE', 'CRM', 
                        'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'IBM', 'CSCO', 'UBER', 'LYFT', 'SNOW'],
                default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX'],
                help="Select stocks to train the ML model"
            )
            
            model_type = st.selectbox(
                "Model Type",
                options=["Random Forest", "Gradient Boosting", "Neural Network"],
                index=0,
                help="Machine learning algorithm"
            )
            
            if st.button("🔧 Train ML Model", type="secondary"):
                if len(training_universe) < 5:
                    st.warning("Please select at least 5 stocks for training")
                else:
                    with st.spinner("Training ML model... This may take a moment."):
                        try:
                            training_results = self.ml_model.build_valuation_model(training_universe)
                            st.session_state.ml_training_results = training_results
                            st.session_state.ml_model_trained = True
                            st.success("✅ Model trained successfully!")
                            
                            # Display training results
                            metrics_col1, metrics_col2 = st.columns(2)
                            with metrics_col1:
                                st.metric("R² Score", f"{training_results['r2_score']:.3f}")
                            with metrics_col2:
                                st.metric("Training Samples", training_results['training_samples'])
                            
                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")
        
        with col2:
            st.markdown("#### Prediction Results")
            
            if st.session_state.get('ml_model_trained', False):
                if st.button("🎯 Predict Stock Price", type="primary"):
                    with st.spinner("Generating ML prediction..."):
                        try:
                            prediction_results = self.ml_model.predict_valuation(target_symbol)
                            st.session_state.ml_prediction_results = prediction_results
                            
                            # Display prediction
                            pred_col1, pred_col2 = st.columns(2)
                            
                            with pred_col1:
                                st.metric(
                                    "Current Price",
                                    f"${prediction_results['current_price']:.2f}"
                                )
                                st.metric(
                                    "ML Predicted Price",
                                    f"${prediction_results['predicted_price']:.2f}"
                                )
                            
                            with pred_col2:
                                upside_color = "normal" if prediction_results['upside_downside'] < 0 else "inverse"
                                st.metric(
                                    "Upside/Downside",
                                    f"{prediction_results['upside_downside']:+.1f}%"
                                )
                                st.metric(
                                    "Confidence Interval",
                                    f"${prediction_results['confidence_interval_low']:.0f} - ${prediction_results['confidence_interval_high']:.0f}"
                                )
                            
                            # Feature importance chart
                            if hasattr(self.ml_model, 'feature_importance') and self.ml_model.feature_importance:
                                fig_importance = px.bar(
                                    x=list(self.ml_model.feature_importance.values()),
                                    y=list(self.ml_model.feature_importance.keys()),
                                    orientation='h',
                                    title="Feature Importance",
                                    labels={'x': 'Importance', 'y': 'Features'}
                                )
                                fig_importance.update_layout(height=300, template='plotly_white')
                                st.plotly_chart(fig_importance, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error generating prediction: {str(e)}")
            else:
                st.info("👆 Please train the ML model first")
    
    def _render_sentiment_analysis(self, target_symbol: str):
        """Render sentiment analysis interface"""
        st.markdown("### 📰 News Sentiment Analysis")
        st.markdown("*Market sentiment and news flow analysis*")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Analysis Parameters")
            
            analysis_period = st.selectbox(
                "Analysis Period",
                options=[7, 14, 30, 60, 90],
                index=2,
                format_func=lambda x: f"Last {x} days"
            )
            
            include_social = st.checkbox(
                "Include Social Media",
                value=True,
                help="Include Twitter, Reddit sentiment"
            )
            
            if st.button("📊 Analyze Sentiment", type="primary"):
                with st.spinner("Analyzing news sentiment..."):
                    try:
                        sentiment_results = self.sentiment_analyzer.analyze_news_sentiment(
                            target_symbol, days_back=analysis_period
                        )
                        st.session_state.sentiment_results = sentiment_results
                        
                    except Exception as e:
                        st.error(f"Error analyzing sentiment: {str(e)}")
        
        with col2:
            if 'sentiment_results' in st.session_state:
                sentiment = st.session_state.sentiment_results
                
                # Sentiment gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = sentiment['sentiment_score'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Sentiment Score"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "gray"},
                            {'range': [70, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig_gauge.update_layout(height=300, template='plotly_white')
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Sentiment metrics
                met_col1, met_col2, met_col3 = st.columns(3)
                
                with met_col1:
                    st.metric("Sentiment Label", sentiment['sentiment_label'])
                
                with met_col2:
                    st.metric("Confidence", f"{sentiment['confidence']:.1%}")
                
                with met_col3:
                    st.metric("Articles Analyzed", sentiment['articles_analyzed'])
    
    def _render_real_options(self, target_symbol: str):
        """Render real options valuation interface"""
        st.markdown("### ⚡ Real Options Valuation")
        st.markdown("*Value growth opportunities and strategic flexibility*")
        
        options_model = RealOptionsValuation()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Option Parameters")
            
            option_type = st.selectbox(
                "Option Type",
                options=["Expansion Option", "Abandonment Option", "Timing Option"],
                index=0
            )
            
            if option_type == "Expansion Option":
                project_npv = st.number_input(
                    "Current Project NPV ($M)",
                    min_value=1.0,
                    value=100.0,
                    step=10.0
                ) * 1e6
                
                expansion_cost = st.number_input(
                    "Expansion Investment ($M)",
                    min_value=1.0,
                    value=150.0,
                    step=10.0
                ) * 1e6
                
                volatility = st.slider(
                    "Project Volatility",
                    min_value=0.1,
                    max_value=0.8,
                    value=0.3,
                    step=0.05,
                    format="%.0%%"
                )
                
                time_to_expand = st.slider(
                    "Time to Expand (Years)",
                    min_value=1,
                    max_value=10,
                    value=3
                )
                
                risk_free_rate = st.slider(
                    "Risk-Free Rate",
                    min_value=0.01,
                    max_value=0.08,
                    value=0.03,
                    step=0.005,
                    format="%.1%%"
                )
        
        with col2:
            st.markdown("#### Option Valuation")
            
            if st.button("⚡ Calculate Option Value", type="primary"):
                if option_type == "Expansion Option":
                    try:
                        option_results = options_model.calculate_expansion_option(
                            project_npv=project_npv,
                            expansion_cost=expansion_cost,
                            volatility=volatility,
                            time_to_expand=time_to_expand,
                            risk_free_rate=risk_free_rate
                        )
                        
                        # Display results
                        opt_col1, opt_col2 = st.columns(2)
                        
                        with opt_col1:
                            st.metric(
                                "Option Value",
                                f"${option_results['option_value']/1e6:.1f}M"
                            )
                            st.metric(
                                "Project NPV",
                                f"${option_results['project_npv']/1e6:.1f}M"
                            )
                        
                        with opt_col2:
                            st.metric(
                                "Total Project Value",
                                f"${option_results['total_project_value']/1e6:.1f}M"
                            )
                            st.metric(
                                "Option Premium",
                                f"{option_results['option_premium']:.1f}%"
                            )
                        
                        # Option value sensitivity
                        self._create_option_sensitivity_chart(options_model, option_results)
                        
                    except Exception as e:
                        st.error(f"Error calculating option value: {str(e)}")
    
    def _create_option_sensitivity_chart(self, options_model, base_results):
        """Create option sensitivity analysis chart"""
        volatility_range = np.arange(0.1, 0.8, 0.05)
        option_values = []
        
        for vol in volatility_range:
            try:
                result = options_model.calculate_expansion_option(
                    project_npv=base_results['project_npv'],
                    expansion_cost=base_results.get('expansion_cost', base_results['project_npv'] * 1.5),
                    volatility=vol,
                    time_to_expand=3,
                    risk_free_rate=0.03
                )
                option_values.append(result['option_value'] / 1e6)
            except:
                option_values.append(0)
        
        fig_sensitivity = go.Figure()
        fig_sensitivity.add_trace(go.Scatter(
            x=volatility_range * 100,
            y=option_values,
            mode='lines+markers',
            name='Option Value',
            line=dict(color='blue', width=2)
        ))
        
        fig_sensitivity.update_layout(
            title="Option Value Sensitivity to Volatility",
            xaxis_title="Volatility (%)",
            yaxis_title="Option Value ($M)",
            template='plotly_white',
            height=300
        )
        
        st.plotly_chart(fig_sensitivity, use_container_width=True)
    
    def _render_risk_analytics(self, target_symbol: str):
        """Render risk analytics interface"""
        st.markdown("### ⚠️ Risk Analytics & Stress Testing")
        st.markdown("*Comprehensive risk assessment and scenario analysis*")
        
        tab1, tab2 = st.tabs(["Portfolio Risk", "Stress Testing"])
        
        with tab1:
            self._render_portfolio_risk()
        
        with tab2:
            self._render_stress_testing(target_symbol)
    
    def _render_portfolio_risk(self):
        """Render portfolio risk analysis"""
        st.markdown("#### 📊 Portfolio VaR Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Portfolio Composition**")
            
            # Portfolio input
            portfolio_symbols = st.text_input(
                "Portfolio Symbols (comma-separated)",
                value="AAPL,MSFT,GOOGL,AMZN",
                help="Enter stock symbols separated by commas"
            )
            
            symbols = [s.strip().upper() for s in portfolio_symbols.split(',') if s.strip()]
            
            if symbols:
                # Weight input
                weights = []
                for i, symbol in enumerate(symbols):
                    weight = st.number_input(
                        f"{symbol} Weight (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=100.0/len(symbols),
                        step=1.0,
                        key=f"weight_{symbol}"
                    )
                    weights.append(weight/100)
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w/total_weight for w in weights]
        
        with col2:
            if symbols and st.button("⚠️ Calculate Portfolio Risk", type="primary"):
                with st.spinner("Calculating portfolio risk metrics..."):
                    try:
                        risk_metrics = self.risk_analyzer.calculate_var_cvar(
                            symbols=symbols,
                            weights=weights,
                            confidence_level=0.05
                        )
                        
                        # Display risk metrics
                        risk_col1, risk_col2 = st.columns(2)
                        
                        with risk_col1:
                            st.metric("95% VaR", f"{risk_metrics['var_95']:.1%}")
                            st.metric("95% CVaR", f"{risk_metrics['cvar_95']:.1%}")
                        
                        with risk_col2:
                            st.metric("Portfolio Volatility", f"{risk_metrics['portfolio_volatility']:.1%}")
                            st.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.2f}")
                        
                    except Exception as e:
                        st.error(f"Error calculating risk metrics: {str(e)}")
    
    def _render_stress_testing(self, target_symbol: str):
        """Render stress testing interface"""
        st.markdown("#### 🎭 Stress Test Scenarios")
        
        # Predefined scenarios
        scenario_names = list(DEFAULT_STRESS_SCENARIOS.keys())
        selected_scenarios = st.multiselect(
            "Select Stress Test Scenarios",
            options=scenario_names,
            default=scenario_names[:3],
            help="Choose stress test scenarios to run"
        )
        
        if st.button("⚡ Run Stress Tests", type="primary") and selected_scenarios:
            with st.spinner("Running stress tests..."):
                try:
                    # Filter scenarios
                    scenarios_to_run = {k: v for k, v in DEFAULT_STRESS_SCENARIOS.items() 
                                      if k in selected_scenarios}
                    
                    stress_results = self.risk_analyzer.stress_test_scenarios(
                        symbol=target_symbol,
                        scenarios=scenarios_to_run
                    )
                    
                    # Create stress test results chart
                    scenario_names = list(stress_results.keys())
                    price_changes = [stress_results[scenario]['price_change_percent'] 
                                   for scenario in scenario_names]
                    
                    # Color bars based on impact
                    colors = ['red' if change < -20 else 'orange' if change < -10 else 'yellow' 
                             for change in price_changes]
                    
                    fig_stress = go.Figure()
                    fig_stress.add_trace(go.Bar(
                        x=scenario_names,
                        y=price_changes,
                        marker_color=colors,
                        text=[f"{change:+.1f}%" for change in price_changes],
                        textposition='auto'
                    ))
                    
                    fig_stress.update_layout(
                        title=f"Stress Test Results - {target_symbol}",
                        xaxis_title="Stress Scenario",
                        yaxis_title="Price Change (%)",
                        template='plotly_white',
                        height=400
                    )
                    
                    # Add zero line
                    fig_stress.add_hline(y=0, line_dash="dash", line_color="black")
                    
                    st.plotly_chart(fig_stress, use_container_width=True)
                    
                    # Detailed results table
                    stress_df = pd.DataFrame([
                        {
                            'Scenario': scenario,
                            'Current Price': f"${results['current_price']:.2f}",
                            'Stressed Price': f"${results['stressed_price']:.2f}",
                            'Price Change': f"{results['price_change_percent']:+.1f}%",
                            'Description': DEFAULT_STRESS_SCENARIOS[scenario]['description']
                        }
                        for scenario, results in stress_results.items()
                    ])
                    
                    st.dataframe(stress_df, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error(f"Error running stress tests: {str(e)}")
