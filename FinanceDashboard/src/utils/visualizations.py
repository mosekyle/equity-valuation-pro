import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

def create_market_overview_chart(market_data: Dict) -> go.Figure:
    """Create market overview chart with indices performance."""
    
    if not market_data:
        return go.Figure()
    
    # Extract data
    indices = list(market_data.keys())
    prices = [data.get('price', 0) for data in market_data.values()]
    changes = [data.get('change_percent', 0) for data in market_data.values()]
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Index Levels', 'Daily Change %'),
        vertical_spacing=0.1
    )
    
    # Index levels (top chart)
    fig.add_trace(
        go.Bar(x=indices, y=prices, name='Index Level', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Daily changes (bottom chart)
    colors = ['green' if change >= 0 else 'red' for change in changes]
    fig.add_trace(
        go.Bar(x=indices, y=changes, name='Daily Change %', marker_color=colors),
        row=2, col=1
    )
    
    fig.update_layout(
        title="Market Overview",
        height=500,
        showlegend=False
    )
    
    return fig

def create_performance_chart(performance_data: pd.DataFrame) -> go.Figure:
    """Create performance comparison chart."""
    
    if performance_data.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    periods = ['1 Day', '1 Week', '1 Month', '3 Months', '1 Year']
    
    for period in periods:
        if period in performance_data.columns:
            fig.add_trace(go.Scatter(
                x=performance_data.index,
                y=performance_data[period],
                mode='lines+markers',
                name=period,
                line=dict(width=2),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title="Performance Comparison Across Time Periods",
        xaxis_title="Companies",
        yaxis_title="Return (%)",
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_dcf_waterfall_chart(valuation_components: Dict) -> go.Figure:
    """Create DCF valuation waterfall chart."""
    
    components = [
        'PV of Projection Period',
        'PV of Terminal Value',
        'Enterprise Value',
        'Less: Debt',
        'Plus: Cash',
        'Equity Value'
    ]
    
    values = [
        valuation_components.get('pv_projection_period', 0) / 1e6,
        valuation_components.get('pv_terminal_value', 0) / 1e6,
        0,  # This will be calculated
        -valuation_components.get('total_debt', 0) / 1e6,
        valuation_components.get('total_cash', 0) / 1e6,
        0   # This will be calculated
    ]
    
    # Calculate enterprise value and equity value
    values[2] = values[0] + values[1]  # Enterprise value
    values[5] = values[2] + values[3] + values[4]  # Equity value
    
    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
        name="DCF Valuation",
        orientation="v",
        measure=["relative", "relative", "total", "relative", "relative", "total"],
        x=components,
        textposition="auto",
        text=[f"${v:.0f}M" for v in values],
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="DCF Valuation Waterfall",
        showlegend=False,
        height=500,
        yaxis_title="Value ($ Millions)"
    )
    
    return fig

def create_sensitivity_heatmap(sensitivity_data: pd.DataFrame, 
                              x_label: str = "Terminal Growth Rate", 
                              y_label: str = "Discount Rate") -> go.Figure:
    """Create sensitivity analysis heatmap."""
    
    if sensitivity_data.empty:
        return go.Figure()
    
    fig = go.Figure(data=go.Heatmap(
        z=sensitivity_data.values,
        x=sensitivity_data.columns,
        y=sensitivity_data.index,
        colorscale='RdYlGn',
        text=sensitivity_data.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Sensitivity Analysis Heatmap",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=400
    )
    
    return fig

def create_comparable_analysis_chart(comp_data: pd.DataFrame, 
                                   x_metric: str, y_metric: str, 
                                   size_metric: str = None,
                                   target_company: str = None) -> go.Figure:
    """Create comparable company analysis scatter plot."""
    
    if comp_data.empty or x_metric not in comp_data.columns or y_metric not in comp_data.columns:
        return go.Figure()
    
    # Filter out invalid data
    valid_data = comp_data[(comp_data[x_metric] > 0) & (comp_data[y_metric] > 0)].copy()
    
    if valid_data.empty:
        return go.Figure()
    
    fig = px.scatter(
        valid_data,
        x=x_metric,
        y=y_metric,
        size=size_metric if size_metric and size_metric in valid_data.columns else None,
        hover_name=valid_data.index,
        title=f"{y_metric} vs {x_metric}",
        labels={x_metric: x_metric.replace('_', ' ').title(),
                y_metric: y_metric.replace('_', ' ').title()}
    )
    
    # Highlight target company if specified
    if target_company and target_company in valid_data.index:
        target_row = valid_data.loc[target_company]
        fig.add_trace(go.Scatter(
            x=[target_row[x_metric]],
            y=[target_row[y_metric]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name=f'{target_company} (Target)',
            showlegend=True
        ))
    
    fig.update_layout(height=500)
    return fig

def create_sector_performance_chart(sector_data: Dict) -> go.Figure:
    """Create sector performance comparison chart."""
    
    if not sector_data:
        return go.Figure()
    
    # Extract performance data
    companies = []
    performance_1m = []
    performance_1y = []
    market_caps = []
    
    for ticker, data in sector_data.items():
        if 'performance' in data:
            companies.append(data.get('longName', ticker)[:20])
            performance_1m.append(data['performance'].get('1mo', 0))
            performance_1y.append(data['performance'].get('1y', 0))
            market_caps.append(data.get('marketCap', 0))
    
    if not companies:
        return go.Figure()
    
    # Create bubble chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=performance_1m,
        y=performance_1y,
        mode='markers+text',
        marker=dict(
            size=[mc/1e9 for mc in market_caps],  # Size by market cap (billions)
            sizemode='area',
            sizeref=2.*max(market_caps)/1e9/(40.**2),
            sizemin=4,
            color=performance_1y,
            colorscale='RdYlGn',
            colorbar=dict(title="1-Year Return (%)"),
            line=dict(width=2)
        ),
        text=companies,
        textposition="middle center",
        textfont=dict(size=10),
        name="Companies"
    ))
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Sector Performance Analysis (Size = Market Cap)",
        xaxis_title="1-Month Return (%)",
        yaxis_title="1-Year Return (%)",
        height=600,
        showlegend=False
    )
    
    return fig

def create_valuation_comparison_chart(valuation_data: pd.DataFrame) -> go.Figure:
    """Create valuation multiples comparison chart."""
    
    if valuation_data.empty:
        return go.Figure()
    
    # Select key valuation metrics
    metrics = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda']
    metric_names = ['P/E Ratio', 'P/B Ratio', 'P/S Ratio', 'EV/EBITDA']
    
    # Filter valid data
    valid_metrics = []
    valid_names = []
    
    for metric, name in zip(metrics, metric_names):
        if metric in valuation_data.columns:
            valid_data = valuation_data[valuation_data[metric] > 0][metric]
            if len(valid_data) > 1:
                valid_metrics.append(valid_data)
                valid_names.append(name)
    
    if not valid_metrics:
        return go.Figure()
    
    # Create box plots
    fig = go.Figure()
    
    for i, (data, name) in enumerate(zip(valid_metrics, valid_names)):
        fig.add_trace(go.Box(
            y=data,
            name=name,
            boxpoints='outliers',
            marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
        ))
    
    fig.update_layout(
        title="Valuation Multiples Distribution",
        yaxis_title="Multiple (x)",
        height=500,
        showlegend=False
    )
    
    return fig

def create_financial_health_radar(health_metrics: Dict, company_name: str = "") -> go.Figure:
    """Create financial health radar chart."""
    
    if not health_metrics:
        return go.Figure()
    
    # Define metrics for radar chart
    metrics_mapping = {
        'ROE': 'return_on_equity',
        'ROA': 'return_on_assets',
        'Gross Margin': 'gross_margin',
        'Operating Margin': 'operating_margin',
        'Profit Margin': 'profit_margin',
        'Current Ratio': 'current_ratio'
    }
    
    categories = []
    values = []
    
    for display_name, metric_key in metrics_mapping.items():
        if metric_key in health_metrics:
            categories.append(display_name)
            # Normalize values for radar chart
            value = health_metrics[metric_key]
            if 'margin' in metric_key.lower() or 'roe' in metric_key.lower() or 'roa' in metric_key.lower():
                values.append(value * 100)  # Convert to percentage
            else:
                values.append(value)
    
    if not categories:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=company_name or 'Company',
        line=dict(color='blue', width=2),
        fillcolor='rgba(0, 100, 255, 0.1)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1] if values else [0, 100]
            )),
        showlegend=True,
        title=f"Financial Health Profile{' - ' + company_name if company_name else ''}",
        height=500
    )
    
    return fig

def create_growth_analysis_chart(growth_data: pd.DataFrame) -> go.Figure:
    """Create growth analysis visualization."""
    
    if growth_data.empty:
        return go.Figure()
    
    # Create 2x2 subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Revenue vs Earnings Growth',
            'Growth Distribution',
            'Growth vs Valuation',
            'Growth Correlation'
        ),
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Revenue vs Earnings Growth scatter
    if all(col in growth_data.columns for col in ['revenue_growth', 'earnings_growth']):
        fig.add_trace(
            go.Scatter(
                x=growth_data['revenue_growth'],
                y=growth_data['earnings_growth'],
                mode='markers',
                text=growth_data.index,
                name='Companies'
            ),
            row=1, col=1
        )
    
    # Growth distribution histogram
    if 'revenue_growth' in growth_data.columns:
        fig.add_trace(
            go.Histogram(
                x=growth_data['revenue_growth'],
                name='Revenue Growth Distribution',
                nbinsx=20
            ),
            row=1, col=2
        )
    
    # Growth vs Valuation
    if all(col in growth_data.columns for col in ['revenue_growth', 'pe_ratio']):
        fig.add_trace(
            go.Scatter(
                x=growth_data['revenue_growth'],
                y=growth_data['pe_ratio'],
                mode='markers',
                text=growth_data.index,
                name='Growth vs P/E'
            ),
            row=2, col=1
        )
    
    # Growth correlation matrix
    growth_cols = [col for col in growth_data.columns if 'growth' in col.lower()]
    if len(growth_cols) > 1:
        corr_matrix = growth_data[growth_cols].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdYlBu',
                name='Growth Correlation'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Growth Analysis Dashboard",
        height=800,
        showlegend=False
    )
    
    return fig

def create_portfolio_optimization_chart(returns: np.ndarray, risks: np.ndarray, 
                                      sharpe_ratios: np.ndarray) -> go.Figure:
    """Create efficient frontier chart for portfolio optimization."""
    
    fig = go.Figure()
    
    # Efficient frontier
    fig.add_trace(go.Scatter(
        x=risks,
        y=returns,
        mode='markers',
        marker=dict(
            size=8,
            color=sharpe_ratios,
            colorscale='viridis',
            colorbar=dict(title="Sharpe Ratio"),
            showscale=True
        ),
        text=[f"Sharpe: {sr:.3f}" for sr in sharpe_ratios],
        name='Portfolio Combinations'
    ))
    
    # Highlight optimal portfolio (max Sharpe ratio)
    max_sharpe_idx = np.argmax(sharpe_ratios)
    fig.add_trace(go.Scatter(
        x=[risks[max_sharpe_idx]],
        y=[returns[max_sharpe_idx]],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Optimal Portfolio'
    ))
    
    fig.update_layout(
        title="Efficient Frontier - Portfolio Optimization",
        xaxis_title="Risk (Standard Deviation)",
        yaxis_title="Expected Return",
        height=500
    )
    
    return fig

def create_risk_return_scatter(risk_return_data: pd.DataFrame, 
                             risk_col: str = 'risk', 
                             return_col: str = 'return',
                             size_col: str = None) -> go.Figure:
    """Create risk-return scatter plot."""
    
    if risk_return_data.empty:
        return go.Figure()
    
    fig = px.scatter(
        risk_return_data,
        x=risk_col,
        y=return_col,
        size=size_col,
        hover_name=risk_return_data.index,
        title="Risk vs Return Analysis",
        labels={
            risk_col: risk_col.replace('_', ' ').title(),
            return_col: return_col.replace('_', ' ').title()
        }
    )
    
    # Add risk-free rate line (if applicable)
    if return_col in risk_return_data.columns:
        max_return = risk_return_data[return_col].max()
        max_risk = risk_return_data[risk_col].max()
        
        fig.add_shape(
            type="line",
            x0=0, y0=0,
            x1=max_risk, y1=max_return * 0.3,  # Approximate risk-free rate
            line=dict(color="red", width=2, dash="dash"),
        )
    
    fig.update_layout(height=500)
    return fig

def create_monte_carlo_results(simulation_results: List[float], 
                             current_price: float = 0,
                             confidence_levels: List[float] = [0.05, 0.95]) -> go.Figure:
    """Create Monte Carlo simulation results visualization."""
    
    if not simulation_results:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Distribution', 'Confidence Intervals'),
        vertical_spacing=0.1
    )
    
    # Histogram of results
    fig.add_trace(
        go.Histogram(
            x=simulation_results,
            nbinsx=50,
            name='Price Distribution',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Add current price line
    if current_price > 0:
        fig.add_vline(
            x=current_price,
            line_dash="dash",
            line_color="red",
            annotation_text="Current Price",
            row=1, col=1
        )
    
    # Add confidence interval markers
    percentiles = [np.percentile(simulation_results, cl*100) for cl in confidence_levels]
    
    for i, (cl, percentile) in enumerate(zip(confidence_levels, percentiles)):
        fig.add_vline(
            x=percentile,
            line_dash="dot",
            line_color="green",
            annotation_text=f"{cl*100:.0f}% CI",
            row=1, col=1
        )
    
    # Box plot
    fig.add_trace(
        go.Box(
            x=simulation_results,
            name='Price Range',
            orientation='h'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="Monte Carlo Simulation Results",
        height=600,
        showlegend=False
    )
    
    return fig

def create_dividend_analysis_chart(dividend_data: pd.DataFrame) -> go.Figure:
    """Create dividend analysis visualization."""
    
    if dividend_data.empty:
        return go.Figure()
    
    # Create subplot for dividend metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Dividend Yield vs Payout Ratio',
            'Dividend Yield Distribution',
            'Payout Ratio Distribution',
            'Dividend Growth Rate'
        )
    )
    
    # Dividend yield vs payout ratio scatter
    if all(col in dividend_data.columns for col in ['dividend_yield', 'payout_ratio']):
        fig.add_trace(
            go.Scatter(
                x=dividend_data['dividend_yield'],
                y=dividend_data['payout_ratio'],
                mode='markers',
                text=dividend_data.index,
                name='Companies',
                marker=dict(size=10)
            ),
            row=1, col=1
        )
    
    # Dividend yield distribution
    if 'dividend_yield' in dividend_data.columns:
        fig.add_trace(
            go.Histogram(
                x=dividend_data['dividend_yield'],
                nbinsx=20,
                name='Yield Distribution'
            ),
            row=1, col=2
        )
    
    # Payout ratio distribution
    if 'payout_ratio' in dividend_data.columns:
        fig.add_trace(
            go.Histogram(
                x=dividend_data['payout_ratio'],
                nbinsx=20,
                name='Payout Distribution'
            ),
            row=2, col=1
        )
    
    # Dividend growth (if available)
    if 'dividend_growth' in dividend_data.columns:
        fig.add_trace(
            go.Bar(
                x=dividend_data.index,
                y=dividend_data['dividend_growth'],
                name='Dividend Growth'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Dividend Analysis Dashboard",
        height=700,
        showlegend=False
    )
    
    return fig
