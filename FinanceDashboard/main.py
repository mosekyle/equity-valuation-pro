import streamlit as st
import sys
import os
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from src.dashboard import overview, company_analysis, sector_comparison

# Page configuration
st.set_page_config(
    page_title="Equity Valuation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
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
    }
    .sidebar-section {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">📊 Equity Valuation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Professional-Grade Investment Analysis Platform**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    # Page selection
    pages = {
        "🏠 Dashboard Overview": "overview",
        "🔍 Company Analysis": "company_analysis",
        "📊 Sector Comparison": "sector_comparison"
    }
    
    selected_page = st.sidebar.selectbox(
        "Select Analysis Type",
        list(pages.keys()),
        index=0
    )
    
    # Display selected page
    page_key = pages[selected_page]
    
    if page_key == "overview":
        overview.show()
    elif page_key == "company_analysis":
        company_analysis.show()
    elif page_key == "sector_comparison":
        sector_comparison.show()
    
    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard provides institutional-grade equity valuation "
        "capabilities including DCF modeling, comparable analysis, "
        "and real-time financial data integration."
    )
    
    st.sidebar.markdown("### Features")
    st.sidebar.markdown("""
    - 📈 Real-time market data
    - 💰 DCF valuation models
    - 📊 Comparable analysis
    - 🎯 Sensitivity analysis
    - 📋 Export capabilities
    """)

if __name__ == "__main__":
    main()
