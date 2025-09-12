# Overview

The Equity Valuation Dashboard is a professional-grade investment analysis platform built with Streamlit that provides comprehensive equity valuation capabilities. The application implements institutional-level financial models including Discounted Cash Flow (DCF) analysis, comparable company analysis, and sector comparison tools. It's designed to help investment professionals, analysts, and sophisticated investors perform detailed equity research and valuation analysis with real-time market data integration.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
The application uses Streamlit as the primary web framework with a modular dashboard architecture. The frontend is organized into three main sections: overview dashboard, individual company analysis, and sector comparison. Each dashboard component is implemented as a separate module in the `src/dashboard/` directory, promoting code reusability and maintainability. Custom CSS styling provides a professional appearance with metric cards, color-coded indicators, and responsive layouts.

## Backend Architecture
The backend follows a layered architecture pattern with clear separation of concerns:

- **Models Layer**: Contains valuation models (DCF, comparable analysis) implementing institutional-grade methodologies
- **Data Layer**: Handles market data fetching and financial statement processing using Yahoo Finance API
- **Utils Layer**: Provides shared utilities for calculations, visualizations, and export functionality
- **Dashboard Layer**: Contains UI components and user interaction logic

The architecture emphasizes modularity with each component having specific responsibilities and well-defined interfaces.

## Data Processing Pipeline
Financial data processing is centralized through the `FinancialDataProcessor` class which handles data fetching, cleaning, and standardization. The system implements caching mechanisms to reduce API calls and improve performance. Data validation and error handling ensure robust operation with various market conditions and data availability scenarios.

## Visualization Strategy
All charts and visualizations are built using Plotly, providing interactive and professional-grade financial charts. The visualization system includes DCF waterfall charts, sensitivity analysis heatmaps, sector performance comparisons, and market overview dashboards. Chart components are modularized for reuse across different dashboard sections.

## Export Capabilities
The system includes comprehensive export functionality supporting Excel format with multiple worksheets for different analysis components. Export data includes summary metrics, detailed projections, assumptions, and formatted reports suitable for professional presentations and documentation.

# External Dependencies

## Market Data Provider
- **Yahoo Finance API (yfinance)**: Primary source for real-time stock prices, financial statements, company information, and market indices
- **Purpose**: Provides comprehensive market data including historical prices, financial statements, key metrics, and company fundamentals

## Python Libraries
- **Streamlit**: Web application framework for creating the dashboard interface
- **Plotly**: Interactive charting and visualization library for financial charts
- **Pandas/NumPy**: Data manipulation and numerical computation
- **SciPy**: Advanced financial calculations including optimization for IRR calculations
- **OpenPyXL**: Excel file generation for analysis exports

## Data Sources
- **Market Indices**: Real-time data for major market indices (S&P 500, NASDAQ, etc.)
- **Financial Statements**: Income statements, balance sheets, and cash flow statements
- **Company Fundamentals**: Valuation ratios, financial metrics, and company information
- **Sector Data**: Industry classifications and peer company identification

## Containerization
- **Docker**: Application containerization for consistent deployment across environments
- **Streamlit Configuration**: Custom configuration for production deployment settings

The system is designed to be self-contained with minimal external dependencies, relying primarily on Yahoo Finance for data and standard Python libraries for computation and visualization.