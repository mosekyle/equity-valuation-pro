import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
import csv
from datetime import datetime
import io
import base64
from pathlib import Path

class ExportManager:
    """
    Handle various export formats for analysis results.
    """
    
    def __init__(self):
        self.export_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export_to_excel(self, data_dict: Dict, filename: str = None) -> bytes:
        """Export analysis results to Excel format."""
        
        if filename is None:
            filename = f"equity_analysis_{self.export_timestamp}.xlsx"
        
        # Create a BytesIO buffer
        buffer = io.BytesIO()
        
        try:
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                
                # Summary sheet
                if 'summary' in data_dict:
                    summary_df = self._create_summary_dataframe(data_dict['summary'])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # DCF Analysis
                if 'dcf_analysis' in data_dict:
                    dcf_data = data_dict['dcf_analysis']
                    
                    # DCF Assumptions
                    if 'assumptions' in dcf_data:
                        assumptions_df = self._create_assumptions_dataframe(dcf_data['assumptions'])
                        assumptions_df.to_excel(writer, sheet_name='DCF_Assumptions', index=False)
                    
                    # Financial Projections
                    if 'projections' in dcf_data:
                        projections_df = dcf_data['projections']
                        if isinstance(projections_df, pd.DataFrame):
                            projections_df.to_excel(writer, sheet_name='DCF_Projections')
                    
                    # Valuation Results
                    if 'valuation_results' in dcf_data:
                        valuation_df = self._create_valuation_dataframe(dcf_data['valuation_results'])
                        valuation_df.to_excel(writer, sheet_name='DCF_Valuation', index=False)
                    
                    # Sensitivity Analysis
                    if 'sensitivity_analysis' in dcf_data:
                        sensitivity_df = dcf_data['sensitivity_analysis']
                        if isinstance(sensitivity_df, pd.DataFrame):
                            sensitivity_df.to_excel(writer, sheet_name='Sensitivity_Analysis')
                
                # Comparable Analysis
                if 'comparable_analysis' in data_dict:
                    comp_data = data_dict['comparable_analysis']
                    
                    # Comp Table
                    if 'comp_table' in comp_data:
                        comp_table_df = comp_data['comp_table']
                        if isinstance(comp_table_df, pd.DataFrame):
                            comp_table_df.to_excel(writer, sheet_name='Comparable_Companies')
                    
                    # Peer Statistics
                    if 'peer_statistics' in comp_data:
                        peer_stats_df = self._create_peer_stats_dataframe(comp_data['peer_statistics'])
                        peer_stats_df.to_excel(writer, sheet_name='Peer_Statistics', index=False)
                    
                    # Relative Valuation
                    if 'relative_valuation' in comp_data:
                        rel_val_df = self._create_relative_valuation_dataframe(comp_data['relative_valuation'])
                        rel_val_df.to_excel(writer, sheet_name='Relative_Valuation', index=False)
                
                # Financial Data
                if 'financial_data' in data_dict:
                    financial_df = self._create_financial_data_dataframe(data_dict['financial_data'])
                    financial_df.to_excel(writer, sheet_name='Financial_Data', index=False)
                
                # Market Data
                if 'market_data' in data_dict:
                    market_df = self._create_market_data_dataframe(data_dict['market_data'])
                    market_df.to_excel(writer, sheet_name='Market_Data', index=False)
            
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            # Return empty bytes if export fails
            return b''
    
    def export_to_csv(self, dataframe: pd.DataFrame, filename: str = None) -> str:
        """Export DataFrame to CSV format."""
        
        if filename is None:
            filename = f"equity_data_{self.export_timestamp}.csv"
        
        try:
            csv_buffer = io.StringIO()
            dataframe.to_csv(csv_buffer, index=True)
            return csv_buffer.getvalue()
        except Exception as e:
            return ""
    
    def export_to_json(self, data_dict: Dict, filename: str = None) -> str:
        """Export analysis results to JSON format."""
        
        if filename is None:
            filename = f"equity_analysis_{self.export_timestamp}.json"
        
        try:
            # Convert numpy types to native Python types for JSON serialization
            cleaned_data = self._clean_data_for_json(data_dict)
            return json.dumps(cleaned_data, indent=2, default=str)
        except Exception as e:
            return "{}"
    
    def create_analysis_report(self, analysis_results: Dict, company_info: Dict) -> str:
        """Create a comprehensive analysis report in text format."""
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("EQUITY VALUATION ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Company Information
        report_lines.append("COMPANY INFORMATION")
        report_lines.append("-" * 50)
        report_lines.append(f"Company Name: {company_info.get('longName', 'N/A')}")
        report_lines.append(f"Ticker Symbol: {company_info.get('symbol', 'N/A')}")
        report_lines.append(f"Sector: {company_info.get('sector', 'N/A')}")
        report_lines.append(f"Industry: {company_info.get('industry', 'N/A')}")
        report_lines.append(f"Current Price: ${company_info.get('currentPrice', 0):.2f}")
        report_lines.append(f"Market Cap: ${company_info.get('marketCap', 0):,.0f}")
        report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Executive Summary
        if 'executive_summary' in analysis_results:
            summary = analysis_results['executive_summary']
            report_lines.append("EXECUTIVE SUMMARY")
            report_lines.append("-" * 50)
            
            if 'recommendation' in summary:
                report_lines.append(f"Investment Recommendation: {summary['recommendation']}")
            
            if 'target_price' in summary:
                report_lines.append(f"Target Price: ${summary['target_price']:.2f}")
            
            if 'upside_downside' in summary:
                report_lines.append(f"Upside/Downside: {summary['upside_downside']:+.1f}%")
            
            report_lines.append("")
        
        # DCF Analysis
        if 'dcf_analysis' in analysis_results:
            dcf_data = analysis_results['dcf_analysis']
            report_lines.append("DCF VALUATION ANALYSIS")
            report_lines.append("-" * 50)
            
            if 'valuation_results' in dcf_data:
                val_results = dcf_data['valuation_results']
                report_lines.append(f"Fair Value per Share: ${val_results.get('fair_value_per_share', 0):.2f}")
                report_lines.append(f"Enterprise Value: ${val_results.get('enterprise_value', 0):,.0f}")
                report_lines.append(f"Equity Value: ${val_results.get('equity_value', 0):,.0f}")
                report_lines.append(f"PV of Projection Period: ${val_results.get('pv_projection_period', 0):,.0f}")
                report_lines.append(f"PV of Terminal Value: ${val_results.get('pv_terminal_value', 0):,.0f}")
            
            if 'assumptions' in dcf_data:
                assumptions = dcf_data['assumptions']
                report_lines.append("")
                report_lines.append("Key Assumptions:")
                report_lines.append(f"  - Terminal Growth Rate: {assumptions.get('terminal_growth', 0)*100:.2f}%")
                report_lines.append(f"  - Discount Rate (WACC): {assumptions.get('discount_rate', 0)*100:.2f}%")
                report_lines.append(f"  - Tax Rate: {assumptions.get('tax_rate', 0)*100:.2f}%")
            
            report_lines.append("")
        
        # Comparable Analysis
        if 'comparable_analysis' in analysis_results:
            comp_data = analysis_results['comparable_analysis']
            report_lines.append("COMPARABLE COMPANY ANALYSIS")
            report_lines.append("-" * 50)
            
            if 'relative_valuation' in comp_data:
                rel_val = comp_data['relative_valuation']
                report_lines.append("Relative Valuation Results:")
                
                if 'average_implied_value' in rel_val:
                    report_lines.append(f"  - Average Implied Value: ${rel_val['average_implied_value']:.2f}")
                
                if 'median_implied_value' in rel_val:
                    report_lines.append(f"  - Median Implied Value: ${rel_val['median_implied_value']:.2f}")
                
                if 'upside_downside_median' in rel_val:
                    report_lines.append(f"  - Upside/Downside (Median): {rel_val['upside_downside_median']:+.1f}%")
            
            if 'peer_statistics' in comp_data:
                report_lines.append("")
                report_lines.append("Peer Group Statistics (Median):")
                peer_stats = comp_data['peer_statistics']
                
                for metric in ['pe_ratio', 'pb_ratio', 'ev_ebitda', 'ps_ratio']:
                    if metric in peer_stats and 'median' in peer_stats[metric]:
                        metric_name = metric.replace('_', '/').upper()
                        report_lines.append(f"  - {metric_name}: {peer_stats[metric]['median']:.2f}x")
            
            report_lines.append("")
        
        # Risk Factors
        report_lines.append("KEY RISK FACTORS")
        report_lines.append("-" * 50)
        
        beta = company_info.get('beta', 1.0)
        if beta > 1.5:
            report_lines.append(f"• High Beta Risk: Beta of {beta:.2f} indicates high market sensitivity")
        elif beta > 1.2:
            report_lines.append(f"• Moderate Beta Risk: Beta of {beta:.2f} indicates above-average market sensitivity")
        
        debt_to_equity = company_info.get('debtToEquity', 0) / 100 if company_info.get('debtToEquity') else 0
        if debt_to_equity > 1.0:
            report_lines.append(f"• High Leverage Risk: Debt-to-Equity ratio of {debt_to_equity:.2f}")
        elif debt_to_equity > 0.5:
            report_lines.append(f"• Moderate Leverage Risk: Debt-to-Equity ratio of {debt_to_equity:.2f}")
        
        pe_ratio = company_info.get('trailingPE', 0)
        if pe_ratio > 30:
            report_lines.append(f"• Valuation Risk: High P/E ratio of {pe_ratio:.1f}x may indicate overvaluation")
        
        report_lines.append("")
        
        # Investment Thesis
        report_lines.append("INVESTMENT THESIS")
        report_lines.append("-" * 50)
        
        # Generate basic thesis based on metrics
        thesis_points = []
        
        # Growth analysis
        revenue_growth = company_info.get('revenueGrowth', 0)
        if revenue_growth > 0.15:
            thesis_points.append(f"• Strong revenue growth of {revenue_growth*100:.1f}% indicates business expansion")
        elif revenue_growth > 0.05:
            thesis_points.append(f"• Moderate revenue growth of {revenue_growth*100:.1f}% shows steady business performance")
        
        # Profitability analysis
        roe = company_info.get('returnOnEquity', 0)
        if roe > 0.20:
            thesis_points.append(f"• Excellent ROE of {roe*100:.1f}% demonstrates efficient capital utilization")
        elif roe > 0.15:
            thesis_points.append(f"• Strong ROE of {roe*100:.1f}% indicates good profitability")
        
        # Market position
        market_cap = company_info.get('marketCap', 0)
        if market_cap > 100e9:
            thesis_points.append("• Large-cap company with established market position")
        elif market_cap > 10e9:
            thesis_points.append("• Mid-cap company with growth potential")
        
        for point in thesis_points:
            report_lines.append(point)
        
        if not thesis_points:
            report_lines.append("• Analysis based on current financial metrics and market conditions")
        
        report_lines.append("")
        
        # Disclaimer
        report_lines.append("DISCLAIMER")
        report_lines.append("-" * 50)
        report_lines.append("This analysis is for informational purposes only and should not be considered")
        report_lines.append("as investment advice. Past performance does not guarantee future results.")
        report_lines.append("Please consult with a qualified financial advisor before making investment decisions.")
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def _create_summary_dataframe(self, summary_data: Dict) -> pd.DataFrame:
        """Create summary DataFrame for export."""
        
        summary_items = []
        for key, value in summary_data.items():
            summary_items.append({
                'Metric': key.replace('_', ' ').title(),
                'Value': value
            })
        
        return pd.DataFrame(summary_items)
    
    def _create_assumptions_dataframe(self, assumptions: Dict) -> pd.DataFrame:
        """Create DCF assumptions DataFrame for export."""
        
        assumption_items = []
        for key, value in assumptions.items():
            if isinstance(value, list):
                for i, v in enumerate(value):
                    assumption_items.append({
                        'Assumption': f"{key.replace('_', ' ').title()} - Year {i+1}",
                        'Value': v
                    })
            else:
                assumption_items.append({
                    'Assumption': key.replace('_', ' ').title(),
                    'Value': value
                })
        
        return pd.DataFrame(assumption_items)
    
    def _create_valuation_dataframe(self, valuation_results: Dict) -> pd.DataFrame:
        """Create valuation results DataFrame for export."""
        
        valuation_items = []
        for key, value in valuation_results.items():
            valuation_items.append({
                'Component': key.replace('_', ' ').title(),
                'Value': value
            })
        
        return pd.DataFrame(valuation_items)
    
    def _create_peer_stats_dataframe(self, peer_stats: Dict) -> pd.DataFrame:
        """Create peer statistics DataFrame for export."""
        
        stats_data = []
        for metric, stats in peer_stats.items():
            if isinstance(stats, dict):
                stats_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Mean': stats.get('mean', 0),
                    'Median': stats.get('median', 0),
                    'Min': stats.get('min', 0),
                    'Max': stats.get('max', 0),
                    'Std Dev': stats.get('std', 0),
                    'Count': stats.get('count', 0)
                })
        
        return pd.DataFrame(stats_data)
    
    def _create_relative_valuation_dataframe(self, relative_val: Dict) -> pd.DataFrame:
        """Create relative valuation DataFrame for export."""
        
        rel_val_items = []
        for key, value in relative_val.items():
            rel_val_items.append({
                'Valuation Method': key.replace('_', ' ').title(),
                'Implied Value': value
            })
        
        return pd.DataFrame(rel_val_items)
    
    def _create_financial_data_dataframe(self, financial_data: Dict) -> pd.DataFrame:
        """Create financial data DataFrame for export."""
        
        financial_items = []
        for key, value in financial_data.items():
            financial_items.append({
                'Financial Metric': key.replace('_', ' ').title(),
                'Value': value
            })
        
        return pd.DataFrame(financial_items)
    
    def _create_market_data_dataframe(self, market_data: Dict) -> pd.DataFrame:
        """Create market data DataFrame for export."""
        
        market_items = []
        for key, value in market_data.items():
            market_items.append({
                'Market Metric': key.replace('_', ' ').title(),
                'Value': value
            })
        
        return pd.DataFrame(market_items)
    
    def _clean_data_for_json(self, data):
        """Clean data for JSON serialization."""
        
        if isinstance(data, dict):
            return {key: self._clean_data_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._clean_data_for_json(item) for item in data]
        elif isinstance(data, pd.DataFrame):
            return data.to_dict('records')
        elif isinstance(data, pd.Series):
            return data.to_dict()
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif pd.isna(data):
            return None
        else:
            return data
    
    def get_download_link(self, file_content: bytes, filename: str, file_type: str) -> str:
        """Generate download link for file content."""
        
        # Encode file content in base64
        b64_content = base64.b64encode(file_content).decode()
        
        # Create download link
        href = f'<a href="data:{file_type};base64,{b64_content}" download="{filename}">Download {filename}</a>'
        return href

# Utility functions for export operations
def export_analysis_to_excel(analysis_data: Dict, filename: str = None) -> bytes:
    """Export complete analysis to Excel file."""
    
    export_manager = ExportManager()
    return export_manager.export_to_excel(analysis_data, filename)

def export_dataframe_to_csv(dataframe: pd.DataFrame, filename: str = None) -> str:
    """Export DataFrame to CSV string."""
    
    export_manager = ExportManager()
    return export_manager.export_to_csv(dataframe, filename)

def create_comprehensive_report(analysis_results: Dict, company_info: Dict) -> str:
    """Create comprehensive analysis report."""
    
    export_manager = ExportManager()
    return export_manager.create_analysis_report(analysis_results, company_info)

def prepare_export_data(dcf_model, comp_analysis, company_info: Dict) -> Dict:
    """Prepare data for export from analysis models."""
    
    export_data = {
        'company_info': company_info,
        'export_timestamp': datetime.now().isoformat()
    }
    
    # Add DCF data if available
    if dcf_model and hasattr(dcf_model, 'assumptions'):
        try:
            valuation_results = dcf_model.calculate_fair_value()
            export_data['dcf_analysis'] = {
                'assumptions': dcf_model.assumptions,
                'projections': dcf_model.projections,
                'valuation_results': valuation_results
            }
        except:
            pass
    
    # Add comparable analysis data if available
    if comp_analysis:
        try:
            comp_table = comp_analysis.create_comp_table()
            peer_stats = comp_analysis.calculate_peer_statistics()
            relative_val = comp_analysis.relative_valuation()
            
            export_data['comparable_analysis'] = {
                'comp_table': comp_table,
                'peer_statistics': peer_stats,
                'relative_valuation': relative_val
            }
        except:
            pass
    
    return export_data
