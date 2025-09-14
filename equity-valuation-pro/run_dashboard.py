"""
Launch script for Equity Valuation Dashboard
Easy way to start the application with proper configuration
"""

import os
import sys
import subprocess
import webbrowser
import time
from datetime import datetime

def print_banner():
    """Print application banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║                  📊 EQUITY VALUATION PRO 📊                  ║
    ║                                                               ║
    ║            Professional Investment Analysis Platform          ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_streamlit():
    """Check if Streamlit is installed"""
    try:
        import streamlit
        return True
    except ImportError:
        return False


def check_dependencies():
    """Check critical dependencies"""
    critical_deps = ['pandas', 'numpy', 'plotly', 'yfinance']
    missing = []
    
    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    return missing


def get_available_port():
    """Find an available port for the Streamlit app"""
    import socket
    
    for port in range(8501, 8600):  # Try ports 8501-8599
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    return 8501  # Default fallback


def launch_streamlit():
    """Launch the Streamlit application"""
    print("🚀 Starting Equity Valuation Dashboard...")
    print("⏳ Please wait while the application loads...")
    print()
    
    # Find available port
    port = get_available_port()
    
    # Set environment variables for better performance
    env = os.environ.copy()
    env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    env['STREAMLIT_SERVER_HEADLESS'] = 'true'
    
    # Build command
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 
        'src/main.py',
        '--server.port', str(port),
        '--server.address', 'localhost',
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false'
    ]
    
    try:
        # Start Streamlit
        process = subprocess.Popen(cmd, env=env)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open browser
        url = f"http://localhost:{port}"
        print(f"🌐 Opening dashboard at: {url}")
        webbrowser.open(url)
        
        print()
        print("✅ Dashboard launched successfully!")
        print(f"📊 Access your dashboard at: {url}")
        print()
        print("💡 Pro Tips:")
        print("   • Try stocks like: AAPL, MSFT, GOOGL, AMZN, TSLA")
        print("   • Build DCF models with custom assumptions")
        print("   • Run scenario analysis for comprehensive valuation")
        print("   • Use Ctrl+C to stop the server")
        print()
        print("🎯 Ready to impress recruiters with professional analysis!")
        print("="*60)
        
        # Keep the process running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down dashboard...")
            process.terminate()
            process.wait()
            print("✅ Dashboard stopped.")
            
    except Exception as e:
        print(f"❌ Error launching dashboard: {str(e)}")
        return False
    
    return True


def run_quick_setup():
    """Run quick setup if needed"""
    print("🔧 Running quick setup check...")
    
    # Check if main files exist
    required_files = ['src/main.py', 'requirements.txt']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        print("Please run setup.py first or check your project structure.")
        return False
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Installing missing dependencies...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_deps)
            print("✅ Dependencies installed!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies automatically.")
            print("Please run: pip install -r requirements.txt")
            return False
    
    print("✅ Setup check complete!")
    return True


def display_system_info():
    """Display system information"""
    print("🖥️  System Information:")
    print(f"   Python version: {sys.version.split()[0]}")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check Streamlit version
    try:
        import streamlit as st
        print(f"   Streamlit version: {st.__version__}")
    except:
        print("   Streamlit: Not installed")
    
    print()


def show_menu():
    """Show interactive menu"""
    print("📋 Launch Options:")
    print("   1. 🚀 Launch Dashboard (Default)")
    print("   2. 🛠️  Run Setup First")
    print("   3. 📊 Launch with Sample Data")
    print("   4. 🧪 Run Tests")
[O    print("   5. 📖 Show Documentation")
    print("   6. ❌ Exit")
    print()
    
    choice = input("Select option [1]: ").strip()
    return choice if choice else "1"


def run_tests():
    """Run basic application tests"""
    print("🧪 Running application tests...")
    
    try:
        # Import test
        sys.path.append('src')
        from data.market_data import MarketDataProvider
        from models.dcf import DCFModel
        from utils.calculations import FinancialCalculations
        
        print("✅ All modules imported successfully")
        
        # Basic functionality test
        provider = MarketDataProvider()
        print("✅ Market data provider initialized")
        
        # Test DCF model
        dcf = DCFModel("TEST")
        print("✅ DCF model initialized")
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def show_documentation():
    """Display quick documentation"""
    docs = """
    📖 QUICK START GUIDE
    ═══════════════════════════════════════════════════════════════
    
    🎯 Getting Started:
    1. Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)
    2. Click "Load Company Data"
    3. Explore company overview and charts
    4. Build DCF models with custom assumptions
    
    📊 Key Features:
    • Real-time market data integration
    • Professional DCF modeling with scenario analysis
    • Interactive price charts and technical indicators
    • Comprehensive financial ratio analysis
    • Export capabilities for presentations
    
    🚀 Pro Tips for IB Applications:
    • Use real companies you're interested in
    • Experiment with different assumptions
    • Compare scenarios (Bear/Base/Bull)
    • Take screenshots of your analysis
    • Understand the methodology behind calculations
    
    🔧 Troubleshooting:
    • If data doesn't load: Check internet connection
    • If calculations fail: Verify input assumptions
    • If app crashes: Check console for error messages
    • For help: Check README.md or GitHub issues
    
    ═══════════════════════════════════════════════════════════════
    """
    print(docs)
    input("\nPress Enter to continue...")


def main():
    """Main launcher function"""
    print_banner()
    display_system_info()
    
    # Check if Streamlit is available
    if not check_streamlit():
        print("❌ Streamlit is not installed!")
        print("Please install it with: pip install streamlit")
        return
    
    while True:
        choice = show_menu()
        
        if choice == "1":
            # Launch dashboard
            if run_quick_setup():
                launch_streamlit()
            break
            
        elif choice == "2":
            # Run setup
            print("🛠️  Running setup...")
            try:
                import setup
                setup.main()
            except ImportError:
                print("❌ setup.py not found. Please ensure it exists in the project root.")
            input("\nPress Enter to continue...")
            
        elif choice == "3":
            # Launch with sample data
            print("📊 Launching with sample data...")
            if run_quick_setup():
                # Set environment variable for sample mode
                os.environ['USE_SAMPLE_DATA'] = 'true'
                launch_streamlit()
            break
            
        elif choice == "4":
            # Run tests
            run_tests()
            input("\nPress Enter to continue...")
            
        elif choice == "5":
            # Show documentation
            show_documentation()
            
        elif choice == "6":
            # Exit
            print("👋 Goodbye! Good luck with your investment banking applications!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-6.")
            time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("Please check your setup and try again.")
