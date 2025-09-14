"""
Setup script for Equity Valuation Dashboard
Run this to verify your installation and create sample data
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Force UTF-8 encoding for Windows
if os.name == 'nt':  # Windows
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'yfinance',
        'scipy', 'scikit-learn', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n🎉 All dependencies are installed!")
    return True


def create_sample_data():
    """Create sample data for testing"""
    print("📊 Creating sample data...")
    
    # Create sample directory if it doesn't exist
    sample_dir = "data/sample"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Sample company data
    sample_companies = {
        'AAPL': {
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'market_cap': 3000000000000,
            'current_price': 175.43
        },
        'MSFT': {
            'name': 'Microsoft Corporation',
            'sector': 'Technology',
            'market_cap': 2800000000000,
            'current_price': 378.85
        },
        'GOOGL': {
            'name': 'Alphabet Inc.',
            'sector': 'Technology',
            'market_cap': 1700000000000,
            'current_price': 138.21
        }
    }
    
    # Save sample data
    sample_df = pd.DataFrame(sample_companies).T
    sample_df.to_csv(f"{sample_dir}/sample_companies.csv")
    
    # Create sample stock price data
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='D')
    
    for symbol in sample_companies.keys():
        # Generate random walk stock prices
        np.random.seed(hash(symbol) % 2**32)  # Reproducible random data
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [100]  # Starting price
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        stock_data = pd.DataFrame({
            'date': dates,
            'close': prices[:len(dates)],
            'volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        stock_data.to_csv(f"{sample_dir}/{symbol}_sample_data.csv", index=False)
    
    print(f"✅ Sample data created in {sample_dir}/")


def test_market_data():
    """Test market data functionality"""
    print("🔄 Testing market data connection...")
    
    try:
        import yfinance as yf
        
        # Test with Apple
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        
        if 'currentPrice' in info:
            print(f"✅ Successfully connected to market data")
            print(f"   AAPL current price: ${info.get('currentPrice', 'N/A')}")
            return True
        else:
            print("⚠️  Market data connection successful but limited data")
            return True
            
    except Exception as e:
        print(f"❌ Market data connection failed: {str(e)}")
        print("   You can still use the app with sample data")
        return False


def verify_project_structure():
    """Verify project directory structure"""
    print("📁 Verifying project structure...")
    
    required_dirs = [
        'src',
        'src/models',
        'src/data',
        'src/dashboard',
        'src/utils',
        'data',
        'data/raw',
        'data/processed',
        'data/sample',
        'notebooks',
        'tests',
        'docs'
    ]
    
    required_files = [
        'src/main.py',
        'src/models/dcf.py',
        'src/data/market_data.py',
        'src/utils/calculations.py',
        'requirements.txt',
        'README.md',
        '.gitignore',
        '.streamlit/config.toml'
    ]
    
    missing_items = []
    
    # Check directories
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/ - OK")
        else:
            print(f"❌ {directory}/ - MISSING")
            missing_items.append(directory)
            # Create missing directory
            os.makedirs(directory, exist_ok=True)
            print(f"   📁 Created {directory}/")
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} - OK")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_items.append(file_path)
    
    if not missing_items:
        print("🎉 Project structure is complete!")
    else:
        print(f"\n⚠️  Some items were missing but directories have been created.")
    
    return len(missing_items) == 0


def create_init_files():
    """Create __init__.py files for Python packages"""
    print("🐍 Creating Python package files...")
    
    init_files = [
        'src/__init__.py',
        'src/models/__init__.py',
        'src/data/__init__.py',
        'src/dashboard/__init__.py',
        'src/utils/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('"""Package initialization file"""\n')
            print(f"✅ Created {init_file}")
        else:
            print(f"✅ {init_file} already exists")


def run_basic_tests():
    """Run basic functionality tests"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        sys.path.append('src')
        
        print("  Testing market data module...")
        from data.market_data import MarketDataProvider
        provider = MarketDataProvider()
        print("  ✅ Market data module imported successfully")
        
        print("  Testing calculations module...")
        from utils.calculations import FinancialCalculations
        calc = FinancialCalculations()
        print("  ✅ Calculations module imported successfully")
        
        print("  Testing DCF model...")
        from models.dcf import DCFModel, DCFAssumptions
        dcf = DCFModel("TEST")
        print("  ✅ DCF model imported successfully")
        
        print("🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def display_next_steps():
    """Display next steps for the user"""
    print("\n" + "="*60)
    print("🚀 SETUP COMPLETE! Next Steps:")
    print("="*60)
    print()
    print("1. 📊 Start the dashboard:")
    print("   streamlit run src/main.py")
    print()
    print("2. 🌐 Open your browser to:")
    print("   http://localhost:8501")
    print()
    print("3. 🎯 Try these sample stocks:")
    print("   • AAPL (Apple)")
    print("   • MSFT (Microsoft)")
    print("   • GOOGL (Google)")
    print("   • AMZN (Amazon)")
    print("   • TSLA (Tesla)")
    print()
    print("4. 📈 Build your first DCF model:")
    print("   • Load a company")
    print("   • Adjust assumptions in the DCF section")
    print("   • Run scenario analysis")
    print()
    print("5. 📚 Learn more:")
    print("   • Check out the notebooks/ folder for examples")
    print("   • Read the README.md for detailed documentation")
    print("   • Explore the src/ folder to understand the code")
    print()
    print("🎉 Happy analyzing! Your professional equity valuation")
    print("   platform is ready to impress recruiters!")
    print("="*60)


def main():
    """Main setup function"""
    print("🏗️  Setting up Equity Valuation Pro Dashboard")
    print("="*50)
    print()
    
    # Step 1: Verify project structure
    print("Step 1: Verifying project structure...")
    verify_project_structure()
    print()
    
    # Step 2: Create init files
    print("Step 2: Creating package files...")
    create_init_files()
    print()
    
    # Step 3: Check dependencies
    print("Step 3: Checking dependencies...")
    deps_ok = check_dependencies()
    print()
    
    # Step 4: Create sample data
    print("Step 4: Creating sample data...")
    create_sample_data()
    print()
    
    # Step 5: Test market data connection
    print("Step 5: Testing market data connection...")
    market_ok = test_market_data()
    print()
    
    # Step 6: Run basic tests
    print("Step 6: Running basic tests...")
    tests_ok = run_basic_tests()
    print()
    
    # Final status
    if deps_ok and tests_ok:
        print("✅ SETUP SUCCESSFUL!")
        display_next_steps()
    else:
        print("⚠️  SETUP COMPLETED WITH WARNINGS")
        print("Some components may not work properly.")
        print("Please check the errors above and install missing dependencies.")
        print()
        if not deps_ok:
            print("Run: pip install -r requirements.txt")
        print()
        print("You can still start the app with: streamlit run src/main.py")


if __name__ == "__main__":
    main()
