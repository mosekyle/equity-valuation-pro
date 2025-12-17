# 🚀 Deployment Guide - Equity Valuation Pro

This guide covers multiple deployment options for your professional equity valuation platform, from local development to production-ready cloud deployment.

---

## 🎯 **Quick Start (Local Development)**

### **Prerequisites**
- Python 3.9+ installed
- Git installed
- 8GB+ RAM recommended
- Stable internet connection (for market data)

### **Setup Steps**
```bash
# 1. Clone or create project directory
mkdir equity-valuation-pro
cd equity-valuation-pro

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run setup
python setup.py

# 5. Launch application
python run_dashboard.py
```

### **Access Your Dashboard**
- Local URL: `http://localhost:8501`
- The dashboard will automatically open in your browser

---

## ☁️ **Cloud Deployment Options**

### **Option 1: Streamlit Cloud (Easiest - Free)**

**Perfect for:** Portfolio demonstrations, sharing with recruiters

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Equity Valuation Pro"
   git branch -M main
   git remote add origin https://github.com/yourusername/equity-valuation-pro.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set main file path: `src/main.py`
   - Click "Deploy"

3. **Your Live Demo:**
   - URL: `https://yourusername-equity-valuation-pro-srcmain-xyz123.streamlit.app`
   - Share this link in job applications!

### **Option 2: Heroku (Professional)**

**Perfect for:** Custom domain, professional presentations

1. **Install Heroku CLI** and login:
   ```bash
   heroku login
   ```

2. **Create Heroku app:**
   ```bash
   heroku create your-app-name-equity-valuation
   ```

3. **Add Procfile:**
   ```text
   web: sh setup.sh && streamlit run src/main.py --server.port=$PORT --server.address=0.0.0.0
   ```

4. **Add setup.sh:**
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [general]\n\
   email = \"your-email@domain.com\"\n\
   " > ~/.streamlit/credentials.toml
   echo "\
   [server]\n\
   headless = true\n\
   enableCORS=false\n\
   port = \$PORT\n\
   " > ~/.streamlit/config.toml
   ```

5. **Deploy:**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

### **Option 3: AWS EC2 (Enterprise)**

**Perfect for:** Large scale deployment, custom configurations

1. **Launch EC2 Instance:**
   - Ubuntu 22.04 LTS
   - t3.medium or larger
   - Security group: HTTP (80), HTTPS (443), SSH (22)

2. **Setup on EC2:**
   ```bash
   # SSH into instance
   ssh -i your-key.pem ubuntu@your-ec2-ip

   # Install dependencies
   sudo apt update
   sudo apt install python3-pip nginx -y
   
   # Clone repository
   git clone https://github.com/yourusername/equity-valuation-pro.git
   cd equity-valuation-pro
   
   # Setup Python environment
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # Run setup
   python setup.py
   ```

3. **Configure Nginx:**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

4. **Start with PM2:**
   ```bash
   npm install -g pm2
   pm2 start "streamlit run src/main.py" --name equity-valuation
   pm2 startup
   pm2 save
   ```

### **Option 4: Docker Deployment**

**Perfect for:** Consistent environments, containerized deployment

1. **Build Docker image:**
   ```bash
   docker build -t equity-valuation-pro .
   ```

2. **Run locally:**
   ```bash
   docker run -p 8501:8501 equity-valuation-pro
   ```

3. **Deploy to cloud:**
   ```bash
   # Tag for registry
   docker tag equity-valuation-pro your-registry/equity-valuation-pro
   
   # Push to registry
   docker push your-registry/equity-valuation-pro
   
   # Deploy on cloud platform
   # (AWS ECS, Google Cloud Run, Azure Container Instances)
   ```

---

## 🔧 **Configuration & Optimization**

### **Environment Variables**
Create `.streamlit/secrets.toml`:
```toml
[data_sources]
alpha_vantage_api_key = "your_api_key"
finnhub_api_key = "your_api_key"

[app_settings]
max_peer_companies = 15
cache_duration_minutes = 30
default_risk_free_rate = 0.035
```

### **Performance Optimization**
```python
# In src/main.py
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_market_data(symbol):
    return market_provider.get_company_info(symbol)

@st.cache_resource
def initialize_models():
    return {
        'dcf_calculator': DCFModel,
        'comps_analyzer': ComparableAnalyzer
    }
```

### **Memory Management**
```bash
# Set memory limits for production
export STREAMLIT_SERVER_MAXUPLOADSIZE=200
export STREAMLIT_SERVER_MAXMESSAGESIZE=200
```

---

## 📊 **Monitoring & Analytics**

### **Application Monitoring**
```python
# Add to main.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### **Usage Analytics**
```python
# Track user interactions
def log_user_action(action, symbol=None, user_ip=None):
    logging.info(f"User action: {action}, Symbol: {symbol}, IP: {user_ip}")
```

### **Health Checks**
```python
# Add health check endpoint
def health_check():
    try:
        # Test data connection
        test_data = market_provider.get_stock_data('AAPL', period='1d')
        return {"status": "healthy", "data_connection": "ok"}
    except:
        return {"status": "degraded", "data_connection": "error"}
```

---

## 🛡️ **Security & Best Practices**

### **API Security**
```python
# Rate limiting
from streamlit_extras.app_logo import add_logo
import time

if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = 0

def rate_limit_check():
    current_time = time.time()
    if current_time - st.session_state.last_request_time < 1:  # 1 second between requests
        st.warning("Please wait before making another request")
        return False
    st.session_state.last_request_time = current_time
    return True
```

### **Data Validation**
```python
def validate_stock_symbol(symbol):
    if not symbol or len(symbol) > 10:
        return False
    return symbol.isalpha()

def sanitize_user_input(user_input):
    return re.sub(r'[^A-Za-z0-9.\-_]', '', user_input)
```

### **HTTPS Configuration**
```bash
# For production deployment
sudo certbot --nginx -d your-domain.com
```

---

## 📱 **Mobile Optimization**

### **Responsive Design**
```python
# Add mobile-friendly CSS
st.markdown("""
<style>
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .metric-card {
        margin: 0.25rem 0;
        padding: 0.5rem;
    }
}
</style>
""", unsafe_allow_html=True)
```

---

## 🔄 **CI/CD Pipeline**

### **GitHub Actions Workflow**
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python -m pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to Streamlit Cloud
      run: echo "Deploying to production..."
```

---

## 📈 **Scaling Considerations**

### **For High Traffic**
1. **Load Balancing:** Use multiple Streamlit instances behind a load balancer
2. **Database:** Move from file-based to PostgreSQL/Redis for data storage
3. **Caching:** Implement Redis for shared caching across instances
4. **CDN:** Use CloudFlare or AWS CloudFront for static assets

### **Cost Optimization**
- **Streamlit Cloud:** Free tier suitable for portfolios
- **Heroku:** ~$7/month for hobby tier
- **AWS EC2:** ~$10-30/month depending on instance size
- **Docker + DigitalOcean:** ~$5-10/month for small droplets

---

## 🎯 **Success Metrics**

### **For Investment Banking Applications**
- ✅ **Uptime:** 99%+ availability
- ✅ **Performance:** <3 second load times
- ✅ **Functionality:** All features working correctly
- ✅ **Professional appearance:** Clean, bug-free interface
- ✅ **Data accuracy:** Real market data integration

### **Pre-Deployment Checklist**
- [ ] All tests passing
- [ ] Professional styling applied
- [ ] Error handling implemented
- [ ] Performance optimized
- [ ] Security measures in place
- [ ] Documentation complete
- [ ] Live demo working
- [ ] Screenshots/videos prepared
- [ ] GitHub repository polished

---

## 🆘 **Troubleshooting**

### **Common Issues**

**Issue:** Streamlit app won't start
```bash
# Solution: Check Python version and dependencies
python --version
pip install --upgrade streamlit
streamlit --version
```

**Issue:** Market data not loading
```bash
# Solution: Check internet connection and API limits
pip install --upgrade yfinance
python -c "import yfinance as yf; print(yf.Ticker('AAPL').info['currentPrice'])"
```

**Issue:** Memory errors with large datasets
```python
# Solution: Implement data pagination and caching
@st.cache_data(max_entries=10, ttl=3600)
def load_data(symbol):
    # Your data loading logic
    pass
```

### **Getting Help**
- 📚 **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)
- 💬 **Community:** [discuss.streamlit.io](https://discuss.streamlit.io)
- 🐛 **Issues:** GitHub Issues on your repository

---

## 🏆 **Final Deployment Tips**

1. **Test thoroughly** before sharing with recruiters
2. **Use a custom domain** for professional appearance
3. **Include usage instructions** in your README
4. **Monitor performance** and fix issues promptly
5. **Keep dependencies updated** for security
6. **Backup your work** regularly

**🎉 Your professional equity valuation platform is ready to impress investment banking recruiters! Good luck with your applications!**
