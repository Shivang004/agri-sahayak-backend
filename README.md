# Agri-Sahayak Backend

A comprehensive multi-agent agricultural assistance system built with FastAPI and the Agno framework. This backend provides conversational AI capabilities and direct data access for farmers.

## 🚀 Features

- **Multi-Agent System**: Four specialist agents working together under a "Sarpanch" coordinator
- **Conversational AI**: Natural language processing for agricultural queries
- **Real-time Data**: Market prices, weather forecasts, and agricultural information
- **Knowledge Base**: FAISS-powered RAG system for government schemes and farming practices
- **Crop Health Analysis**: Multi-modal image analysis for plant disease diagnosis
- **RESTful API**: Comprehensive endpoints for both conversational and direct data access

## 🤖 Agent Team

### 1. **Vayu** - Weather Specialist
- Provides weather forecasts and climate information
- Analyzes weather patterns for agricultural planning
- Uses Open-Meteo API for real-time weather data

### 2. **Bazaar** - Market Analyst
- Provides commodity prices and market trends
- Analyzes market dynamics and price fluctuations
- Uses Agmarknet API for market data

### 3. **Yojana** - Policy & Knowledge Specialist
- Provides information about government schemes and subsidies
- Shares best farming practices and techniques
- Uses FAISS knowledge base for comprehensive information

### 4. **Dr. Fasal** - Crop Doctor
- Analyzes plant images for disease diagnosis
- Provides treatment recommendations
- Uses multi-modal LLM capabilities

## 📁 Project Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── team.py                 # Agno agents and team configuration
├── models.py               # Pydantic models for API requests/responses
├── ingest.py               # FAISS index creation script
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
├── tools/                  # Agent tools directory
│   ├── __init__.py
│   ├── weather_tools.py    # Weather API integration
│   ├── market_tools.py     # Market data integration
│   └── rag_tools.py        # Knowledge base tools
├── routers/                # FastAPI routers
│   ├── __init__.py
│   └── data_endpoints.py   # Direct data access endpoints
├── faiss_index/            # FAISS index storage (auto-created)
└── knowledge_base/         # PDF documents for ingestion (add your files)
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Google API Key for Gemini
- Data.gov.in API Key (optional, has fallback)

### Setup

1. **Clone and navigate to the backend directory:**
   ```bash
   cd "Agri Sahayak/backend"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create environment file:**
   ```bash
   # Create .env file with your API keys
   echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
   echo "DATA_GOV_API_KEY=your_data_gov_api_key_here" >> .env
   echo "DEBUG=True" >> .env
   echo "HOST=0.0.0.0" >> .env
   echo "PORT=8000" >> .env
   ```

4. **Build the knowledge base:**
   ```bash
   python ingest.py
   ```

5. **Run the application:**
   ```bash
   python main.py
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# API Keys
GOOGLE_API_KEY=your_google_api_key_here
DATA_GOV_API_KEY=your_data_gov_api_key_here

# Application Settings
DEBUG=True
HOST=0.0.0.0
PORT=8000

# FAISS Index Settings
FAISS_INDEX_PATH=./faiss_index
KNOWLEDGE_BASE_PATH=./knowledge_base
```

### Knowledge Base Setup

1. **Add PDF documents** to the `knowledge_base/` directory
2. **Run the ingestion script:**
   ```bash
   python ingest.py
   ```
3. The script will create a FAISS index and document database

## 📡 API Endpoints

### Conversational AI
- `POST /api/query` - Process natural language queries

### Direct Data Access
- `GET /api/data/commodities` - Get available commodities and states
- `POST /api/data/prices` - Get market prices for specific commodity/state
- `POST /api/data/weather` - Get weather data for location
- `GET /api/data/market-summary` - Get market data summary

### System Information
- `GET /` - Root endpoint with system info
- `GET /api/agents` - Get information about all agents
- `GET /api/health` - Health check
- `GET /docs` - Interactive API documentation

## 💬 Usage Examples

### Conversational Query
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the weather forecast for Delhi and what are the current wheat prices in Punjab?",
    "imageUrl": null
  }'
```

### Market Data
```bash
curl -X POST "http://localhost:8000/api/data/prices" \
  -H "Content-Type: application/json" \
  -d '{
    "commodity": "Wheat",
    "state": "Punjab"
  }'
```

### Weather Data
```bash
curl -X POST "http://localhost:8000/api/data/weather" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Delhi"
  }'
```

## 🔍 Knowledge Base Management

### Adding New Documents

1. **Place PDF files** in the `knowledge_base/` directory
2. **Run ingestion:**
   ```bash
   python ingest.py
   ```

### Sample Documents Included

The system includes sample agricultural information covering:
- Government schemes (PM-KISAN, KCC)
- Farming practices (crop rotation, soil testing)
- Irrigation methods (drip irrigation)
- Pest management (IPM)
- Fertilizer application
- Crop insurance

## 🚀 Deployment

### Development
```bash
python main.py
```

### Production
```bash
# Using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000

# Using gunicorn (recommended for production)
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python ingest.py

EXPOSE 8000
CMD ["python", "main.py"]
```

## 🔧 Troubleshooting

### Common Issues

1. **Agno Framework Not Available**
   - The system includes a mock implementation for development
   - Install Agno when available: `pip install agno`

2. **API Key Issues**
   - Ensure Google API Key is set in `.env`
   - Check API key permissions and quotas

3. **Knowledge Base Not Loading**
   - Run `python ingest.py` to create the FAISS index
   - Check file permissions for `faiss_index/` directory

4. **Memory Issues**
   - FAISS index can be large; ensure sufficient RAM
   - Consider using `faiss-cpu` instead of `faiss-gpu`

### Logs and Debugging

- Set `DEBUG=True` in `.env` for detailed logs
- Check console output for initialization messages
- Use `/api/health` endpoint to verify system status

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the API documentation at `/docs`
- Review the troubleshooting section
- Open an issue on the repository

---

**Agri-Sahayak Backend** - Empowering farmers with AI-driven agricultural assistance.
