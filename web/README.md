# Himalayan Herbal AI - Web Interface

Complete web application for the antimicrobial compound prediction system.

## Architecture

```
┌─────────────────────────────────────────┐
│     Frontend (HTML/CSS/JavaScript)      │
│         Port: 5000                      │
│  - Compound prediction form              │
│  - Batch prediction interface            │
│  - Dataset explorer                      │
│  - Prediction history                    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Backend (Flask Web Server)             │
│  Port: 5000/api                         │
│  - Session management                    │
│  - Request routing                       │
│  - Error handling                        │
│  - CORS support                          │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│   Main AI API (FastAPI)                 │
│   Port: 8000                            │
│  - Model inference                       │
│  - Dataset management                    │
│  - Predictions                           │
└─────────────────────────────────────────┘
```

## Quick Start

### 1. Start Everything

```bash
cd /Users/dipalshrestha/himalayan-herbal-ai
bash web/start.sh
```

This will:
- ✓ Start main API on port 8000 (if not running)
- ✓ Install web backend dependencies
- ✓ Start web backend on port 5000
- ✓ Open browser to http://localhost:5000

### 2. Manual Start

**Terminal 1 - Main API:**
```bash
cd /Users/dipalshrestha/himalayan-herbal-ai
export KMP_DUPLICATE_LIB_OK=TRUE
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Web Backend:**
```bash
cd /Users/dipalshrestha/himalayan-herbal-ai/web/backend
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

**Browser:**
```
http://localhost:5000
```

## File Structure

```
web/
├── backend/
│   ├── app.py                 # Flask application
│   ├── requirements.txt       # Python dependencies
│   └── venv/                 # Virtual environment
├── frontend/
│   └── index.html            # Single-page application
├── start.sh                  # Startup script
└── README.md                 # This file
```

## Features

### 🧪 Single Compound Prediction
- Enter SMILES string
- Get bioactivity and toxicity predictions
- View drug-likeness assessment
- See recommendations

### ⚡ Batch Predictions
- Predict multiple compounds at once (up to 1000)
- One SMILES per line
- Bulk export results
- Identify promising candidates quickly

### 🔍 Dataset Explorer
- Browse all 34 Himalayan compounds
- Filter by active/inactive status
- View plant sources
- Molecular properties
- Quick selection for prediction

### 📊 Real-time Status
- API health indicator
- Model load status
- Connection monitoring
- Error feedback

### 📜 Prediction History
- Track all predictions
- Review past results
- Clear history

## API Endpoints

### Backend (Port 5000)

```
GET  /                           Main page
GET  /api/health                Server health & API status
GET  /api/dataset               Dataset statistics
GET  /api/compounds             List compounds (paginated)
GET  /api/compounds?active_only Filter active compounds
POST /api/predict               Single prediction
POST /api/batch-predict         Batch predictions
GET  /api/history               Prediction history
POST /api/history/clear         Clear history
GET  /api/models/status         Model status
GET  /api/info                  API information
```

### Main API (Port 8000)

See `../API_DEPLOYMENT.md` for complete documentation.

## Configuration

### Environment Variables

Create `.env` file in `web/backend/`:

```bash
# Flask configuration
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
DEBUG=False

# API configuration
API_BASE_URL=http://localhost:8000
API_TIMEOUT=30

# Server
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
```

### Backend Settings (app.py)

```python
API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:8000')
API_TIMEOUT = 30
```

## Development

### Install Dependencies

```bash
cd web/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Development Server

```bash
cd web/backend
source venv/bin/activate
python app.py
```

Server will run at `http://localhost:5000` with auto-reload enabled.

### Frontend Development

Edit `web/frontend/index.html` directly. Changes reload automatically in browser.

## Troubleshooting

### Port Already in Use

```bash
# Kill process on port 5000
lsof -ti :5000 | xargs kill -9

# Kill process on port 8000
lsof -ti :8000 | xargs kill -9
```

### Cannot Connect to API

1. Check main API is running: `curl http://localhost:8000/health`
2. Check `API_BASE_URL` in `.env`
3. Check firewall settings
4. Review logs: `tail -f /tmp/backend.log`

### Slow Predictions

1. Check model is loaded: `curl http://localhost:8000/models/status`
2. Monitor CPU usage: `top -l1 | grep -E "python|java"`
3. Check network latency: `ping localhost`

### Batch Prediction Fails

1. Check SMILES format (one per line)
2. Limit to 1000 compounds max
3. Check error message in UI
4. Review backend logs

## Performance Tips

1. **Caching:** Backend caches dataset info for 1 hour
2. **Batch Processing:** Use batch endpoint for 10+ predictions
3. **Connection Pooling:** Requests library reuses connections
4. **Session Storage:** Flask sessions stored server-side in memory

## Security

### Production Deployment

1. Set `DEBUG=False` in `.env`
2. Generate strong `SECRET_KEY`
3. Use environment variables for sensitive data
4. Enable HTTPS with reverse proxy (Nginx, Apache)
5. Implement API authentication
6. Use rate limiting

Example Nginx config:

```nginx
server {
    listen 443 ssl;
    server_name example.com;
    
    ssl_certificate /path/to/cert;
    ssl_certificate_key /path/to/key;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /api {
        proxy_pass http://localhost:8000;
    }
}
```

## Testing

### Test Health Endpoint

```bash
curl http://localhost:5000/api/health | python -m json.tool
```

### Test Prediction

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "CC(=O)Cc1ccc(O)c(O)cc1",
    "compound_name": "Aspirin"
  }'
```

### Test Batch Prediction

```bash
curl -X POST http://localhost:5000/api/batch-predict \
  -H "Content-Type: application/json" \
  -d '[
    {"smiles": "CC(=O)Cc1ccc(O)c(O)cc1"},
    {"smiles": "c1ccc(O)c(O)c1"}
  ]'
```

## Deployment

### Docker

Create `web/Dockerfile`:

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY web/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY web/backend .
COPY web/frontend templates/

EXPOSE 5000

CMD ["python", "app.py"]
```

Build and run:

```bash
docker build -t himalayan-herbal-web .
docker run -p 5000:5000 -e API_BASE_URL=http://host.docker.internal:8000 himalayan-herbal-web
```

### Heroku

1. Create `Procfile`:
```
web: python web/backend/app.py
```

2. Deploy:
```bash
git push heroku main
```

## License

Same as main project.

## Support

For issues:
1. Check logs: `/tmp/backend.log`
2. Review main API status: `curl http://localhost:8000/health`
3. Check browser console for JavaScript errors
4. Enable Flask debug mode for detailed errors

---

Last Updated: May 27, 2026  
Status: ✅ Ready for use
