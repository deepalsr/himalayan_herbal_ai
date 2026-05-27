"""
Web Backend for Himalayan Herbal AI
====================================
Flask wrapper around the FastAPI service with session management
"""

from flask import Flask, render_template, request, jsonify, session, send_file
from flask_cors import CORS
import requests
import json
from datetime import datetime
from functools import wraps
import os
from pathlib import Path

# Configure template and static folders
base_dir = Path(__file__).parent.parent
template_dir = str(base_dir / 'frontend')
static_dir = str(base_dir / 'frontend')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir, static_url_path='')
app.secret_key = os.environ.get('SECRET_KEY', 'himalayan-herbal-ai-dev-key')
CORS(app)

# Configuration
API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:8000')
API_TIMEOUT = 30

# Decorators
def handle_api_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except requests.exceptions.ConnectionError:
            return jsonify({'error': 'Cannot connect to API server'}), 503
        except requests.exceptions.Timeout:
            return jsonify({'error': 'API request timeout'}), 504
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return decorated_function

# Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
@handle_api_errors
def health():
    """Check API and backend health"""
    try:
        response = requests.get(f'{API_BASE_URL}/health', timeout=API_TIMEOUT)
        api_status = response.json() if response.status_code == 200 else {'status': 'error'}
    except:
        api_status = {'status': 'disconnected'}
    
    return jsonify({
        'backend': 'healthy',
        'api': api_status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/dataset', methods=['GET'])
@handle_api_errors
def dataset_info():
    """Get dataset information"""
    response = requests.get(f'{API_BASE_URL}/dataset', timeout=API_TIMEOUT)
    return jsonify(response.json())

@app.route('/api/compounds', methods=['GET'])
@handle_api_errors
def list_compounds():
    """List compounds with filters"""
    limit = request.args.get('limit', 50, type=int)
    active_only = request.args.get('active_only', False, type=bool)
    
    params = {'limit': limit, 'active_only': active_only}
    response = requests.get(
        f'{API_BASE_URL}/compounds',
        params=params,
        timeout=API_TIMEOUT
    )
    return jsonify(response.json())

@app.route('/api/predict', methods=['POST'])
@handle_api_errors
def predict():
    """Make a prediction"""
    data = request.get_json()
    
    # Validate input
    if not data or 'smiles' not in data:
        return jsonify({'error': 'SMILES string required'}), 400
    
    response = requests.post(
        f'{API_BASE_URL}/predict',
        json=data,
        timeout=API_TIMEOUT
    )
    
    if response.status_code == 200:
        prediction = response.json()
        # Store in session for history
        if 'prediction_history' not in session:
            session['prediction_history'] = []
        
        session['prediction_history'].append({
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction
        })
        session.modified = True
        
        return jsonify(prediction)
    else:
        return jsonify({'error': 'Prediction failed'}), response.status_code

@app.route('/api/batch-predict', methods=['POST'])
@handle_api_errors
def batch_predict():
    """Make batch predictions"""
    data = request.get_json()
    
    if not isinstance(data, list) or len(data) == 0:
        return jsonify({'error': 'Array of compounds required'}), 400
    
    if len(data) > 1000:
        return jsonify({'error': 'Maximum 1000 compounds per batch'}), 400
    
    response = requests.post(
        f'{API_BASE_URL}/batch-predict',
        json=data,
        timeout=API_TIMEOUT
    )
    
    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({'error': 'Batch prediction failed'}), response.status_code

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get prediction history"""
    history = session.get('prediction_history', [])
    return jsonify({
        'count': len(history),
        'predictions': history[-20:]  # Last 20
    })

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """Clear prediction history"""
    session['prediction_history'] = []
    session.modified = True
    return jsonify({'status': 'cleared'})

@app.route('/api/models/status', methods=['GET'])
@handle_api_errors
def models_status():
    """Get model status"""
    response = requests.get(f'{API_BASE_URL}/models/status', timeout=API_TIMEOUT)
    return jsonify(response.json())

@app.route('/api/info', methods=['GET'])
@handle_api_errors
def api_info():
    """Get API information"""
    response = requests.get(f'{API_BASE_URL}/info', timeout=API_TIMEOUT)
    return jsonify(response.json())

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
