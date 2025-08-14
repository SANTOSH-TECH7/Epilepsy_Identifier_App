from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import joblib
import numpy as np
import pandas as pd
import json
from datetime import datetime
import threading
import time
import random
import os
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

class SeizureDetectionSystem:
    def __init__(self, model_path):
        """Initialize the seizure detection system"""
        self.model_package = None
        self.is_monitoring = False
        self.prediction_history = deque(maxlen=100)  # Store last 100 predictions
        self.confidence_threshold = 0.7
        self.seizure_detected = False
        self.last_prediction_time = None
        
        # Load the trained model
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load the trained XGBoost model"""
        try:
            self.model_package = joblib.load(model_path)
            logger.info("Model loaded successfully!")
            
            # Handle different model package structures
            if 'training_info' in self.model_package:
                logger.info(f"Model type: {self.model_package['training_info']['model_type']}")
                logger.info(f"Feature count: {self.model_package['training_info']['feature_count']}")
            else:
                logger.info("Model type: XGBoost (inferred)")
                logger.info(f"Model object type: {type(self.model_package.get('model', 'Unknown'))}")
            
            if 'class_names' in self.model_package:
                logger.info(f"Classes: {self.model_package['class_names']}")
            else:
                logger.info("Classes: Binary classification (inferred)")
                
            # Verify essential components
            required_components = ['model', 'scaler', 'label_encoder']
            missing_components = [comp for comp in required_components if comp not in self.model_package]
            
            if missing_components:
                logger.error(f"Missing required components: {missing_components}")
                raise ValueError(f"Model package missing: {missing_components}")
            
            logger.info("All required model components found!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_data(self, eeg_data):
        """Preprocess incoming EEG data"""
        try:
            # Convert to numpy array if it's not already
            if isinstance(eeg_data, list):
                eeg_data = np.array(eeg_data)
            
            # Reshape if necessary (ensure it's 2D for sklearn)
            if eeg_data.ndim == 1:
                eeg_data = eeg_data.reshape(1, -1)
            
            # Apply the same scaling used during training
            scaled_data = self.model_package['scaler'].transform(eeg_data)
            
            return scaled_data
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def predict_seizure(self, eeg_data):
        """Make seizure prediction from EEG data"""
        try:
            # Preprocess the data
            processed_data = self.preprocess_data(eeg_data)
            
            # Make prediction
            prediction = self.model_package['model'].predict(processed_data)[0]
            prediction_proba = self.model_package['model'].predict_proba(processed_data)[0]
            
            # Convert prediction back to original labels
            predicted_class = self.model_package['label_encoder'].inverse_transform([prediction])[0]
            
            # Calculate confidence
            confidence = max(prediction_proba)
            
            # Determine if seizure is detected (handle different class encodings)
            # Check if predicted_class is numeric or string
            if isinstance(predicted_class, (int, float)):
                # Assuming higher numbers indicate seizure
                is_seizure = predicted_class > 0
            else:
                # Handle string labels
                seizure_labels = ['seizure', 'epileptic', '1', 'positive', 'abnormal']
                is_seizure = str(predicted_class).lower() in seizure_labels
            
            # Store prediction in history
            prediction_result = {
                'timestamp': datetime.now().isoformat(),
                'predicted_class': int(predicted_class) if isinstance(predicted_class, (int, float)) else str(predicted_class),
                'confidence': float(confidence),
                'is_seizure': is_seizure,
                'probabilities': prediction_proba.tolist()
            }
            
            self.prediction_history.append(prediction_result)
            self.last_prediction_time = datetime.now()
            
            # Update seizure status
            if is_seizure and confidence > self.confidence_threshold:
                self.seizure_detected = True
            else:
                self.seizure_detected = False
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            logger.error(f"EEG data shape: {np.array(eeg_data).shape}")
            logger.error(f"Model components available: {list(self.model_package.keys())}")
            raise
    
    def get_system_status(self):
        """Get current system status"""
        return {
            'is_monitoring': self.is_monitoring,
            'seizure_detected': self.seizure_detected,
            'last_prediction': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'prediction_count': len(self.prediction_history),
            'model_loaded': self.model_package is not None
        }

# Initialize the seizure detection system
# Update this path to match your model location
MODEL_PATH = os.path.join(os.path.dirname(__file__), "epilepsy_seizure_model_joblib.pkl")
seizure_detector = SeizureDetectionSystem(MODEL_PATH)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify(seizure_detector.get_system_status())

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a single prediction"""
    try:
        data = request.json
        eeg_data = data.get('eeg_data')
        
        if not eeg_data:
            return jsonify({'error': 'No EEG data provided'}), 400
        
        # Make prediction
        result = seizure_detector.predict_seizure(eeg_data)
        
        # Emit real-time update via WebSocket
        socketio.emit('prediction_update', result)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history')
def get_prediction_history():
    """Get prediction history"""
    return jsonify({
        'history': list(seizure_detector.prediction_history),
        'total_count': len(seizure_detector.prediction_history)
    })

@socketio.on('start_monitoring')
def start_monitoring():
    """Start continuous monitoring"""
    seizure_detector.is_monitoring = True
    emit('monitoring_status', {'status': 'started'})
    logger.info("Monitoring started")

@socketio.on('stop_monitoring')
def stop_monitoring():
    """Stop continuous monitoring"""
    seizure_detector.is_monitoring = False
    emit('monitoring_status', {'status': 'stopped'})
    logger.info("Monitoring stopped")

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected")
    emit('system_status', seizure_detector.get_system_status())

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected")

# Simulated data generator for testing
def generate_sample_eeg_data():
    """Generate sample EEG data for testing"""
    # Create realistic EEG-like data with 178 features (X1 to X178, excluding 'Unnamed' and 'y')
    base_signal = np.random.randn(178)
    
    # Add some seizure-like patterns randomly
    if random.random() < 0.3:  # 30% chance of seizure-like pattern
        # Add high frequency components and amplitude changes
        base_signal += np.random.randn(178) * 2
        base_signal *= np.random.uniform(1.5, 2.5)
        
        # Add some spikes to simulate seizure activity
        spike_indices = np.random.choice(178, size=int(178 * 0.1), replace=False)
        base_signal[spike_indices] *= np.random.uniform(2, 4, size=len(spike_indices))
    
    return base_signal.tolist()

@app.route('/api/simulate_data', methods=['POST'])
def simulate_data():
    """Generate and process simulated EEG data"""
    try:
        # Generate sample data
        eeg_data = generate_sample_eeg_data()
        
        # Make prediction
        result = seizure_detector.predict_seizure(eeg_data)
        
        # Emit real-time update
        socketio.emit('prediction_update', result)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'data_length': len(eeg_data)
        })
        
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
