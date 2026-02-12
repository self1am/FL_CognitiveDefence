#!/usr/bin/env python3
"""
Simple Flask API for monitoring experiments
Provides JSON endpoints for external monitoring and dashboard integration
Usage: python scripts/monitoring_api.py
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
from pathlib import Path
from datetime import datetime
import psutil

app = Flask(__name__)
CORS(app)  # Enable CORS for browser access

LOG_DIR = Path("logs/experiments")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def find_status_files():
    """Find all experiment status files"""
    return list(LOG_DIR.glob("*_status.json"))


def read_status(status_file):
    """Read status from JSON file"""
    try:
        with open(status_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def parse_experiment_progress(log_file):
    """Parse log file to extract progress"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        total_rounds = None
        current_round = 0
        last_accuracy = None
        last_loss = None
        rounds_data = []
        
        for line in lines:
            if 'num_rounds=' in line:
                try:
                    total_rounds = int(line.split('num_rounds=')[1].split(',')[0])
                except (ValueError, IndexError):
                    pass
            
            if '[ROUND' in line:
                try:
                    current_round = int(line.split('[ROUND')[1].split(']')[0].strip())
                except (ValueError, IndexError):
                    pass
            
            if 'Accuracy:' in line:
                try:
                    acc = float(line.split('Accuracy:')[1].split()[0])
                    last_accuracy = acc
                    
                    # Extract loss if on same line
                    if 'Loss:' in line:
                        loss = float(line.split('Loss:')[1].split(',')[0].strip())
                        last_loss = loss
                        rounds_data.append({
                            'round': current_round,
                            'accuracy': acc,
                            'loss': loss
                        })
                except (ValueError, IndexError):
                    pass
        
        return {
            'total_rounds': total_rounds,
            'current_round': current_round,
            'last_accuracy': last_accuracy,
            'last_loss': last_loss,
            'rounds_data': rounds_data
        }
    except FileNotFoundError:
        return {}


def is_process_running(pid):
    """Check if process is running"""
    try:
        process = psutil.Process(int(pid))
        return process.is_running()
    except (psutil.NoSuchProcess, ValueError):
        return False


@app.route('/api/experiments', methods=['GET'])
def list_experiments():
    """List all experiments"""
    status_files = find_status_files()
    experiments = []
    
    for status_file in status_files:
        status = read_status(status_file)
        if not status:
            continue
        
        exp_name = status_file.stem.replace('_status', '')
        pid = status.get('pid')
        
        # Check if actually running
        is_running = is_process_running(pid) if pid else False
        
        log_file = status.get('log_file')
        progress = parse_experiment_progress(log_file) if log_file else {}
        
        experiments.append({
            'name': exp_name,
            'pid': pid,
            'status': 'running' if is_running else 'stopped',
            'config_file': status.get('config_file'),
            'log_file': log_file,
            'memory_mb': status.get('memory_mb'),
            'cpu_percent': status.get('cpu_percent'),
            'last_update': status.get('last_update'),
            'progress': progress
        })
    
    return jsonify({
        'experiments': experiments,
        'count': len(experiments),
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/api/experiments/<experiment_name>', methods=['GET'])
def get_experiment(experiment_name):
    """Get details for specific experiment"""
    status_file = LOG_DIR / f"{experiment_name}_status.json"
    
    if not status_file.exists():
        return jsonify({'error': 'Experiment not found'}), 404
    
    status = read_status(status_file)
    if not status:
        return jsonify({'error': 'Failed to read status'}), 500
    
    log_file = status.get('log_file')
    progress = parse_experiment_progress(log_file) if log_file else {}
    
    pid = status.get('pid')
    is_running = is_process_running(pid) if pid else False
    
    return jsonify({
        'name': experiment_name,
        'pid': pid,
        'status': 'running' if is_running else 'stopped',
        'config_file': status.get('config_file'),
        'log_file': log_file,
        'memory_mb': status.get('memory_mb'),
        'cpu_percent': status.get('cpu_percent'),
        'last_update': status.get('last_update'),
        'progress': progress
    })


@app.route('/api/experiments/<experiment_name>/logs', methods=['GET'])
def get_experiment_logs(experiment_name):
    """Get recent logs for experiment"""
    status_file = LOG_DIR / f"{experiment_name}_status.json"
    
    if not status_file.exists():
        return jsonify({'error': 'Experiment not found'}), 404
    
    status = read_status(status_file)
    log_file = status.get('log_file')
    
    if not log_file or not os.path.exists(log_file):
        return jsonify({'error': 'Log file not found'}), 404
    
    # Get number of lines (default 100)
    lines = request.args.get('lines', 100, type=int)
    
    try:
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:]
        
        return jsonify({
            'logs': ''.join(recent_lines),
            'total_lines': len(all_lines),
            'returned_lines': len(recent_lines)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/system', methods=['GET'])
def system_info():
    """Get system resource information"""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    disk = psutil.disk_usage('/')
    
    return jsonify({
        'memory': {
            'total_mb': memory.total / 1024 / 1024,
            'available_mb': memory.available / 1024 / 1024,
            'used_mb': memory.used / 1024 / 1024,
            'percent': memory.percent
        },
        'cpu': {
            'count': psutil.cpu_count(),
            'percent_per_core': cpu_percent,
            'percent_avg': sum(cpu_percent) / len(cpu_percent)
        },
        'disk': {
            'total_gb': disk.total / 1024 / 1024 / 1024,
            'used_gb': disk.used / 1024 / 1024 / 1024,
            'free_gb': disk.free / 1024 / 1024 / 1024,
            'percent': disk.percent
        },
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/')
def index():
    """Simple HTML dashboard"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FL Experiment Monitor</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            .status { display: inline-block; padding: 5px 10px; border-radius: 4px; font-weight: bold; }
            .status.running { background: #4CAF50; color: white; }
            .status.stopped { background: #f44336; color: white; }
            .metric { margin: 10px 0; }
            .progress-bar { width: 100%; height: 20px; background: #ddd; border-radius: 4px; overflow: hidden; }
            .progress-fill { height: 100%; background: #4CAF50; transition: width 0.3s; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 FL Cognitive Defence - Experiment Monitor</h1>
            <div id="experiments"></div>
            <div id="system" class="card"></div>
        </div>
        <script>
            function updateDashboard() {
                // Fetch experiments
                fetch('/api/experiments')
                    .then(r => r.json())
                    .then(data => {
                        const html = data.experiments.map(exp => `
                            <div class="card">
                                <h2>${exp.name}</h2>
                                <div class="status ${exp.status}">${exp.status.toUpperCase()}</div>
                                <div class="metric"><strong>PID:</strong> ${exp.pid}</div>
                                <div class="metric"><strong>Memory:</strong> ${exp.memory_mb || 'N/A'} MB</div>
                                <div class="metric"><strong>CPU:</strong> ${exp.cpu_percent || 'N/A'}%</div>
                                ${exp.progress.total_rounds ? `
                                    <div class="metric">
                                        <strong>Progress:</strong> Round ${exp.progress.current_round}/${exp.progress.total_rounds}
                                        <div class="progress-bar">
                                            <div class="progress-fill" style="width: ${(exp.progress.current_round/exp.progress.total_rounds)*100}%"></div>
                                        </div>
                                    </div>
                                ` : ''}
                                ${exp.progress.last_accuracy ? `<div class="metric"><strong>Accuracy:</strong> ${exp.progress.last_accuracy.toFixed(4)}</div>` : ''}
                                ${exp.progress.last_loss ? `<div class="metric"><strong>Loss:</strong> ${exp.progress.last_loss.toFixed(4)}</div>` : ''}
                            </div>
                        `).join('');
                        document.getElementById('experiments').innerHTML = html || '<div class="card">No experiments running</div>';
                    });
                
                // Fetch system info
                fetch('/api/system')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('system').innerHTML = `
                            <h2>System Resources</h2>
                            <div class="metric"><strong>Memory:</strong> ${data.memory.used_mb.toFixed(0)} / ${data.memory.total_mb.toFixed(0)} MB (${data.memory.percent.toFixed(1)}%)</div>
                            <div class="metric"><strong>CPU:</strong> ${data.cpu.percent_avg.toFixed(1)}% (${data.cpu.count} cores)</div>
                            <div class="metric"><strong>Disk:</strong> ${data.disk.used_gb.toFixed(1)} / ${data.disk.total_gb.toFixed(1)} GB (${data.disk.percent.toFixed(1)}%)</div>
                        `;
                    });
            }
            
            updateDashboard();
            setInterval(updateDashboard, 5000);  // Update every 5 seconds
        </script>
    </body>
    </html>
    """


if __name__ == '__main__':
    print("=" * 80)
    print("🚀 Starting FL Experiment Monitoring API")
    print("=" * 80)
    print(f"API will be available at: http://0.0.0.0:5000")
    print(f"Dashboard: http://0.0.0.0:5000/")
    print(f"API Endpoints:")
    print(f"  - GET /api/experiments")
    print(f"  - GET /api/experiments/<name>")
    print(f"  - GET /api/experiments/<name>/logs")
    print(f"  - GET /api/system")
    print(f"  - GET /api/health")
    print("=" * 80)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
