# src/utils/logging_utils.py
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class ExplainableDecision:
    """Represents an explainable decision made by the system"""
    decision: str
    confidence: float
    reasoning: str
    evidence: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class ExperimentLogger:
    """Enhanced logging for federated learning experiments"""
    
    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logger()
        self.decision_log = []
        self.metrics_log = []
        
    def setup_logger(self):
        """Setup structured logging"""
        log_file = self.log_dir / f"{self.experiment_name}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(self.experiment_name)
    
    def log_decision(self, decision: ExplainableDecision):
        """Log an explainable decision"""
        self.decision_log.append(decision)
        self.logger.info(f"DECISION: {decision.decision} | Confidence: {decision.confidence:.3f} | {decision.reasoning}")
    
    def log_round_summary(self, round_num: int, metrics: Dict[str, Any], decisions: List[ExplainableDecision]):
        """Log aggregated round summary"""
        summary = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'metrics': self._convert_numpy_types(metrics),
            'decisions_summary': {
                'total_decisions': len(decisions),
                'avg_confidence': np.mean([d.confidence for d in decisions]) if decisions else 0,
                'decision_types': list(set([d.decision for d in decisions]))
            }
        }
        
        self.metrics_log.append(summary)
        self.logger.info(f"ROUND {round_num} SUMMARY: {json.dumps(summary, indent=2)}")
    
    def save_experiment_log(self):
        """Save complete experiment log"""
        log_data = {
            'experiment_name': self.experiment_name,
            'decisions': [asdict(d) for d in self.decision_log],
            'metrics': self.metrics_log
        }
        
        log_file = self.log_dir / f"{self.experiment_name}_complete.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    @staticmethod
    def _convert_numpy_types(obj):
        """Convert numpy types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: ExperimentLogger._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ExperimentLogger._convert_numpy_types(item) for item in obj]
        else:
            return obj