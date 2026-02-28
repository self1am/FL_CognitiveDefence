import numpy as np
import sys
sys.path.append('.')
from src.defences.vert_defence import VERTDefenceStrategy

vert = VERTDefenceStrategy(kappa=5, history_size=10, projection_dim=100, learning_rate=0.01, min_history_rounds=3)

# Mock some gradients
grad_dim = 1000
for i in range(10):
    client_updates = {}
    for j in range(10):
        # Generate some synthetic gradients with varying norms to trigger issues
        g = [np.random.randn(500) * (j+1) * 10, np.random.randn(500) * (j+1) * 10]
        client_updates[f"client_{j}"] = (g, 100, {})
    
    agged, decs = vert.aggregate_updates(client_updates)
    if vert.predictor_weights is not None:
        print(f"Round {i} weight norm:", np.linalg.norm(vert.predictor_weights))
        if np.isnan(vert.predictor_weights).any():
            print("NaNs detected!")
            break
