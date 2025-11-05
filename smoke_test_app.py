import os
import json
import numpy as np

from app import app  # imports and loads the model


def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    encoded_path = os.path.join(data_dir, 'encoded_test.npy')
    if not os.path.exists(encoded_path):
        raise FileNotFoundError(f"Missing test features: {encoded_path}")
    x = np.load(encoded_path)
    if x.shape[1] != 32:
        raise ValueError(f"Expected encoded feature dim 32, got {x.shape[1]}")
    vec = x[0].tolist()

    payload = {
        'vector': vec,
        'threshold': 0.5
    }

    with app.test_client() as c:
        rv = c.post('/predict_encoded', json=payload)
        print('Status:', rv.status_code)
        print('JSON:', json.dumps(rv.get_json(), indent=2))


if __name__ == '__main__':
    main()

