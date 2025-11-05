import os
import json
from typing import Any, Dict

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib
from build_model import build_SAE


# Optional: enable GPU memory growth if GPU is available
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, 'models', 'best_model.hdf5')
DATA_DIR = os.path.join(APP_DIR, 'data')
FEATURE_COLUMNS_PATH = os.path.join(DATA_DIR, 'feature_columns.json')
SCALER_PATH = os.path.join(DATA_DIR, 'scaler.pkl')


def load_classifier():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)
    # Inspect input shape to validate time_steps
    # Expected (None, 1, 32) because trained with time_steps=1
    return model


classifier_model = load_classifier()


def load_ssae_encoder():
    # Build SSAE and load pre-trained AE weights to the encoder
    if not (
        os.path.exists(os.path.join(APP_DIR, 'saved_ae_1', 'best_ae_1.hdf5')) and
        os.path.exists(os.path.join(APP_DIR, 'saved_ae_2', 'best_ae_2.hdf5')) and
        os.path.exists(os.path.join(APP_DIR, 'saved_ae_3', 'best_ae_3.hdf5'))
    ):
        raise FileNotFoundError('Missing saved AE weights in saved_ae_*/ directories.')

    ae1, enc1, ae2, enc2, ae3, enc3, ssae, ssae_encoder = build_SAE(rho=0.04)

    ae1.load_weights(os.path.join(APP_DIR, 'saved_ae_1', 'best_ae_1.hdf5'))
    ae2.load_weights(os.path.join(APP_DIR, 'saved_ae_2', 'best_ae_2.hdf5'))
    ae3.load_weights(os.path.join(APP_DIR, 'saved_ae_3', 'best_ae_3.hdf5'))

    # Pass weights to stacked encoder exactly as in training script
    ssae_encoder.layers[1].set_weights(ae1.layers[2].get_weights())
    ssae_encoder.layers[2].set_weights(ae1.layers[3].get_weights())
    ssae_encoder.layers[3].set_weights(ae2.layers[2].get_weights())
    ssae_encoder.layers[4].set_weights(ae2.layers[3].get_weights())
    ssae_encoder.layers[5].set_weights(ae3.layers[2].get_weights())
    ssae_encoder.layers[6].set_weights(ae3.layers[3].get_weights())

    return ssae_encoder


ssae_encoder = load_ssae_encoder()


def preprocess_dataframe(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    # Load artifacts
    if not (os.path.exists(FEATURE_COLUMNS_PATH) and os.path.exists(SCALER_PATH)):
        raise FileNotFoundError('Preprocessing artifacts not found. Run preprocess_artifacts.py first.')
    with open(FEATURE_COLUMNS_PATH, 'r', encoding='utf-8') as f:
        feature_columns = json.load(f)['feature_columns']
    scaler = joblib.load(SCALER_PATH)

    # Preserve labels if present
    labels = None
    if 'label' in df.columns:
        labels = df['label'].values.reshape(-1, 1)

    # One-hot encode categorical columns (if missing in input, add empty ones later)
    for col in ['proto', 'service', 'state']:
        if col not in df.columns:
            df[col] = ''  # placeholder for get_dummies
    df_ohe = pd.get_dummies(data=df, columns=['proto', 'service', 'state'])

    # Drop non-feature columns if present
    for drop_col in ['label', 'attack_cat']:
        if drop_col in df_ohe.columns:
            df_ohe.drop([drop_col], axis=1, inplace=True)

    # Align columns to training schema
    for col in feature_columns:
        if col not in df_ohe.columns:
            df_ohe[col] = 0
    # Remove unexpected columns
    extra_cols = [c for c in df_ohe.columns if c not in feature_columns]
    if extra_cols:
        df_ohe.drop(columns=extra_cols, inplace=True)
    df_ohe = df_ohe[feature_columns]

    # Scale
    X = scaler.transform(df_ohe.values)
    return X, labels


def predict_encoded_batch(seqs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    # seqs shape must be (batch, time_steps, 32); in our trained model time_steps == 1
    if seqs.ndim != 3 or seqs.shape[2] != 32:
        raise ValueError("Input must have shape (batch, time_steps, 32)")
    probs = classifier_model.predict(seqs, verbose=0)
    labels = (probs > threshold).astype(int)
    return {
        'probabilities': probs.reshape(-1).tolist(),
        'predictions': labels.reshape(-1).tolist(),
        'threshold': threshold,
        'batch_size': int(seqs.shape[0])
    }


app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': True})


@app.route('/predict_encoded', methods=['POST'])
def predict_encoded():
    """
    Request JSON formats supported:
    - Single vector (time_steps=1): {"vector": [32 floats], "threshold": 0.5}
      -> Internally expanded to shape (1, 1, 32)
    - Batch of vectors (time_steps=1): {"batch": [[32 floats], ...], "threshold": 0.5}
      -> Internally expanded to shape (batch, 1, 32)
    - Explicit sequences: {"sequences": [[[...32...]], [[...32...]], ...], "threshold": 0.5}
      -> Must have shape (batch, time_steps, 32). Note: current model trained with time_steps=1.
    """
    try:
        payload = request.get_json(force=True)  # type: ignore
    except Exception as e:
        return jsonify({'error': f'Invalid JSON: {e}'}), 400

    threshold = float(payload.get('threshold', 0.5))

    if 'vector' in payload:
        vec = np.array(payload['vector'], dtype=np.float32)
        if vec.shape != (32,):
            return jsonify({'error': 'vector must be length-32 list'}), 400
        seqs = vec.reshape(1, 1, 32)
    elif 'batch' in payload:
        batch = np.array(payload['batch'], dtype=np.float32)
        if batch.ndim != 2 or batch.shape[1] != 32:
            return jsonify({'error': 'batch must be a list of length-32 lists'}), 400
        seqs = batch.reshape(batch.shape[0], 1, 32)
    elif 'sequences' in payload:
        seqs = np.array(payload['sequences'], dtype=np.float32)
        if seqs.ndim != 3 or seqs.shape[2] != 32:
            return jsonify({'error': 'sequences must have shape (batch, time_steps, 32)'}), 400
    else:
        return jsonify({'error': 'Missing key: provide one of vector, batch, or sequences'}), 400

    try:
        result = predict_encoded_batch(seqs, threshold)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Time-related IDS Evaluator</title>
    <style>
      :root {
        --bg:#eef2f7;
        --fg:#111827;
        --muted:#6b7280;
        --card:#ffffff;
        --border:#d4d9e3;
        --accent:#2563eb;
        --accent-soft:#e0e9ff;
        --pos:#0ea5e9;
        --neg:#f97316;
      }
      * { box-sizing:border-box; }
      body { margin:0; font-family:'Segoe UI', Roboto, -apple-system, BlinkMacSystemFont, sans-serif; background:var(--bg); color:var(--fg); }
      header { padding:26px 34px; background:linear-gradient(135deg,#1d4ed8,#3b82f6); color:#fff; box-shadow:0 8px 22px rgba(37,99,235,0.25); }
      header h1 { margin:0; font-size:28px; letter-spacing:0.02em; }
      header p { margin:8px 0 0; color:rgba(255,255,255,0.85); font-size:15px; }
      main { padding:32px; max-width:1120px; margin:0 auto; display:grid; gap:24px; }
      .card { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:24px 26px; box-shadow:0 12px 32px rgba(15,23,42,0.08); }
      .row { margin-bottom:16px; }
      label { font-weight:600; font-size:15px; margin-bottom:6px; display:block; }
      input[type=file], input[type=number] { width:100%; padding:11px; font-size:15px; border-radius:12px; border:1px solid var(--border); background:#fbfcfe; }
      .btn { margin-top:12px; background:var(--accent); color:#fff; border:none; padding:12px 20px; font-size:15px; border-radius:10px; cursor:pointer; box-shadow:0 10px 24px rgba(37,99,235,0.25); transition:transform .15s ease, box-shadow .15s ease; }
      .btn:hover { transform:translateY(-1px); box-shadow:0 14px 28px rgba(37,99,235,0.25); }
      .muted { color:var(--muted); font-size:14px; }
      .result-layout { display:grid; gap:22px; }
      .metrics-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; }
      .metric-card { background:var(--accent-soft); border-radius:14px; padding:14px 16px; border:1px solid rgba(37,99,235,0.15); box-shadow:0 8px 20px rgba(37,99,235,0.12); }
      .metric-card strong { font-size:22px; display:block; margin-top:4px; letter-spacing:0.01em; }
      .metric-card .label { display:inline-flex; align-items:center; gap:6px; padding:4px 10px; font-size:13px; font-weight:600; border-radius:999px; background:#fff; color:var(--accent); box-shadow:0 6px 16px rgba(37,99,235,0.18); }
      .metric-card.pos { background:#ecfeff; border-color:#bae6fd; }
      .metric-card.pos strong { color:var(--pos); }
      .metric-card.neg { background:#fff7ed; border-color:#fed7aa; }
      .metric-card.neg strong { color:var(--neg); }
      .section-title { font-size:16px; text-transform:uppercase; letter-spacing:0.05em; color:var(--muted); margin:0 0 10px; }
      table { width:100%; border-collapse:separate; border-spacing:0; border-radius:12px; overflow:hidden; box-shadow:0 10px 30px rgba(15,23,42,0.10); }
      th, td { padding:12px 14px; text-align:center; border-bottom:1px solid var(--border); background:#f9fbff; font-size:15px; }
      th { background:#e5edff; font-weight:700; letter-spacing:0.03em; }
      tr:last-child td { border-bottom:none; }
      .mono { font-family:'SFMono-Regular', Consolas, 'Liberation Mono', monospace; font-size:13px; }
      pre { background:#0f172a; color:#c7d2fe; padding:18px; border-radius:14px; font-size:13px; line-height:1.45; box-shadow:0 16px 34px rgba(15,23,42,0.35); }
      .error-card { border:1px solid #fecaca; background:linear-gradient(140deg,#fee2e2,#ffe4e6); color:#991b1b; }
      footer { text-align:center; padding:18px; font-size:13px; color:var(--muted); }
    </style>
  </head>
  <body>
    <header>
      <h1>Time-related Intrusion Detection</h1>
      <p>Evaluate custom CSV datasets using the pre-trained SSAE + BiLSTM pipeline.</p>
    </header>
    <main>
      <div class="card">
        <h2 style="margin-top:0;">Upload CSV</h2>
        <form method="POST" action="/upload_csv" enctype="multipart/form-data">
          <div class="row">
            <label>Dataset CSV (UNSW_NB15 compatible)</label>
            <input name="file" type="file" accept=".csv" required>
          </div>
          <div class="row" style="max-width:240px;">
            <label>Decision Threshold</label>
            <input name="threshold" type="number" step="0.01" value="0.5">
          </div>
          <button class="btn" type="submit">Run Evaluation</button>
          <div class="muted" style="margin-top:12px;">Categorical columns (proto/service/state) are aligned automatically. Include 'label' for metrics.</div>
        </form>
      </div>

      {% if result %}
        {% if result.error is defined %}
          <div class="card error-card">
            <h3 style="margin-top:0;">Evaluation Failed</h3>
            <p>{{ result.error }}</p>
          </div>
        {% else %}
          <div class="card">
            <div class="result-layout">
              <div>
                <div class="section-title">Summary</div>
                <div class="metrics-grid">
                  <div class="metric-card">
                    <span class="label">Samples</span>
                    <strong>{{ result.count }}</strong>
                    <div class="muted">Rows evaluated</div>
                  </div>
                  <div class="metric-card">
                    <span class="label">Threshold</span>
                    <strong>{{ '%.2f'|format(result.threshold) }}</strong>
                    <div class="muted">Decision boundary</div>
                  </div>
                  <div class="metric-card pos">
                    <span class="label">Predicted Positive</span>
                    <strong>{{ result.positives }}</strong>
                    <div class="muted">Samples flagged anomalous</div>
                  </div>
                  <div class="metric-card neg">
                    <span class="label">Predicted Negative</span>
                    <strong>{{ result.negatives }}</strong>
                    <div class="muted">Samples flagged normal</div>
                  </div>
                  {% if result.accuracy is defined %}
                  <div class="metric-card">
                    <span class="label">Accuracy</span>
                    <strong>{{ '%.4f'|format(result.accuracy) }}</strong>
                    <div class="muted">Overall correctness</div>
                  </div>
                  {% endif %}
                  {% if result.metrics is defined %}
                  <div class="metric-card">
                    <span class="label">Precision⁺</span>
                    <strong>{{ '%.4f'|format(result.metrics.precision_pos) }}</strong>
                    <div class="muted">TP / (TP + FP)</div>
                  </div>
                  <div class="metric-card">
                    <span class="label">Recall⁺</span>
                    <strong>{{ '%.4f'|format(result.metrics.recall_pos) }}</strong>
                    <div class="muted">TP / (TP + FN)</div>
                  </div>
                  <div class="metric-card">
                    <span class="label">Specificity⁻</span>
                    <strong>{{ '%.4f'|format(result.metrics.tnr_neg) }}</strong>
                    <div class="muted">TN / (TN + FP)</div>
                  </div>
                  {% endif %}
                </div>
              </div>

              {% if result.confusion_matrix is defined %}
              <div>
                <div class="section-title">Confusion Matrix</div>
                <table>
                  <tr><th></th><th>Predicted 0</th><th>Predicted 1</th></tr>
                  <tr><th>True 0</th><td>{{ result.confusion_matrix[0][0] }}</td><td>{{ result.confusion_matrix[0][1] }}</td></tr>
                  <tr><th>True 1</th><td>{{ result.confusion_matrix[1][0] }}</td><td>{{ result.confusion_matrix[1][1] }}</td></tr>
                </table>
              </div>
              {% endif %}

              {% if sample_rows %}
              <div>
                <div class="section-title">Sample Predictions (first 10)</div>
                <table>
                  <tr><th>#</th><th>Probability</th><th>Decision</th></tr>
                  {% for idx, prob, pred in sample_rows %}
                  <tr>
                    <td class="mono">{{ idx }}</td>
                    <td class="mono">{{ '%.6f'|format(prob) }}</td>
                    <td class="mono">{{ pred }}</td>
                  </tr>
                  {% endfor %}
                </table>
              </div>
              {% endif %}

              {% if classification_report %}
              <div>
                <div class="section-title">Classification Report</div>
                <pre class="mono">{{ classification_report }}</pre>
              </div>
              {% endif %}
            </div>
          </div>
        {% endif %}
      {% endif %}
    </main>
    <footer>SSAE + BiLSTM powered evaluation • Flask integration demo</footer>
  </body>
</html>
"""


@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML, result=None, sample_rows=[], classification_report=None)


@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    file = request.files.get('file')
    threshold = float(request.form.get('threshold', 0.5))
    if not file:
        return render_template_string(
            INDEX_HTML,
            result={'error': 'No file provided.'},
            sample_rows=[],
            classification_report=None,
        )

    try:
        # Read CSV to DataFrame; prefer using 'id' as index when present
        df = pd.read_csv(file)
        if 'id' in df.columns:
            df = df.set_index('id')

        # Preprocess to training schema and scale
        X, labels = preprocess_dataframe(df)

        # Encode to 32-dim then to sequences
        encoded = ssae_encoder.predict(X, verbose=0)
        seqs = encoded.reshape(-1, 1, 32)

        # Predict
        probs = classifier_model.predict(seqs, verbose=0).reshape(-1)
        preds = (probs > threshold).astype(int)

        # Compose output
        out = {
            'count': int(len(preds)),
            'threshold': float(threshold),
            'positives': int(preds.sum()),
            'negatives': int((1 - preds).sum()),
            'sample': {
                'probabilities_first_10': probs[:10].tolist(),
                'predictions_first_10': preds[:10].tolist()
            }
        }

        sample_rows = [
            (idx, prob, pred)
            for idx, (prob, pred) in enumerate(
                zip(out['sample']['probabilities_first_10'], out['sample']['predictions_first_10'])
            )
        ]

        # If labels available, compute metrics
        classification_report_text = None
        if labels is not None:
            from sklearn.metrics import confusion_matrix, classification_report
            m = min(len(preds), len(labels))
            # Force 2x2 ordering [0,1]
            cm = confusion_matrix(labels[:m], preds[:m], labels=[0, 1])
            out['confusion_matrix'] = cm.tolist()
            # Metrics
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            acc = (tp + tn) / total if total else 0.0
            precision_pos = tp / (tp + fp) if (tp + fp) else 0.0
            recall_pos = tp / (tp + fn) if (tp + fn) else 0.0
            tnr_neg = tn / (tn + fp) if (tn + fp) else 0.0
            out['accuracy'] = acc
            out['metrics'] = {
                'precision_pos': precision_pos,
                'recall_pos': recall_pos,
                'tnr_neg': tnr_neg,
            }
            classification_report_text = classification_report(labels[:m], preds[:m])

        return render_template_string(
            INDEX_HTML,
            result=out,
            classification_report=classification_report_text,
            sample_rows=sample_rows,
        )
    except Exception as e:
        return render_template_string(
            INDEX_HTML,
            result={'error': str(e)},
            sample_rows=[],
            classification_report=None,
        )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=False)
