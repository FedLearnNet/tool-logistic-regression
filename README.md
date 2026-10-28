# tool-logistic-regression

Trains logistic regression models and exports predictions, coefficients, and feature explanations.

## Runtime inputs

Supply input files according to the schemas in `app.yml`.

## Inputs and outputs

### Inputs

| Name | Type | Description |
| --- | --- | --- |
| `data` | CSV | Training CSV containing numeric features and a target column (target_column). Optional first column 'id'. |

### Outputs

| Name | Type | Description |
| --- | --- | --- |
| `explanation` | HTML | Human-readable explainability report (top positive/negative coefficients per class). |
| `report` | TEXT | Small textual run report. |
| `predictions` | CSV | Predictions CSV including label and probabilities for each class (p_<class> columns). |
| `coefficients` | CSV | Long-form coefficient table per class (includes intercept). |

## Configuration

| Name | Default | Description |
| --- | --- | --- |
| `target_column` | `col_5` | Name of the target/label column in the training CSV. |
| `standardize` | `1` | Standardize numeric features before training and prediction. |
| `C` | `1.0` | Inverse regularization strength (smaller => stronger regularization). |
| `max_iter` | `200` | Maximum number of solver iterations. |
| `solver` | `lbfgs` | Optimizer/solver used for Logistic Regression. |
| `multi_class` | `auto` | Multi-class handling strategy. |
| `class_weight` | `balanced` | Class weighting strategy. Use balanced for imbalanced datasets. |
| `random_state` | `42` | Random seed. |
| `explain_top_k` | `15` | Number of top positive/negative coefficients per class to show in explanation.html. |

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# Set APP_ID and platform URLs before connecting.
python main.py
```

The Docker base image provides the FL-Net API. `requirements.txt` lists additional
runtime dependencies; `requirements-dev.txt` includes the API for local development.
Runtime datasets and trained models must be supplied separately; see `data/README.md`.

Build with `docker build -t tool-logistic-regression .`.
