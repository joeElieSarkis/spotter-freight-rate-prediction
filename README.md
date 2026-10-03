# Freight rate prediction

Predict the total posted rate in dollars for the 12,000 November-December loads supplied in the assessment.

## Setup

Use Python 3.12. From this folder on Windows:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

On macOS/Linux, replace `.venv\Scripts\python` with `.venv/bin/python`.

## Files

- `assessment/`: original instructions and scorer requirements.
- `data/`: supplied CSV files, with hyphens changed to underscores in filenames.
- `score.py`: the unmodified supplied output validator and chart generator.
- `freight/`: data checks, features, training, and prediction code.
- `tests/`: checks for the data and prediction contracts.
- `artifacts/`: reproducible audit and evaluation results.

## Validation plan

The labeled data covers January-October 2025. Final predictions cover November-December, so validation uses whole future months:

| Use | Train through | Evaluate on |
| --- | --- | --- |
| Model selection | April 30 | May-June |
| Model selection | June 30 | July-August |
| Final local evaluation | August 31 | September-October |
| Submission fit | October 31 | November-December, labels unavailable |

Choose models using mean dollar MAE across the two selection windows. Also report RMSE, WAPE, and R-squared. Keep every evaluation label, including unusually large or small rates. The provided scorer does not measure accuracy; Spotter calculates the final metrics privately.

The fixed December scenario omits quote signals, market indices, and coordinates. Its model must be evaluated using the same available inputs. Do not invent future market values or copy quotes from unrelated loads.
