"""Model retraining, split by responsibility.

settings     thresholds and switches that decide whether a model is accepted
data         model paths, trading costs and market-history download
gates        quality gate, strategy simulation, walk-forward check, calibration log
threshold    prediction-threshold selection
candidate    train a candidate model and run it through every gate
incumbent    score the model already in Model/ (and the classifier challenger)
status       the buy/block decision and the Model/model_status.json file
report       the Telegram retraining report

The command-line entry point stays ``python retrain.py``.
"""
