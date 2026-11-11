"""Thresholds and switches that decide whether a retrained model is accepted."""

THRESHOLD_PERCENTILES = (55, 60, 65, 70, 75)
MIN_THRESHOLD_PRECISION = 0.35
MIN_THRESHOLD_RECALL = 0.35

# --- Evidence gates: judge models on money, not only on accuracy ---------------
# Estimated cost of one round trip as a fraction of the position. Alpaca crypto
# charges about 0.25% per side; US stocks are commission-free, so only a small
# allowance for spread and slippage is assumed. Adjust if your costs differ.
ROUND_TRIP_COST = {"crypto": 0.005, "stock": 0.0005}
# Mirrors of the live entry bars in run_execution.py (REGRESSOR_ENTRY_MIN and
# CLASSIFIER_ENTRY_MIN). Keep these in sync with that file.
REGRESSOR_ENTRY_MIN = 0.006
CLASSIFIER_ENTRY_MIN = 0.60
# The holdout back-test of the live rule needs at least this many trades and a
# positive average net return per trade.
MIN_BACKTEST_TRADES = 10
# Walk-forward: sequential expanding-window folds over the whole history.
WALK_FORWARD_FOLDS = 4
WALK_FORWARD_MIN_POSITIVE_FOLDS = 3  # folds whose ROC-AUC must beat 0.5
WALK_FORWARD_MIN_MEAN_AUC = 0.52
MIN_FOLD_TRAIN_ROWS = 150
# When True, a candidate must also pass the evidence gates to replace the live model.
ENFORCE_EVIDENCE_GATES = True
# When True, an asset whose live model shows no demonstrated edge is marked "blocked"
# in Model/model_status.json and run_execution.py stops opening new positions in it.
BLOCK_BUYING_ON_FAILURE = True
STATUS_FILE_NAME = "model_status.json"
# A blocked asset only reopens after this many consecutive weekly evidence passes.
REQUIRED_PASS_STREAK = 2
# Forward test: the live model is replayed on bars that arrived after tracking began
# (after the model was last trained), so the result is genuinely out of sample. Once it
# has this many trades it overrides the in-sample checks: a positive mean net return
# allows buying, a non-positive one blocks it.
MIN_FORWARD_TRADES = 10
# Classifier challenger for assets whose live model is a legacy classifier. It is trained
# and scored like the regressors, but only reported: set True only if models.py can save
# an XGBClassifier in the format load_trading_model() reads.
DEPLOY_CLASSIFIER_CHALLENGER = False
CLASSIFIER_EXIT_BAR = 0.55  # mirrors LEGACY_CLASSIFIER_THRESHOLD
