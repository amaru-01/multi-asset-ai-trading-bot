# Security Policy

This is an educational trading-bot prototype that can hold broker API keys and place
orders, so security problems matter even though it trades on paper by default.

## Supported versions

Only the latest code on the `main` branch is supported. There are no versioned releases.

## Reporting a vulnerability

Please **do not open a public issue or pull request** for a security problem.

Use GitHub's private reporting instead: open the repository's **Security** tab and choose
**Report a vulnerability**. If that option is not available, open a normal issue that says
only that you have a security concern, with no technical details, and ask for a private way
to share them.

Helpful details: what you found, where (file and line if you can), how to reproduce it, and
what you think the impact is. You will get an acknowledgement as soon as the maintainer can
respond; this is a personal project, so there is no guaranteed response time.

## What is in scope

- Code paths that could place orders, or bypass the live-trading safeguards, the kill
  switch, the daily-loss circuit breaker or the live capital cap in `run_execution.py`
- Ways credentials (Alpaca keys, Telegram tokens, the GitHub dispatch token) could leak,
  for example into logs, messages, commits or workflow output
- Unsafe handling of untrusted input, such as the market data and model files the bot loads
- Insecure GitHub Actions workflow configuration (excess permissions, injection risks)

Trading losses, poor strategy performance and bugs without a security impact are not
security issues; please open a normal issue for those.

## Keeping your own deployment safe

- **Never commit secrets.** Keep keys in `.env` (already git-ignored) or in GitHub Actions
  secrets. If a key is ever committed or pasted somewhere public, revoke and replace it
  immediately; deleting the commit is not enough.
- **Use paper-trading keys** until you have a deliberate reason not to. Live trading needs
  `ALPACA_PAPER=false`, `ALPACA_LIVE_CONFIRM` and `ALPACA_LIVE_MAX_CAPITAL`; use a broker
  key with the smallest permissions and the smallest balance that works.
- **Scope the scheduler token.** The token an external timer (such as cron-job.org) uses to
  trigger workflows should be a fine-grained GitHub token limited to this one repository
  with only "Actions: read and write", and it should have an expiry date.
- **Keep a kill switch ready.** Setting the `TRADING_HALTED` repository variable to `true`
  stops new entries on the next run (see the README).
- **Be careful with model files.** The bot loads native XGBoost JSON models from `Model/`.
  Only use models you trained yourself or fully trust. The repository also contains an old
  `Model/xgb_trading_model.pkl`; the current code does not load it, and you should never
  unpickle a `.pkl` file from a source you do not trust.
