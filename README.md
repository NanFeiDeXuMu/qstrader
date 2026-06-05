# PPO Portfolio Optimisation with QSTrader

> **CS377 Team 12** — Nurtilek Duishobaev · Tattep Lakmuang · Liu Yixuan
>
> **Research question:** Does deep reinforcement learning (PPO) learn to allocate a multi-asset portfolio better than naïve baselines (buy-and-hold, equal-weight) in a clean, fully controlled setup?

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Architecture & Data Flow](#3-architecture--data-flow)
4. [Quick Start](#4-quick-start)
5. [Training Pipeline Details](#5-training-pipeline-details)
6. [Backtest Pipeline Details](#6-backtest-pipeline-details)
7. [Key Design Decisions (read before changing anything)](#7-key-design-decisions-read-before-changing-anything)
8. [Known Limitations & Next Steps](#8-known-limitations--next-steps)
9. [Bug-Fix History](#9-bug-fix-history)

---

## 1. Project Overview

| Dimension | Value |
|-----------|-------|
| Asset universe | SPY, AGG, GLD, IEI, TLT (5 ETFs) |
| Train | 2010–2018 (random 252-day sub-windows per episode) |
| Validation | 2019 (fixed window, used by EvalCallback) |
| Test (out-of-sample) | 2020–2023 |
| RL algorithm | PPO (Stable-Baselines3) + MlpPolicy |
| Backtest framework | QSTrader (daily OHLCV CSV data) |
| Action space | `Box(-1e4, +1e4, shape=(5,))` — raw logits; softmax inside env |
| Observation space | `Box(-1e4, +1e4, shape=(15,))` — 5 assets × 3 rolling features |
| Reward | Daily log return `log(V_t / V_{t-1})` |

The codebase adapts [QSTrader](https://github.com/mhallsmoore/qstrader) as the simulation engine and adds a Gymnasium-compatible wrapper so Stable-Baselines3 can drive portfolio rebalancing directly.

---

## 2. Repository Layout

```
qstrader/
├── main.py                              # CLI entry point (--mode train/backtest/both)
├── ppo_final_model.zip                  # Saved PPO weights (output of training)
├── ppo_vecnormalize.pkl                 # VecNormalize obs statistics (must ship with model)
├── ppo_checkpoints/                     # Per-10k-step checkpoints + best/ subdir
├── backtest_comparison.png              # Output chart (3-strategy equity curves + metrics)
├── examples/                            # CSV price data (SPY.csv, AGG.csv, …)
│
└── qstrader/
    ├── alpha_model/
    │   ├── env_setup.py                 # Gymnasium env wrapping QSTrader (★ core RL file)
    │   ├── ppo_training.py              # PPO training entry point
    │   ├── ppo_model.py                 # PPO inference adapter (AlphaModel interface)
    │   ├── feature_handler.py           # Market data → 15-dim state vector
    │   ├── alpha_model.py               # AlphaModel abstract base class
    │   └── fixed_signals.py             # Fixed-weight baseline alpha model
    │
    ├── trading/backtest.py              # BacktestTradingSession (simulation orchestrator)
    ├── system/qts.py                    # QuantTradingSystem (PCM + Execution)
    ├── system/rebalance/                # Rebalance schedules (daily/weekly/EOM/buy_and_hold)
    ├── portcon/
    │   ├── pcm.py                       # PortfolioConstructionModel
    │   ├── optimiser/fixed_weight.py    # Pass-through optimiser (weights unchanged)
    │   └── order_sizer/dollar_weighted.py  # Long-only share-quantity calculator
    ├── broker/simulated_broker.py       # Simulated broker (positions, cash, order fill)
    ├── data/
    │   ├── backtest_data_handler.py     # Price access layer (bid/ask/historical closes)
    │   └── daily_bar_csv.py             # CSV data source (OHLCV → bid/ask DataFrame)
    └── simulation/daily_bday.py         # Daily market event engine (open/close)
```

---

## 3. Architecture & Data Flow

### 3.1 How the RL loop connects to QSTrader

```
PPO.learn()
  └─ QSTraderExecutionEnv.step(action)          # env_setup.py
       │
       ├─ proxyAlphaModel.current_actions = action   # store raw logits
       │
       └─ _advance_to_market_open()
            ├─ market_close event
            │    └─ BacktestTradingSession.qts(dt)
            │         └─ PortfolioConstructionModel(dt)
            │              ├─ proxyAlphaModel(dt)     # softmax(logits) → {asset: weight}
            │              ├─ FixedWeightOptimiser     # pass-through
            │              └─ DollarWeightedOrderSizer → rebalance Orders
            │                   └─ SimulatedBroker.submit_order()
            │
            └─ market_open event
                 └─ FeatureHandler(dt) → obs (15,)
```

**Key invariant:** The PPO policy never touches QSTrader internals directly. It only writes logits into `proxyAlphaModel.current_actions`; QSTrader reads them via the normal `AlphaModel.__call__` interface.

### 3.2 Episode structure

Each `reset()` samples a **random 252-trading-day sub-window** from 2010–2018 (start date drawn uniformly from all valid business days that leave room for a full 252-day window and have at least 25 days of prior price history for the feature lookback).

```
[  25 burn-in days excluded from valid start range  ]
[ ── 252-day episode ── ]
  day 1 … day 252
  each day = 1 RL step = 1 (market_open obs → action → market_close rebalance → reward)
```

### 3.3 Action space & weight mapping

```
PPO outputs logits a ∈ [-1e4, 1e4]^5  (finite but functionally unbounded)
         ↓  softmax  (inside proxyAlphaModel.__call__ and PPOModel._action_to_weights)
weights w_i = exp(a_i) / Σ exp(a_j)   ∈ (0,1),  Σ w_i = 1
```

Using a large but finite logit space gives the policy a unique, bijective gradient signal for every weight vector. The old `[0,1]` box caused gradient degeneracy: many distinct raw actions normalised to the same weight vector. Bounds of `±1e4` are used instead of `±∞` because Stable-Baselines3 ≥2.3 requires finite action-space bounds; softmax differences beyond ~10 already saturate to near-one-hot weights, so the practical difference is negligible.

### 3.4 Observation (state) vector — 15 dimensions

For each of the 5 assets, `FeatureHandler` computes over a 20-day rolling window:

| Index | Feature | Description |
|-------|---------|-------------|
| 3i | `latest_log_return` | `log(p_t) - log(p_{t-1})` |
| 3i+1 | `mean_log_return` | Mean of 20-day log returns (momentum proxy) |
| 3i+2 | `std_log_return` | Std of 20-day log returns (volatility proxy) |

If fewer than 2 prices are available (e.g. at the start of a CSV), the fallback uses the last known `mean` and `std` for that asset (accumulated lazily once a full window has been seen), with `latest_log_return = 0`. This avoids the all-zero observation problem that corrupted warm-up steps in the previous implementation.

### 3.5 Reward

```python
reward = log(portfolio_value_t / portfolio_value_{t-1})
```

Log return is scale-invariant and additive over time. The previous implementation used absolute dollar P&L, which caused Critic instability because the magnitude varied by orders of magnitude across market regimes.

---

## 4. Quick Start

### 4.1 Dependencies

```bash
pip install stable-baselines3 gymnasium pandas numpy matplotlib pytz tabulate python-docx
```

### 4.2 Data

Place `SPY.csv, AGG.csv, GLD.csv, IEI.csv, TLT.csv` in `examples/`. Format (Yahoo Finance daily):

```
Date,Open,High,Low,Close,Volume
2010-01-04,112.37,113.39,111.51,113.33,118944600
```

### 4.3 Run

```bash
# Train PPO on 2010–2018, save model + VecNormalize stats
python main.py --mode train

# Backtest trained model on 2020–2023 vs two baselines, save backtest_comparison.png
python main.py --mode backtest

# Train then immediately backtest
python main.py --mode both

# Custom data directory
QSTRADER_CSV_DATA_DIR=/path/to/csvs python main.py --mode both
```

### 4.4 Outputs

| File | Description |
|------|-------------|
| `ppo_final_model.zip` | Trained PPO policy weights |
| `ppo_vecnormalize.pkl` | Running obs normalisation stats — **must be kept alongside the model** |
| `ppo_checkpoints/` | Per-10k-step checkpoints; `best/` subdirectory holds the best validation checkpoint |
| `backtest_comparison.png` | 3-strategy normalised equity curves + performance metrics table |

---

## 5. Training Pipeline Details

**Entry point:** `qstrader/alpha_model/ppo_training.py`

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `state_dim` | 15 | 5 assets × 3 features |
| `action_dim` | 5 | One logit per asset |
| `learning_rate` | 3e-4 | Adam |
| `n_steps` | 2048 | Rollout buffer length before each update |
| `batch_size` | 64 | Mini-batch size for gradient updates |
| `n_epochs` | 10 | PPO update epochs per rollout |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_range` | 0.2 | PPO clipping ε |
| `ent_coef` | **0.01** | Entropy bonus — keeps exploration alive, prevents premature convergence to equal-weight |
| `total_timesteps` | 500,000 | ≈ 1,984 random 252-day episodes |
| `VecNormalize` | `norm_obs=True, norm_reward=True, clip_obs=10.0` | Running normalisation of obs and rewards |

### Checkpointing & evaluation

- Checkpoint saved every 10,000 steps → `ppo_checkpoints/ppo_model_<step>_steps.zip`
- `EvalCallback` evaluates on fixed 2019 window every 50,000 steps; best model saved to `ppo_checkpoints/best/`
- `ppo_vecnormalize.pkl` saved at end of training — required for consistent inference

---

## 6. Backtest Pipeline Details

**Entry point:** `main.py → run_backtest()`

Three strategies run over the **2020–2023 out-of-sample period**:

| Strategy | Alpha Model | Rebalance |
|----------|------------|-----------|
| PPO Agent | `PPOModel` (trained network, softmax action) | Daily |
| Buy & Hold SPY | `FixedSignalsAlphaModel({'EQ:SPY': 1.0})` | Initial buy only |
| Equal-Weight | `FixedSignalsAlphaModel({each asset: 0.2})` | End of month |

Each strategy gets its own independent `BacktestDataHandler` instance to avoid `cumulative_offsets` state leaking between runs.

**Metrics computed:** Annualised return, annualised volatility, Sharpe ratio, max drawdown, Calmar ratio, total return.

---

## 7. Key Design Decisions (read before changing anything)

### 7.1 `CSVDailyBarDataSource` is created once in `QSTraderExecutionEnv.__init__`

The data source is shared across all `reset()` calls. Creating a new instance per reset caused LRU-cache memory growth (the `@functools.lru_cache` on `get_bid`/`get_ask` accumulates ~tens of MB per episode). Only the thin `BacktestDataHandler` wrapper is re-created each reset.

### 7.2 `BacktestDataHandler.get_asset_latest_mid_price` does NOT add `cumulative_offsets`

The `cumulative_offsets` mechanism exists for slippage simulation. Mid-price is used only for position marking (unrealised P&L). Adding offsets there created a runaway positive-feedback loop (`total_equity → ∞`). Bid and ask prices do add offsets (intentionally, for execution price simulation).

### 7.3 `VecNormalize` statistics must travel with the model

`ppo_final_model.zip` and `ppo_vecnormalize.pkl` are a matched pair. If you retrain, both files are overwritten together. At inference time, `PPOModel.__init__` loads `ppo_vecnormalize.pkl` and applies `normalize_obs()` before calling `model.predict()`. Deploying the model without the `.pkl` will silently degrade performance because the policy was trained on normalised observations.

### 7.4 Random episode sampling requires `_valid_starts`

`_valid_starts` is computed once in `__init__` and excludes the first `BURN_IN_DAYS=25` business days from the training range (so `FeatureHandler` always has ≥20 days of history) and any start date that would end the episode beyond `ending_day`. Changing `EPISODE_LENGTH_DAYS` or `BURN_IN_DAYS` constants at the top of `env_setup.py` affects how many valid start dates exist.

---

## 8. Known Limitations & Next Steps

The following are the most important directions for future work, roughly in order of expected impact.

### 8.1 Transaction cost penalty in the reward (highest priority — Proposal extension)

The Proposal explicitly names this as the primary extension. Currently `fee_model=ZeroFeeModel()` is passed to every `BacktestTradingSession`. The simulation infrastructure already supports `PercentFeeModel`.

**To add transaction cost to PPO training:**

1. In `BacktestTradingSession.__init__`, replace `ZeroFeeModel()` with `PercentFeeModel(commission=0.001)` (10 bps).
2. Alternatively, compute the cost explicitly inside `QSTraderExecutionEnv.step()`:

```python
# Approximate turnover-based cost penalty
prev_weights = np.array(list(self.alpha_model(dt).values()))  # before rebalance
# ... advance simulation ...
turnover = np.sum(np.abs(new_weights - prev_weights))
cost_penalty = 0.001 * turnover   # 10 bps × turnover fraction
reward = log_return - cost_penalty
```

3. Re-train with the penalty and compare against the zero-cost run. The expected result is that the agent learns to trade less frequently (lower turnover), which validates that the reward signal is being used correctly.

### 8.2 Multiple random seeds (3 seeds minimum per Proposal)

The Proposal specifies training with **3 random seeds** and reporting mean ± std across seeds. The current codebase trains a single model. To run multi-seed:

```python
# In ppo_training.py, wrap main() with a seed loop:
for seed in [42, 123, 7]:
    model = PPO(..., seed=seed)
    model.learn(...)
    model.save(f'ppo_model_seed{seed}.zip')
    train_env.save(f'ppo_vecnormalize_seed{seed}.pkl')
```

Report the mean Sharpe / cumulative return across seeds, not just the single best run, to avoid cherry-picking.

### 8.3 Expand state features

The current 15-dim state captures only short-term momentum and volatility. Consider adding:

| Feature | Rationale |
|---------|-----------|
| Current portfolio weights `w_t` | Allows agent to reason about turnover cost |
| 60-day momentum (`mean_log_return`, `lookback=60`) | Medium-term trend signal |
| Pairwise rolling correlation (upper triangle, 10 values) | Regime-dependent diversification signal |
| Drawdown from peak per asset | Risk-off signal |

If you add features, update `state_dim` in `training_config` in both `ppo_training.py` and `main.py`, and re-train from scratch (the saved model is tied to the old `state_dim=15`).

### 8.4 Increase training timesteps

500,000 steps ÷ 252 steps/episode ≈ 1,984 episodes. Financial RL typically requires 5–10× more episodes to converge, especially after adding transaction costs. Consider `total_timesteps=2_000_000` if compute allows.

### 8.5 Expand asset universe to Dow Jones 30

The Proposal mentions using Dow Jones 30 stocks. The current code uses 5 ETFs because they are liquid and diversified across asset classes. Switching to 30 stocks requires:

1. Download 30 CSVs into `examples/` (or a new directory).
2. Update `SYMBOLS` and `ASSETS` lists in `main.py`.
3. Update `state_dim` (30 × 3 = 90) and `action_dim` (30) in `training_config`.
4. The `FixedWeightOptimiser` and `DollarWeightedOrderSizer` require no changes — they are already asset-count agnostic.

### 8.6 Tune `ent_coef` and `learning_rate`

The current `ent_coef=0.01` was chosen to prevent collapse to equal-weight; it may be too high (too much randomness) or too low (still converging prematurely) for the transaction-cost setting. A simple grid search: `ent_coef ∈ {0.005, 0.01, 0.02}` × `learning_rate ∈ {1e-4, 3e-4}`.

### 8.7 Current benchmark gap context

In the 2020–2023 test period, Buy & Hold SPY outperforms the PPO agent. This is **partially expected** for structural reasons (see below), but the technical fixes in this codebase should significantly narrow the gap compared to the unfixed version:

| Root cause | Status |
|-----------|--------|
| Identical episode repeated 220× (no generalisation) | Fixed — random sub-windows |
| Dollar P&L reward (Critic instability) | Fixed — log return reward |
| `ent_coef=0.0` (zero exploration) | Fixed — `ent_coef=0.01` |
| Action space degeneracy | Fixed — softmax logit space |
| No obs normalisation | Fixed — `VecNormalize` |
| Train/test domain shift (QE era vs rate-hike era) | **Structural** — add macro features (§8.3) |
| Daily rebalancing friction vs zero-cost buy-and-hold | **Structural** — add transaction cost (§8.1) |

---

## 9. Bug-Fix History

The table below documents all fixes applied to the original skeleton code, for traceability.

### 9.1 Original skeleton bugs (from `CS377_Project_Plan.md`)

| File | Bug | Fix |
|------|-----|-----|
| `portcon/order_sizer/dollar_weighted.py` | Division by zero when price is NaN/0/negative; `int(inf)` overflow | Added price validity check; `np.isfinite` guard before `int()` |
| `portcon/order_sizer/long_short.py` | Only checked NaN, missed `price=0` | Extded check to `NaN or <= 0.0` |
| `data/backtest_data_handler.py` | `get_asset_latest_bid_ask_price` double-added `cumulative_offsets` | Removed redundant second addition |
| `data/backtest_data_handler.py` | `get_asset_latest_mid_price` added `cumulative_offsets`, causing `total_equity → ∞` | Removed offset from mid-price path |
| `alpha_model/env_setup.py` | Rebalance errors during training crashed the episode | Wrapped `qts()` in `try/except (ValueError, OverflowError)` |
| `alpha_model/ppo_model.py` | Missing `__call__`; wrong `feature_handler` call signature | Implemented `__call__(dt)`; fixed to `feature_handler(dt)` |
| `data/backtest_data_handler.py` | `get_assets_historical_closes` received unsupported `adjusted` kwarg | Removed the argument |

### 9.2 RL training quality bugs (fixed in current version)

| File | Bug | Fix |
|------|-----|-----|
| `alpha_model/env_setup.py` | Every episode replayed identical 2010–2018 window → memorisation, no generalisation | `reset()` now samples a random 252-day sub-window; `CSVDailyBarDataSource` created once in `__init__` to avoid LRU memory leak |
| `alpha_model/env_setup.py` | Reward was absolute dollar P&L → scale instability across regimes | Changed to `log(V_t / V_{t-1})` |
| `alpha_model/env_setup.py` | Action space `Box(0,1)` caused gradient degeneracy (many raw actions → same normalised weights); later `Box(-inf,inf)` rejected by SB3 ≥2.3 (`isfinite` assertion) | Changed to `Box(-1e4, 1e4)`; softmax mapping in `proxyAlphaModel.__call__` |
| `alpha_model/ppo_training.py` | `ent_coef=0.0` → zero exploration, converged to equal-weight after first pass | Set `ent_coef=0.01` |
| `alpha_model/ppo_training.py` | No obs/reward normalisation → Critic unstable with log-return rewards | Added `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0)` |
| `alpha_model/ppo_model.py` | Inference used `clip+normalise` for weights; training used no explicit normalisation → train/infer mismatch | Both now use identical softmax; `ppo_vecnormalize.pkl` applied at inference |
| `alpha_model/feature_handler.py` | Episode warm-up (first ~20 steps) emitted all-zero observation vectors | Fallback now uses lazily-accumulated historical mean/std per asset instead of zeros |

---

*QSTrader original framework: © 2015–2024 QuantStart.com, QuarkGluon Ltd. MIT License.*
