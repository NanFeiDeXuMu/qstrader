"""
main.py — CS377 Project entry point

Three modes (select via --mode):
  train    Train the PPO agent (train set 2010–2018, validation set 2019)
  backtest Run out-of-sample backtest (2020–2023) with the trained model and compare to benchmarks
  both     Run train then backtest sequentially

Usage examples:
  python main.py --mode train
  python main.py --mode backtest
  python main.py --mode both
  QSTRADER_CSV_DATA_DIR=/data python main.py --mode both
"""

import argparse
import os
import datetime

import numpy as np
import pandas as pd
import pytz
import matplotlib
matplotlib.use('Agg')          # safe in headless environments; change to 'TkAgg' or remove if a display is available
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── QSTrader core modules ─────────────────────────────────────────
from qstrader.asset.equity import Equity
from qstrader.asset.universe.static import StaticUniverse
from qstrader.alpha_model.fixed_signals import FixedSignalsAlphaModel
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession

# ── PPO modules ───────────────────────────────────────────────────
from qstrader.alpha_model.ppo_model import PPOModel
from qstrader.alpha_model.feature_handler import FeatureHandler
from qstrader.alpha_model.ppo_training import main as train_ppo
from qstrader import settings as qstrader_settings


# ─────────────────────────────────────────────────────────────────
# Global configuration
# ─────────────────────────────────────────────────────────────────

SYMBOLS = ['SPY', 'AGG', 'GLD', 'SHY', 'TLT']
ASSETS  = ['EQ:SPY', 'EQ:AGG', 'EQ:GLD', 'EQ:SHY', 'EQ:TLT']

# Out-of-sample test window (train 2010-2018, validation 2019, test 2020-2023)
TEST_START = pd.Timestamp('2020-01-01 14:30:00', tz=pytz.UTC)
TEST_END   = pd.Timestamp('2023-12-31 23:59:00', tz=pytz.UTC)

PPO_MODEL_PATH = 'ppo_final_model.zip'
PPO_VECNORM_PATH = 'ppo_vecnormalize.pkl'
TRADING_DAYS_PER_YEAR = 252


# ─────────────────────────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────────────────────────

def build_data_handler(universe, symbols):
    """Load CSVs from QSTRADER_CSV_DATA_DIR (or examples/) and return a BacktestDataHandler."""
    _default_csv = os.path.join(os.path.dirname(__file__), 'examples')
    csv_dir = os.environ.get('QSTRADER_CSV_DATA_DIR', _default_csv)
    data_source = CSVDailyBarDataSource(
        csv_dir, Equity, csv_symbols=symbols, adjust_prices=False
    )
    return BacktestDataHandler(universe, data_sources=[data_source])


# ─────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────

def run_train():
    """Invoke ppo_training.main() to train and save the model."""
    print("=== [train] Starting PPO training (train 2010-2018, validation 2019) ===")
    train_ppo()
    print(f"=== [train] Done — model saved to {PPO_MODEL_PATH} ===")


# ─────────────────────────────────────────────────────────────────
# Backtest: three strategies
# ─────────────────────────────────────────────────────────────────

def run_ppo_backtest(data_handler, universe):
    """Run the trained PPO model on the test set and return an equity curve DataFrame."""
    feature_handler = FeatureHandler(
        data_handler=data_handler,
        assets=ASSETS,
        lookback=20
    )
    alpha_model = PPOModel(
        ppo_model_path=PPO_MODEL_PATH,
        assets=ASSETS,
        feature_handler=feature_handler,
        vecnormalize_path=PPO_VECNORM_PATH
    )
    backtest = BacktestTradingSession(
        start_dt=TEST_START,
        end_dt=TEST_END,
        universe=universe,
        alpha_model=alpha_model,
        rebalance='daily',
        long_only=True,
        cash_buffer_percentage=0.01,
        data_handler=data_handler
    )
    backtest.run()
    return backtest.get_equity_curve()


def run_benchmark_buyhold_spy():
    """Buy-and-hold SPY with its own Universe and DataHandler."""
    spy_universe = StaticUniverse(['EQ:SPY'])
    spy_handler  = build_data_handler(spy_universe, ['SPY'])
    alpha_model  = FixedSignalsAlphaModel({'EQ:SPY': 1.0})
    backtest = BacktestTradingSession(
        start_dt=TEST_START,
        end_dt=TEST_END,
        universe=spy_universe,
        alpha_model=alpha_model,
        rebalance='buy_and_hold',
        long_only=True,
        cash_buffer_percentage=0.01,
        data_handler=spy_handler
    )
    backtest.run()
    return backtest.get_equity_curve()


def run_benchmark_equal_weight(data_handler, universe):
    """Five-asset equal-weight portfolio rebalanced at end of month."""
    equal_weights = {asset: 1.0 / len(ASSETS) for asset in ASSETS}
    alpha_model   = FixedSignalsAlphaModel(equal_weights)
    backtest = BacktestTradingSession(
        start_dt=TEST_START,
        end_dt=TEST_END,
        universe=universe,
        alpha_model=alpha_model,
        rebalance='end_of_month',
        long_only=True,
        cash_buffer_percentage=0.01,
        data_handler=data_handler
    )
    backtest.run()
    return backtest.get_equity_curve()


# ─────────────────────────────────────────────────────────────────
# Performance metrics
# ─────────────────────────────────────────────────────────────────

def _equity_to_returns(equity_curve: pd.DataFrame) -> pd.Series:
    """Extract daily percentage returns from an equity curve DataFrame."""
    col = 'Equity'
    if col not in equity_curve.columns:
        col = equity_curve.columns[0]
    return equity_curve[col].pct_change().dropna()


def compute_metrics(equity_curve: pd.DataFrame, name: str) -> dict:
    """
    Compute and return standard portfolio performance metrics:
      - Annualised return
      - Annualised volatility
      - Sharpe ratio (risk-free rate = 0)
      - Maximum drawdown
      - Calmar ratio (annualised return / |max drawdown|)
      - Total cumulative return
    """
    col = 'Equity'
    if col not in equity_curve.columns:
        col = equity_curve.columns[0]

    equity = equity_curve[col].dropna()
    daily_ret = equity.pct_change().dropna()

    ann_ret  = (1 + daily_ret).prod() ** (TRADING_DAYS_PER_YEAR / len(daily_ret)) - 1
    ann_vol  = daily_ret.std()  * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan

    # Maximum drawdown
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max
    max_dd      = drawdown.min()                        # negative number

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1.0

    return {
        'Strategy':       name,
        'Ann. Return':    f'{ann_ret:.2%}',
        'Ann. Volatility':f'{ann_vol:.2%}',
        'Sharpe':         f'{sharpe:.3f}',
        'Max Drawdown':   f'{max_dd:.2%}',
        'Calmar':         f'{calmar:.3f}',
        'Total Return':   f'{total_ret:.2%}',
    }


def print_metrics_table(metrics_list: list):
    """Print a right-aligned metrics table to stdout."""
    if not metrics_list:
        return
    keys = list(metrics_list[0].keys())
    col_widths = {k: max(len(k), max(len(str(m[k])) for m in metrics_list)) for k in keys}

    header = '  '.join(k.ljust(col_widths[k]) for k in keys)
    sep    = '  '.join('-' * col_widths[k] for k in keys)
    print('\n' + header)
    print(sep)
    for m in metrics_list:
        print('  '.join(str(m[k]).ljust(col_widths[k]) for k in keys))
    print()


# ─────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────

def _normalise(equity_curve: pd.DataFrame) -> pd.Series:
    """Normalise an equity curve so the starting value equals 1."""
    col = 'Equity'
    if col not in equity_curve.columns:
        col = equity_curve.columns[0]
    equity = equity_curve[col].dropna()
    return equity / equity.iloc[0]


def plot_comparison(
    ppo_eq:   pd.DataFrame,
    spy_eq:   pd.DataFrame,
    ew_eq:    pd.DataFrame,
    metrics:  list,
    save_path: str = None
):
    """
    Two-panel comparison chart:
      Top:    normalised equity curves (PPO / Buy & Hold SPY / Equal-Weight)
      Bottom: PPO cumulative excess return vs SPY
    Performance metrics table shown to the right.
    Saved as PNG; display attempted if a GUI backend is available.
    """
    ppo_norm = _normalise(ppo_eq)
    spy_norm = _normalise(spy_eq)
    ew_norm  = _normalise(ew_eq)

    # Align on shared dates
    common_idx = ppo_norm.index.intersection(spy_norm.index)
    excess = ppo_norm.loc[common_idx] - spy_norm.loc[common_idx]

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(
        2, 2,
        width_ratios=[3, 1],
        height_ratios=[2, 1],
        hspace=0.35,
        wspace=0.35
    )

    # ── Top-left: equity curves ──────────────────────────────────
    ax_main = fig.add_subplot(gs[0, 0])
    ax_main.plot(ppo_norm.index, ppo_norm.values,  label='PPO Agent',         color='#1f77b4', linewidth=1.6)
    ax_main.plot(spy_norm.index, spy_norm.values,  label='Buy & Hold SPY',    color='#ff7f0e', linewidth=1.2, linestyle='--')
    ax_main.plot(ew_norm.index,  ew_norm.values,   label='Equal-Weight (EOM)', color='#2ca02c', linewidth=1.2, linestyle=':')
    ax_main.set_title('Normalised Equity Curves (2020–2023)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Portfolio Value (start = 1)')
    ax_main.legend(fontsize=9)
    ax_main.grid(True, alpha=0.3)

    # ── Bottom-left: excess return ───────────────────────────────
    ax_excess = fig.add_subplot(gs[1, 0])
    ax_excess.fill_between(
        excess.index, excess.values, 0,
        where=(excess.values >= 0), color='#2ca02c', alpha=0.4, label='Outperform'
    )
    ax_excess.fill_between(
        excess.index, excess.values, 0,
        where=(excess.values < 0),  color='#d62728', alpha=0.4, label='Underperform'
    )
    ax_excess.axhline(0, color='black', linewidth=0.8)
    ax_excess.set_title('PPO Excess Return vs Buy & Hold SPY', fontsize=10)
    ax_excess.set_ylabel('Excess (normalised)')
    ax_excess.legend(fontsize=8)
    ax_excess.grid(True, alpha=0.3)

    # ── Right: metrics table ─────────────────────────────────────
    ax_table = fig.add_subplot(gs[:, 1])
    ax_table.axis('off')

    if metrics:
        keys       = [k for k in metrics[0].keys() if k != 'Strategy']
        row_labels = [m['Strategy'] for m in metrics]
        col_labels = keys
        cell_data  = [[m[k] for k in keys] for m in metrics]

        tbl = ax_table.table(
            cellText=cell_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.6)

        # Highlight header row and row-label column
        for (row, col), cell in tbl.get_celld().items():
            if row == 0 or col == -1:
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor('#EBF3FB' if row % 2 == 0 else 'white')

        ax_table.set_title('Performance Metrics\n(2020–2023)', fontsize=10, fontweight='bold', pad=12)

    fig.suptitle(
        'PPO Portfolio Optimisation vs Benchmarks',
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[chart] Saved to {save_path}")
    plt.close()
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Backtest entry point
# ─────────────────────────────────────────────────────────────────

def run_backtest():
    """Run all three strategies, print a metrics table, and save the comparison chart."""
    qstrader_settings.set_print_events(False)
    universe = StaticUniverse(ASSETS)

    # Each strategy gets its own data_handler to prevent cumulative_offsets state
    # from leaking across runs.
    print("=== [backtest] Strategy 1/3: PPO Agent ===")
    ppo_equity = run_ppo_backtest(build_data_handler(universe, SYMBOLS), universe)

    print("=== [backtest] Strategy 2/3: Buy & Hold SPY (benchmark) ===")
    spy_equity = run_benchmark_buyhold_spy()

    print("=== [backtest] Strategy 3/3: Equal-Weight 5-asset (end-of-month rebalance) ===")
    ew_equity  = run_benchmark_equal_weight(build_data_handler(universe, SYMBOLS), universe)

    # ── Compute and print performance metrics ────────────────────
    metrics = [
        compute_metrics(ppo_equity, 'PPO Agent'),
        compute_metrics(spy_equity, 'Buy & Hold SPY'),
        compute_metrics(ew_equity,  'Equal-Weight'),
    ]
    print("\n========== Backtest performance comparison (out-of-sample 2020–2023) ==========")
    print_metrics_table(metrics)

    # ── Multi-strategy comparison chart ─────────────────────────
    charts_dir = os.path.join(os.path.dirname(__file__), 'backtest_charts')
    os.makedirs(charts_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_path = os.path.join(charts_dir, f'backtest_comparison_{timestamp}.png')
    plot_comparison(ppo_equity, spy_equity, ew_equity, metrics, save_path=chart_path)

    # ── QSTrader native tearsheet (PPO vs SPY) ──────────────────
    tearsheet = TearsheetStatistics(
        strategy_equity=ppo_equity,
        benchmark_equity=spy_equity,
        title='PPO Trading Agent vs Buy & Hold SPY (2020–2023)'
    )
    tearsheet.plot_results()


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='CS377 PPO Trading Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python main.py --mode train\n'
            '  python main.py --mode backtest\n'
            '  QSTRADER_CSV_DATA_DIR=/data python main.py --mode both\n'
        )
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'backtest', 'both'],
        default='both',
        help='Run mode: train / backtest / both (default: both)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'train':
        run_train()
    elif args.mode == 'backtest':
        run_backtest()
    else:  # both
        run_train()
        run_backtest()
