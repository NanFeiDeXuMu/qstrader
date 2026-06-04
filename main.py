"""
main.py — CS377 Project 主程序

三个模式（通过命令行 --mode 切换）：
  train    训练 PPO 智能体（训练集 2010–2018，验证集 2019）
  backtest 用训练好的模型做 out-of-sample 回测（2020–2023），与基准对比
  both     顺序执行 train 然后 backtest

用法示例：
  python main.py --mode train
  python main.py --mode backtest
  python main.py --mode both
  QSTRADER_CSV_DATA_DIR=/data python main.py --mode both
"""

import argparse
import os

import numpy as np
import pandas as pd
import pytz
import matplotlib
matplotlib.use('Agg')          # 无 GUI 环境下安全；有 GUI 时改为 'TkAgg' 或删除此行
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── QSTrader 核心模块 ─────────────────────────────────────────────
from qstrader.asset.equity import Equity
from qstrader.asset.universe.static import StaticUniverse
from qstrader.alpha_model.fixed_signals import FixedSignalsAlphaModel
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession

# ── PPO 相关模块 ──────────────────────────────────────────────────
from qstrader.alpha_model.ppo_model import PPOModel
from qstrader.alpha_model.feature_handler import FeatureHandler
from qstrader.alpha_model.ppo_training import main as train_ppo


# ─────────────────────────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────────────────────────

SYMBOLS = ['SPY', 'AGG', 'GLD', 'IEI', 'TLT']
ASSETS  = ['EQ:SPY', 'EQ:AGG', 'EQ:GLD', 'EQ:IEI', 'EQ:TLT']

# out-of-sample 测试区间（训练 2010-2018，验证 2019，测试 2020-2023）
TEST_START = pd.Timestamp('2020-01-01 14:30:00', tz=pytz.UTC)
TEST_END   = pd.Timestamp('2023-12-31 23:59:00', tz=pytz.UTC)

PPO_MODEL_PATH = 'ppo_final_model.zip'
TRADING_DAYS_PER_YEAR = 252


# ─────────────────────────────────────────────────────────────────
# 数据工具
# ─────────────────────────────────────────────────────────────────

def build_data_handler(universe, symbols):
    """从环境变量或项目 examples/ 目录加载 CSV，返回 BacktestDataHandler。"""
    _default_csv = os.path.join(os.path.dirname(__file__), 'examples')
    csv_dir = os.environ.get('QSTRADER_CSV_DATA_DIR', _default_csv)
    data_source = CSVDailyBarDataSource(
        csv_dir, Equity, csv_symbols=symbols, adjust_prices=False
    )
    return BacktestDataHandler(universe, data_sources=[data_source])


# ─────────────────────────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────────────────────────

def run_train():
    """调用 ppo_training.main() 训练并保存模型。"""
    print("=== [训练] 开始 PPO 训练（训练集 2010-2018，验证集 2019）===")
    train_ppo()
    print(f"=== [训练] 完成，模型已保存至 {PPO_MODEL_PATH} ===")


# ─────────────────────────────────────────────────────────────────
# 回测：三种策略
# ─────────────────────────────────────────────────────────────────

def run_ppo_backtest(data_handler, universe):
    """用训练好的 PPO 模型在测试集上做回测，返回 equity curve DataFrame。"""
    feature_handler = FeatureHandler(
        data_handler=data_handler,
        assets=ASSETS,
        lookback=20
    )
    alpha_model = PPOModel(
        ppo_model_path=PPO_MODEL_PATH,
        assets=ASSETS,
        feature_handler=feature_handler
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
    """买入持有 SPY（独立 Universe 和 DataHandler，buy_and_hold 频率）。"""
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
    """五资产等权重组合（月末再平衡）。"""
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
# 性能指标计算
# ─────────────────────────────────────────────────────────────────

def _equity_to_returns(equity_curve: pd.DataFrame) -> pd.Series:
    """从 equity curve 中提取每日百分比收益率。"""
    col = 'Equity'
    if col not in equity_curve.columns:
        col = equity_curve.columns[0]
    return equity_curve[col].pct_change().dropna()


def compute_metrics(equity_curve: pd.DataFrame, name: str) -> dict:
    """
    计算并返回常用组合绩效指标：
      - 年化收益率
      - 年化波动率
      - Sharpe Ratio（无风险利率 = 0）
      - 最大回撤
      - Calmar Ratio（年化收益 / |最大回撤|）
      - 累积收益率
    """
    col = 'Equity'
    if col not in equity_curve.columns:
        col = equity_curve.columns[0]

    equity = equity_curve[col].dropna()
    daily_ret = equity.pct_change().dropna()

    ann_ret  = daily_ret.mean() * TRADING_DAYS_PER_YEAR
    ann_vol  = daily_ret.std()  * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan

    # 最大回撤
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max
    max_dd      = drawdown.min()                        # 负数

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
    """将指标列表格式化打印为对齐表格。"""
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
# 可视化
# ─────────────────────────────────────────────────────────────────

def _normalise(equity_curve: pd.DataFrame) -> pd.Series:
    """将 equity curve 归一化到起始值 = 1，便于多策略对比。"""
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
    save_path: str = 'backtest_comparison.png'
):
    """
    绘制双面板对比图：
      上：归一化净值曲线（PPO / Buy&Hold SPY / Equal-Weight）
      下：PPO 相对于 SPY 的累积超额收益
    图旁附绩效指标表格。
    保存为 PNG 并尝试显示。
    """
    ppo_norm = _normalise(ppo_eq)
    spy_norm = _normalise(spy_eq)
    ew_norm = _normalise(ew_eq)

    # 对齐日期索引（以共有日期为准）
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

    # ── 上左：净值曲线 ───────────────────────────────────────────
    ax_main = fig.add_subplot(gs[0, 0])
    ax_main.plot(ppo_norm.index, ppo_norm.values,  label='PPO Agent',        color='#1f77b4', linewidth=1.6)
    ax_main.plot(spy_norm.index, spy_norm.values,  label='Buy & Hold SPY',   color='#ff7f0e', linewidth=1.2, linestyle='--')
    ax_main.plot(ew_norm.index,  ew_norm.values,   label='Equal-Weight (EOM)',color='#2ca02c', linewidth=1.2, linestyle=':')
    ax_main.set_title('Normalised Equity Curves (2020–2023)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Portfolio Value (start = 1)')
    ax_main.legend(fontsize=9)
    ax_main.grid(True, alpha=0.3)

    # ── 下左：超额收益 ───────────────────────────────────────────
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

    # ── 右侧：指标表格 ───────────────────────────────────────────
    ax_table = fig.add_subplot(gs[:, 1])
    ax_table.axis('off')

    if metrics:
        keys      = [k for k in metrics[0].keys() if k != 'Strategy']
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

        # 高亮表头
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
    print(f"[图表] 已保存至 {save_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# 回测主入口
# ─────────────────────────────────────────────────────────────────

def run_backtest():
    """运行全部策略，输出指标表格与对比图。"""
    print("=== [回测] 构建五资产 Universe 与数据源 ===")
    universe     = StaticUniverse(ASSETS)
    data_handler = build_data_handler(universe, SYMBOLS)

    # ── 运行三种策略 ─────────────────────────────────────────────
    print("=== [回测] 策略 1/3：PPO Agent ===")
    ppo_equity = run_ppo_backtest(data_handler, universe)

    print("=== [回测] 策略 2/3：买入持有 SPY（基准）===")
    spy_equity = run_benchmark_buyhold_spy()

    print("=== [回测] 策略 3/3：五资产等权重（月末再平衡）===")
    ew_equity  = run_benchmark_equal_weight(data_handler, universe)

    # ── 计算并打印性能指标 ───────────────────────────────────────
    metrics = [
        compute_metrics(ppo_equity, 'PPO Agent'),
        compute_metrics(spy_equity, 'Buy & Hold SPY'),
        compute_metrics(ew_equity,  'Equal-Weight'),
    ]
    print("\n========== 回测绩效对比（out-of-sample 2020–2023）==========")
    print_metrics_table(metrics)

    # ── 多策略对比图 ─────────────────────────────────────────────
    plot_comparison(ppo_equity, spy_equity, ew_equity, metrics)

    # ── QSTrader 原生 Tearsheet（PPO vs SPY）───────────────────
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
            '示例:\n'
            '  python main.py --mode train\n'
            '  python main.py --mode backtest\n'
            '  QSTRADER_CSV_DATA_DIR=/data python main.py --mode both\n'
        )
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'backtest', 'both'],
        default='both',
        help='运行模式：train / backtest / both（默认 both）'
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
