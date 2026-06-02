"""
main.py — CS377 Project 主程序

三个模式（通过命令行 --mode 切换）：
  train    训练 PPO 智能体
  backtest 用训练好的模型做 out-of-sample 回测，并与基准对比
  both     顺序执行 train 然后 backtest
"""

import argparse
import os

import pandas as pd
import pytz

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

TEST_START = pd.Timestamp('2020-01-01 14:30:00', tz=pytz.UTC)
TEST_END   = pd.Timestamp('2023-12-31 23:59:00', tz=pytz.UTC)

PPO_MODEL_PATH = 'ppo_final_model.zip'


# ─────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────

def build_data_handler(universe, symbols):
    """从环境变量或当前目录加载 CSV 数据，返回 BacktestDataHandler。"""
    csv_dir = os.environ.get('QSTRADER_CSV_DATA_DIR', '.')
    data_source = CSVDailyBarDataSource(
        csv_dir, Equity, csv_symbols=symbols, adjust_prices=False
    )
    return BacktestDataHandler(universe, data_sources=[data_source])


# ─────────────────────────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────────────────────────

def run_train():
    """调用 ppo_training.main() 训练并保存模型。"""
    print("=== [训练] 开始 PPO 训练 ===")
    train_ppo()
    print(f"=== [训练] 完成，模型已保存至 {PPO_MODEL_PATH} ===")


# ─────────────────────────────────────────────────────────────────
# 回测：PPO 策略
# ─────────────────────────────────────────────────────────────────

def run_ppo_backtest(data_handler, universe):
    """用训练好的 PPO 模型在测试集上做回测，返回 equity curve。"""
    # TODO: FeatureHandler 需要 data_handler，在回测阶段从 backtest 对象获取后注入
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


# ─────────────────────────────────────────────────────────────────
# 回测：基准策略
# ─────────────────────────────────────────────────────────────────

def run_benchmark_buyhold(data_handler):
    """买入持有 SPY 基准，返回 equity curve。"""
    universe = StaticUniverse(['EQ:SPY'])
    alpha_model = FixedSignalsAlphaModel({'EQ:SPY': 1.0})
    backtest = BacktestTradingSession(
        start_dt=TEST_START,
        end_dt=TEST_END,
        universe=universe,
        alpha_model=alpha_model,
        rebalance='buy_and_hold',
        long_only=True,
        cash_buffer_percentage=0.01,
        data_handler=data_handler
    )
    backtest.run()
    return backtest.get_equity_curve()


def run_benchmark_equal_weight(data_handler, universe):
    """等权重投资组合基准，返回 equity curve。"""
    equal_weights = {asset: 1.0 / len(ASSETS) for asset in ASSETS}
    alpha_model = FixedSignalsAlphaModel(equal_weights)
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
# 回测入口
# ─────────────────────────────────────────────────────────────────

def run_backtest():
    """运行全部策略并输出 tearsheet 对比图。"""
    print("=== [回测] 构建数据和 Universe ===")
    universe = StaticUniverse(ASSETS)
    data_handler = build_data_handler(universe, SYMBOLS)

    print("=== [回测] PPO 策略 ===")
    ppo_equity = run_ppo_backtest(data_handler, universe)

    print("=== [回测] 基准：买入持有 SPY ===")
    spy_equity = run_benchmark_buyhold(data_handler)

    print("=== [回测] 基准：等权重组合 ===")
    eq_equity = run_benchmark_equal_weight(data_handler, universe)

    # ── 输出结果 ─────────────────────────────────────────────────
    # TODO: TearsheetStatistics 目前只支持一条基准曲线；
    #       后续可扩展为多曲线对比图
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
    parser = argparse.ArgumentParser(description='CS377 PPO Trading Project')
    parser.add_argument(
        '--mode',
        choices=['train', 'backtest', 'both'],
        default='both',
        help='运行模式：train / backtest / both（默认）'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'train':
        run_train()
    elif args.mode == 'backtest':
        run_backtest()
    elif args.mode == 'both':
        run_train()
        run_backtest()
