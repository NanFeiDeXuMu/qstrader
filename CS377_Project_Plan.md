# CS377 Project 实现方案：基于 PPO 的强化学习量化交易系统

## 项目目标

在 QSTrader 回测框架中集成 PPO（近端策略优化）算法，训练一个自动管理投资组合权重的交易智能体，并与买入持有、等权重等基准策略比较表现。

---

## 当前问题总结

现有 PPO 模块有 10 个 bug，分为三类：

| 类别 | Bug | 影响 |
|------|-----|------|
| 致命 | `super().__init__(signal_weights)` 参数错误 | 初始化即崩溃 |
| 致命 | `proxyAlphaModel.__call__` 返回 numpy array 而非 dict | PCM 无法解析权重 |
| 致命 | `sim_engine` 生成器被预先耗尽 | 训练循环无法运行 |
| API | `reset()` 无返回值；`step()` 返回 4 个值而非 5 个 | Gymnasium ≥0.26 不兼容 |
| 逻辑 | `_run_one_step` 返回标量而非向量；while 循环处理旧 event；done 判断时机提前 | 状态维度崩溃、训练逻辑错误 |
| 其他 | 相对导入路径；`__main__` 构造参数不匹配 | 包外运行失败 |

---

## 实现方案

### 阶段四：回测评估（ppo_model.py）

运行标准 QSTrader 回测，与以下基准对比：
- 买入持有 SPY（`buy_and_hold.py` 改编）
- 等权重投资组合（`sixty_forty.py` 改编）

对比指标：年化收益、Sharpe Ratio、最大回撤、Calmar Ratio。

---

## 文件改动总览

```
qstrader/alpha_model/
├── env_setup.py       ← ✅ 已修复全部 10 个 bug
├── feature_handler.py ← ✅ 已实现真实特征提取逻辑
├── ppo_training.py    ← ✅ 已补全 config，修复导入，添加 eval callback
└── ppo_model.py       ← ✅ 已统一 __call__ 接口
```

---

## 时间估计

| 任务 | 预计工时 |
|------|---------|
| Bug 修复（env_setup.py） | ✅ 已完成 |
| FeatureHandler 实现 | ✅ 已完成 |
| 训练运行（含调参） | 视算力，4–8 h |
| 回测评估与作图 | 2–3 h |
| **合计** | **12–20 h** |
