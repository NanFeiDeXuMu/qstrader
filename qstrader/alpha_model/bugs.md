Bug                                                                                                                  1. proxyAlphaModel.__init__ 调用错误
  super().__init__(signal_weights)  # 错误：object.__init__ 不接受参数
  AlphaModel 没有定义 __init__，所以 super().__init__() 实际上调用的是 object.__init__()，传入
  signal_weights 会直接报 TypeError。应改为：
  super().__init__()
  self.current_actions = signal_weights

  2. proxyAlphaModel.__call__ 返回类型错误
  return self.current_actions  # 返回 numpy array
  AlphaModel.__call__ 按照 qstrader 约定必须返回 dict[asset_str -> float]，PCM
  依赖这个格式做权重分配。应返回：
  return {asset: float(w) for asset, w in zip(self.assets, self.current_actions)}
  这意味着 proxyAlphaModel 需要持有 assets 列表。

  3. sim_engine 生成器被提前耗尽（最核心的架构问题）

  reset() 第 67 行：
  self.events = [event for event in self.backtest.sim_engine]  # 耗尽生成器
  之后 _run_one_step 第 81 行又调用：
  self.backtest.run(external_event=event)  # 内部再次 for event in self.sim_engine → 空的
  生成器已经耗尽，backtest.run() 里的循环不会执行任何事情。两种方案二选一：
  - 不预先消费 events，改为逐步驱动 sim_engine
  - 或者完全不调用 backtest.run()，自己手动处理每个 event

  4. _run_one_step 返回的 next_state 类型错误
  next_price = self.backtest.data_handler.get_asset_latest_mid_price(...)
  return next_price, reward, done  # next_price 是单个 float
  observation_space 定义为 shape=(state_dim,) 的向量，但返回的是一个标量。PPO 模型收到的 obs
  维度不匹配会直接崩溃。

  ---
  Gymnasium API 不兼容

  5. reset() 没有返回值（gymnasium >= 0.26 要求）
  # 当前：没有 return
  # 应该：return obs, info

  6. step() 返回值数量错误
  return next_state, reward, done, {}  # 旧 gym API，4个值
  # gymnasium 要求 5 个值：
  return next_state, reward, terminated, truncated, {}

  ---
  逻辑问题

  7. _run_one_step 的 while 循环逻辑
  while event.event_type != "market_open":
      self.event_idx += 1
      self.backtest.run(external_event=event)  # 处理的是旧 event，不是新的
      event = self.events[self.event_idx]
  event_idx 递增后取了新 event，但 backtest.run() 处理的还是循环开始时的旧 event。另外循环结束后
  market_open 事件本身没有被处理（没有推进 broker 状态）。

  8. done 判断时机
  done = self.event_idx >= len(self.events) - 1
  这个判断在处理完当前 event 之前就做了，可能提前一步触发终止。

  ---
  其他问题

  9. 相对导入路径
  from feature_handler import FeatureHandler  # 相对导入，在包外运行会失败
  # 应改为：
  from qstrader.alpha_model.feature_handler import FeatureHandler

  10. __main__ 块的构造参数与类定义不匹配
  env = QSTraderExecutionEnv(training_config={...}, alpha_model=ppo_model, pcm=None, ...)
  # 类的 __init__ 只接受 training_config 一个参数
  # training_config 里也缺少 'symbols' 和 'assets' 键

  ---
  最核心的问题是 第3条（生成器耗尽）和 第2条（alpha model
  返回格式），这两个会导致整个训练循环无法运行。建议先解决这两个，再处理 gymnasium API 兼容性。
