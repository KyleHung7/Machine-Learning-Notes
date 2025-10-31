# Buffer 定義

## 什麼是 Buffer
Buffer 就是模型的「記憶張量」，用來保存狀態或統計數據，而不是模型要學習的參數。

- **不參與梯度計算** → 不會被 optimizer 更新
- **可選擇保存到模型 state_dict**（persistent=True 表示會保存）

## Buffer 常用用途
Buffer 通常用來存儲模型狀態或統計數據，例如：

- **BatchNorm**
  - `running_mean`：跑動平均的均值
  - `running_var`：跑動平均的方差
- **RNN / LSTM**
  - 中間 hidden states（隱藏狀態）
- **其他中間數據**
  - 累計計數、滑動平均值等需要保留的數據
