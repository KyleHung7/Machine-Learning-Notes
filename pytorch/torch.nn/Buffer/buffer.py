import torch
from torch.nn import Module
from torch.nn.parameter import Buffer

# ----------------------------
# 定義模型
# ----------------------------
class MyModule(Module):
    def __init__(self):
        super().__init__()  # 初始化父類別 Module，管理參數和 Buffer
        
        # 建立 Buffer 存「跑動平均均值」
        # 初始值為 [0,0,0]，不會參與梯度計算，也不會被 optimizer 更新
        # Buffer 會被保存到 state_dict，可隨模型儲存或載入
        self.running_mean = Buffer(torch.zeros(3), persistent=True)
        
        # 建立 Buffer 存「跑動平均方差」
        # 初始值為 [1,1,1]，同樣不參與梯度計算
        self.running_var  = Buffer(torch.ones(3), persistent=True)

    def forward(self, x):
        # 前向運算：對輸入做標準化
        # 公式：(x - running_mean) / sqrt(running_var + epsilon)
        # 1e-5 避免除以 0
        return (x - self.running_mean) / torch.sqrt(self.running_var + 1e-5)


# ----------------------------
# 使用範例
# ----------------------------
model = MyModule()

# 模擬一個 batch 的輸入
x = torch.tensor([[1.0, 2.0, 3.0],
                  [2.0, 3.0, 4.0],
                  [3.0, 4.0, 5.0]])

# 計算 batch 統計量（每個特徵的平均和方差）
batch_mean = x.mean(dim=0)          # [2.0, 3.0, 4.0]
batch_var  = x.var(dim=0, unbiased=False)  # [0.6667, 0.6667, 0.6667]

# ----------------------------
# 手動更新 Buffer（滑動平均）
# ----------------------------
momentum = 0.1

# 更新 running_mean：使用滑動平均公式
# (1 - momentum) * 舊值 + momentum * 當前 batch 均值
model.running_mean = (1 - momentum) * model.running_mean + momentum * batch_mean

# 更新 running_var：同理
model.running_var  = (1 - momentum) * model.running_var + momentum * batch_var

# ----------------------------
# 使用更新後的 Buffer 做前向運算
# ----------------------------
output = model(x)

# ----------------------------
# 印出結果
# ----------------------------
print("更新後的 running_mean:", model.running_mean)
print("更新後的 running_var:", model.running_var)
print("模型輸出:", output)
