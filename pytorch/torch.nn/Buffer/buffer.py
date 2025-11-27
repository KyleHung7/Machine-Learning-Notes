import torch
from torch.nn import Module
from torch.nn.parameter import Buffer

# ----------------------------
# 定義模型
# ----------------------------
class MyModule(Module):
    def __init__(self):
        super().__init__()  # 初始化父類別 Module，管理參數和 Buffer
        
        # ------------------------
        # 建立 Buffer 存「跑動平均均值」
        # ------------------------
        # 初始值為 [0,0,0]，代表三個特徵的均值初始為 0
        # Buffer 的特性：
        # 1. 不會參與梯度計算 (不會被 optimizer 更新)
        # 2. 會被包含在 state_dict 中，方便儲存和載入
        self.running_mean = Buffer(torch.zeros(3), persistent=True)
        
        # ------------------------
        # 建立 Buffer 存「跑動平均方差」
        # ------------------------
        # 初始值為 [1,1,1]，代表三個特徵的方差初始為 1
        self.running_var  = Buffer(torch.ones(3), persistent=True)

    def forward(self, x):
        # 前向運算：對輸入做標準化 (Normalization)
        # 標準化公式：
        # y = (x - μ) / sqrt(σ^2 + ε)
        # 其中 μ = running_mean, σ^2 = running_var
        # 加上小常數 epsilon (1e-5) 避免除以 0
        return (x - self.running_mean) / torch.sqrt(self.running_var + 1e-5)


# ----------------------------
# 使用範例
# ----------------------------
model = MyModule()

# 模擬一個 batch 的輸入 (3 個樣本，每個樣本 3 個特徵)
x = torch.tensor([[1.0, 2.0, 3.0],
                  [2.0, 3.0, 4.0],
                  [3.0, 4.0, 5.0]])

# 計算 batch 統計量（每個特徵的平均和方差）
# mean(dim=0) -> 對列取平均，得到每個特徵的均值
batch_mean = x.mean(dim=0)  # 計算步驟：
# 第 1 個特徵： (1+2+3)/3 = 2.0
# 第 2 個特徵： (2+3+4)/3 = 3.0
# 第 3 個特徵： (3+4+5)/3 = 4.0

# var(dim=0, unbiased=False) -> 對列取方差，使用 N 而非 N-1
batch_var  = x.var(dim=0, unbiased=False)  # 計算步驟：
# 第 1 個特徵： [(1-2)^2 + (2-2)^2 + (3-2)^2]/3 = (1+0+1)/3 = 0.6667
# 第 2 個特徵： [(2-3)^2 + (3-3)^2 + (4-3)^2]/3 = 0.6667
# 第 3 個特徵： [(3-4)^2 + (4-4)^2 + (5-4)^2]/3 = 0.6667

# ----------------------------
# 手動更新 Buffer（滑動平均）
# ----------------------------
momentum = 0.1  # 更新係數，控制新值對舊值的影響

# 更新 running_mean：滑動平均公式
# 新的 running_mean = (1 - momentum) * 舊值 + momentum * 當前 batch 均值
# 計算步驟：
# running_mean = (1-0.1)*[0,0,0] + 0.1*[2.0,3.0,4.0] = [0.2,0.3,0.4]
model.running_mean = (1 - momentum) * model.running_mean + momentum * batch_mean

# 更新 running_var：同理
# running_var = (1-0.1)*[1,1,1] + 0.1*[0.6667,0.6667,0.6667] 
#             = [0.9667,0.9667,0.9667]
model.running_var  = (1 - momentum) * model.running_var + momentum * batch_var

# ----------------------------
# 使用更新後的 Buffer 做前向運算
# ----------------------------
# 計算標準化結果
# y = (x - running_mean) / sqrt(running_var + 1e-5)
# 例如第 1 個特徵第一個樣本：
# (1 - 0.2)/sqrt(0.9667 + 1e-5) ≈ 0.814
output = model(x)

# ----------------------------
# 印出結果
# ----------------------------
print("更新後的 running_mean:", model.running_mean)  # [0.2, 0.3, 0.4]
print("更新後的 running_var:", model.running_var)    # [0.9667,0.9667,0.9667]
print("模型輸出:", output)
