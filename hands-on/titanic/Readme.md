# 🚢 Titanic 生存預測 — 神經網路模型 (titanic_nn.py / titanic_nn_improved.py)

[toc]

本專案使用經典 **Kaggle Titanic 資料集**，建立並比較兩個版本的神經網路模型（**Baseline** 與 **Improved**），用以預測乘客是否存活。

專案重點在於**資料前處理**、**模型架構設計**、**性能比較**與**模型改進方法**。

---

## 📁 1. 專案內容

本專案包含兩份程式：

* **`titanic_nn.py`**
    * ➜ **基礎版神經網路模型（Baseline）**
* **`titanic_nn_improved.py`**
    * ➜ **加強版神經網路模型（Improved）**

兩者皆使用相同資料，但 Improved 版加入**更完整的資料清理、正規化、模型調整與正則化**，因此預期能得到更高的準確率。

---

## 📦 2. 檔案說明

### 🔹 `titanic_nn.py` — Baseline（基礎版）

內容包含：

* 基本資料處理
* 缺失值填補（`Age`、`Embarked`）
* One-Hot Encoding（分類變數）
* 移除不必要欄位（如 `PassengerId`）
* 簡單的**多層感知器 (MLP)** 架構
* 固定 `random_state=777` 的 **80/20 訓練/驗證切分**
* 輸出訓練與驗證準確率

> 💡 此版本作為比對的**基準模型**。

### 🔹 `titanic_nn_improved.py` — Improved（加強版）

相較於 baseline，改善內容包括：

* **更完善的資料前處理**
    * 額外刪除高缺失欄位（`Cabin`、`Ticket`）
    * 數值欄位**標準化**（`StandardScaler`）
* **更深或更多神經元**的網路架構
* 加入 **Dropout** 防止過度擬合
* 加入 **Batch Normalization**
* 加入 **Early Stopping**
* 更佳的模型泛化能力

> 🚀 通常可達到 **更高的 Validation Accuracy（80%~90%）**。

---

## 🧹 3. 資料前處理流程

兩份程式都會進行以下步驟（Improved 版更完整）：

1.  讀取 `train.csv`
2.  移除不必要欄位（例如 `PassengerId`）
3.  處理缺失值
    * `Age` → 中位數填補
    * `Embarked` → 眾數填補
4.  類別欄位編碼（One-Hot Encoding）
5.  **欄位縮放（Improved 版特有）**
    * 使用 `StandardScaler`
6.  **切分資料：**
    * 訓練集：80%
    * 驗證集：20%
    * `random_state=777`（固定隨機性）

---

## 🧠 4. 模型架構

### 🔹 Baseline (`titanic_nn.py`)

* 全連接層 (Dense)
* ReLU 啟動函數
* Sigmoid 輸出層（二元分類）
* Loss：`binary_crossentropy`
* Optimizer：`Adam`
* 簡單的 **2～3 層架構**

### 🔹 Improved (`titanic_nn_improved.py`)

* **更多神經元與更深層數**
* 使用 **Dropout** 減少 overfitting
* **Batch Normalization** 加速收斂
* **EarlyStopping** 防止訓練過久
* 更好的泛化能力與驗證效能