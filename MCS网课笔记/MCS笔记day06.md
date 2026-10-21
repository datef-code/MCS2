日期： 2026-09-27

# day 06  笔记
# 主题：梯度下降


![[Pasted image 20260927212545.png]]

![[Pasted image 20260927212702.png]]

损失函数 MSE：
$$L(w,b）=\frac{1}{n}\sum_i(w\cdot x_i+b-y_i)^2$$

对 w 和 b 求偏导（这就是梯度！）：
$$\frac{\partial L}{\partial w} = \frac{2}{n}\sum_i x_i(wx_i+b-y_i)$$
 $$\frac{\partial L}{\partial b} = \frac{2}{n}\sum_i(wx_i+b-y_i)$$
## sklearn 内部数据集 diabetes（糖尿病数据集）
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
  
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
print('success')
```
### 加载 diabetes 数据集(糖尿病数据集)
- **diabetes 数据集是 sklearn 内置的经典回归数据集**
- **样本数** : 442
- **特征数** : 10 \['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'] (s1-s6是6个血清测量值)
- **目标** : 一年后疾病进展的量化指标(连续值适合做回归)
- **关键优点** : 特征已经标准化(均值 0, 方差 1), 对于梯度下降非常重要——否则不同量纲的特征会让 loss 曲面变成"狭长峡谷", 梯度下降会震荡, 难以收敛;
- **命名约定** : 互不干扰
```python
data = load_diabetes()
X, y = data['data'], data['target']
```
### 一元线性回归
```python
true_w, true_b = 2.5, 1.0
xs = np.linspace(0, 4, 50)
ys = true_w * xs + true_b + np.random.randn(50)*0.8

plt.scatter(x=xs, y=ys, s=20, alpha=0.7, label='数据')
plt.plot(xs, true_w * xs + true_b, 'r--', label=f'真直线 w={true_w}, b={true_b}')
plt.legend()
plt.grid(visible=True)
plt.title(label='合成数据')
plt.show()
```
### 手写梯度下降

```python
def gradient_descent_1d(xs, ys, lr=0.1, epochs=50): # epochs 迭代次数
    w, b = 0.0, 0.0
    hist = [(w, b)]
    losses = []     # 损失值(均方误差)
    n = len(xs)
    for _ in range(epochs):
        pred = w * xs + b   # 预测值
        err = pred - ys     # ys: 真实标签; err: 残差
        dw = (2/n)*np.sum(xs*err)   # 计算梯度: dw(L(w, b))
        db = (2/n)*np.sum(err)      # 计算梯度: db(L(w, b))
        w -= lr * dw    # lr 学习率; w: 反向更新
        b -= lr * db    # b: 反向更新  
        hist.append((w, b))
        losses.append(np.mean(err**2))
    return w, b, np.array(hist), np.array(losses)  

w1, b1, hist, losses1 = gradient_descent_1d(xs, ys, lr=0.1, epochs=50)
```

```python
W = np.linspace(-1, 5, 80)
B = np.linspace(-2, 4, 80)
Wg, Bg = np.meshgrid(W, B)
Lg = np.mean((Wg[: , : , None]*xs[None, None, : ]+ Bg[: , : , None] -ys[None, None, : ])**2, axis=2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.contour(Wg, Bg, Lg, 30, cmap='viridis')
ax.plot(hist[: , 0], hist[: , 1], 'r.-', markersize=4, label='梯度下降路径')
ax.plot(true_w, true_b, 'g*', markersize=15,  label='真值')
ax.set_xlabel('W')
ax.set_ylabel('B')
ax.set_title('loss等高线与梯度下降路径')
ax.legend()
ax.grid(True)
axes[1].plot(losses1, 'b.-')
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('loss(MSE)')
axes[1].set_title('loss下降曲线')
axes[1].grid(True)
plt.tight_layout()
plt.show()
```

### 多元线性回归
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def gradient_descent_multi(X, y, lr=0.5, epochs=500): # epochs 迭代次数
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = []     # 损失值(均方误差)
    for _ in range(epochs):
        pred = X @ w + b   # 预测值
        err = pred - y     # ys: 真实标签; err: 残差
        dw = (2/n)*np.sum(X.T @ err)   # 计算梯度: dw(L(w, b))
        db = (2/n)*np.sum(err)      # 计算梯度: db(L(w, b))
        w -= lr * dw    # lr 学习率; w: 反向更新
        b -= lr * db    # b: 反向更新
        losses.append(np.mean(err**2))
    return w, b, np.array(losses)

w, b, losses2 = gradient_descent_multi(X, y, lr=0.5, epochs=500)
pred_test = X_test @ w + b
test_mse = mean_squared_error(y_test, pred_test)
```

```python
plt.figure(figsize=(8, 4))
plt.plot(losses2, 'b-')
plt.xlabel('epoch')
plt.ylabel('loss(MSE)')
plt.title('diabetes 多元线性回归: 手写梯度下降 loss 曲线')
plt.grid(True)
plt.show()
```

### sklearn 基准
```python
sk = LinearRegression()
sk.fit(X_train, y_train)
sk_pred = sk.predict(X_test)
sk_mse = mean_squared_error(y_test, sk_pred)
```

```python
_, _, losses_big = gradient_descent_multi(X_train, y_train, lr=1.2, epochs=100)
print(f'lr=1.2 时 loss 序列前10: {losses_big[:10].round(2)}')
print(f'lr=1.2 时 最终loss序列: {losses_big[-1]:.2f}(若为nan/inf 说明发散)')
  
plt.figure(figsize=(8, 4))
plt.plot(losses2, 'b-', label='lr=0.5(正常)')
plt.plot(losses_big, 'r-', label='lr=1.2(发散)')
plt.ylim(0, min(5000, max(losses2[-1]*3, 1)))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('学习率的影响')
plt.legend()
plt.grid(True)
plt.show()
```