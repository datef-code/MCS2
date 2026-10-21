日期： 2026-09-29

# day 08 笔记
# 主题：
L-BFGS 算法(Limited-memory BFGS)
two-loop recursion 双循环递归(只能算近似)

奇异 Hessian: det(H)=0 存在 0 特征值，矩阵不可逆，H-1 根本不存在(nan)
病态 Hessian: 最大特征值/最小特征值的比值(条件数)极大，代表不同方向的曲率差几个数量级(方向完全错误)；小特征值是对应损失平摊的方向(错误得到完全不收敛情况)

纯牛顿法: 必须加**阻尼修正**，


- 牛顿法与梯度下降法的核心差异

本质差异：牛顿法使用二阶泰勒展开，梯度下降法使用一阶泰勒展开。二阶展开包含了函数的曲率信息，使其方向更新更贴近真实函数，收敛更快。
收敛速度：牛顿法通常表现为二阶收敛，而梯度下降法为线性收敛，因此在迭代次数上牛顿法显著优于梯度下降法。

-  牛顿法的代价与挑战

计算代价：计算二阶导数（Hessian 矩阵）及其逆矩阵的计算量远大于梯度下降法，尤其在高维问题中，存储和计算 Hessian 矩阵的逆矩阵（求逆代价）变得不可行。
数值稳定性问题：Hessian 矩阵可能为奇异或接近奇异（条件数很大），导致矩阵不可逆，从而引发算法崩溃或更新方向错误。
解决方案：为解决上述问题，提出了拟牛顿法，通过迭代更新近似 Hessian 矩阵的逆矩阵，避免直接计算二阶导数。
代价：
1. 算力代价： $O(n^2)$，
2. 存储代价：求逆为 $O(n^3)$，
3. 奇异 Hessian：（矩阵不可逆，H-1 根本不存在(nan)），
4. 病态 Hessian：最大特征值/最小特征值的比值(条件数)极大，代表不同方向的曲率差几个数量级(方向完全错误)；小特征值是对应损失平摊的方向(错误得到完全不收敛情况)


-  拟牛顿法的代表算法

BFGS 算法：一种经典的拟牛顿法，通过迭代更新近似 Hessian 矩阵的逆矩阵，避免了直接计算二阶导数。
L-BFGS（Broyden–Fletcher–Goldfarb–Shanno） 算法：为解决 BFGS 算法在大规模问题中存储和计算开销过大的问题而提出。它只保留最近 M 次迭代的信息（参数更新增量和梯度增量），通过双循环递归（To Loop Resolution）算法来近似计算 Hessian 矩阵的逆，从而大幅减少内存占用。

-  牛顿法的数值修正

当 Hessian 矩阵为奇异或接近奇异时，必须进行阻尼修正，例如通过加入对角项来保证矩阵可逆，或限制最大步长，以确保算法的数值稳定性。


- 泰勒展开: 多项式逼近
```python
def taylor_sin(x, n):
    s = 0.0
    for k in range(n):
        s += (-1)**k * x**(2*k+1) / factorial(2*k+1)
    return s
  
x = np.linspace(-4, 4, 200)
plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), 'k-', linewidth=3, label='sin(x) 真值')
for n in [1, 3, 5, 7]:
    plt.plot(x, taylor_sin(x, n), '--', label=f' 泰勒 n={n}(取{n}项)')
plt.ylim(-2.5, 2.5)
plt.legend()
plt.grid(True)
plt.title('sin(x)的泰勒逼近: 阶数越高越准')
plt.show()
```
![[Pasted image 20260930102417.png]]
```python
def taylor_exp(x, n):
    return sum(x**k / factorial(k) for k in range(n))
  
x = np.linspace(-2, 2, 200)
plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), 'k-', linewidth=3, label='e^x 真值')
for n in[1, 2, 3, 5,8]:
    plt.plot(x, taylor_exp(x, n), '--', label=f'n={n}')

plt.ylim(-1, 8)
plt.legend()
plt.grid(True)
plt.title('e^x的麦克劳林逼近')
plt.show()
```
![[Pasted image 20260930102454.png]]
```python
def f(x): return x**2 -2
def fp(x): return 2*x
x = np.linspace(0.5, 2.5, 200)
plt.figure(figsize=(10, 6))
plt.plot(x, f(x), 'b-', linewidth=2, label='f(x)=x^2-2')
plt.axvline(0, color='gray',linewidth=1)
x_cur = 1.5
for i in range(10):
    fx, fpx = f(x_cur), fp(x_cur)
    x_tan = np.linspace(x_cur-0.7, x_cur+0.7, 50)
    plt.plot(x_tan, fx + fpx * (x_tan - x_cur), 'r--', alpha=0.7)
    plt.plot(x_cur, fx, 'go', markersize=8)
    x_next = x_cur - fx / fpx
    plt.annotate('', xy=(x_next, 0), xytext=(x_next, f(x_next)), arrowprops=dict(arrowstyle='->', color='green'))
    x_cur = x_next

plt.plot(2**0.5, 0, 'k*', markersize=15, label='真根✓2')
plt.ylim(-2.5, 3)
plt.legend()
plt.grid(True)
plt.title('牛顿法求根')
plt.show()
```
![[Pasted image 20260930102627.png]]