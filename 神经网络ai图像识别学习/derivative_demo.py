"""
求导演示 - 从零开始理解导数和梯度计算

这个文件演示导数的基本概念和计算方法，帮助理解神经网络中的反向传播算法。
导数是函数变化率的度量，在神经网络中用于计算梯度，指导参数更新。

作者：AI助手
日期：2026-06-05
"""

import numpy as np

# ==================== 1. 导数的基本概念 ====================
# 导数表示函数在某一点的变化率
# 几何意义：函数曲线在该点的切线斜率
# 物理意义：瞬时速度（位移对时间的导数）
#
# 例如：f(x) = x² 的导数是 f'(x) = 2x
# 表示当x变化时，f(x)的变化速度是2x

print("=" * 60)
print("求导演示 - 理解导数和梯度计算")
print("=" * 60)

# ==================== 2. 符号求导（使用SymPy） ====================
# SymPy是Python的符号计算库，可以进行符号求导
# 符号求导：得到导数的精确表达式

print("\n1. 符号求导（使用SymPy）")
print("-" * 40)

try:
    import sympy as sp
    
    # 定义符号变量
    x = sp.Symbol('x')
    
    # 示例1：多项式函数 f(x) = x² + 2x + 1
    f1 = x**2 + 2*x + 1
    f1_prime = sp.diff(f1, x)  # 对x求导
    print(f"函数: f(x) = {f1}")
    print(f"导数: f'(x) = {f1_prime}")
    
    # 示例2：三角函数 f(x) = sin(x)
    f2 = sp.sin(x)
    f2_prime = sp.diff(f2, x)
    print(f"\n函数: f(x) = {f2}")
    print(f"导数: f'(x) = {f2_prime}")
    
    # 示例3：指数函数 f(x) = eˣ
    f3 = sp.exp(x)
    f3_prime = sp.diff(f3, x)
    print(f"\n函数: f(x) = {f3}")
    print(f"导数: f'(x) = {f3_prime}")
    
    # 示例4：复合函数 f(x) = (x² + 1)³
    f4 = (x**2 + 1)**3
    f4_prime = sp.diff(f4, x)
    print(f"\n函数: f(x) = {f4}")
    print(f"导数: f'(x) = {f4_prime}")
    print("说明: 使用链式法则求导")
    
except ImportError:
    print("SymPy 未安装，跳过符号求导演示")
    print("要安装 SymPy: pip install sympy")

# ==================== 3. 数值求导（使用NumPy） ====================
# 数值求导：用差分近似导数
# 公式：f'(x) ≈ (f(x+h) - f(x)) / h，h是很小的数

print("\n\n2. 数值求导（使用NumPy）")
print("-" * 40)

def numerical_derivative(f, x, h=1e-5):
    """
    数值求导函数
    
    参数:
        f: 要求导的函数
        x: 求导点
        h: 微小增量（默认1e-5）
    
    返回:
        导数的近似值
    """
    return (f(x + h) - f(x)) / h

# 示例函数
def f_quadratic(x):
    """二次函数: f(x) = x² + 2x + 1"""
    return x**2 + 2*x + 1

def f_sin(x):
    """正弦函数: f(x) = sin(x)"""
    return np.sin(x)

def f_exp(x):
    """指数函数: f(x) = eˣ"""
    return np.exp(x)

# 测试数值求导
print("数值求导示例:")
x_val = 2.0  # 求导点

# 二次函数在x=2处的导数
deriv_quad = numerical_derivative(f_quadratic, x_val)
print(f"f(x) = x^2 + 2x + 1 在 x={x_val} 处的导数: {deriv_quad:.6f}")
print(f"理论值: f'(x) = 2x + 2 = {2*x_val + 2}")

# 正弦函数在x=π/4处的导数
x_val = np.pi / 4
deriv_sin = numerical_derivative(f_sin, x_val)
print(f"\nf(x) = sin(x) 在 x={x_val:.4f} 处的导数: {deriv_sin:.6f}")
print(f"理论值: f'(x) = cos(x) = {np.cos(x_val):.6f}")

# 指数函数在x=1处的导数
x_val = 1.0
deriv_exp = numerical_derivative(f_exp, x_val)
print(f"\nf(x) = e^x 在 x={x_val} 处的导数: {deriv_exp:.6f}")
print(f"理论值: f'(x) = e^x = {np.exp(x_val):.6f}")

# ==================== 4. 偏导数和梯度 ====================
# 偏导数：多元函数对某个变量的导数
# 梯度：所有偏导数组成的向量，指向函数增长最快的方向

print("\n\n3. 偏导数和梯度")
print("-" * 40)

def f_multivariable(x, y):
    """二元函数: f(x,y) = x² + y² + 2xy"""
    return x**2 + y**2 + 2*x*y

def partial_derivative(f, x, y, var='x', h=1e-5):
    """
    计算偏导数
    
    参数:
        f: 二元函数
        x, y: 求导点
        var: 对哪个变量求导 ('x' 或 'y')
        h: 微小增量
    
    返回:
        偏导数的近似值
    """
    if var == 'x':
        return (f(x + h, y) - f(x, y)) / h
    else:  # var == 'y'
        return (f(x, y + h) - f(x, y)) / h

# 计算偏导数
x_val, y_val = 1.0, 2.0

df_dx = partial_derivative(f_multivariable, x_val, y_val, 'x')
df_dy = partial_derivative(f_multivariable, x_val, y_val, 'y')

print(f"函数: f(x,y) = x^2 + y^2 + 2xy")
print(f"在点 ({x_val}, {y_val}) 处:")
print(f"df/dx = {df_dx:.6f}")
print(f"df/dy = {df_dy:.6f}")

# 梯度向量
gradient = np.array([df_dx, df_dy])
print(f"梯度向量: grad_f = [{df_dx:.6f}, {df_dy:.6f}]")

# 梯度的模（长度）
gradient_magnitude = np.linalg.norm(gradient)
print(f"梯度模长: |grad_f| = {gradient_magnitude:.6f}")

# 理论值验证
# f(x,y) = x² + y² + 2xy
# ∂f/∂x = 2x + 2y = 2*1 + 2*2 = 6
# ∂f/∂y = 2y + 2x = 2*2 + 2*1 = 6
print(f"\n理论值:")
print(f"df/dx = 2x + 2y = {2*x_val + 2*y_val}")
print(f"df/dy = 2y + 2x = {2*y_val + 2*x_val}")

# ==================== 5. 链式法则 ====================
# 链式法则：复合函数的求导法则
# 如果 y = f(g(x))，则 dy/dx = f'(g(x)) * g'(x)

print("\n\n4. 链式法则")
print("-" * 40)

def chain_rule_demo():
    """链式法导演示"""
    # 复合函数: f(x) = sin(x²)
    # 外层函数: sin(u)，内层函数: u = x²
    # 链式法则: f'(x) = cos(x²) * 2x
    
    x = sp.Symbol('x')
    
    # 复合函数
    f = sp.sin(x**2)
    
    # 用SymPy求导
    f_prime = sp.diff(f, x)
    
    print(f"复合函数: f(x) = sin(x^2)")
    print(f"链式法则求导: f'(x) = cos(x^2) * 2x")
    print(f"SymPy计算结果: f'(x) = {f_prime}")
    
    # 数值验证
    x_val = 1.0
    numerical_deriv = numerical_derivative(lambda x: np.sin(x**2), x_val)
    theoretical_deriv = np.cos(x_val**2) * 2 * x_val
    
    print(f"\n在 x = {x_val} 处:")
    print(f"数值求导: {numerical_deriv:.6f}")
    print(f"理论值: {theoretical_deriv:.6f}")

try:
    import sympy as sp
    chain_rule_demo()
except ImportError:
    print("需要SymPy库来演示链式法则")

# ==================== 6. 在神经网络中的应用 ====================
# 在神经网络中，导数用于计算梯度，指导参数更新
# 反向传播算法就是应用链式法则计算梯度

print("\n\n5. 在神经网络中的应用")
print("-" * 60)

print("""
在神经网络中，导数（梯度）的作用：

1. 损失函数对参数的导数：
   - 表示参数变化对损失的影响程度
   - 梯度方向：损失增长最快的方向
   - 梯度大小：影响程度的大小

2. 梯度下降法：
   - 参数更新公式：theta = theta - learning_rate * gradient
   - theta：模型参数
   - learning_rate：学习率（步长）
   - gradient：损失函数对参数的梯度

3. 反向传播算法：
   - 从输出层开始，逐层计算梯度
   - 使用链式法则：dL/dw = dL/dy * dy/dw
   - 高效计算所有参数的梯度

4. 激活函数的导数：
   - Sigmoid: sigmoid'(x) = sigmoid(x)(1-sigmoid(x))
   - ReLU: f'(x) = 1 if x>0, else 0
   - Tanh: f'(x) = 1 - tanh(x)^2
""")

# ==================== 7. 导数的几何意义 ====================
print("6. 导数的几何意义")
print("-" * 60)

print("""
导数的几何意义：
1. 函数曲线在某点的切线斜率
2. 正导数：函数在该点递增
3. 负导数：函数在该点递减
4. 零导数：函数在该点有极值（最大值或最小值）

在优化中的应用：
- 找到函数的最小值：沿着梯度反方向移动
- 梯度为零的点：可能是最小值、最大值或鞍点
- 学习率的选择：太小收敛慢，太大可能震荡
""")

# ==================== 8. 简单的可视化（可选） ====================
print("\n7. 可视化（可选）")
print("-" * 60)

try:
    import matplotlib.pyplot as plt
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 函数和导数
    x = np.linspace(-2, 2, 100)
    y = x**2 + 2*x + 1
    y_deriv = 2*x + 2  # 导数
    
    axes[0, 0].plot(x, y, 'b-', label='f(x) = x² + 2x + 1', linewidth=2)
    axes[0, 0].plot(x, y_deriv, 'r--', label="f'(x) = 2x + 2", linewidth=2)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('函数和它的导数')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 切线示例
    x_point = 1.0
    y_point = x_point**2 + 2*x_point + 1
    slope = 2*x_point + 2  # 该点的导数（斜率）
    
    # 切线方程: y - y_point = slope * (x - x_point)
    tangent_line = slope * (x - x_point) + y_point
    
    axes[0, 1].plot(x, y, 'b-', label='f(x) = x² + 2x + 1', linewidth=2)
    axes[0, 1].plot(x, tangent_line, 'r--', label=f'切线 (斜率={slope})', linewidth=2)
    axes[0, 1].plot(x_point, y_point, 'go', markersize=10, label=f'点({x_point}, {y_point})')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_title('导数的几何意义：切线斜率')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 梯度下降示例
    x = np.linspace(-3, 3, 100)
    y = x**2  # 简单的二次函数
    
    # 梯度下降路径
    learning_rate = 0.1
    x_current = 2.5  # 起始点
    gradient_path_x = [x_current]
    gradient_path_y = [x_current**2]
    
    for _ in range(10):
        gradient = 2 * x_current  # 导数
        x_current = x_current - learning_rate * gradient  # 更新
        gradient_path_x.append(x_current)
        gradient_path_y.append(x_current**2)
    
    axes[1, 0].plot(x, y, 'b-', label='f(x) = x²', linewidth=2)
    axes[1, 0].plot(gradient_path_x, gradient_path_y, 'ro-', markersize=8, label='梯度下降路径')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title('梯度下降法示例')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 激活函数和导数
    x = np.linspace(-5, 5, 100)
    
    # Sigmoid函数
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_deriv = sigmoid * (1 - sigmoid)  # 导数
    
    axes[1, 1].plot(x, sigmoid, 'b-', label='Sigmoid', linewidth=2)
    axes[1, 1].plot(x, sigmoid_deriv, 'r--', label="Sigmoid'", linewidth=2)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_title('Sigmoid函数和它的导数')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('derivative_demo.png', dpi=150, bbox_inches='tight')
    print("可视化已保存为 'derivative_demo.png'")
    
except ImportError:
    print("matplotlib 未安装，跳过可视化")
    print("要安装 matplotlib: pip install matplotlib")

print("\n" + "=" * 60)
print("演示完成！")
print("=" * 60)
print("\n下一步学习建议：")
print("1. 尝试修改函数，观察导数变化")
print("2. 学习多元函数的偏导数和梯度")
print("3. 理解链式法则在反向传播中的应用")
print("4. 学习梯度下降法的变体（如Adam、RMSprop等）")
print("5. 实现一个简单的神经网络，理解梯度计算")