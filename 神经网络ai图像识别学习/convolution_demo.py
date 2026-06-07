"""
卷积操作演示 - GUI版本

这个文件演示最基础的2D卷积操作，使用GUI界面展示结果。
卷积是神经网络中提取特征的基本操作。

作者：AI助手
日期：2026-06-05
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# ==================== 1. 卷积函数 ====================
def conv2d(input_matrix, kernel, stride=1, padding=0):
    """
    实现简单的2D卷积操作
    
    参数:
        input_matrix: 输入矩阵（如图像像素值）
        kernel: 卷积核（滤波器）
        stride: 步长（卷积核每次滑动的像素数，默认为1）
        padding: 填充（在输入边缘添加的0的层数，默认为0）
    
    返回:
        output: 卷积后的特征图
    """
    # 获取输入和卷积核的尺寸
    input_h, input_w = input_matrix.shape
    kernel_h, kernel_w = kernel.shape
    
    # 如果有填充，在输入边缘添加0
    if padding > 0:
        padded_input = np.zeros((input_h + 2*padding, input_w + 2*padding))
        padded_input[padding:padding+input_h, padding:padding+input_w] = input_matrix
        input_matrix = padded_input
        input_h, input_w = input_matrix.shape
    
    # 计算输出尺寸
    output_h = (input_h - kernel_h) // stride + 1
    output_w = (input_w - kernel_w) // stride + 1
    
    # 初始化输出矩阵
    output = np.zeros((output_h, output_w))
    
    # 执行卷积操作
    for i in range(output_h):
        for j in range(output_w):
            receptive_field = input_matrix[i*stride:i*stride+kernel_h, 
                                          j*stride:j*stride+kernel_w]
            output[i, j] = np.sum(receptive_field * kernel)
    
    return output

# ==================== 2. GUI应用 ====================
class ConvolutionDemoApp:
    """卷积演示GUI应用"""
    
    def __init__(self, root):
        """初始化GUI"""
        self.root = root
        self.root.title("卷积操作演示")
        self.root.geometry("1000x700")
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(2, weight=1)
        
        # 创建输入区域
        self.create_input_section()
        
        # 创建控制区域
        self.create_control_section()
        
        # 创建显示区域
        self.create_display_section()
        
        # 创建状态栏
        self.create_status_bar()
        
        # 初始化数据
        self.init_data()
        
        # 初始显示
        self.update_display()
    
    def create_input_section(self):
        """创建输入区域"""
        # 输入矩阵框架
        input_frame = ttk.LabelFrame(self.main_frame, text="输入矩阵 (5x5)", padding="5")
        input_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 输入矩阵文本框
        self.input_text = tk.Text(input_frame, height=5, width=30, font=('Courier', 10))
        self.input_text.grid(row=0, column=0, padx=(0, 10))
        
        # 卷积核框架
        kernel_frame = ttk.LabelFrame(self.main_frame, text="卷积核 (3x3)", padding="5")
        kernel_frame.grid(row=0, column=2, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 卷积核文本框
        self.kernel_text = tk.Text(kernel_frame, height=3, width=20, font=('Courier', 10))
        self.kernel_text.grid(row=0, column=0, padx=(0, 10))
        
        # 预设卷积核选择
        preset_frame = ttk.Frame(kernel_frame)
        preset_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Label(preset_frame, text="预设卷积核:").grid(row=0, column=0, padx=(0, 5))
        
        self.preset_var = tk.StringVar(value="水平边缘检测")
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var, 
                                   values=["水平边缘检测", "垂直边缘检测", "锐化", "模糊"], 
                                   state="readonly", width=15)
        preset_combo.grid(row=0, column=1)
        preset_combo.bind("<<ComboboxSelected>>", self.on_preset_change)
    
    def create_control_section(self):
        """创建控制区域"""
        control_frame = ttk.LabelFrame(self.main_frame, text="卷积参数", padding="5")
        control_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 步长控制
        ttk.Label(control_frame, text="步长:").grid(row=0, column=0, padx=(0, 5))
        self.stride_var = tk.IntVar(value=1)
        stride_spin = ttk.Spinbox(control_frame, from_=1, to=3, textvariable=self.stride_var, width=5)
        stride_spin.grid(row=0, column=1, padx=(0, 20))
        
        # 填充控制
        ttk.Label(control_frame, text="填充:").grid(row=0, column=2, padx=(0, 5))
        self.padding_var = tk.IntVar(value=0)
        padding_spin = ttk.Spinbox(control_frame, from_=0, to=2, textvariable=self.padding_var, width=5)
        padding_spin.grid(row=0, column=3, padx=(0, 20))
        
        # 运行按钮
        run_button = ttk.Button(control_frame, text="运行卷积", command=self.run_convolution)
        run_button.grid(row=0, column=4, padx=(20, 0))
        
        # 重置按钮
        reset_button = ttk.Button(control_frame, text="重置数据", command=self.reset_data)
        reset_button.grid(row=0, column=5, padx=(10, 0))
    
    def create_display_section(self):
        """创建显示区域"""
        # 创建matplotlib图形
        self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 4))
        self.fig.tight_layout(pad=3.0)
        
        # 将matplotlib图形嵌入tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 结果文本框
        result_frame = ttk.LabelFrame(self.main_frame, text="卷积结果", padding="5")
        result_frame.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.result_text = tk.Text(result_frame, height=4, width=80, font=('Courier', 10))
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text.configure(yscrollcommand=scrollbar.set)
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def init_data(self):
        """初始化数据"""
        # 默认输入矩阵
        self.input_matrix = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25]
        ], dtype=np.float32)
        
        # 默认卷积核（水平边缘检测）
        self.kernel = np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ], dtype=np.float32)
        
        # 更新文本框
        self.update_text_boxes()
    
    def update_text_boxes(self):
        """更新文本框内容"""
        # 更新输入矩阵文本框
        self.input_text.delete(1.0, tk.END)
        for row in self.input_matrix:
            self.input_text.insert(tk.END, " ".join([f"{x:6.1f}" for x in row]) + "\n")
        
        # 更新卷积核文本框
        self.kernel_text.delete(1.0, tk.END)
        for row in self.kernel:
            self.kernel_text.insert(tk.END, " ".join([f"{x:6.1f}" for x in row]) + "\n")
    
    def on_preset_change(self, event):
        """预设卷积核改变事件"""
        preset = self.preset_var.get()
        
        if preset == "水平边缘检测":
            self.kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        elif preset == "垂直边缘检测":
            self.kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        elif preset == "锐化":
            self.kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        elif preset == "模糊":
            self.kernel = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]], dtype=np.float32)
        
        self.update_text_boxes()
        self.status_var.set(f"已选择预设卷积核: {preset}")
    
    def run_convolution(self):
        """运行卷积操作"""
        try:
            # 获取参数
            stride = self.stride_var.get()
            padding = self.padding_var.get()
            
            # 验证参数
            if stride < 1:
                messagebox.showerror("错误", "步长必须大于0")
                return
            if padding < 0:
                messagebox.showerror("错误", "填充不能为负数")
                return
            
            # 执行卷积
            self.output = conv2d(self.input_matrix, self.kernel, stride, padding)
            
            # 更新显示
            self.update_display()
            
            # 更新状态栏
            self.status_var.set(f"卷积完成 - 输出尺寸: {self.output.shape}")
            
        except Exception as e:
            messagebox.showerror("错误", f"卷积操作失败: {str(e)}")
            self.status_var.set("卷积操作失败")
    
    def update_display(self):
        """更新显示"""
        # 清除旧图形
        for ax in self.axes:
            ax.clear()
        
        # 显示输入矩阵
        self.axes[0].imshow(self.input_matrix, cmap='gray', interpolation='nearest')
        self.axes[0].set_title('输入矩阵')
        self.axes[0].set_xlabel('宽度')
        self.axes[0].set_ylabel('高度')
        
        # 显示卷积核
        self.axes[1].imshow(self.kernel, cmap='RdBu', interpolation='nearest')
        self.axes[1].set_title('卷积核')
        self.axes[1].set_xlabel('宽度')
        self.axes[1].set_ylabel('高度')
        
        # 显示输出（如果有）
        if hasattr(self, 'output'):
            self.axes[2].imshow(self.output, cmap='gray', interpolation='nearest')
            self.axes[2].set_title('卷积结果')
            self.axes[2].set_xlabel('宽度')
            self.axes[2].set_ylabel('高度')
            
            # 更新结果文本框
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"输出尺寸: {self.output.shape}\n")
            self.result_text.insert(tk.END, "输出矩阵:\n")
            for row in self.output:
                self.result_text.insert(tk.END, " ".join([f"{x:8.2f}" for x in row]) + "\n")
        else:
            self.axes[2].text(0.5, 0.5, '点击"运行卷积"\n查看结果', 
                            ha='center', va='center', transform=self.axes[2].transAxes)
            self.axes[2].set_title('卷积结果')
        
        # 刷新画布
        self.canvas.draw()
    
    def reset_data(self):
        """重置数据"""
        self.init_data()
        if hasattr(self, 'output'):
            delattr(self, 'output')
        self.update_display()
        self.status_var.set("数据已重置")

# ==================== 3. 主程序 ====================
if __name__ == "__main__":
    try:
        # 创建主窗口
        root = tk.Tk()
        
        # 创建应用
        app = ConvolutionDemoApp(root)
        
        # 运行主循环
        root.mainloop()
        
    except Exception as e:
        print(f"程序发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("程序运行结束。")