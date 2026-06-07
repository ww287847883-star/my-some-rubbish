import tkinter as tk
import numpy as np

def draw_color_matrix(canvas, matrix, cell_size=50):
    """绘制带颜色的矩阵"""
    rows, cols = matrix.shape
    
    # 找到最大最小值用于归一化
    min_val = matrix.min()
    max_val = matrix.max()
    
    for i in range(rows):
        for j in range(cols):
              # 归一化计算灰度值
            if max_val == min_val:
                gray = 200  # 默认中等灰度
            else:
                gray = int(100 + (matrix[i, j] - min_val) / (max_val - min_val) * 155)  # 100-255范围
            
            # 灰度颜色
            color = f"#{gray:02x}{gray:02x}{gray:02x}"
            
            # 绘制矩形
            x1 = j * cell_size
            y1 = i * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="white")
            

# 示例
root = tk.Tk()
root.title("卷积核变化面板")
mart1 = np.array([[1, 1.1, 3], [4, 5, 6], [7, 8, 9]])
matrix = np.array(mart1)

canvas = tk.Canvas(root, width=300, height=300)
canvas.pack(padx=100, pady=10)

draw_color_matrix(canvas, matrix)

root.mainloop()