import tkinter as tk
import numpy as np

def draw_matrix(canvas, matrix, cell_size=100):
    rows, cols = matrix.shape

    min_val = 0
    max_val = 100

    for i in range(rows):
        for j in range(cols):
            box = max_val - min_val
            matrix[i,j] = float((matrix[i,j] - min_val)/box)
            if min_val == max_val:
                gray = 200
            else:
                gray = int(255*(matrix[i,j]))
            color = f"#{gray:02x}{gray:02x}{gray:02x}"

            start_x = j * cell_size
            start_y = i * cell_size
            end_x = start_x + cell_size
            end_y = start_y + cell_size
            canvas.create_rectangle(start_x,start_y,end_x,end_y,fill=color,
            outline="white")
root = tk.Tk()
root.title("卷积运行情况显示")
mart1 = np.array([[1, 1.1, 3], [4, 5, 6], [7, 8, 90]])
matrix = np.array(mart1)

canvas = tk.Canvas(root ,width=500 ,height=500)
canvas.pack(padx=10, pady=10)

draw_matrix(canvas, matrix)

root.mainloop()