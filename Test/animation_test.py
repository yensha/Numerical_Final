import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()       # 建立單一圖表
ax.set_xlim(0,20)              # x 座標範圍設定 0～20
ax.set_ylim(-1.5,1.5)          # y 座標範圍設定 -1.5～1.5

n = [i/5 for i in range(100)]  # 使用串列升成式產生 0～20 共 100 筆資料
x, y = [], []                  # 設定 x 和 y 變數為空串列
line, = ax.plot(x, y)          # 定義 line 變數為折線圖物件 ( 注意 line 後方有逗號 )

def run(data):
    x.append(data)             # 添加 x 資料點
    y.append(math.sin(data))   # 添加 y 資料點
    line.set_data(x, y)        # 重新設定資料點

ani = animation.FuncAnimation(fig, run, frames=n, interval=30)
ani.save('animation.gif', fps=30)
plt.show()