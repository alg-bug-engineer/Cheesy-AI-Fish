import turtle
import time

# 设置屏幕
wn = turtle.Screen()
wn.title("大灰狼追小红帽")
wn.bgcolor("white")
wn.setup(width=600, height=700)  # 增加高度以容纳输入框和历史命令
wn.tracer(0)  # 关闭自动屏幕更新

# 游戏区域大小
GAME_HEIGHT = 500
GAME_WIDTH = 600

# 网格设置
GRID_SIZE = 40  # 每个格子的像素大小
GRID_WIDTH = 11  # 网格宽度（格子数）
GRID_HEIGHT = 11  # 网格高度（格子数）

# 创建大灰狼
wolf = turtle.Turtle()
wolf.shape("triangle")
wolf.color("gray")
wolf.penup()
wolf.speed(0)
wolf.goto(-GRID_SIZE * 2, 0)  # 初始位置

# 创建小红帽
red_hood = turtle.Turtle()
red_hood.shape("circle")
red_hood.color("red")
red_hood.penup()
red_hood.speed(0)
red_hood.goto(GRID_SIZE * 3, 0)  # 初始位置，与大灰狼相距5个格子

# 创建网格绘制工具
grid_pen = turtle.Turtle()
grid_pen.hideturtle()
grid_pen.penup()
grid_pen.speed(0)
grid_pen.color("lightgray")

# 创建界面分隔线
separator = turtle.Turtle()
separator.hideturtle()
separator.penup()
separator.speed(0)
separator.color("black")
separator.pensize(2)

# 创建命令历史显示工具
history_pen = turtle.Turtle()
history_pen.hideturtle()
history_pen.penup()
history_pen.speed(0)
history_pen.color("blue")

# 创建输入提示和输入框
input_prompt = turtle.Turtle()
input_prompt.hideturtle()
input_prompt.penup()
input_prompt.speed(0)
input_prompt.color("black")

# 用于输入框光标闪烁的turtle
cursor = turtle.Turtle()
cursor.hideturtle()
cursor.penup()
cursor.speed(0)
cursor.color("black")

# 存储命令历史
command_history = []
max_history_display = 5  # 最多显示的历史命令数

# 游戏状态
game_active = True
last_update_time = time.time()
current_input = ""
cursor_visible = True
last_cursor_time = time.time()

# 绘制游戏界面框架
def draw_interface():
    # 绘制分隔线
    separator.clear()
    separator.penup()
    separator.goto(-300, -GAME_HEIGHT/2)
    separator.pendown()
    separator.goto(300, -GAME_HEIGHT/2)
    
    # 绘制输入框边框
    separator.penup()
    separator.goto(-280, -GAME_HEIGHT/2 - 80)
    separator.pendown()
    for _ in range(2):
        separator.forward(560)
        separator.right(90)
        separator.forward(30)
        separator.right(90)

# 绘制网格函数
def draw_grid():
    grid_pen.clear()
    
    # 设置网格边界
    left = -GRID_WIDTH * GRID_SIZE // 2
    right = GRID_WIDTH * GRID_SIZE // 2
    top = GAME_HEIGHT // 2 - 50
    bottom = top - GRID_HEIGHT * GRID_SIZE
    
    # 绘制垂直线
    for x in range(-GRID_WIDTH // 2, GRID_WIDTH // 2 + 1):
        grid_pen.penup()
        grid_pen.goto(x * GRID_SIZE, bottom)
        grid_pen.pendown()
        grid_pen.goto(x * GRID_SIZE, top)
    
    # 绘制水平线
    for y in range(GRID_HEIGHT + 1):
        y_pos = top - y * GRID_SIZE
        grid_pen.penup()
        grid_pen.goto(left, y_pos)
        grid_pen.pendown()
        grid_pen.goto(right, y_pos)

# 像素坐标转网格坐标
def pixel_to_grid(x, y):
    top = GAME_HEIGHT // 2 - 50
    grid_x = round(x / GRID_SIZE)
    grid_y = round((top - y) / GRID_SIZE)
    return (grid_x, grid_y)

# 网格坐标转像素坐标
def grid_to_pixel(grid_x, grid_y):
    top = GAME_HEIGHT // 2 - 50
    x = grid_x * GRID_SIZE
    y = top - grid_y * GRID_SIZE
    return (x, y)

# 显示命令历史
def display_history():
    history_pen.clear()
    
    # 历史命令区域位置
    history_top = -GAME_HEIGHT/2 - 10
    
    # 显示最新的几条命令
    display_commands = command_history[-max_history_display:] if len(command_history) > max_history_display else command_history
    
    for i, cmd in enumerate(display_commands):
        history_pen.goto(-270, history_top - i * 20)
        history_pen.write(f"命令 {len(command_history) - len(display_commands) + i + 1}: {cmd}", font=("Arial", 10, "normal"))

# 显示输入框
def display_input_box():
    input_prompt.clear()
    input_y = -GAME_HEIGHT/2 - 95
    input_prompt.goto(-270, input_y)
    input_prompt.write(f"> {current_input}", font=("Arial", 12, "normal"))
    
    # 更新光标位置
    cursor_x = -270 + 10 + len(current_input) * 10  # 估算的字符宽度
    cursor.clear()
    if cursor_visible:
        cursor.goto(cursor_x, input_y)
        cursor.write("|", font=("Arial", 12, "normal"))

# 处理用户输入
def process_input(key):
    global current_input, command_history
    
    if not game_active:
        return
    
    if key == "Return":
        if current_input in ["上", "下", "左", "右"]:
            # 添加到历史记录
            command_history.append(current_input)
            # 移动小红帽
            move_red_hood(current_input)
            # 清空输入
            current_input = ""
        else:
            current_input = ""  # 清空无效输入
    elif key == "BackSpace":
        if current_input:
            current_input = current_input[:-1]
    elif key == "Escape":
        current_input = ""  # 清空输入
    elif key in ["上", "下", "左", "右"]:
        current_input = key
    elif key in ["w", "a", "s", "d"]:
        # 支持WASD按键
        key_map = {"w": "上", "s": "下", "a": "左", "d": "右"}
        current_input = key_map[key]
    
    display_input_box()
    display_history()

# 更新光标闪烁
def update_cursor():
    global cursor_visible, last_cursor_time
    
    current_time = time.time()
    if current_time - last_cursor_time >= 0.5:  # 每0.5秒闪烁一次
        cursor_visible = not cursor_visible
        last_cursor_time = current_time
        display_input_box()

# 小红帽的移动函数
def move_red_hood(direction):
    x, y = pixel_to_grid(red_hood.xcor(), red_hood.ycor())
    moved = False
    
    # 根据方向移动
    if direction == "上":
        if y > 0:
            new_x, new_y = grid_to_pixel(x, y - 1)
            red_hood.goto(new_x, new_y)
            moved = True
    elif direction == "下":
        if y < GRID_HEIGHT - 1:
            new_x, new_y = grid_to_pixel(x, y + 1)
            red_hood.goto(new_x, new_y)
            moved = True
    elif direction == "左":
        if x > -GRID_WIDTH // 2:
            new_x, new_y = grid_to_pixel(x - 1, y)
            red_hood.goto(new_x, new_y)
            moved = True
    elif direction == "右":
        if x < GRID_WIDTH // 2:
            new_x, new_y = grid_to_pixel(x + 1, y)
            red_hood.goto(new_x, new_y)
            moved = True

# 绑定键盘事件
def setup_keyboard():
    wn.onkeypress(lambda: process_input("上"), "Up")
    wn.onkeypress(lambda: process_input("下"), "Down")
    wn.onkeypress(lambda: process_input("左"), "Left")
    wn.onkeypress(lambda: process_input("右"), "Right")
    wn.onkeypress(lambda: process_input("w"), "w")
    wn.onkeypress(lambda: process_input("s"), "s")
    wn.onkeypress(lambda: process_input("a"), "a")
    wn.onkeypress(lambda: process_input("d"), "d")
    wn.onkeypress(lambda: process_input("Return"), "Return")
    wn.onkeypress(lambda: process_input("BackSpace"), "BackSpace")
    wn.onkeypress(lambda: process_input("Escape"), "Escape")
    wn.listen()

# 大灰狼AI - 沿最优路径追小红帽
def wolf_move():
    # 获取当前网格坐标
    wolf_x, wolf_y = pixel_to_grid(wolf.xcor(), wolf.ycor())
    red_hood_x, red_hood_y = pixel_to_grid(red_hood.xcor(), red_hood.ycor())
    
    # 计算距离差
    dx = red_hood_x - wolf_x
    dy = red_hood_y - wolf_y
    
    # 沿最优路径移动一步（曼哈顿距离）
    if dx != 0:
        # 优先水平移动
        wolf_x += 1 if dx > 0 else -1
    elif dy != 0:
        # 若水平距离为零，则垂直移动
        wolf_y += 1 if dy > 0 else -1
    
    # 更新大灰狼位置
    new_x, new_y = grid_to_pixel(wolf_x, wolf_y)
    wolf.goto(new_x, new_y)
    
    # 检查是否捉到小红帽
    if wolf_x == red_hood_x and wolf_y == red_hood_y:
        return True
    return False

# 游戏结束函数
def game_over():
    global game_active
    game_active = False
    
    pen = turtle.Turtle()
    pen.hideturtle()
    pen.penup()
    pen.color("red")
    pen.goto(0, 0)
    pen.write("游戏结束！大灰狼捉到了小红帽！", align="center", font=("Arial", 16, "normal"))

# 游戏更新函数 - 定时调用
def update_game():
    global last_update_time, game_active
    
    if not game_active:
        return
    
    current_time = time.time()
    
    # 更新光标闪烁
    update_cursor()
    
    # 每秒更新一次游戏状态
    if current_time - last_update_time >= 1:
        # 移动大灰狼
        if wolf_move():
            game_over()
        
        last_update_time = current_time
    
    # 刷新屏幕
    wn.update()
    
    # 持续更新游戏
    if game_active:
        wn.ontimer(update_game, 50)  # 约20帧/秒

# 初始化游戏
def init_game():
    draw_interface()
    draw_grid()
    display_input_box()
    display_history()
    setup_keyboard()
    
    # 启动游戏更新循环
    update_game()

# 启动游戏
if __name__ == "__main__":
    init_game()
    wn.mainloop()