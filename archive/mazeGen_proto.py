from __future__ import annotations

import random
from rc_agents.ui.streamlit_ui.text_num import text_num


def generate_maze(width, height):
    # Create a grid with all walls
    maze_one = [[1 for _ in range(width)] for _ in range(height)]
    
    # Set the starting position
    start_x, start_y = 1, 1
    maze_one[start_y][start_x] = 0
    
    # Stack for backtracking
    stack = [(start_x, start_y)]
    
    # Directions: right, down, left, up
    directions = [(2, 0), (0, 2), (-2, 0), (0, -2)]
    
    while stack:
        current_x, current_y = stack[-1]
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = current_x + dx, current_y + dy
            
            if 1 <= nx < width-1 and 1 <= ny < height-1 and maze_one[ny][nx] == 1:
                maze_one[ny][nx] = 0
                maze_one[current_y + dy//2][current_x + dx//2] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()
    
    # Set the end position
    end_x, end_y = width - 2, height - 2
    maze_one[end_y][end_x] = 0
    
    return maze_one

def print_maze_one(maze_one):
    for row in maze_one:
        print(''.join(['#' if cell == 1 else ' ' for cell in row]))

def generate_maze_two(size):
    maze_two = []
    for _ in range(size):
        row = "".join*(random.choice(["#", "."]) for _ in range(size))
        maze_two.append(row)
    maze_two[0] = maze_two[0][:1] + "S" + maze_two[0][2:] # Start
    maze_two[-1] = maze_two[-1][:1] + "E" # End
    return maze_two

def print_maze_two(maze_two):
    for row in maze_two:
        print(row)

size = rows_i, cols_i
maze_one = generate_maze(rows_i, cols_i)
maze_two = generate_maze_two(size)