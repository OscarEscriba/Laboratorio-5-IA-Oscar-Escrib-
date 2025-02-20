import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt

def load_maze(image_path, scale=10):
    """Carga una imagen y la convierte en una matriz discreta"""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (scale, scale), interpolation=cv2.INTER_NEAREST)
    maze = np.zeros((scale, scale), dtype=int)
    
    start, goals = None, []
    for y in range(scale):
        for x in range(scale):
            pixel = img[y, x]
            if np.array_equal(pixel, [255, 255, 255]):  # Blanco -> Camino libre
                maze[y, x] = 0
            elif np.array_equal(pixel, [0, 0, 0]):  # Negro -> Pared
                maze[y, x] = 1
            elif np.array_equal(pixel, [0, 255, 0]):  # Verde -> Meta
                maze[y, x] = 2
                goals.append((y, x))
            elif np.array_equal(pixel, [0, 0, 255]):  # Rojo -> Inicio
                maze[y, x] = 3
                start = (y, x)
    return maze, start, goals

class MazeProblem:
    def __init__(self, maze, start, goals):
        self.maze = maze
        self.start = start
        self.goals = set(goals)
        self.rows, self.cols = maze.shape
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Derecha, Abajo, Izquierda, Arriba
    
    def is_goal(self, state):
        return state in self.goals
    
    def get_neighbors(self, state):
        y, x = state
        neighbors = []
        for dy, dx in self.actions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.rows and 0 <= nx < self.cols and self.maze[ny, nx] != 1:
                neighbors.append((ny, nx))
        return neighbors
    
    def step_cost(self, state, action, next_state):
        return 1  # Todas las transiciones tienen costo 1
    
def breadth_first_search(problem):
    frontier = [(problem.start, [])]
    explored = set()
    
    while frontier:
        state, path = frontier.pop(0)
        if state in explored:
            continue
        explored.add(state)
        if problem.is_goal(state):
            return path + [state]
        for neighbor in problem.get_neighbors(state):
            frontier.append((neighbor, path + [state]))
    return None

def depth_first_search(problem):
    frontier = [(problem.start, [])]
    explored = set()
    
    while frontier:
        state, path = frontier.pop()
        if state in explored:
            continue
        explored.add(state)
        if problem.is_goal(state):
            return path + [state]
        for neighbor in problem.get_neighbors(state):
            frontier.append((neighbor, path + [state]))
    return None

def a_star_search(problem, heuristic):
    frontier = [(0, problem.start, [])]
    explored = set()
    
    while frontier:
        cost, state, path = heapq.heappop(frontier)
        if state in explored:
            continue
        explored.add(state)
        if problem.is_goal(state):
            return path + [state]
        for neighbor in problem.get_neighbors(state):
            new_cost = len(path) + 1 + heuristic(neighbor, problem)
            heapq.heappush(frontier, (new_cost, neighbor, path + [state]))
    return None

# Heurísticas para A*
def manhattan_heuristic(state, problem):
    return min(abs(state[0] - g[0]) + abs(state[1] - g[1]) for g in problem.goals)

def euclidean_heuristic(state, problem):
    return min(np.sqrt((state[0] - g[0])**2 + (state[1] - g[1])**2) for g in problem.goals)

### **Task 1.4 - Visualización de la Solución**
def draw_solution(maze, path):
    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap='gray_r')
    
    for y, x in path:
        plt.scatter(x, y, c='blue', s=100)
    plt.show()

# Cargar el laberinto y resolverlo
maze, start, goals = load_maze("maze.png", scale=20)
problem = MazeProblem(maze, start, goals)

solution_bfs = breadth_first_search(problem)
solution_dfs = depth_first_search(problem)
solution_a_star = a_star_search(problem, manhattan_heuristic)

print("Solución BFS:", solution_bfs)
print("Solución DFS:", solution_dfs)
print("Solución A*:", solution_a_star)

draw_solution(maze, solution_a_star)
