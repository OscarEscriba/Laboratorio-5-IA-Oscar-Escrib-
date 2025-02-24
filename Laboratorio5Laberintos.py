import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
import heapq
import time
import sys

def discretize_maze(image_path, grid_size=10):
    # Leer la imagen
    img = cv2.imread(image_path)
    
    # Convertir a RGB si está en BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Obtener dimensiones
    height, width, _ = img.shape
    
    grid_h = height // grid_size
    grid_w = width // grid_size
    
    # Crear matriz para el laberinto discretizado
    maze = np.zeros((grid_h, grid_w), dtype=int)
    
    # Discretizar la imagen
    for i in range(grid_h):
        for j in range(grid_w):
            # Obtener la región correspondiente a esta celda
            cell_img = img[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size]
            
            # Calcular los colores promedio en la celda
            avg_color = np.mean(cell_img, axis=(0, 1))
            
            # Determinar si es principalmente negro (pared)
            if avg_color[0] < 50 and avg_color[1] < 50 and avg_color[2] < 50:
                maze[i, j] = 1  # Pared (negro)
            # Determinar si es principalmente verde (meta)
            elif avg_color[1] > 150 and avg_color[0] < 100 and avg_color[2] < 100:
                maze[i, j] = 2  # Meta (verde)
            # Determinar si es principalmente rojo (inicio)
            elif avg_color[0] > 150 and avg_color[1] < 100 and avg_color[2] < 100:
                maze[i, j] = 3  # Inicio (rojo)
            else:
                maze[i, j] = 0  # Camino libre (blanco)
    
    return maze

def visualize_discrete_maze(maze):

    colors = {
        0: [1, 1, 1],      
        1: [0, 0, 0],      
        2: [0, 1, 0],      
        3: [1, 0, 0]      
    }
    
    # Crear una imagen RGB
    h, w = maze.shape
    rgb_maze = np.zeros((h, w, 3))
    
    # Asignar colores
    for i in range(h):
        for j in range(w):
            rgb_maze[i, j] = colors[maze[i, j]]
    
    # Mostrar
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_maze)
    plt.title("Laberinto Discretizado")
    plt.grid(False)
    plt.show()

# 2. FRAMEWORK DE PROBLEMAS
class Problem:    
    def initial_state(self):
        raise NotImplementedError("Las subclases deben implementar este método")
    
    def goal_test(self, state):
        raise NotImplementedError("Las subclases deben implementar este método")
    
    def actions(self, state):
        raise NotImplementedError("Las subclases deben implementar este método")
    
    def result(self, state, action):
        raise NotImplementedError("Las subclases deben implementar este método")
    
    def step_cost(self, state, action, next_state):
        raise NotImplementedError("Las subclases deben implementar este método")


class MazeProblem(Problem):
    
    def __init__(self, maze):
        self.maze = maze
        self.height, self.width = maze.shape
        
        # Encontrar el estado inicial (posición roja)
        initial_positions = np.where(maze == 3)
        if len(initial_positions[0]) == 0:
            raise ValueError("No se encontró el punto de inicio (rojo) en el laberinto")
        self.initial_pos = (initial_positions[0][0], initial_positions[1][0])
        
        # Encontrar los estados objetivo (posiciones verdes)
        goal_positions = np.where(maze == 2)
        self.goal_positions = set(zip(goal_positions[0], goal_positions[1]))
        if not self.goal_positions:
            raise ValueError("No se encontraron puntos objetivo (verdes) en el laberinto")
    
    def initial_state(self):
        return self.initial_pos
    
    def goal_test(self, state):
        return state in self.goal_positions
    
    def actions(self, state):
        row, col = state
        possible_actions = []
        
        # Definir los cuatro movimientos posibles (arriba, derecha, abajo, izquierda)
        movements = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        for dr, dc in movements:
            new_row, new_col = row + dr, col + dc
            
            # Verificar si la nueva posición está dentro de los límites
            if 0 <= new_row < self.height and 0 <= new_col < self.width:
                # Verificar si no es una pared
                if self.maze[new_row, new_col] != 1:
                    possible_actions.append((dr, dc))
        
        return possible_actions
    
    def result(self, state, action):
        row, col = state
        dr, dc = action
        return (row + dr, col + dc)
    
    def step_cost(self, state, action, next_state):
        return 1
    
    def heuristic(self, state, heuristic_type="manhattan"):
        row, col = state
        
        if heuristic_type == "manhattan":
            # Distancia Manhattan al objetivo más cercano
            min_distance = float('inf')
            for goal_row, goal_col in self.goal_positions:
                distance = abs(row - goal_row) + abs(col - goal_col)
                min_distance = min(min_distance, distance)
            return min_distance
        
        elif heuristic_type == "euclidean":
            # Distancia Euclidiana al objetivo más cercano
            min_distance = float('inf')
            for goal_row, goal_col in self.goal_positions:
                distance = ((row - goal_row) ** 2 + (col - goal_col) ** 2) ** 0.5
                min_distance = min(min_distance, distance)
            return min_distance
        
        else:
            raise ValueError(f"Tipo de heurística no reconocido: {heuristic_type}")

# 3. ALGORITMOS DE BÚSQUEDA EN GRAFOS
class GraphSearch:    
    def __init__(self, problem):
        self.problem = problem
    
    def search(self):
        raise NotImplementedError("Las subclases deben implementar este método")
    
    def build_solution(self, node):
        path = []
        actions = []
        
        # Reconstruir el camino desde el nodo objetivo hasta el inicio
        while node:
            path.append(node['state'])
            if 'action' in node:
                actions.append(node['action'])
            node = node.get('parent')
        
        # Invertir las listas para tener el camino desde el inicio hasta el objetivo
        path.reverse()
        actions.reverse()
        
        return path, actions


class BFS(GraphSearch):
    def search(self):
        initial_state = self.problem.initial_state()
        
        # Si el estado inicial ya es un objetivo, devolver solución vacía
        if self.problem.goal_test(initial_state):
            return [initial_state], []
        
        # Cola FIFO para BFS
        frontier = deque([{'state': initial_state}])
        
        # Conjunto de estados explorados
        explored = set()
        
        while frontier:
            node = frontier.popleft()
            state = node['state']
            
            # Marcar el estado como explorado
            explored.add(state)
            
            # Expandir el nodo actual
            for action in self.problem.actions(state):
                child_state = self.problem.result(state, action)
                
                if child_state not in explored and not any(n['state'] == child_state for n in frontier):
                    # Verificar si es un estado objetivo
                    if self.problem.goal_test(child_state):
                        child_node = {
                            'state': child_state,
                            'parent': node,
                            'action': action
                        }
                        return self.build_solution(child_node)
                    
                    # Agregar el estado hijo a la frontera
                    child_node = {
                        'state': child_state,
                        'parent': node,
                        'action': action
                    }
                    frontier.append(child_node)
        
        # No se encontró solución
        return None


class DFS(GraphSearch):
    def search(self):

        initial_state = self.problem.initial_state()
        
        # Si el estado inicial ya es un objetivo, devolver solución vacía
        if self.problem.goal_test(initial_state):
            return [initial_state], []
        
        # Pila para DFS
        frontier = [{'state': initial_state}]
        
        # Conjunto de estados explorados
        explored = set()
        
        while frontier:
            node = frontier.pop()  # LIFO para DFS
            state = node['state']
            
            if state in explored:
                continue
            
            explored.add(state)
            
            # Expandir el nodo actual
            for action in self.problem.actions(state):
                child_state = self.problem.result(state, action)
                
                # Verificar si el estado hijo no ha sido explorado
                if child_state not in explored:
                    # Verificar si es un estado objetivo
                    if self.problem.goal_test(child_state):
                        child_node = {
                            'state': child_state,
                            'parent': node,
                            'action': action
                        }
                        return self.build_solution(child_node)
                    
                    # Agregar el estado hijo a la frontera
                    child_node = {
                        'state': child_state,
                        'parent': node,
                        'action': action
                    }
                    frontier.append(child_node)
        
        # No se encontró solución
        return None


class AStar(GraphSearch):
    def __init__(self, problem, heuristic_type="manhattan"):
        super().__init__(problem)
        self.heuristic_type = heuristic_type
    
    def search(self):
        initial_state = self.problem.initial_state()
        
        # Si el estado inicial ya es un objetivo, devolver solución vacía
        if self.problem.goal_test(initial_state):
            return [initial_state], []
        
        frontier = []
        initial_node = {
            'state': initial_state,
            'parent': None,
            'action': None,
            'g': 0,  # Costo acumulado desde el inicio
            'h': self.problem.heuristic(initial_state, self.heuristic_type),  # Heurística
            'f': self.problem.heuristic(initial_state, self.heuristic_type)   # f = g + h
        }
        
        heapq.heappush(frontier, (initial_node['f'], id(initial_node), initial_node))
        
        # Diccionario para llevar un registro de los nodos en la frontera y sus costos
        frontier_states = {initial_state: initial_node['g']}
        
        explored = set()
        
        while frontier:
            _, _, node = heapq.heappop(frontier)
            state = node['state']
            
            # Eliminar el estado de frontier_states
            if state in frontier_states:
                del frontier_states[state]
            
            # Verificar si es un estado objetivo
            if self.problem.goal_test(state):
                return self.build_solution(node)
            
            # Marcar el estado como explorado
            explored.add(state)
            
            # Expandir el nodo actual
            for action in self.problem.actions(state):
                child_state = self.problem.result(state, action)
                
                # Calcular el costo acumulado hasta el estado hijo
                g = node['g'] + self.problem.step_cost(state, action, child_state)
                
                # Verificar si el estado hijo no ha sido explorado o si tiene un mejor costo
                if child_state not in explored and (child_state not in frontier_states or g < frontier_states[child_state]):
                    h = self.problem.heuristic(child_state, self.heuristic_type)
                    f = g + h
                    
                    child_node = {
                        'state': child_state,
                        'parent': node,
                        'action': action,
                        'g': g,
                        'h': h,
                        'f': f
                    }
                    
                    # Agregar el estado hijo a la frontera
                    heapq.heappush(frontier, (f, id(child_node), child_node))
                    frontier_states[child_state] = g
        
        # No se encontró solución
        return None

# 4. VISUALIZACIÓN DE LA SOLUCIÓN
def visualize_solution(maze, path, title="Solución del Laberinto"):
    """
    Visualiza la solución del laberinto.
    
    Args:
        maze: Matriz discreta del laberinto
        path: Lista de estados (posiciones) que forman el camino desde el inicio hasta el objetivo
        title: Título de la visualización
    """
    solution_maze = maze.copy()
    
    for row, col in path[1:-1]:  # Excluimos el inicio y la meta
        solution_maze[row, col] = 4
    
    # Definir colores para la visualización
    colors = np.array([
        [1, 1, 1],     
        [0, 0, 0],      
        [0, 1, 0],      
        [1, 0, 0],      
        [1, 1, 0]      
    ])
    
    # Crear una imagen RGB
    h, w = solution_maze.shape
    rgb_maze = np.zeros((h, w, 3))
    
    # Asignar colores
    for i in range(h):
        for j in range(w):
            rgb_maze[i, j] = colors[solution_maze[i, j]]
    
    # Mostrar
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_maze)
    plt.title(title)
    plt.grid(False)
    plt.show()

# 5. PROGRAMA PRINCIPAL
def solve_and_visualize(maze_path, grid_size=10, algorithm="astar", heuristic_type="manhattan"):
    print(f"Discretizando el laberinto con tamaño de celda {grid_size}...")
    maze = discretize_maze(maze_path, grid_size)
    
    print("Laberinto discretizado:")
    visualize_discrete_maze(maze)
    
    print("Creando el problema...")
    problem = MazeProblem(maze)
    
    print(f"Resolviendo el laberinto con {algorithm.upper()}...")
    start_time = time.time()
    
    if algorithm.lower() == "bfs":
        search_algo = BFS(problem)
    elif algorithm.lower() == "dfs":
        search_algo = DFS(problem)
    elif algorithm.lower() == "astar":
        search_algo = AStar(problem, heuristic_type)
    else:
        raise ValueError(f"Algoritmo no reconocido: {algorithm}")
    
    # Ejecutar la búsqueda
    result = search_algo.search()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    if result:
        path, actions = result
        print(f"Solución encontrada en {execution_time:.2f} segundos!")
        print(f"Longitud del camino: {len(path)}")
        
        # Visualizar la solución
        algo_name = algorithm.upper() if algorithm.lower() != "astar" else f"A* ({heuristic_type})"
        visualize_solution(maze, path, title=f"Solución con {algo_name}")
        
        return path, actions
    else:
        print(f"No se encontró solución después de {execution_time:.2f} segundos.")
        return None

# Programa principal
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python maze_solver.py <ruta_del_laberinto.bmp> [tamaño_de_celda] [algoritmo] [heuristica]")
        print("  tamaño_de_celda: Tamaño de cada celda discreta (por defecto: 10)")
        print("  algoritmo: bfs, dfs, astar (por defecto: astar)")
        print("  heuristica: manhattan, euclidean (por defecto: manhattan)")
        
        
        maze_path = "Test2.bmp"  
        grid_size = 10
        algorithm = "astar"  
        heuristic_type = "manhattan"  
        print(f"Ejecutando con valores predeterminados: {maze_path}, tamaño_celda={grid_size}, algoritmo={algorithm}, heurística={heuristic_type}")
    else:
        maze_path = sys.argv[1]
        grid_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        algorithm = sys.argv[3] if len(sys.argv) > 3 else "astar"
        heuristic_type = sys.argv[4] if len(sys.argv) > 4 else "manhattan"
    
    # Resolver y visualizar
    solve_and_visualize(maze_path, grid_size, algorithm, heuristic_type)
