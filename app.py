from flask import Flask, render_template, request
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from collections import deque

app = Flask(__name__)

# Función para generar una matriz n x n aleatoria con una densidad de conexiones, respetando las restricciones
def generate_random_matrix(n):
    matrix = np.zeros((n, n), dtype=int)  # Inicializa la matriz de ceros como enteros
    
    # Para cada vértice (excepto el último), aseguramos que tenga entre 1 y 4 conexiones
    for i in range(n - 1):  # El último vértice no tendrá conexiones salientes
        possible_connections = list(range(i + 1, n))  # Conectar a vértices posteriores (asegura grafo dirigido)
        
        # Limitar num_connections al número de posibles conexiones
        num_connections = min(np.random.randint(1, 5), len(possible_connections))  # Entre 1 y 4 conexiones, ajustado al máximo posible
        
        # Elegir aleatoriamente los vértices a los que se conectará
        chosen_connections = np.random.choice(possible_connections, size=num_connections, replace=False)
        
        for j in chosen_connections:
            matrix[i, j] = np.random.randint(1, 10)  # Valores entre 1 y 9

    # Aseguramos que no haya conexiones bidireccionales en la matriz (si hay una i->j, no debe haber j->i)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[j][i] > 0:  # Si hay una conexión de j a i, eliminamos cualquier posible conexión de i a j
                matrix[i][j] = 0

    return matrix

# Función para generar el grafo dirigido a partir de una matriz de adyacencia
def create_graph_from_matrix(matrix):
    G = nx.DiGraph()  # Crear un grafo dirigido
    size = len(matrix)
    
    # Agregar nodos y aristas en función de la matriz
    for i in range(size):
        for j in range(size):
            if matrix[i][j] > 0:  # Solo agregar aristas con capacidades mayores a 0
                G.add_edge(i, j, capacity=matrix[i][j])
    
    return G

# Búsqueda en anchura (BFS) para encontrar caminos aumentantes
def bfs(G, source, sink, parent):
    visited = {node: False for node in G.nodes()}
    queue = deque([source])
    visited[source] = True
    
    while queue:
        u = queue.popleft()
        
        for v in G[u]:
            if visited[v] == False and G[u][v]['capacity'] - G[u][v].get('flow', 0) > 0:
                queue.append(v)
                visited[v] = True
                parent[v] = u
                if v == sink:
                    return True
    return False

# Implementación del algoritmo de Ford-Fulkerson
def ford_fulkerson(G, source, sink):
    # Inicializar el flujo a 0 para cada arista
    for u in G:
        for v in G[u]:
            G[u][v]['flow'] = 0
    
    parent = {}
    max_flow = 0
    
    # Mientras haya un camino aumentante, aumentamos el flujo
    while bfs(G, source, sink, parent):
        # Encontrar la capacidad mínima a lo largo del camino aumentante
        path_flow = float('Inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, G[parent[s]][s]['capacity'] - G[parent[s]][s]['flow'])
            s = parent[s]
        
        # Actualizar las capacidades residuales de las aristas y las aristas inversas a lo largo del camino
        v = sink
        while v != source:
            u = parent[v]
            G[u][v]['flow'] += path_flow
            if G.has_edge(v, u):  # Arista inversa
                G[v][u]['flow'] -= path_flow
            else:
                G.add_edge(v, u, flow=-path_flow, capacity=0)  # Agregar arista inversa
            v = parent[v]
        
        max_flow += path_flow
    
    return max_flow

# Función para generar la imagen del grafo con origen a la izquierda y destino a la derecha
def plot_graph_flow(G, n):
    # Crear una distribución geométrica tipo "red de flujos"
    pos = {}
    
    # El origen siempre será el nodo 0 (izquierda)
    pos[0] = (0, 0)  # Colocar el origen en la izquierda
    
    # El destino siempre será el último nodo n-1 (derecha)
    pos[n-1] = (n-1, 0)  # Colocar el destino en la derecha
    
    # Distribuir los nodos intermedios de manera uniforme entre el origen y el destino
    middle_nodes = list(range(1, n-1))
    for i, node in enumerate(middle_nodes):
        pos[node] = (i + 1, np.random.uniform(-1, 1))  # Distribuir los nodos intermedios en posiciones y-aleatorias
    
    # Obtener las etiquetas de las aristas con los pesos (solo aquellas con capacidad > 0)
    weights = nx.get_edge_attributes(G, 'capacity')

    # Ajustar el tamaño de la figura para acomodar mejor un grafo grande
    plt.figure(figsize=(10, 8))

    # Dibujar el grafo dirigido con el layout de flujo
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, edge_color='gray', font_weight='bold', arrows=True)

    # Dibujar solo las etiquetas de las aristas con capacidad mayor a 0
    weights = {k: v for k, v in weights.items() if v > 0}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_size=8, label_pos=0.3)

    # Guardar la imagen en un buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_matrix', methods=['POST'])
def select_matrix():
    n = int(request.form['matrix_size'])  # Obtener el tamaño de la matriz

    if request.form['matrix_type'] == 'random':
        # Generar matriz aleatoria respetando las restricciones
        matrix = generate_random_matrix(n)
        return render_template('matrix_input.html', matrix=matrix, n=n, random=True)
    else:
        # Preparar la tabla para ingreso manual
        return render_template('matrix_input.html', matrix=None, n=n, random=False)

@app.route('/generate_graph', methods=['POST'])
def generate_graph():
    n = int(request.form['n'])
    matrix = []
    
    # Obtener los valores de la matriz del formulario (manual o aleatorio)
    for i in range(n):
        row = []
        for j in range(n):
            # Convertir los valores a float primero y luego a entero para manejar decimales
            value = int(float(request.form[f'matrix_{i}_{j}']))
            row.append(value)
        matrix.append(row)
    
    matrix = np.array(matrix, dtype=int)
    
    # Crear el grafo a partir de la matriz
    G = create_graph_from_matrix(matrix)

    # Calcular el flujo máximo usando Ford-Fulkerson
    max_flow_value = ford_fulkerson(G, 0, n-1)

    # Generar la imagen del grafo (red de flujos)
    graph_img = plot_graph_flow(G, n)

    # Convertir la matriz a una cadena para mostrarla en la página web
    matrix_str = np.array2string(matrix)

    return render_template('graph.html', graph_img=graph_img, matrix_str=matrix_str, max_flow=max_flow_value)

if __name__ == '__main__':
    plt.switch_backend('Agg')  # Cambia el backend a uno no interactivo para evitar problemas
    app.run(debug=True)
