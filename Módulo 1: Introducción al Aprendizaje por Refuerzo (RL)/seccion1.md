# Qué es el aprendizaje por refuerzo (RL)?

El aprendizaje por refuerzo (Reinforcement Learning, RL) es una rama de la inteligencia artificial donde un agente aprende a tomar decisiones óptimas interactuando con un entorno. El objetivo es maximizar las recompensas acumuladas a lo largo del tiempo.


## Concepto Clave
- El agente experimenta, recibe retroalimentación y mejora sus decisiones con el tiempo.


## Ejemplo Práctico

Un robot que aprende a caminar:

1) Empieza dando pasos al azar
2) Recibe recompensas cuando avanza y penalizaciones si se cae.
3) Ajusta su comportamiento para maximizar el número de pasos avanzados.


### **Sección 1.2: Componentes del Aprendizaje por Refuerzo**

| **Concepto**             | **Definición**                                                                                         | **Ejemplo**                                                                                                                                                     |
|---------------------------|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Agente**                | Toma decisiones.                                                                                     | Un robot, un programa que juega ajedrez.                                                                                                                       |
| **Entorno**               | El espacio donde opera el agente.                                                                    | Un tablero de ajedrez, un simulador de vuelo.                                                                                                                  |
| **Estado (\(s\))**         | Representa una instantánea del entorno.                                                              | La posición actual de las piezas en el ajedrez.                                                                                                                |
| **Acción (\(a\))**         | Decisión tomada por el agente.                                                                       | Mover una pieza en el ajedrez.                                                                                                                                 |
| **Recompensa (\(r\))**     | Retroalimentación del entorno después de una acción.                                                 | \(+10\) puntos por ganar el juego, \(-5\) por un movimiento inválido.                                                                                          |
| **Política (\(\pi\))**     | Estrategia del agente para tomar decisiones.                                                         | "Si hay un camino libre hacia la derecha, muévete a la derecha."                                                                                               |
| **Función de Valor (\(V(s)\))** | Estima la utilidad de un estado considerando recompensas futuras.                                        | Saber que un estado cercano a la meta es más valioso que un estado inicial.                                                                                    |
| **Función de Acción-Valor (\(Q(s, a)\))** | Estima la utilidad de realizar una acción \(a\) en un estado \(s\).                                     | En ajedrez, mover una torre puede ser más valioso que mover un peón desde una misma posición.                                                                   |

---

### **Sección 1.3: Flujo del Aprendizaje por Refuerzo**

El RL se basa en un ciclo de interacción entre el **agente** y el **entorno**:

1. El **agente** observa el estado actual (\(s\)).
2. Selecciona una acción (\(a\)) basada en su política (\(\pi\)).
3. El **entorno**:
   - Cambia al nuevo estado (\(s'\)).
   - Proporciona una recompensa (\(r\)).
4. El agente actualiza su política para maximizar recompensas futuras.

Este ciclo continúa hasta que se alcanza un objetivo o se completa un episodio.

---

### **Sección 1.4: RL vs Otros Métodos**

| **Característica**       | **Supervisado**                      | **No Supervisado**                | **Reforzamiento (RL)**          |
|--------------------------|--------------------------------------|-----------------------------------|---------------------------------|
| **Tipo de datos**         | Etiquetados                         | Sin etiquetas                     | Recompensas                    |
| **Objetivo**              | Minimizar error de predicción       | Descubrir patrones ocultos        | Maximizar recompensa acumulada |
| **Ejemplo**               | Clasificar correos spam             | Agrupación de clientes            | Aprender a jugar videojuegos   |


# proyecto práctico: Entrenando un agente en CartPole-v1


### **Explicación Paso a Paso del Código**

`import gym` y `from stable_baselines3 import PPO`:

- Importamos OpenAI Gym para crear el entorno y Stable Baselines3 para entrenar al agente.

`env = gym.make('CartPole-v1')`:

- Creamos el entorno `CartPole-v1`, donde el objetivo es mantener un palo en equilibrio.

`model = PPO("MlpPolicy", env, verbose=1')`:

- Usamos el algoritmo **PPO** con una política basada en redes neuronales (`MlpPolicy`).

`model.learn(total_timesteps=10000')`:

- Entrenamos el modelo durante 10,000 pasos.

**Bucle de Evaluación**:

- Usamos `model.predict(obs)` para que el modelo tome decisiones.
- Visualizamos la interacción usando `env.render()`.





### **Modos de Renderizado (Render Modes)**

Gymnasium ofrece diferentes modos de renderizado para personalizar cómo se visualizan los entornos durante el entrenamiento o la evaluación. Aquí están los principales modos:

#### **1. `human`**
- **Descripción**: Muestra el entorno en tiempo real en una ventana gráfica interactiva.
- **Uso principal**: Evaluar visualmente el comportamiento del agente mientras interactúa con el entorno.
- **Ejemplo**:
  ```python
  env = gym.make('CartPole-v1', render_mode="human")
  env.render()
  ```

#### **2. `rgb_array`**
- **Descripción**: Devuelve la visualización del entorno como una matriz de píxeles en formato RGB.
- **Uso principal**: Capturar imágenes del entorno para análisis o procesamiento posterior.
- **Ejemplo**:
  ```python
  env = gym.make('CartPole-v1', render_mode="rgb_array")
  frame = env.render()
  print(frame.shape)  # Matriz con las dimensiones de la imagen
  ```

#### **3. `rgb_array_list`**
- **Descripción**: Genera una lista de matrices RGB (imágenes) correspondientes a cada fotograma de un episodio.
- **Uso principal**: Grabar episodios completos para crear animaciones o videos.
- **Ejemplo**:
  ```python
  env = gym.make('CartPole-v1', render_mode="rgb_array_list")
  frames = []
  obs, info = env.reset()
  done = False
  while not done:
      frames.append(env.render())
      action = env.action_space.sample()
      obs, reward, done, truncated, info = env.step(action)
  ```

#### **4. `ansi`**
- **Descripción**: Devuelve una representación en texto ASCII del entorno.
- **Uso principal**: Visualización en terminal para entornos basados en texto.
- **Ejemplo**:
  ```python
  env = gym.make('Taxi-v3', render_mode="ansi")
  print(env.render())
  ```

#### **5. `None`**
- **Descripción**: No renderiza nada.
- **Uso principal**: Entrenamiento puro sin visualización para mejorar el rendimiento.
- **Ejemplo**:
  ```python
  env = gym.make('CartPole-v1')
  ```

---

### **Resumen de Usos por Modo**

| **Modo**        | **Descripción**                                  | **Uso Principal**                                                   |
|------------------|--------------------------------------------------|----------------------------------------------------------------------|
| `human`         | Ventana interactiva en tiempo real               | Evaluar visualmente el comportamiento del agente.                   |
| `rgb_array`     | Matriz de píxeles RGB                            | Procesamiento o análisis de imágenes.                               |
| `rgb_array_list`| Lista de fotogramas en formato RGB               | Grabar episodios completos para videos o animaciones.               |
| `ansi`          | Representación ASCII en texto                   | Entornos basados en texto o terminal.                               |
| `None`          | No renderiza nada                                | Entrenamiento sin visualización para maximizar el rendimiento.      |
