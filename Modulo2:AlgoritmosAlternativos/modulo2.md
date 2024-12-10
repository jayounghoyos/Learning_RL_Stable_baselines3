# Algoritmos Alternativos de RL
El objetivo es explorar algoritmos diferentes y comprender cómo funcionan en comparación con `PPO`. Empezaremos con `DQN (Deep Q-Networks)`, uno de los algoritmos fundamentales en el aprendizaje por refuerzo.


# Sección 2.1: Introducción a DQN

## ¿Qué es DQN?
### **Deep Q-Networks (DQN)**

Deep Q-Networks (DQN) es un algoritmo basado en Q-learning que usa redes neuronales para aproximar la función Q (\(Q(s, a)\)) en entornos discretos. Fue introducido por DeepMind y es famoso por su éxito en juegos de Atari.

---

### **Conceptos Clave**

#### **Función Q (\(Q(s, a)\))**
- **Descripción**: Estima el valor esperado de realizar una acción \(a\) en un estado \(s\).
- **Fórmula de Ajuste**:
  \[
  Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
  \]

---

#### **Red Neuronal en DQN**
- **Descripción**: En lugar de almacenar \(Q(s, a)\) en una tabla, se usa una red neuronal para aproximarlo.

---

#### **Replay Buffer**
- **Descripción**: Guarda experiencias (\(s, a, r, s'\)) para entrenar la red neuronal.
- **Beneficio**: Reduce la correlación entre datos consecutivos, mejorando la estabilidad del entrenamiento.

---

#### **Target Network**
- **Descripción**: Una red separada para calcular el objetivo de actualización, estabilizando el entrenamiento.
