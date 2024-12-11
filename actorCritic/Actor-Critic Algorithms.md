# **Actor-Critic Algorithms (A2C/A3C)**

## **Conceptos Clave**

### **Actor y Critic**
- **Actor**: Es la parte de la red neuronal que decide qué acción tomar dado un estado (\( \pi(a|s) \)).
- **Critic**: Evalúa qué tan buena es la acción tomada calculando el valor del estado (\( V(s) \)) o la ventaja (\( A(s, a) = Q(s, a) - V(s) \)).

---

### **Ventaja (\( A(s, a) \))**
- **Descripción**: Indica cuánto mejor es una acción en comparación con el promedio esperado del estado.
- **Fórmula**:
  \[
  A(s, a) = r + \gamma V(s') - V(s)
  \]

---

### **A2C (Advantage Actor-Critic)**
- **Descripción**: Variante síncrona de A3C, que ejecuta múltiples copias del entorno en paralelo para mejorar la eficiencia.
- **Beneficio**: Reduce la varianza en las estimaciones de las políticas y valores.
