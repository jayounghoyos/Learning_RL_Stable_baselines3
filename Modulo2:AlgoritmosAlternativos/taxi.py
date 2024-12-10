import os
import gymnasium as gym
from stable_baselines3 import DQN

# Nombre del archivo del modelo
MODEL_FILENAME = "dqn_taxi_model.zip"

# Crear el entorno con render_mode para visualización ASCII
env = gym.make('Taxi-v3', render_mode="ansi")  # 'ansi' para renderizar en la consola

# Verificar si el modelo ya existe
if os.path.exists(MODEL_FILENAME):
    # Cargar el modelo existente
    model = DQN.load(MODEL_FILENAME, env=env)
    print("Modelo cargado exitosamente.")
else:
    # Entrenar un nuevo modelo
    model = DQN("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=20000)  # Ajusta el número de pasos según sea necesario
    # Guardar el modelo
    model.save(MODEL_FILENAME)
    print("Modelo entrenado y guardado exitosamente.")

# Evaluar el modelo (usando el modelo existente o recién entrenado)
obs, info = env.reset()
total_reward = 0
print("Estado inicial del entorno:")
print(env.render())  # Mostrar el estado inicial del entorno en ASCII

for _ in range(100):  # Ejecuta 100 pasos
    action, _ = model.predict(obs)
    action = int(action)  # Convertir la acción en un entero para evitar el error
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    print(f"\nRecompensa obtenida: {reward}")
    print(env.render())  # Mostrar el entorno después de cada acción
    if done or truncated:
        print("Episodio terminado. Reiniciando entorno...\n")
        obs, info = env.reset()
        print(env.render())

env.close()
print(f"Recompensa total obtenida por el agente: {total_reward}")
