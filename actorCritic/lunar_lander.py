import os
import gymnasium as gym
from stable_baselines3 import A2C

# Nombre del archivo del modelo
MODEL_FILENAME = "a2c_lunarlander_model.zip"

# Crear el entorno con la versión actualizada
env = gym.make("LunarLander-v3")  # Cambiado a v3

# Verificar si el modelo ya existe
if os.path.exists(MODEL_FILENAME):
    # Cargar el modelo existente
    model = A2C.load(MODEL_FILENAME, env=env)
    print("Modelo cargado exitosamente.")
else:
    # Entrenar un nuevo modelo
    model = A2C("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=50000)  # Ajusta los pasos según sea necesario
    # Guardar el modelo
    model.save(MODEL_FILENAME)
    print("Modelo entrenado y guardado exitosamente.")

# Evaluar el modelo
obs, info = env.reset()
total_reward = 0
for _ in range(1000):  # Ejecuta un episodio
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    env.render()  # Muestra la visualización del entorno
    if done or truncated:
        break
env.close()
print(f"Recompensa total obtenida por el agente: {total_reward}")
