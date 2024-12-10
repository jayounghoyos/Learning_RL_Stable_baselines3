import gymnasium as gym
from stable_baselines3 import PPO

# Crear el entorno con el modo de renderizado especificado
env = gym.make('CartPole-v1', render_mode="human")

# Crear el modelo PPO y especificar el uso de la CPU
model = PPO("MlpPolicy", env, verbose=1, device="cpu")

# Entrenar el modelo
model.learn(total_timesteps=10000)

# Evaluar el modelo
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()
env.close()
