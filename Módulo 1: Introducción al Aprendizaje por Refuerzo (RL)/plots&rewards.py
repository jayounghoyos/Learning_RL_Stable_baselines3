import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import matplotlib.pyplot as plt

# Nombre del archivo del modelo
MODEL_FILENAME = "ppo_cartpole_model.zip"
TENSORBOARD_LOG = "./tensorboard_logs/"

# Crear el entorno
env = gym.make('CartPole-v1', render_mode="human")

# Callbacks para métricas y checkpoints
eval_callback = EvalCallback(
    env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=5000,  # Evaluar cada 5000 pasos
    deterministic=True,
    render=False,
)
checkpoint_callback = CheckpointCallback(
    save_freq=5000,  # Guardar el modelo cada 5000 pasos
    save_path="./checkpoints/",
    name_prefix="ppo_checkpoint",
)

# Verificar si el modelo ya existe
if os.path.exists(MODEL_FILENAME):
    # Cargar el modelo existente
    model = PPO.load(MODEL_FILENAME, env=env, tensorboard_log=TENSORBOARD_LOG)
    print("Modelo cargado exitosamente.")
else:
    # Entrenar un nuevo modelo
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=TENSORBOARD_LOG)
    model.learn(
        total_timesteps=10000,
        callback=[eval_callback, checkpoint_callback]
    )
    # Guardar el modelo
    model.save(MODEL_FILENAME)
    print("Modelo entrenado y guardado exitosamente.")

# Evaluar el modelo (usando el modelo existente o recién entrenado)
obs, info = env.reset()
total_reward = 0
for _ in range(1000):  # Ejecuta un episodio
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    env.render()
    if done or truncated:
        break
env.close()
print(f"Recompensa total obtenida por el agente: {total_reward}")

# Mostrar un mensaje sobre TensorBoard
print("Puedes ver las métricas en TensorBoard ejecutando:")
print("tensorboard --logdir ./tensorboard_logs/")
