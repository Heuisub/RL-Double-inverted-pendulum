import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# 보상 기록 및 실시간 플로팅을 위한 콜백
class PlottingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PlottingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # 벡터화된 환경에서는 dones를 사용
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.locals['infos'][0]['episode']['r'])
            self.episode_lengths.append(self.locals['infos'][0]['episode']['l'])
            plt.clf()  
            plt.plot(self.episode_rewards, label='Total Reward per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Training Progress')
            plt.pause(0.01)
        return True

# 환경 초기화
def make_env():
    env = gym.make('InvertedDoublePendulum-v4', render_mode="human")
    env = Monitor(env)  # 모니터링을 위해 래핑
    return env

# 벡터화된 환경으로 전환
env = DummyVecEnv([make_env])

# Action noise (DDPG는 continuous action space이므로 exploration을 위해 노이즈를 추가)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# 모델 저장 디렉토리
log_dir = "./ddpg_inverted_double_pendulum/"
os.makedirs(log_dir, exist_ok=True)

# DDPG 모델 생성 (정책 네트워크는 MlpPolicy)
model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, 
             tensorboard_log=log_dir, 
             learning_rate=1e-3,  # 학습률 조정 가능
             gamma=0.99,  # 할인 계수
             buffer_size=1000000,  # 경험 리플레이 버퍼 크기
             learning_starts=10000,  # 학습 시작 단계
             batch_size=64,  # 배치 사이즈
             tau=0.005  # 타겟 네트워크 업데이트 비율 (soft update)
             )

# 모델 학습
plot_callback = PlottingCallback()
model.learn(total_timesteps=200000, callback=plot_callback)

# 학습된 모델 저장
model.save(log_dir + "ddpg_inverted_double_pendulum")

# 학습된 모델 로드 및 테스트
model = DDPG.load(log_dir + "ddpg_inverted_double_pendulum", env=env)

# 학습된 모델을 테스트하고 환경 시각화
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()  # 환경 시각화
    if done:
        obs = env.reset()

env.close()
