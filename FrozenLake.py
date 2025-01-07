import gym
from gym.envs.registration import register

# Register FrozenLake with is_slippery False
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)

# 방향 키 매핑
arrow_keys = {
    'w': 0,  # 위쪽 이동
    'a': 1,  # 왼쪽 이동
    's': 2,  # 아래쪽 이동
    'd': 3   # 오른쪽 이동
}

def get_user_input():
    """사용자 입력을 받아 방향 키로 변환"""
    while True:
        key = input("Enter move (w/a/s/d): ").strip().lower()
        if key in arrow_keys:
            return key
        print("Invalid input! Use w/a/s/d for moves.")

# FrozenLake 환경 초기화
env = gym.make('FrozenLake-v3', render_mode="human")
observation = env.reset()
env.render()

# 게임 루프
while True:
    # 사용자 입력
    key = get_user_input()
    action = arrow_keys[key]

    # Gym의 반환값 처리 (5개 값 처리)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

    # 상태 출력
    print(f"State: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

    # 에피소드 종료 조건
    if terminated or truncated:
        if reward > 0:
            print("Congratulations! You reached the goal!")
        else:
            print("You fell into a hole! Better luck next time!")
        break

env.close()
