from rl.env_UAV import QuadrotorEnv

def make_env(initial_position=[0, 0, 0], target_position=[1, 1, 2], random_target=True):
    """創建四旋翼機環境"""
    env = QuadrotorEnv(
        initial_position=initial_position,
        target_position=target_position,
        random_target=random_target
    )
    return env