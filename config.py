"""配置檔案，存放模擬的所有參數設定"""
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# 無人機設定
DEFAULT_DRONE = DroneModel("cf2x")  # 默認無人機型號: Crazyflie 2.x
DEFAULT_NUM_DRONES = 1  # 默認只有一架無人機

# 環境設定
DEFAULT_PHYSICS = Physics("pyb")  # 默認物理引擎: PyBullet
DEFAULT_GUI = True  # 默認啟用GUI顯示
DEFAULT_RECORD_VISION = False  # 默認不記錄視頻
DEFAULT_PLOT = True  # 默認繪製結果圖表
DEFAULT_USER_DEBUG_GUI = False  # 默認不顯示調試GUI
DEFAULT_OBSTACLES = False  # 默認不添加障礙物，創建平地環境

# 模擬設定
DEFAULT_SIMULATION_FREQ_HZ = 240  # 默認模擬頻率: 240Hz
DEFAULT_CONTROL_FREQ_HZ = 48  # 默認控制頻率: 48Hz
DEFAULT_DURATION_SEC = 5  # 默認模擬時長: 5秒
DEFAULT_OUTPUT_FOLDER = 'results'  # 默認輸出資料夾
DEFAULT_COLAB = False  # 默認非Colab環境運行

# 位置設定
DEFAULT_INITIAL_X = 0.0
DEFAULT_INITIAL_Y = 0.0
DEFAULT_INITIAL_Z = 0.1
DEFAULT_TARGET_X = 1.0
DEFAULT_TARGET_Y = 1.0
DEFAULT_TARGET_Z = 0.5