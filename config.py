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
DEFAULT_INITIAL_Z = 0.5
DEFAULT_TARGET_X = 0.0
DEFAULT_TARGET_Y = 0.0
DEFAULT_TARGET_Z = 2

# 滑模控制器參數
SMC_LAMBDA_POS = 0.5      # 位置控制滑動面斜率
SMC_ETA_POS = 0.5         # 位置控制增益
SMC_LAMBDA_ALT = 2.3      # 高度控制滑動面斜率
SMC_ETA_ALT = 25.0        # 高度控制增益
SMC_LAMBDA_ATT = 30       # 姿態控制滑動面斜率
SMC_ETA_ATT = 25          # 姿態控制增益
SMC_LAMBDA_ATT_YAW = 30   # 偏航控制滑動面斜率
SMC_ETA_ATT_YAW = 9.0     # 偏航控制增益
SMC_K_SMOOTH = 0.5        # 平滑因子
SMC_K_SMOOTH_POS = 0.5    # 位置控制平滑因子
SMC_MAX_ANGLE = 30 * 3.14159/180  # 最大傾角(弧度)