import numpy as np
import os
import pickle

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(CURRENT_DIR, "0202g1.pkl")
motion_files = [dataset_path]  # 文件列表

# 初始化存储容器
all_states = []
all_next_states = []

# 遍历处理每个文件
for file_path in motion_files:
    # 修正点：使用 with open 以二进制模式打开文件
    try:
        with open(file_path, 'rb') as f:  # 正确获取文件对象 'f'
            motion_data = pickle.load(f)  # 将文件对象 'f' 传递给 pickle.load
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        continue

    if 'dof_pos' not in motion_data:
        print(f"警告: {os.path.basename(file_path)} 缺少 'dof_pos' 键，跳过处理")
        continue

    # 提取数据
    dof_data = motion_data['dof_pos']
    states = dof_data[:-1]
    next_states = dof_data[1:]

    # 将当前文件的数据添加到总列表
    all_states.append(states)
    all_next_states.append(next_states)

# 合并所有文件的数据（如果有多个文件）
if all_states:
    all_states = np.vstack(all_states)
    all_next_states = np.vstack(all_next_states)
    
    # 保存数据集
    output_path = os.path.join(CURRENT_DIR, "state_next_state_dataset.npz")
    np.savez(output_path, states=all_states, next_states=all_next_states)
    
    print(f"处理完成！共处理 {len(motion_files)} 个文件")
    print(f"数据集形状: states {all_states.shape}, next_states {all_next_states.shape}")
    print(f"保存至: {output_path}")
else:
    print("错误: 无有效数据可保存")
