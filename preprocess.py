import numpy as np
import pandas as pd
import os, time, sys

# sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
np.set_printoptions(linewidth=400)

qws = pd.read_csv('data/qws2.CSV', header=None)

pd_data = qws.iloc[1:, :8].values.astype(float)
item_size = pd_data.shape[0]
data_dimensions = pd_data.shape[1]
print(f'item_size: {item_size}\ndata_dimensions: {data_dimensions}')

# 每一列的最大值和最小值
column_max = np.max(pd_data, axis=0)
column_min = np.min(pd_data, axis=0)
# print(f'column_max: {column_max}\ncolumn_min: {column_min}')
# 正负属性，如延迟越高越不好，归一化
pos_or_neg = ['-', '+', '+', '+', '+', '+', '+', '-']
for (index, value) in enumerate(pos_or_neg):
  if value == '-':  # 如果是负属性，改变方向
    pd_data[:, index] = (pd_data[:, index] - column_min[index]) / (column_max[index] - column_min[index])
  else:
    pd_data[:, index] = (column_max[index] - pd_data[:, index]) / (column_max[index] - column_min[index])
all_services = pd_data
print(f'all_services: \n{all_services}\nall_services.shape: {all_services.shape}')
# 数据集的划分
constraints_index = np.random.choice(item_size, 6, replace=False)  # 随机选择6个索引
constraints = pd_data[constraints_index, :]  # 选择6个索引的数据，作为约束集
print(f"constraints_service: \n{constraints}\nconstraints_service.shape:{constraints.shape}")
candidates = np.delete(pd_data, constraints_index, axis=0)  # 删除6个索引的数据，作为候选集
print(f"candidates_service: \n{candidates}\ncandidates_service.shape:{candidates.shape}")

# 生成历史调用记录，随机从所有服务中挑选出 10 - 15 个服务
gen_histories = lambda allServices: all_services[
  np.random.choice(all_services.shape[0], np.random.randint(10, 15 + 1), replace=False)]

# 生成很多个用户的历史调用记录
gen_users_histories = lambda user_count, allServices: np.array([gen_histories(all_services) for _ in range(user_count)],
                                                               dtype=list)

users_histories = gen_users_histories(6, all_services)

get_local_time = lambda: time.strftime("%Y%m%d%H%M%S", time.localtime())

# 保存候选集，约束集，历史记录
np.savez(f'data/data_{get_local_time()}', constraints=constraints, candidates=candidates, histories=users_histories)
