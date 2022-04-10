import logging

import pandas as pd, requests, sys, os
from tqdm import tqdm
import numpy as np, math, time
from itertools import combinations, permutations, product
from scipy import spatial
from scipy import stats
from collections import Counter

# 定义预先处理好的数据路径
data_path = 'data/data_20220410011614.npz'
# 设置显示宽度
np.set_printoptions(linewidth=400)


def load_data(path):
  data = np.load(path, allow_pickle=True)
  constraints = data['constraints']
  candidates = data['candidates']
  histories = data['histories']
  return constraints, candidates, histories


def get_similarity(item1, item2, dimensions, alpha=0.5):
  """
  计算两个物品的相似度
  param: item1, item2: 两个 d 维的向量
  param: dimensions: 向量的维度
  param: alpha: 平衡因子，默认为 0.5
  return: similarity: 两个物品的相似度
  """
  distance = spatial.distance.euclidean(item1, item2)
  tau, p = stats.kendalltau(item1, item2)
  similarity = alpha * (1.0 - distance / np.sqrt(dimensions)) + (1.0 - alpha) * tau
  return similarity


def get_shannon_entropies(user_call_histories):
  """
  获取用户访问历史的 Shannon Entropy 香浓信息熵，表示用户的多样性需求
  :param user_call_histories: 用户的服务调用历史记录
  :rtype list: 服务调用记录的熵值列表
  """
  indexs = [[np.argmax(i) for i in item] for item in user_call_histories]
  shannon_entropies = []
  for item in indexs:
    shannon_entropy = np.abs(
      sum([count / len(item) * (np.math.log2(count / len(item))) for count in Counter(item).values()]))
    shannon_entropies.append(shannon_entropy)
  return np.asarray(shannon_entropies)


def get_diversity_parameter(shannon_entropies, H0=1):
  """
  根据服务调用历史记录信息熵计算多样性程度
  :param shannon_entropies: 用户的服务调用历史记录多样性熵值列表
  :param H0: 超参数
  :return: 每一个用户的多样化参数列表
  """
  H_max = np.max(shannon_entropies)
  H_min = np.min(shannon_entropies)
  return np.asarray([(item - H_min + H0) / (H_max - H_min + H0) for item in shannon_entropies])


def get_kernel_matrix(Constraint, Candidates, dimensions, fu=1, alpha=0.9):
  """
  根据约束和候选集合，计算核矩阵
  param: constraints: 约束集合
  param: candidates: 候选集合
  param: fu: 多样化系数
  param: alpha: 多样化因子
  param: dimensions: 向量的维度
  return: kernel_matrix: 核矩阵
  """
  similarities = np.asarray([get_similarity(item, Constraint, dimensions=dimensions) for item in Candidates])
  kernel_matrix = np.diag(np.square(similarities))
  comb = [(i, j) for (i, j) in list(combinations(range(len(Candidates)), 2))]
  for (i, j) in tqdm(comb):
    kernel_matrix[i, j] = fu * alpha * similarities[i] * similarities[j] * get_similarity(Candidates[i],
                                                                                          Candidates[j],
                                                                                          dimensions=dimensions)
    kernel_matrix[j, i] = kernel_matrix[i, j]
  return kernel_matrix


def dpp(kernel_matrix, max_length, epsilon=1E-10):
  """
  使用行列式点过程，生成推荐列表
  param: kernel_matrix: 包含了每个物品的相似度和个性化参数的核矩阵
  param: max_length: 推荐列表的长度
  param: epsilon: 两个物品的相似度最小值
  return: 推荐物品的下标
  """
  item_size = kernel_matrix.shape[0]
  cis = np.zeros((max_length, item_size))
  di2s = np.copy(np.diag(kernel_matrix))
  selected_items = list()
  selected_item = np.argmax(di2s)
  selected_items.append(selected_item)
  while len(selected_items) < max_length:
    k = len(selected_items) - 1
    ci_optimal = cis[:k, selected_item]
    di_optimal = math.sqrt(di2s[selected_item])
    elements = kernel_matrix[selected_item, :]
    eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
    cis[k, :] = eis
    di2s -= np.square(eis)
    di2s[selected_item] = -np.inf
    selected_item = np.argmax(di2s)
    if di2s[selected_item] < epsilon:
      break
    selected_items.append(selected_item)
  return selected_items


def get_dcg_value(constraint, result_list, dimensions):
  """
  获取推荐列表的 DCG 值
  param: constraint: 推荐列表的约束集
  param: result_list: 推荐列表本身
  return: dcg_value: 推荐列表的 DCG 值
  """
  gain = lambda score, rank: (np.power(2, score) - 1) / np.log2(1 + rank)
  dcg = np.sum(
    [gain(get_similarity(item, constraint, dimensions=dimensions), index + 1) for (index, item) in
     enumerate(result_list)])
  return dcg


def get_diversity_of_list(hlist, d):
  """
  获取推荐列表的多样性，使用累计不相似度度量
  param: hlist: 推荐列表
  param: dimensions: 向量的维度
  return: diversity: 推荐列表的多样性
  """
  if len(hlist) <= 1:
    # 如果推荐列表只有一个元素，则返回 0，表示完全没有多样性
    return 0
  return 2 / (len(hlist) * (len(hlist) - 1)) * np.sum(
    [1 - get_similarity(hlist[i], hlist[j], dimensions=d) for i, j in list(permutations(range(len(hlist)), 2))])


def get_rmdse_of_lists(historical_list, recommend_list, top_k, dimensions):
  """
  计算推荐列表与用户历史记录的多样性是否符合
  param: historical_list： 用户的服务调用历史记录列表
  param: recommend_list: 使用模型推荐的服务列表
  param: top_k: 推荐服务的数量
  param: dimensions: 向量的维度
  return: 推荐列表与历史记录的均方根误差 RMDSE
  """
  historical_diversity = get_diversity_of_list(historical_list, d=dimensions)
  recommend_diversity = get_diversity_of_list(recommend_list, d=dimensions)
  print(f'历史多样性: {historical_diversity} 推荐多样性: {recommend_diversity}')
  return np.sqrt(np.square((historical_diversity - recommend_diversity)) / top_k)


def send_msg(message):
  """
  使用飞书 WebHook 向群组发送消息
  param: message: 需要发送的消息
  """
  msg = {'msg_type': 'text', 'content': {'text': f'{message}\n{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'}}
  url = 'https://open.feishu.cn/open-apis/bot/v2/hook/36921b6c-7587-4781-8645-2d794b78bf3b'
  headers = {'Content-Type': 'application/json'}
  requests.post(url, json=msg, headers=headers)


def get_exp_list(exp_type):
  """
  生成测试列表
  :param exp_type: 使用小规模或大规模数据
  :return: 实验列表的组合
  """
  t_list = [10, 13]
  r_list = [1000, 1300, 1600, 1900, 2200, 2500]
  d_list = [3, 4, 5, 6, 7, 8]
  top_k = [8]
  # 3, 4, 5, 6, 7,
  if exp_type == 'real':
    return list(product(t_list, d_list, top_k))
  else:
    return list(product(r_list, d_list, top_k))


get_local_time = lambda: time.strftime("%Y%m%d%H%M%S", time.localtime())


def dpp_eva(dimension, topK, con, can, historiesList, fu=1, alpha=0.9):
  """
  根据约束集和候选集，以及 topK 和 数据维度，评估 DPP 模型的 准确度、多样性、偏好一致性
  :param dimension: 考虑向量的维度
  :param topK: 推荐物品的数量
  :param can: 候选集合
  :param con: 约束集合
  :param historiesList: 用户的历史调用记录
  :param fu: 个性化参数
  :param alpha: 个性话超参数
  :return: 返回推荐结果在候选集中的下标，准确度，多样性，偏好一致性
  """
  kMatrix = get_kernel_matrix(con, can, dimension, fu, alpha)
  resultIndex = dpp(kMatrix, topK)
  result = can[resultIndex]
  DCG = get_dcg_value(con, result, dimension)
  Diversity = get_diversity_of_list(result, dimension)
  RMDSE = get_rmdse_of_lists(historiesList, result, topK, dimension)
  return resultIndex, DCG, Diversity, RMDSE


def rs_eva(dimension, topK, con, can, hlist):
  """
  使用 RankingScore 方式评估推荐结果
  :param dimension: 向量的维度
  :param topK: 结果数量
  :param con: 约束集合
  :param can: 候选集合
  :param hlist: 历史调用记录
  :return: 推荐物品下标，准确性，多样性，均方根误差
  """
  scores = np.asarray([get_similarity(con, item, dimension) for item in can])
  # 从 score 中选择分数最高的 topK 个item
  result_index = np.argsort(scores)[::-1][:topK]
  result = can[result_index]
  DCG = get_dcg_value(con, result, dimension)
  Diversity = get_diversity_of_list(result, dimension)
  RMDSE = get_rmdse_of_lists(hlist, result, topK, dimension)
  return result_index, DCG, Diversity, RMDSE
  pass


if __name__ == '__main__':
  constraints, candidates, histories = load_data(data_path)
  logging.critical(f'数据加载完成，约束集:{constraints.shape}, 候选集：{candidates.shape}, 用户调用记录:{len(histories)}')
  exp_list = get_exp_list('test')
  ############参数构建############
  historiesList = histories[0]

  ############使用 DPP 模型评估############
  # dpp_res = {}
  # logging.critical(f'开始使用 DPP 算法计算推荐结果')
  # for (n, d, k) in exp_list:
  #   constraint = constraints[0, :d]
  #   candidate = candidates[:n, :d]
  #   indexs, dcg, div, rmdse = dpp_eva(d, k, constraint, candidate, historiesList[:,:d])
  #   dpp_res[f'dpp_{n}_{d}_{k}'] = {
  #     "n": n,
  #     "d": d,
  #     "k": k,
  #     "indexs": indexs,
  #     "dcg": dcg,
  #     "div": div,
  #     "rmdse": rmdse
  #   }
  # logging.critical(f'DPP 计算完成， 结果共 {len(dpp_res)} 条')
  # pd.DataFrame(dpp_res).to_json(f'data/dpp_res_{get_local_time()}.json')
  # logging.critical("DPP 结果保存完成")
  ############使用 Ranking Score模型评估############
  # rs_res = {}
  # logging.critical(f'开始使用 RankingScore 算法计算推荐结果')
  # for (n, d, k) in exp_list:
  #   constraint = constraints[0, :d]
  #   candidate = candidates[:n, :d]
  #   indexs, dcg, div, rmdse = rs_eva(d, k, constraint, candidate, historiesList[:,:d])
  #   rs_res[f'rs_{n}_{d}_{k}'] = {
  #     "n": n,
  #     "d": d,
  #     "k": k,
  #     "indexs": indexs,
  #     "dcg": dcg,
  #     "div": div,
  #     "rmdse": rmdse
  #   }
  # logging.critical(f'Ranking Score 计算完成，结果共 {len(rs_res)} 条')
  # pd.DataFrame(rs_res).to_json(f'data/rs_res_{get_local_time()}.json')
  # logging.critical('Ranking Score 结果保存完成')
  ############使用 pDPP 模型评估############
  pdpp_res = {}
  logging.critical(f'开始使用 PDPP 方法计算推荐结果')
  shannon_entropies = get_shannon_entropies(histories)
  fus = get_diversity_parameter(shannon_entropies, H0=1)
  for (n, d, k) in exp_list:
    constraint = constraints[0, :d]
    candidate = candidates[:n, :d]
    fu = fus[0]
    indexs, dcg, div, rmdse = dpp_eva(d, k, constraint, candidate, historiesList[:, :d], fu=fu)
    pdpp_res[f'dpp_{n}_{d}_{k}'] = {
      "n": n,
      "d": d,
      "k": k,
      "indexs": indexs,
      "dcg": dcg,
      "div": div,
      "rmdse": rmdse
    }
  logging.critical(f'PDPP 计算完成， 结果共 {len(pdpp_res)} 条')
  pd.DataFrame(pdpp_res).to_json(f'data/pdpp_res_{get_local_time()}.json')
  logging.critical("PDPP 结果保存完成")
