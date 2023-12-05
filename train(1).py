# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training script."""
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import functools
import gc
import time
import json
from absl import app
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
from internal import configs
from internal import datasets
from internal import image
from internal import models
from internal import train_utils
from internal import utils
from internal import vis
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from pathlib import Path, PurePath
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted

configs.define_common_flags()
jax.config.parse_flags_with_absl()

TIME_PRECISION = 1000  # Internally represent integer times in milliseconds.
PATCH_SIZE = 16
NUM_PATCHES = 2000

def patch_apart(image_height, image_width, patch_size, residual):
  '''
  把residual图像分成tiny patch
  image_height, image_width: 输入residual图像的高、宽
  patch_size: patch大小
  '''
  ###############
  # 生成固定的patch坐标，固定是为了方便输出residual的mask
  # 计算patch的行数和列数
  num_patch_rows = image_height // patch_size
  num_patch_cols = image_width // patch_size
  # 生成所有可能的patch的左上角坐标
  rows = np.arange(num_patch_rows) * patch_size
  cols = np.arange(num_patch_cols) * patch_size
  # 使用NumPy的meshgrid函数生成所有可能的patch的左上角坐标
  patch_rows, patch_cols = np.meshgrid(rows, cols, indexing='ij')
  # 将patch的行坐标和列坐标展平，得到所有patch的左上角坐标
  patch_coordinates = np.vstack((patch_cols.ravel(), patch_rows.ravel())).T
  
  # 固定坐标的 patch 坐标
  fixed_patch_coordinates = np.array(list(patch_coordinates))

  # 计算对应的切片索引
  x_indices = (fixed_patch_coordinates[:, 0])[:, np.newaxis] + np.arange(patch_size)[np.newaxis, :]
  y_indices = (fixed_patch_coordinates[:, 1])[:, np.newaxis] + np.arange(patch_size)[np.newaxis, :]

  # 提取小图像块
  residual_patches = residual[:, y_indices[:, :, np.newaxis], x_indices[:, np.newaxis, :]]
  return residual_patches    

def reconstruct_patch(image_height, image_width, patch_size, shape0, reshaped_prob):
  '''
  image_height, image_width: 输入residual图像的高、宽
  patch_size: patch大小
  shape0: residual有几个patch, 取决于卡数量和batch size
  reshaped_prob: 要 reshape mask 的概率值
  '''
  # 16个patch。每个patch里有若干小patch
  length = reshaped_prob.shape[0]
  subset_data = reshaped_prob.reshape(shape0, length//shape0)

  num_patch_rows = image_height // patch_size
  num_patch_cols = image_width // patch_size
  # 生成所有可能的patch的左上角坐标
  rows = np.arange(num_patch_rows) * patch_size
  cols = np.arange(num_patch_cols) * patch_size
  # 使用NumPy的meshgrid函数生成所有可能的patch的左上角坐标
  patch_rows, patch_cols = np.meshgrid(rows, cols, indexing='ij')
  # 将patch的行坐标和列坐标展平，得到所有patch的左上角坐标
  patch_coordinates = np.vstack((patch_cols.ravel(), patch_rows.ravel())).T

  # 将还原的图像块放入对应的位置
  _patch = np.zeros((shape0, image_height, image_width))  # 初始化一个空白图像
  for i in range(len(patch_coordinates)):
      x, y = patch_coordinates[i]
      tiny_patch = np.tile(subset_data[:, i][...,np.newaxis, np.newaxis], (patch_size, patch_size))
      _patch[:, y:y+patch_size, x:x+patch_size] = tiny_patch
  return _patch
      
def pixel_residual(postprocess_fn, rendering, test_case):
  # residual = (jnp.abs(postprocess_fn(rendering['rgb']) - postprocess_fn(test_case.rgb))).mean(axis=-1).reshape(-1, 1) # l1
  residual = ((postprocess_fn(rendering['rgb']) - postprocess_fn(test_case.rgb))**2).mean(axis=-1).reshape(-1, 1)  # l2
  return residual
#####################################################
def rgbchannel_pixel_residual(postprocess_fn, rendering, test_case):
  residual = ((postprocess_fn(rendering['rgb']) - postprocess_fn(test_case.rgb))**2).reshape(-1, 3)  # l2
  return residual
#####################################################

def patch_residual(postprocess_fn, rendering, test_case):
  residual = ((postprocess_fn(rendering['rgb']) - postprocess_fn(test_case.rgb))**2)  # l2
  # 定义图像的大小
  image_height, image_width = test_case.rgb.shape[-3:-1] ## 这里长宽可能会反，需要检查
  # 定义小图像块的大小
  patch_size = PATCH_SIZE
  # 定义想要选择的小图像块的数量
  num_patches = NUM_PATCHES   ### 432/16=27, 27**2=729
  
  ###############
  # 生成固定的patch坐标，固定是为了方便输出residual的mask
  # 计算patch的行数和列数
  num_patch_rows = image_height // patch_size
  num_patch_cols = image_width // patch_size
  # 生成所有可能的patch的左上角坐标
  rows = np.arange(num_patch_rows) * patch_size
  cols = np.arange(num_patch_cols) * patch_size
  # 使用NumPy的meshgrid函数生成所有可能的patch的左上角坐标
  patch_rows, patch_cols = np.meshgrid(rows, cols, indexing='ij')
  # 将patch的行坐标和列坐标展平，得到所有patch的左上角坐标
  patch_coordinates = np.vstack((patch_cols.ravel(), patch_rows.ravel())).T
  ###############
  
  # 生成随机的patch坐标
  random_y = np.random.randint(0, image_height - patch_size, size=num_patches)
  random_x = np.random.randint(0, image_width - patch_size, size=num_patches)
  # 从原始图像中提取小图像块
  # residual_patches = [residual[y:y + patch_size, x:x + patch_size, :] for x, y in patch_coordinates]
  # residual_patches += [residual[y:y + patch_size, x:x + patch_size, :] for y, x in zip(random_y, random_x)]
  # residual_patches = jnp.stack(residual_patches)
  
  ###################################################################################
  # 固定坐标的 patch 坐标
  fixed_patch_coordinates = np.array(list(patch_coordinates))

  # 随机坐标的 patch 坐标
  random_patch_coordinates = np.column_stack((random_x, random_y))

  # 合并固定和随机坐标的 patch 坐标
  all_patch_coordinates = np.vstack((fixed_patch_coordinates, random_patch_coordinates))

  # 计算对应的切片索引
  x_indices = (all_patch_coordinates[:, 0])[:, np.newaxis] + np.arange(patch_size)[np.newaxis, :]
  y_indices = (all_patch_coordinates[:, 1])[:, np.newaxis] + np.arange(patch_size)[np.newaxis, :]

  # 提取小图像块
  residual_patches = residual[y_indices[:, :, np.newaxis], x_indices[:, np.newaxis, :]]

  # print("residual_patches.shape: ", residual_patches.shape)
  # print("residual_patches_.shape: ", residual_patches_.shape)
  # assert jnp.allclose(residual_patches, residual_patches_), "residual_patches != residual_patches_"
  #####################################################################################
  
  residual_patches = residual_patches.mean(axis=(1, 2, 3)).reshape(-1, 1)  # l2
  residual = residual_patches
  return residual
######################################################

def direct_gmm_fitting(all_residual, gmm_start_time):
  # gmm = GaussianMixture(n_components=2, random_state=0)
  gmm_model = GaussianMixture(n_components=2, max_iter=100,tol=1e-2,reg_covar=5e-4)
  gmm_model.fit(all_residual.reshape(-1, 1))
  ## 计算gmm的均值 方差
  gmm_means = gmm_model.means_
  gmm_covariances = gmm_model.covariances_
  gmm_weights = gmm_model.weights_
  gmm_precisions_cholesky = gmm_model.precisions_cholesky_
  # gmm_covariance_type = gmm_model.covariance_type
  str_to_num = {'full': 0, 'tied': 1, 'diag': 2, 'spherical': 3}
  assert gmm_model.covariance_type=='full', "gmm_model.covariance_type!='full'"
  gmm_covariance_type = str_to_num[gmm_model.covariance_type]
  
  check_is_fitted(gmm_model)
  # print(f'gmm1 computed in {(time.time() - gmm_start_time):0.3f}s')
  print("gmm_means: ", gmm_means)
  # print("gmm_covariances: ", gmm_covariances)
  print("gmm_weights: ", gmm_weights)
  # print("gmm_precisions_cholesky: ", gmm_precisions_cholesky)
  # print("gmm_covariance_type: ", gmm_model.covariance_type)
  return gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type
########################################################

def accumulate_gmm_fitting(all_residual, gmm_start_time):
  # 对residual的分布划分成小区间进行统计。
  # 定义区间宽度
  interval_width = 0.001
  # 计算区间数量
  num_intervals = int(1 / interval_width)
  # 初始化区间内元素总和列表
  all_interval_sums = np.zeros(num_intervals)
  all_hist, all_bin_edges = np.histogram(all_residual, bins=np.linspace(0, 1, num_intervals+1))
  # 使用np.digitize将元素分配到区间并计算区间内元素总和
  all_indices = np.digitize(all_residual, all_bin_edges)
  for i in range(1, num_intervals+1):
      all_interval_sums[i-1] += np.sum(all_residual[all_indices == i])
  ### 把每个区间的residual的求和(sum)转换成数量。这里对每个区间直接取整，作为数量，可以理解为对residual大的像素个数进行了放大。
  all_interval_sums = all_interval_sums.reshape(-1, 1)
  ids = interval_width*(np.array(range(1,(num_intervals+1)))-0.5).reshape(-1, 1)
  # numbers = (all_interval_sums/interval_width).astype(int)
  numbers = (all_interval_sums).astype(int)
  statistic = []
  for i, id in enumerate(ids):
      tmp = id * np.ones(numbers[i]).reshape(-1,1)
      statistic.append(tmp)
  statistic_all = np.concatenate(statistic)
  # 拟合GMM模型
  # gmm = GaussianMixture(n_components=2, random_state=0)
  gmm_model = GaussianMixture(n_components=2, max_iter=100,tol=1e-2,reg_covar=5e-4)
  statistic_all = statistic_all.reshape(-1, 1)
  if statistic_all.shape[0] > 20142016:
    statistic_all = statistic_all[np.random.randint(low=0, high=statistic_all.shape[0], size=20142016)]
  gmm_model.fit(statistic_all)
  ## 计算gmm的均值 方差
  gmm_means = gmm_model.means_
  gmm_covariances = gmm_model.covariances_
  gmm_weights = gmm_model.weights_
  gmm_precisions_cholesky = gmm_model.precisions_cholesky_
  # gmm_covariance_type = gmm_model.covariance_type
  str_to_num = {'full': 0, 'tied': 1, 'diag': 2, 'spherical': 3}
  assert gmm_model.covariance_type=='full', "gmm_model.covariance_type!='full'"
  gmm_covariance_type = str_to_num[gmm_model.covariance_type]
  check_is_fitted(gmm_model)
  print(f'gmm2 computed in {(time.time() - gmm_start_time):0.3f}s')
  return gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type
###########################################################

def image_direct_gmm_fitting(all_residual_, gmm_start_time):
  str_to_num = {'full': 0, 'tied': 1, 'diag': 2, 'spherical': 3}
  gmm_means = []
  gmm_covariances = []
  gmm_weights = []
  gmm_precisions_cholesky = []
  gmm_covariance_type = []
  for i in range(all_residual_.shape[0]):  
    gmm_model = GaussianMixture(n_components=2, max_iter=100,tol=1e-2,reg_covar=5e-4)
    gmm_model.fit(all_residual_[i].reshape(-1, 1))
    ## 计算gmm的均值 方差
    gmm_means.append(gmm_model.means_)
    gmm_covariances.append(gmm_model.covariances_)
    gmm_weights.append(gmm_model.weights_)
    gmm_precisions_cholesky.append(gmm_model.precisions_cholesky_)
    assert gmm_model.covariance_type=='full', "gmm_model.covariance_type!='full'"
    gmm_covariance_type.append(str_to_num[gmm_model.covariance_type])
    check_is_fitted(gmm_model)
    print(f'gmm1 computed in {(time.time() - gmm_start_time):0.3f}s')
    print("gmm_means: ", gmm_means)
    print("gmm_covariances: ", gmm_covariances)
    print("gmm_weights: ", gmm_weights)
    print("gmm_precisions_cholesky: ", gmm_precisions_cholesky)
    print("gmm_covariance_type: ", gmm_model.covariance_type)
  
  gmm_means = jnp.stack(gmm_means)
  gmm_covariances = jnp.stack(gmm_covariances)
  gmm_weights = jnp.stack(gmm_weights)
  gmm_precisions_cholesky = jnp.stack(gmm_precisions_cholesky)
  gmm_covariance_type = jnp.stack(gmm_covariance_type)
  
  return gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type

###########################################################
def image_accumulate_gmm_fitting(all_residual_, gmm_start_time):
  str_to_num = {'full': 0, 'tied': 1, 'diag': 2, 'spherical': 3}
  # 对residual的分布划分成小区间进行统计。
  # 定义区间宽度
  interval_width = 0.001
  # 计算区间数量
  num_intervals = int(1 / interval_width)
  
  gmm_means = []
  gmm_covariances = []
  gmm_weights = []
  gmm_precisions_cholesky = []
  gmm_covariance_type = []
  for i in range(all_residual_.shape[0]):  
    # 初始化区间内元素总和列表
    all_interval_sums = np.zeros(num_intervals)
    all_hist, all_bin_edges = np.histogram(all_residual_[i], bins=np.linspace(0, 1, num_intervals+1))
    # 使用np.digitize将元素分配到区间并计算区间内元素总和
    all_indices = np.digitize(all_residual_[i], all_bin_edges)
    for j in range(1, num_intervals+1):
      all_interval_sums[j-1] += np.sum(all_residual_[i][all_indices == j])
    ### 把每个区间的residual的求和(sum)转换成数量。这里对每个区间直接取整，作为数量，可以理解为对residual大的像素个数进行了放大。
    all_interval_sums = all_interval_sums.reshape(-1, 1)
    ids = interval_width*(np.array(range(1,(num_intervals+1)))-0.5).reshape(-1, 1)
    numbers = (all_interval_sums/interval_width).astype(int)
    statistic = []
    for k, id in enumerate(ids):
        tmp = id * np.ones(numbers[k]).reshape(-1,1)
        statistic.append(tmp)
    statistic_all = np.concatenate(statistic)
    # 拟合GMM模型
    # gmm = GaussianMixture(n_components=2, random_state=0)
    gmm_model = GaussianMixture(n_components=2, max_iter=100,tol=1e-2,reg_covar=5e-4)
    gmm_model.fit(statistic_all.reshape(-1, 1))
    ## 计算gmm的均值 方差
    gmm_means.append(gmm_model.means_)
    gmm_covariances.append(gmm_model.covariances_)
    gmm_weights.append(gmm_model.weights_)
    gmm_precisions_cholesky.append(gmm_model.precisions_cholesky_)
    assert gmm_model.covariance_type=='full', "gmm_model.covariance_type!='full'"
    gmm_covariance_type.append(str_to_num[gmm_model.covariance_type])
    check_is_fitted(gmm_model)
    print(f'gmm1 computed in {(time.time() - gmm_start_time):0.3f}s')
    print("gmm_means: ", gmm_means)
    print("gmm_covariances: ", gmm_covariances)
    print("gmm_weights: ", gmm_weights)
    print("gmm_precisions_cholesky: ", gmm_precisions_cholesky)
    print("gmm_covariance_type: ", gmm_model.covariance_type)
  
  gmm_means = jnp.stack(gmm_means)
  gmm_covariances = jnp.stack(gmm_covariances)
  gmm_weights = jnp.stack(gmm_weights)
  gmm_precisions_cholesky = jnp.stack(gmm_precisions_cholesky)
  gmm_covariance_type = jnp.stack(gmm_covariance_type)
  
  return gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type
###########################################################

def rgbchannel_direct_gmm_fitting(all_residual, gmm_start_time):
  str_to_num = {'full': 0, 'tied': 1, 'diag': 2, 'spherical': 3}
  gmm_means = []
  gmm_covariances = []
  gmm_weights = []
  gmm_precisions_cholesky = []
  gmm_covariance_type = []
  for i in range(all_residual.shape[-1]):  ## 应该是 3 才对
    gmm_model = GaussianMixture(n_components=2, max_iter=100,tol=1e-2,reg_covar=5e-4)
    gmm_model.fit(all_residual[..., i].reshape(-1, 1))
    ## 计算gmm的均值 方差
    gmm_means.append(gmm_model.means_)
    gmm_covariances.append(gmm_model.covariances_)
    gmm_weights.append(gmm_model.weights_)
    gmm_precisions_cholesky.append(gmm_model.precisions_cholesky_)
    assert gmm_model.covariance_type=='full', "gmm_model.covariance_type!='full'"
    gmm_covariance_type.append(str_to_num[gmm_model.covariance_type])
    check_is_fitted(gmm_model)
    print(f'gmm1 computed in {(time.time() - gmm_start_time):0.3f}s')
    print("gmm_means: ", gmm_means)
    print("gmm_covariances: ", gmm_covariances)
    print("gmm_weights: ", gmm_weights)
    print("gmm_precisions_cholesky: ", gmm_precisions_cholesky)
    print("gmm_covariance_type: ", gmm_model.covariance_type)
  
  gmm_means = jnp.stack(gmm_means)
  gmm_covariances = jnp.stack(gmm_covariances)
  gmm_weights = jnp.stack(gmm_weights)
  gmm_precisions_cholesky = jnp.stack(gmm_precisions_cholesky)
  gmm_covariance_type = jnp.stack(gmm_covariance_type)
  
  return gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type  # 应该都是3个元素才对

###########################################################
def rgbchannel_accumulate_gmm_fitting(all_residual, gmm_start_time):
  str_to_num = {'full': 0, 'tied': 1, 'diag': 2, 'spherical': 3}
  # 对residual的分布划分成小区间进行统计。
  # 定义区间宽度
  interval_width = 0.001
  # 计算区间数量
  num_intervals = int(1 / interval_width)
  
  gmm_means = []
  gmm_covariances = []
  gmm_weights = []
  gmm_precisions_cholesky = []
  gmm_covariance_type = []
  for i in range(all_residual.shape[-1]):  ## 应该是 3 才对
    # 初始化区间内元素总和列表
    all_interval_sums = np.zeros(num_intervals)
    all_hist, all_bin_edges = np.histogram(all_residual[..., i].reshape(-1, 1), bins=np.linspace(0, 1, num_intervals+1))
    # 使用np.digitize将元素分配到区间并计算区间内元素总和
    all_indices = np.digitize(all_residual[..., i].reshape(-1, 1), all_bin_edges)
    for j in range(1, num_intervals+1):
      all_interval_sums[j-1] += np.sum(all_residual[..., i].reshape(-1, 1)[all_indices == j])
    ### 把每个区间的residual的求和(sum)转换成数量。这里对每个区间直接取整，作为数量，可以理解为对residual大的像素个数进行了放大。
    all_interval_sums = all_interval_sums.reshape(-1, 1)
    ids = interval_width*(np.array(range(1,(num_intervals+1)))-0.5).reshape(-1, 1)
    # numbers = (all_interval_sums/interval_width).astype(int)
    numbers = (all_interval_sums).astype(int)
    statistic = []
    statistic_all = None
    statistic = []
    for k, id in enumerate(ids):
        tmp = id * np.ones(numbers[k]).reshape(-1,1)
        statistic.append(tmp)
    statistic_all = np.concatenate(statistic)
    # 拟合GMM模型
    # gmm = GaussianMixture(n_components=2, random_state=0)
    gmm_model = GaussianMixture(n_components=2, max_iter=100,tol=1e-2,reg_covar=5e-4)
    statistic_all = statistic_all.reshape(-1, 1)
    print("statistic_all.shape: ", statistic_all.shape)
    if statistic_all.shape[0] > 20142016:
      statistic_all = statistic_all[np.random.randint(low = 0, high = statistic_all.shape[0], size = 20142016)]
      print("postprocessing statistic_all.shape: ", statistic_all.shape)
    gmm_model.fit(statistic_all)
    # gmm_model.fit(statistic_all.reshape(-1, 1))
    ## 计算gmm的均值 方差
    gmm_means.append(gmm_model.means_)
    gmm_covariances.append(gmm_model.covariances_)
    gmm_weights.append(gmm_model.weights_)
    gmm_precisions_cholesky.append(gmm_model.precisions_cholesky_)
    assert gmm_model.covariance_type=='full', "gmm_model.covariance_type!='full'"
    gmm_covariance_type.append(str_to_num[gmm_model.covariance_type])
    check_is_fitted(gmm_model)
    print(f'gmm1 computed in {(time.time() - gmm_start_time):0.3f}s')
    print("gmm_means: ", gmm_means)
    print("gmm_covariances: ", gmm_covariances)
    print("gmm_weights: ", gmm_weights)
    print("gmm_precisions_cholesky: ", gmm_precisions_cholesky)
    print("gmm_covariance_type: ", gmm_model.covariance_type)
  
  gmm_means = jnp.stack(gmm_means)
  gmm_covariances = jnp.stack(gmm_covariances)
  gmm_weights = jnp.stack(gmm_weights)
  gmm_precisions_cholesky = jnp.stack(gmm_precisions_cholesky)
  gmm_covariance_type = jnp.stack(gmm_covariance_type)
  
  return gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type # 应该都是3个元素才对
###########################################################


def main(unused_argv):
  rng = random.PRNGKey(20200823)
  # Shift the numpy random seed by host_id() to shuffle data loaded by different
  # hosts.
  np.random.seed(20201473 + jax.host_id())

  config = configs.load_config()

  if config.batch_size % jax.device_count() != 0:
    raise ValueError('Batch size must be divisible by the number of devices.')

  dataset = datasets.load_dataset('train', config.data_dir, config)
  test_dataset = datasets.load_dataset('test', config.data_dir, config)
  residual_dataset = datasets.load_dataset('residual', config.data_dir, config)

  np_to_jax = lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x
  cameras = tuple(np_to_jax(x) for x in dataset.cameras)

  if config.rawnerf_mode:
    postprocess_fn = test_dataset.metadata['postprocess_fn']
  else:
    postprocess_fn = lambda z, _=None: z

  rng, key = random.split(rng)
  setup = train_utils.setup_model(config, key, dataset=dataset)
  model, state, render_eval_pfn, train_pstep, lr_fn = setup

  variables = state.params
  num_params = jax.tree_util.tree_reduce(
      lambda x, y: x + jnp.prod(jnp.array(y.shape)), variables, initializer=0)
  print(f'Number of parameters being optimized: {num_params}')

  if (dataset.size > model.num_glo_embeddings and model.num_glo_features > 0):
    raise ValueError(f'Number of glo embeddings {model.num_glo_embeddings} '
                     f'must be at least equal to number of train images '
                     f'{dataset.size}')

  metric_harness = image.MetricHarness()

  if not utils.isdir(config.checkpoint_dir):
    utils.makedirs(config.checkpoint_dir)
  state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
  # Resume training at the step of the last checkpoint.
  init_step = state.step + 1
  state = flax.jax_utils.replicate(state)

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(config.checkpoint_dir)
    if config.rawnerf_mode:
      for name, data in zip(['train', 'test'], [dataset, test_dataset]):
        # Log shutter speed metadata in TensorBoard for debug purposes.
        for key in ['exposure_idx', 'exposure_values', 'unique_shutters']:
          summary_writer.text(f'{name}_{key}', str(data.metadata[key]), 0)

  # Prefetch_buffer_size = 3 x batch_size.
  pdataset = flax.jax_utils.prefetch_to_device(dataset, 3)
  rng = rng + jax.host_id()  # Make random seed separate across hosts.
  rngs = random.split(rng, jax.local_device_count())  # For pmapping RNG keys.
  gc.disable()  # Disable automatic garbage collection for efficiency.
  total_time = 0
  total_steps = 0
  reset_stats = True
  if config.early_exit_steps is not None:
    num_steps = config.early_exit_steps
  else:
    num_steps = config.max_steps
  if config.varied_threshold or config.always_varied_threshold:
    loss_threshold = config.start_threshold
  else:
    loss_threshold = 1.0
  # all_residual = None
  gmm_means = None
  gmm_covariances = None
  gmm_weights = None
  gmm_precisions_cholesky = None
  gmm_covariance_type = None
  
  gmm_means_patch16 = None
  gmm_covariances_patch16 = None
  gmm_weights_patch16 = None
  gmm_precisions_cholesky_patch16 = None
  gmm_covariance_type_patch16 = None
  
  gmm_means_patch8 = None
  gmm_covariances_patch8 = None
  gmm_weights_patch8 = None
  gmm_precisions_cholesky_patch8 = None
  gmm_covariance_type_patch8 = None
  
  gmm_means_patch4 = None
  gmm_covariances_patch4 = None
  gmm_weights_patch4 = None
  gmm_precisions_cholesky_patch4 = None
  gmm_covariance_type_patch4 = None
  
  gmm_means_pixel= None
  gmm_covariances_pixel = None
  gmm_weights_pixel = None
  gmm_precisions_cholesky_pixel = None
  gmm_covariance_type_pixel = None
  
  # 先选择 residual 计算模式
  # all_residual = []
  pixel_residual_flag = False
  patch_residual_flag = True
  rgbchannel_pixel_residual_flag = False
  # 再选择 gmm fitting 模式
  direct_gmm_fitting_flag = True
  accumulate_gmm_fitting_flag = False
  image_direct_gmm_fitting_flag = False
  image_accumulate_gmm_fitting_flag = False
  rgbchannel_direct_gmm_fitting_flag = False
  rgbchannel_accumulate_gmm_fitting_flag = False
  #######
  # residual_train_all = []
  residual_train_patch16_all = []
  residual_train_patch8_all = []
  residual_train_patch4_all = []
  residual_train_pixel_all = []
  
  ### 如果不是从init_step = 0开始训练的，那就需要先计算gmm模型参数。
  if init_step > 100  and config.test_residual:
    train_frac = jnp.clip((init_step - 1) / (config.max_steps - 1), 0, 1)
    '''
    在训练过程中统计residual的分布情况, 并拟合gmm_model。
    注意在拟合gmm过程中是不需要mask的, 模拟真实应用场景中没有对比图相对的情况。
    '''
    all_residual_dir = config.checkpoint_dir + '/rsidual_statistic/all/fit_gmm_model'
    if not os.path.exists(all_residual_dir):
      os.makedirs(all_residual_dir)

    # We reuse the same random number generator from the optimization step
    # here on purpose so that the visualization matches what happened in
    # training.
    eval_variables = flax.jax_utils.unreplicate(state).params
    statistic_start_time = time.time()
    
    ########
    if pixel_residual_flag:
      all_residual_ = np.zeros((residual_dataset.size, residual_dataset.height * residual_dataset.width, 1))
    elif patch_residual_flag:
      all_residual_ = np.zeros((residual_dataset.size, (residual_dataset.height//PATCH_SIZE) * (residual_dataset.width//PATCH_SIZE)+NUM_PATCHES, 1))
    elif rgbchannel_pixel_residual_flag:
      all_residual_ = np.zeros((residual_dataset.size, residual_dataset.height * residual_dataset.width, 3))
      
      
    for i in range(residual_dataset.size):
    # for i in range(6):
      train_case = next(residual_dataset)
      rendering = models.render_image(
        functools.partial(render_eval_pfn, eval_variables, train_frac),
        train_case.rays, rngs[0], config)
      
      #####################################################################
      # 统计方法1：统计所有的residual 这是拉直的，不是patch
      if pixel_residual_flag:
        residual = pixel_residual(postprocess_fn, rendering, train_case)
        
      
      #####################################################################
      # 统计方法2：采用patch计算residual
      elif patch_residual_flag:
        residual = patch_residual(postprocess_fn, rendering, train_case)
      
      #####################################################################
      # 统计方法3：统计所有的residual,但是分rgb3个通道
      elif rgbchannel_pixel_residual_flag:
        residual = rgbchannel_pixel_residual(postprocess_fn, rendering, train_case)
        
      
        
      ######################################################################
      
      # 比例处理方法1：
      if False:
        '''
        不同的residual计算方法可能会有不同的效果, 因为clean和noisy 像素数量差别太大了。
        '''
        residual = -jnp.log(1-residual)*residual
      ######################################
      
      # all_residual.append(residual)
      all_residual_[train_case.cam_idx] = residual
      print(f'calculate No.{train_case.cam_idx} residual!')
      
    all_residual = jnp.concatenate(all_residual_)
    # all_interval_sums_file_name = f'all_training_{step}_steps_all_residual.npy'
    # np.save(os.path.join(all_residual_dir, all_interval_sums_file_name), all_residual)
    # assert all_residual.shape[-1]==1 , "all_residual shape is wrong!!!!"
    print(f'all_residual shape:', all_residual.shape)
    print(f'Statistics computed in {(time.time() - statistic_start_time):0.3f}s')
    
    gmm_start_time = time.time()
    
    ##############################################################################
    # 拟合GMM模型 方法1：直接使用所有pixel的residual：
    if direct_gmm_fitting_flag:
      gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type = direct_gmm_fitting(all_residual, gmm_start_time)
    
    ##############################################################################
    # 拟合GMM模型 方法2：根据residual在不同小区间内的累加和，转换成数量：
    elif accumulate_gmm_fitting_flag:
      gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type = accumulate_gmm_fitting(all_residual, gmm_start_time)
    
    ##############################################################################
    # 拟合GMM模型 方法3：根据每张图片的 residual_ 分别对每张图估计GMM模型：
    elif image_direct_gmm_fitting_flag:
      gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type = image_direct_gmm_fitting(all_residual_, gmm_start_time)
    
    # 拟合GMM模型 方法4：根据每张图片的 residual_ ,先计算统计量，然后分别对每张图估计GMM模型：
    elif image_accumulate_gmm_fitting_flag:
      gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type = image_accumulate_gmm_fitting(all_residual_, gmm_start_time)
    
    # 拟合GMM模型 方法5：根据每张图片的 rgb 3通道的residual 分别对每张图估计GMM模型：
    elif rgbchannel_direct_gmm_fitting_flag:
      gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type = rgbchannel_direct_gmm_fitting(all_residual_, gmm_start_time)
    
    # 拟合GMM模型 方法6：根据每张图片的 rgb 3通道的residual 先计算统计量，然后分别对每张图估计GMM模型：
    elif rgbchannel_accumulate_gmm_fitting_flag:
      gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type = rgbchannel_accumulate_gmm_fitting(all_residual_, gmm_start_time)
  
  if config.train_residual and init_step > 100:
    gmm_coef_dir = config.checkpoint_dir + '/gmm_coef/train'
    if not os.path.exists(gmm_coef_dir):
      os.makedirs(gmm_coef_dir)
    # 指定 JSON 文件路径
    json_file_path = gmm_coef_dir + '/' + f'all_training_{(init_step - 1)}_steps_gmm_coefficient.json'

    # 读取 JSON 文件
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    # 从 JSON 数据中获取对应的 NumPy 数组
    gmm_means_patch16 = np.array(json_data['gmm_means_patch16'])
    gmm_covariances_patch16 = np.array(json_data['gmm_covariances_patch16'])
    gmm_weights_patch16 = np.array(json_data['gmm_weights_patch16'])
    gmm_precisions_cholesky_patch16 = np.array(json_data['gmm_precisions_cholesky_patch16'])
    gmm_covariance_type_patch16 = int(json_data['gmm_covariance_type_patch16'])
    
    gmm_means_patch8 = np.array(json_data['gmm_means_patch8'])
    gmm_covariances_patch8 = np.array(json_data['gmm_covariances_patch8'])
    gmm_weights_patch8 = np.array(json_data['gmm_weights_patch8'])
    gmm_precisions_cholesky_patch8 = np.array(json_data['gmm_precisions_cholesky_patch8'])
    gmm_covariance_type_patch8 = int(json_data['gmm_covariance_type_patch8'])
    
    gmm_means_patch4 = np.array(json_data['gmm_means_patch4'])
    gmm_covariances_patch4 = np.array(json_data['gmm_covariances_patch4'])
    gmm_weights_patch4 = np.array(json_data['gmm_weights_patch4'])
    gmm_precisions_cholesky_patch4 = np.array(json_data['gmm_precisions_cholesky_patch4'])
    gmm_covariance_type_patch4 = int(json_data['gmm_covariance_type_patch4'])
    
    gmm_means_pixel = np.array(json_data['gmm_means_pixel'])
    gmm_covariances_pixel = np.array(json_data['gmm_covariances_pixel'])
    gmm_weights_pixel = np.array(json_data['gmm_weights_pixel'])
    gmm_precisions_cholesky_pixel = np.array(json_data['gmm_precisions_cholesky_pixel'])
    gmm_covariance_type_pixel = int(json_data['gmm_covariance_type_pixel'])
    
    print("train from checkpoints!!")
    ##############################################################################
    
  for step, batch in zip(range(init_step, num_steps + 1), pdataset):

    if reset_stats and (jax.host_id() == 0):
      stats_buffer = []
      train_start_time = time.time()
      reset_stats = False

    learning_rate = lr_fn(step)
    train_frac = jnp.clip((step - 1) / (config.max_steps - 1), 0, 1)
    
    ### 如果是varied_threshold=Ture，就需要计算 loss_threshold 的值:
    if config.always_varied_threshold and config.train_residual: # 一直变化 threshold
      start_threshold = config.start_threshold
      final_threshold = config.final_threshold
      varied_max_steps = config.varied_max_steps
      if step < varied_max_steps:
        loss_threshold = start_threshold + (final_threshold - start_threshold) * (step - 1) / (varied_max_steps - 1)
      else:
        loss_threshold = final_threshold
        
    elif config.varied_threshold and config.train_residual and config.fit_gmm_train_residual_every > 0 and step % config.fit_gmm_train_residual_every == 0: # fit_gmm_train_residual_every频率
      start_threshold = config.start_threshold
      final_threshold = config.final_threshold
      varied_max_steps = config.varied_max_steps
      if step < varied_max_steps:
        loss_threshold = start_threshold + (final_threshold - start_threshold) * (step - 1) / (varied_max_steps - 1)
      else:
        loss_threshold = final_threshold

    if config.test_residual:
      state, stats, rngs = train_pstep(
          rngs,
          state,
          batch,
          cameras,
          train_frac,
          loss_threshold,
          gmm_means, 
          gmm_covariances, 
          gmm_weights,
          gmm_precisions_cholesky,
          gmm_covariance_type
      )
    elif config.train_residual:
      state, stats, rngs = train_pstep(
          rngs,
          state,
          batch,
          cameras,
          train_frac,
          loss_threshold,
          gmm_means_patch16, 
          gmm_covariances_patch16, 
          gmm_weights_patch16,
          gmm_precisions_cholesky_patch16, 
          gmm_covariance_type_patch16,
          gmm_means_patch8, 
          gmm_covariances_patch8, 
          gmm_weights_patch8, 
          gmm_precisions_cholesky_patch8, 
          gmm_covariance_type_patch8, 
          gmm_means_patch4, 
          gmm_covariances_patch4, 
          gmm_weights_patch4, 
          gmm_precisions_cholesky_patch4, 
          gmm_covariance_type_patch4, 
          gmm_means_pixel, 
          gmm_covariances_pixel, 
          gmm_weights_pixel, 
          gmm_precisions_cholesky_pixel, 
          gmm_covariance_type_pixel
      )
      
    if config.enable_robustnerf_loss and 'loss_threshold' in stats.keys():
        loss_threshold = jnp.mean(stats['loss_threshold'])

    # ######## 从训练代码中拿出residual：
    if config.data_loss_type == 'gmmrobust' and config.train_residual and step > (config.pretrain_steps-1):
      residual_train = stats['data_resid_sq'][0]
      residual_train_patch16 = residual_train.mean(axis=(1, 2, 3)).reshape(-1, 1)
      residual_train_patch16_all.append(residual_train_patch16)
      
      
      residual_train_patch8 = patch_apart(16, 16, 8, residual_train).reshape(-1, 8, 8, 3)
      residual_train_patch8 = residual_train_patch8.mean(axis=(1, 2, 3)).reshape(-1, 1)
      residual_train_patch8_all.append(residual_train_patch8)
      
      residual_train_patch4 = patch_apart(16, 16, 4, residual_train).reshape(-1, 4, 4, 3)
      residual_train_patch4 = residual_train_patch4.mean(axis=(1, 2, 3)).reshape(-1, 1)
      residual_train_patch4_all.append(residual_train_patch4)
      
      residual_train_pixel = residual_train.mean(axis=-1).reshape(-1, 1)
      residual_train_pixel_all.append(residual_train_pixel)
      
      wyk = reconstruct_patch(16, 16, 4, residual_train.shape[0], residual_train_patch4)
      # if 'mask_patch8' in stats.keys():
      #   rrr = stats['mask_patch8'][0]
    ########
    
    if step % config.gc_every == 0:
      gc.collect()  # Disable automatic garbage collection for efficiency.

    # Log training summaries. This is put behind a host_id check because in
    # multi-host evaluation, all hosts need to run inference even though we
    # only use host 0 to record results.
    if jax.host_id() == 0:
      stats = flax.jax_utils.unreplicate(stats)

      stats_buffer.append(stats)

      if step == init_step or step % config.print_every == 0:
        elapsed_time = time.time() - train_start_time
        steps_per_sec = config.print_every / elapsed_time
        rays_per_sec = config.batch_size * steps_per_sec

        # A robust approximation of total training time, in case of pre-emption.
        total_time += int(round(TIME_PRECISION * elapsed_time))
        total_steps += config.print_every
        approx_total_time = int(round(step * total_time / total_steps))

        # Transpose and stack stats_buffer along axis 0.
        fs = [flax.traverse_util.flatten_dict(s, sep='/') for s in stats_buffer]
        stats_stacked = {k: jnp.stack([f[k] for f in fs]) for k in fs[0].keys()}

        # Split every statistic that isn't a vector into a set of statistics.
        stats_split = {}
        for k, v in stats_stacked.items():
          if v.ndim not in [1, 2] and v.shape[0] != len(stats_buffer):
            raise ValueError('statistics must be of size [n], or [n, k].')
          if v.ndim == 1:
            stats_split[k] = v
          elif v.ndim == 2:
            for i, vi in enumerate(tuple(v.T)):
              stats_split[f'{k}/{i}'] = vi

        # Summarize the entire histogram of each statistic.
        for k, v in stats_split.items():
          summary_writer.histogram('train_' + k, v, step)

        # Take the mean and max of each statistic since the last summary.
        avg_stats = {k: jnp.mean(v) for k, v in stats_split.items()}
        max_stats = {k: jnp.max(v) for k, v in stats_split.items()}

        summ_fn = lambda s, v: summary_writer.scalar(s, v, step)  # pylint:disable=cell-var-from-loop

        # Summarize the mean and max of each statistic.
        for k, v in avg_stats.items():
          summ_fn(f'train_avg_{k}', v)
        for k, v in max_stats.items():
          summ_fn(f'train_max_{k}', v)

        summ_fn('train_num_params', num_params)
        summ_fn('train_learning_rate', learning_rate)
        summ_fn('train_steps_per_sec', steps_per_sec)
        summ_fn('train_rays_per_sec', rays_per_sec)

        summary_writer.scalar('train_avg_psnr_timed', avg_stats['psnr'],
                              total_time // TIME_PRECISION)
        summary_writer.scalar('train_avg_psnr_timed_approx', avg_stats['psnr'],
                              approx_total_time // TIME_PRECISION)
        ### 记录 gmm means 变化:
        if gmm_means_patch16 is not None and any(gmm_means_patch16):
          summary_writer.scalar('gmm_means_patch16_0', gmm_means_patch16[0, 0], step)
          summary_writer.scalar('gmm_means_patch16_1', gmm_means_patch16[1, 0], step)
        if gmm_means_patch8 is not None and any(gmm_means_patch8):
          summary_writer.scalar('gmm_means_patch8_0', gmm_means_patch8[0, 0], step)
          summary_writer.scalar('gmm_means_patch8_1', gmm_means_patch8[1, 0], step)
        if gmm_means_pixel is not None and any(gmm_means_pixel):
          summary_writer.scalar('gmm_means_pixel_0', gmm_means_pixel[0, 0], step)
          summary_writer.scalar('gmm_means_pixel_1', gmm_means_pixel[1, 0], step)
        ### 记录 loss_threshold:
        # if config.varied_threshold or config.always_varied_threshold:
        summary_writer.scalar('loss_threshold', loss_threshold, step)
        
        if dataset.metadata is not None and model.learned_exposure_scaling:
          params = state.params['params']
          scalings = params['exposure_scaling_offsets']['embedding'][0]
          num_shutter_speeds = dataset.metadata['unique_shutters'].shape[0]
          for i_s in range(num_shutter_speeds):
            for j_s, value in enumerate(scalings[i_s]):
              summary_name = f'exposure/scaling_{i_s}_{j_s}'
              summary_writer.scalar(summary_name, value, step)

        precision = int(np.ceil(np.log10(config.max_steps))) + 1
        avg_loss = avg_stats['loss']
        avg_psnr = avg_stats['psnr']
        str_losses = {  # Grab each "losses_{x}" field and print it as "x[:4]".
            k[7:11]: (f'{v:0.5f}' if v >= 1e-4 and v < 10 else f'{v:0.1e}')
            for k, v in avg_stats.items()
            if k.startswith('losses/')
        }
        print(f'{step:{precision}d}' + f'/{config.max_steps:d}: ' +
              f'loss={avg_loss:0.5f}, ' + f'psnr={avg_psnr:6.3f}, ' +
              f'lr={learning_rate:0.2e} | ' +
              ', '.join([f'{k}={s}' for k, s in str_losses.items()]) +
              f', {rays_per_sec:0.0f} r/s')

        # Reset everything we are tracking between summarizations.
        reset_stats = True

      if step == 1 or step % config.checkpoint_every == 0:
        state_to_save = jax.device_get(
            flax.jax_utils.unreplicate(state))
        checkpoints.save_checkpoint(
            config.checkpoint_dir, state_to_save, int(step), keep=100)

    # Test-set evaluation.
    if config.train_render_every > 0 and step % config.train_render_every == 0:
      # We reuse the same random number generator from the optimization step
      # here on purpose so that the visualization matches what happened in
      # training.
      eval_start_time = time.time()
      eval_variables = flax.jax_utils.unreplicate(state).params
      test_case = next(test_dataset)
      rendering = models.render_image(
          functools.partial(render_eval_pfn, eval_variables, train_frac),
          test_case.rays, rngs[0], config)

      # Log eval summaries on host 0.
      if jax.host_id() == 0:
        eval_time = time.time() - eval_start_time
        num_rays = jnp.prod(jnp.array(test_case.rays.directions.shape[:-1]))
        rays_per_sec = num_rays / eval_time
        summary_writer.scalar('test_rays_per_sec', rays_per_sec, step)
        print(f'Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec')

        metric_start_time = time.time()
        metric = metric_harness(
            postprocess_fn(rendering['rgb']), postprocess_fn(test_case.rgb))
        print(f'Metrics computed in {(time.time() - metric_start_time):0.3f}s')
        for name, val in metric.items():
          if not np.isnan(val):
            print(f'{name} = {val:.4f}')
            summary_writer.scalar('train_metrics/' + name, val, step)

        if config.vis_decimate > 1:
          d = config.vis_decimate
          decimate_fn = lambda x, d=d: None if x is None else x[::d, ::d]
        else:
          decimate_fn = lambda x: x
        rendering = jax.tree_util.tree_map(decimate_fn, rendering)
        test_case = jax.tree_util.tree_map(decimate_fn, test_case)
        vis_start_time = time.time()
        vis_suite = vis.visualize_suite(rendering, test_case.rays)
        print(f'Visualized in {(time.time() - vis_start_time):0.3f}s')
        if config.rawnerf_mode:
          # Unprocess raw output.
          vis_suite['color_raw'] = rendering['rgb']
          # Autoexposed colors.
          vis_suite['color_auto'] = postprocess_fn(rendering['rgb'], None)
          summary_writer.image('test_true_auto',
                               postprocess_fn(test_case.rgb, None), step)
          # Exposure sweep colors.
          exposures = test_dataset.metadata['exposure_levels']
          for p, x in list(exposures.items()):
            vis_suite[f'color/{p}'] = postprocess_fn(rendering['rgb'], x)
            summary_writer.image(f'test_true_color/{p}',
                                 postprocess_fn(test_case.rgb, x), step)
        summary_writer.image('test_true_color', test_case.rgb, step)
        if config.compute_normal_metrics:
          summary_writer.image('test_true_normals',
                               test_case.normals / 2. + 0.5, step)
        for k, v in vis_suite.items():
          summary_writer.image('test_output_' + k, v, step)
          
    ################################################################################### 统计residual
    if None:
    # if config.train_render_every > 0 and step % config.train_render_every == 0:
      '''
      根据已经算得的mask, 在训练过程中统计residual的分布情况。
      '''
      # 定义区间宽度
      interval_width = 0.01
      # 计算区间数量
      num_intervals = int(1 / interval_width)
      # 初始化区间内元素总和列表
      # interval_sums = [0] * num_intervals
      easy_interval_sums = np.zeros(num_intervals)
      # easy_hist_total = np.zeros(num_intervals)
      
      hard_interval_sums = np.zeros(num_intervals)
      # hard_hist_total = np.zeros(num_intervals)
      
      mask_dir = test_dataset.data_dir +'/noise_mask'
      assert os.path.exists(mask_dir), "mask_dir path doesn't exist!!!"
      easy_residual_dir = config.checkpoint_dir + '/rsidual_statistic/easy'
      if not os.path.exists(easy_residual_dir):
        os.makedirs(easy_residual_dir)
      hard_residual_dir = config.checkpoint_dir + '/rsidual_statistic/hard'
      if not os.path.exists(hard_residual_dir):
        os.makedirs(hard_residual_dir)
      
      # We reuse the same random number generator from the optimization step
      # here on purpose so that the visualization matches what happened in
      # training.
      eval_variables = flax.jax_utils.unreplicate(state).params
      statistic_start_time = time.time()
      
      for i in range(test_dataset.size):
        test_case = next(test_dataset)
        rendering = models.render_image(
          functools.partial(render_eval_pfn, eval_variables, train_frac),
          test_case.rays, rngs[0], config)
        
        mask_filename = PurePath(mask_dir+'/'+test_dataset.test_images_names[test_case.cam_idx]).with_suffix('.npy')
        mask = np.load(mask_filename)
        
        # 分开统计clean像素和noise像素
        all_residual = (np.abs(postprocess_fn(rendering['rgb']) - postprocess_fn(test_case.rgb))).mean(axis=-1)
        hard_residual = all_residual[mask.astype(bool)]
        easy_residual = all_residual[(1-mask).astype(bool)]
        # residual = (np.abs(postprocess_fn(rendering['rgb']) - postprocess_fn(test_case.rgb))).mean(axis=-1).reshape(-1)
        
        
        # 使用np.histogram统计每个区间的元素数量
        easy_hist, easy_bin_edges = np.histogram(easy_residual, bins=np.linspace(0, 1, num_intervals+1))
        # hist_total += hist
        # 使用np.digitize将元素分配到区间并计算区间内元素总和
        easy_indices = np.digitize(easy_residual, easy_bin_edges)
        for i in range(1, num_intervals+1):
            easy_interval_sums[i-1] += np.sum(easy_residual[easy_indices == i])
        
        ## 统计noise像素的residual：
        hard_hist, hard_bin_edges = np.histogram(hard_residual, bins=np.linspace(0, 1, num_intervals+1))
        # hist_total += hist
        # 使用np.digitize将元素分配到区间并计算区间内元素总和
        hard_indices = np.digitize(hard_residual, hard_bin_edges)
        for i in range(1, num_intervals+1):
            hard_interval_sums[i-1] += np.sum(hard_residual[hard_indices == i])
        # 输出每个区间的统计信息
        # for i in range(num_intervals):
        #     start, end = bin_edges[i], bin_edges[i+1]
        #     print(f"区间 [{start:.2f}, {end:.2f}]: 元素数量={hist_total[i]}, 总和={interval_sums[i]:.2f}")
        

        ### 这个方法太慢了
        # statistic_start_time = time.time()
        # # 遍历数组并统计每个区间的元素数量和总和
        # for val in residual:
        #     for i, (start, end) in enumerate(interval_ranges):
        #         if start <= val < end:
        #             if i not in interval_stats:
        #                 interval_stats[i] = 0
        #             interval_stats[i] += 1
        #             interval_sums[i] += val
        #             break

        # # 输出每个区间的统计信息
        # for i, (start, end) in enumerate(interval_ranges):
        #     print(f"区间 [{start:.2f}, {end:.2f}]: 元素数量={interval_stats.get(i, 0)}, 总和={interval_sums[i]:.2f}")
        
        # print(f'Statistics computed in {(time.time() - statistic_start_time):0.3f}s')
        
        # statistic_start_time = time.time()
      # hist_total_file_name = f'training_{step}_steps_residual_statistics_hist_total.npy'
      # np.save(os.path.join(config.checkpoint_dir, hist_total_file_name), hist_total)
      easy_interval_sums_file_name = f'easy_training_{step}_steps_residual_statistics_interval_sums.npy'
      np.save(os.path.join(easy_residual_dir, easy_interval_sums_file_name), easy_interval_sums)
      
      hard_interval_sums_file_name = f'hard_training_{step}_steps_residual_statistics_interval_sums.npy'
      np.save(os.path.join(hard_residual_dir, hard_interval_sums_file_name), hard_interval_sums)
      
      print(f'Statistics computed in {(time.time() - statistic_start_time):0.3f}s')
        
  
    ###################################################################################  用train loss residual计算GMM模型
    if config.train_residual and config.fit_gmm_train_residual_every > 0 and step % config.fit_gmm_train_residual_every == 0 and step > config.pretrain_steps : # 
      '''
      在训练过程中统计residual的分布情况, 并拟合gmm_model。
      注意在拟合gmm过程中是不需要mask的, 模拟真实应用场景中没有对比图相对的情况。
      '''
      # all_residual_dir = config.checkpoint_dir + '/rsidual_statistic/all/fit_gmm_model'
      # if not os.path.exists(all_residual_dir):
      #   os.makedirs(all_residual_dir)
      gmm_coef_dir = config.checkpoint_dir + '/gmm_coef/train'
      if not os.path.exists(gmm_coef_dir):
        os.makedirs(gmm_coef_dir)

      
      # We reuse the same random number generator from the optimization step
      # here on purpose so that the visualization matches what happened in
      # training.
      # eval_variables = flax.jax_utils.unreplicate(state).params
      # statistic_start_time = time.time()
      print("residual_train_patch16_all.len(): ", residual_train_patch16_all.__len__())
      print("residual_train_patch8_all.len(): ", residual_train_patch8_all.__len__())
      print("residual_train_patch4_all.len(): ", residual_train_patch4_all.__len__())
      print("residual_train_pixel_all.len(): ", residual_train_pixel_all.__len__())
      

      ######################################################################
      all_residual_patch16 = jnp.concatenate(residual_train_patch16_all)
      residual_train_patch16_all = []
      all_residual_patch8 = jnp.concatenate(residual_train_patch8_all)
      residual_train_patch8_all = []
      all_residual_patch4 = jnp.concatenate(residual_train_patch4_all)
      residual_train_patch4_all = []
      all_residual_pixel = jnp.concatenate(residual_train_pixel_all)
      residual_train_pixel_all = []
      print(f'all_residual_patch16 shape:', all_residual_patch16.shape)
      print(f'all_residual_patch8 shape:', all_residual_patch8.shape)
      print(f'all_residual_patch4 shape:', all_residual_patch4.shape)
      print(f'all_residual_pixel shape:', all_residual_pixel.shape)

      ###下面这几行改成保存gmm参数！
      ####
      # all_interval_sums_file_name = f'all_training_{step}_steps_all_train_residual.npy'
      # np.save(os.path.join(all_residual_dir, all_interval_sums_file_name), all_residual)
      # # assert all_residual.shape[-1]==1 , "all_residual shape is wrong!!!!"
      # print(f'Statistics computed in {(time.time() - statistic_start_time):0.3f}s')
      ####
      
      gmm_start_time = time.time()
      
      ##############################################################################
      # 拟合GMM模型 方法1：直接使用所有的residual：
      gmm_means_patch16_, \
      gmm_covariances_patch16_, \
      gmm_weights_patch16_, \
      gmm_precisions_cholesky_patch16_, \
      gmm_covariance_type_patch16 = direct_gmm_fitting(all_residual_patch16, gmm_start_time)
      
      # 判断下两个gmm component 均值大小，因为有可能和之前的均值大小关系是反的
      if gmm_means_patch16 is not None and any(gmm_means_patch16) and (gmm_means_patch16[0, 0]>gmm_means_patch16[1, 0]) != (gmm_means_patch16_[0, 0]>gmm_means_patch16_[1, 0]):
        gmm_means_patch16_[0, 0], gmm_means_patch16_[1, 0] = gmm_means_patch16_[1, 0], gmm_means_patch16_[0, 0]  # gmm_means shape: (2,1)
        gmm_covariances_patch16_[0, 0, 0], gmm_covariances_patch16_[1, 0, 0] = gmm_covariances_patch16_[1, 0, 0], gmm_covariances_patch16_[0, 0, 0]  # gmm_covariances shape: (2, 1, 1)
        gmm_weights_patch16_[0], gmm_weights_patch16_[1] = gmm_weights_patch16_[1], gmm_weights_patch16_[0]  # gmm_weights shape: (2, )
        gmm_precisions_cholesky_patch16_[0, 0, 0], gmm_precisions_cholesky_patch16_[1, 0, 0] = gmm_precisions_cholesky_patch16_[1, 0, 0], gmm_precisions_cholesky_patch16_[0, 0, 0]  # gmm_precisions_cholesky shape: (2, 1, 1)
      
      # EMA滑动平均
      if config.ema_gmm_flag and gmm_means_patch16 is not None and any(gmm_means_patch16):
        print("EMA")
        alpha = config.ema_alpha
        gmm_means_patch16 = alpha * gmm_means_patch16_ + (1 - alpha) * gmm_means_patch16
        gmm_covariances_patch16 = alpha * gmm_covariances_patch16_ + (1 - alpha) * gmm_covariances_patch16
        gmm_weights_patch16 = alpha * gmm_weights_patch16_ + (1 - alpha) * gmm_weights_patch16
        gmm_precisions_cholesky_patch16 = alpha * gmm_precisions_cholesky_patch16_ + (1 - alpha) * gmm_precisions_cholesky_patch16
        
      else:
        gmm_means_patch16 = gmm_means_patch16_
        gmm_covariances_patch16 = gmm_covariances_patch16_
        gmm_weights_patch16 = gmm_weights_patch16_
        gmm_precisions_cholesky_patch16 = gmm_precisions_cholesky_patch16_
      ####
      gmm_means_patch8_, \
      gmm_covariances_patch8_, \
      gmm_weights_patch8_, \
      gmm_precisions_cholesky_patch8_, \
      gmm_covariance_type_patch8 = direct_gmm_fitting(all_residual_patch8, gmm_start_time)
      
      if gmm_means_patch8 is not None and any(gmm_means_patch8) and (gmm_means_patch8[0, 0]>gmm_means_patch8[1, 0]) != (gmm_means_patch8_[0, 0]>gmm_means_patch8_[1, 0]):
        gmm_means_patch8_[0, 0], gmm_means_patch8_[1, 0] = gmm_means_patch8_[1, 0], gmm_means_patch8_[0, 0]  # gmm_means shape: (2,1)
        gmm_covariances_patch8_[0, 0, 0], gmm_covariances_patch8_[1, 0, 0] = gmm_covariances_patch8_[1, 0, 0], gmm_covariances_patch8_[0, 0, 0]  # gmm_covariances shape: (2, 1, 1)
        gmm_weights_patch8_[0], gmm_weights_patch8_[1] = gmm_weights_patch8_[1], gmm_weights_patch8_[0]  # gmm_weights shape: (2, )
        gmm_precisions_cholesky_patch8_[0, 0, 0], gmm_precisions_cholesky_patch8_[1, 0, 0] = gmm_precisions_cholesky_patch8_[1, 0, 0], gmm_precisions_cholesky_patch8_[0, 0, 0]  # gmm_precisions_cholesky shape: (2, 1, 1)
      
      if config.ema_gmm_flag and gmm_means_patch8 is not None and any(gmm_means_patch8):
        alpha = config.ema_alpha
        gmm_means_patch8 = alpha * gmm_means_patch8_ + (1 - alpha) * gmm_means_patch8
        gmm_covariances_patch8 = alpha * gmm_covariances_patch8_ + (1 - alpha) * gmm_covariances_patch8
        gmm_weights_patch8 = alpha * gmm_weights_patch8_ + (1 - alpha) * gmm_weights_patch8
        gmm_precisions_cholesky_patch8 = alpha * gmm_precisions_cholesky_patch8_ + (1 - alpha) * gmm_precisions_cholesky_patch8
        
      else:
        gmm_means_patch8 = gmm_means_patch8_
        gmm_covariances_patch8 = gmm_covariances_patch8_
        gmm_weights_patch8 = gmm_weights_patch8_
        gmm_precisions_cholesky_patch8 = gmm_precisions_cholesky_patch8_
      ####
      gmm_means_patch4_, \
      gmm_covariances_patch4_, \
      gmm_weights_patch4_, \
      gmm_precisions_cholesky_patch4_, \
      gmm_covariance_type_patch4 = direct_gmm_fitting(all_residual_patch4, gmm_start_time)
      
      if gmm_means_patch4 is not None and any(gmm_means_patch4) and (gmm_means_patch4[0, 0]>gmm_means_patch4[1, 0]) != (gmm_means_patch4_[0, 0]>gmm_means_patch4_[1, 0]):
        gmm_means_patch4_[0, 0], gmm_means_patch4_[1, 0] = gmm_means_patch4_[1, 0], gmm_means_patch4_[0, 0]  # gmm_means shape: (2,1)
        gmm_covariances_patch4_[0, 0, 0], gmm_covariances_patch4_[1, 0, 0] = gmm_covariances_patch4_[1, 0, 0], gmm_covariances_patch4_[0, 0, 0]  # gmm_covariances shape: (2, 1, 1)
        gmm_weights_patch4_[0], gmm_weights_patch4_[1] = gmm_weights_patch4_[1], gmm_weights_patch4_[0]  # gmm_weights shape: (2, )
        gmm_precisions_cholesky_patch4_[0, 0, 0], gmm_precisions_cholesky_patch4_[1, 0, 0] = gmm_precisions_cholesky_patch4_[1, 0, 0], gmm_precisions_cholesky_patch4_[0, 0, 0]  # gmm_precisions_cholesky shape: (2, 1, 1)
      
      if config.ema_gmm_flag and gmm_means_patch4 is not None and any(gmm_means_patch4):
        alpha = config.ema_alpha
        gmm_means_patch4 = alpha * gmm_means_patch4_ + (1 - alpha) * gmm_means_patch4
        gmm_covariances_patch4 = alpha * gmm_covariances_patch4_ + (1 - alpha) * gmm_covariances_patch4
        gmm_weights_patch4 = alpha * gmm_weights_patch4_ + (1 - alpha) * gmm_weights_patch4
        gmm_precisions_cholesky_patch4 = alpha * gmm_precisions_cholesky_patch4_ + (1 - alpha) * gmm_precisions_cholesky_patch4
        
      else:
        gmm_means_patch4 = gmm_means_patch4_
        gmm_covariances_patch4 = gmm_covariances_patch4_
        gmm_weights_patch4 = gmm_weights_patch4_
        gmm_precisions_cholesky_patch4 = gmm_precisions_cholesky_patch4_
      ####
      gmm_means_pixel_, \
      gmm_covariances_pixel_, \
      gmm_weights_pixel_, \
      gmm_precisions_cholesky_pixel_, \
      gmm_covariance_type_pixel = direct_gmm_fitting(all_residual_pixel, gmm_start_time)
      
      if gmm_means_pixel is not None and any(gmm_means_pixel) and (gmm_means_pixel[0, 0]>gmm_means_pixel[1, 0]) != (gmm_means_pixel_[0, 0]>gmm_means_pixel_[1, 0]):
        gmm_means_pixel_[0, 0], gmm_means_pixel_[1, 0] = gmm_means_pixel_[1, 0], gmm_means_pixel_[0, 0]  # gmm_means shape: (2,1)
        gmm_covariances_pixel_[0, 0, 0], gmm_covariances_pixel_[1, 0, 0] = gmm_covariances_pixel_[1, 0, 0], gmm_covariances_pixel_[0, 0, 0]  # gmm_covariances shape: (2, 1, 1)
        gmm_weights_pixel_[0], gmm_weights_pixel_[1] = gmm_weights_pixel_[1], gmm_weights_pixel_[0]  # gmm_weights shape: (2, )
        gmm_precisions_cholesky_pixel_[0, 0, 0], gmm_precisions_cholesky_pixel_[1, 0, 0] = gmm_precisions_cholesky_pixel_[1, 0, 0], gmm_precisions_cholesky_pixel_[0, 0, 0]  # gmm_precisions_cholesky shape: (2, 1, 1)
      
      if config.ema_gmm_flag and gmm_means_pixel is not None and any(gmm_means_pixel):
        alpha = config.ema_alpha
        gmm_means_pixel = alpha * gmm_means_pixel_ + (1 - alpha) * gmm_means_pixel
        gmm_covariances_pixel = alpha * gmm_covariances_pixel_ + (1 - alpha) * gmm_covariances_pixel
        gmm_weights_pixel = alpha * gmm_weights_pixel_ + (1 - alpha) * gmm_weights_pixel
        gmm_precisions_cholesky_pixel = alpha * gmm_precisions_cholesky_pixel_ + (1 - alpha) * gmm_precisions_cholesky_pixel
        
      else:
        gmm_means_pixel = gmm_means_pixel_
        gmm_covariances_pixel = gmm_covariances_pixel_
        gmm_weights_pixel = gmm_weights_pixel_
        gmm_precisions_cholesky_pixel = gmm_precisions_cholesky_pixel_
      
      print(f'GMM fiting in {(time.time() - gmm_start_time):0.3f}s')
      
      ####
      json_dict = {}
      json_dict['gmm_means_patch16'] = gmm_means_patch16.tolist()
      json_dict['gmm_covariances_patch16'] = gmm_covariances_patch16.tolist()
      json_dict['gmm_weights_patch16'] = gmm_weights_patch16.tolist()
      json_dict['gmm_precisions_cholesky_patch16'] = gmm_precisions_cholesky_patch16.tolist()
      json_dict['gmm_covariance_type_patch16'] = gmm_covariance_type_patch16
      
      json_dict['gmm_means_patch8'] = gmm_means_patch8.tolist()
      json_dict['gmm_covariances_patch8'] = gmm_covariances_patch8.tolist()
      json_dict['gmm_weights_patch8'] = gmm_weights_patch8.tolist()
      json_dict['gmm_precisions_cholesky_patch8'] = gmm_precisions_cholesky_patch8.tolist()
      json_dict['gmm_covariance_type_patch8'] = gmm_covariance_type_patch8
      
      json_dict['gmm_means_patch4'] = gmm_means_patch4.tolist()
      json_dict['gmm_covariances_patch4'] = gmm_covariances_patch4.tolist()
      json_dict['gmm_weights_patch4'] = gmm_weights_patch4.tolist()
      json_dict['gmm_precisions_cholesky_patch4'] = gmm_precisions_cholesky_patch4.tolist()
      json_dict['gmm_covariance_type_patch4'] = gmm_covariance_type_patch4
      
      json_dict['gmm_means_pixel'] = gmm_means_pixel.tolist()
      json_dict['gmm_covariances_pixel'] = gmm_covariances_pixel.tolist()
      json_dict['gmm_weights_pixel'] = gmm_weights_pixel.tolist()
      json_dict['gmm_precisions_cholesky_pixel'] = gmm_precisions_cholesky_pixel.tolist()
      json_dict['gmm_covariance_type_pixel'] = gmm_covariance_type_pixel
      
      gmm_coef_name = gmm_coef_dir + '/' + f'all_training_{step}_steps_gmm_coefficient.json'
      with open(gmm_coef_name, 'w') as json_file:
        json.dump(json_dict, json_file)
      ####
      
      
      ##############################################################################
    ################################################################################### 统计residual 并拟合gmm模型, 更新gmm_model
    if config.test_residual and config.fit_gmm_test_residual_every > 0 and step % config.fit_gmm_test_residual_every == 0 and step > config.pretrain_steps: # and step > 49990 
      '''
      在训练过程中统计residual的分布情况, 并拟合gmm_model。
      注意在拟合gmm过程中是不需要mask的, 模拟真实应用场景中没有对比图相对的情况。
      '''
      all_residual_dir = config.checkpoint_dir + '/rsidual_statistic/all/fit_gmm_model'
      if not os.path.exists(all_residual_dir):
        os.makedirs(all_residual_dir)

      
      # We reuse the same random number generator from the optimization step
      # here on purpose so that the visualization matches what happened in
      # training.
      eval_variables = flax.jax_utils.unreplicate(state).params
      statistic_start_time = time.time()
      
      
      
      if pixel_residual_flag:
        all_residual_ = np.zeros((residual_dataset.size, residual_dataset.height * residual_dataset.width, 1))
      elif patch_residual_flag:
        all_residual_ = np.zeros((residual_dataset.size, (residual_dataset.height//PATCH_SIZE) * (residual_dataset.width//PATCH_SIZE)+NUM_PATCHES, 1))
      elif rgbchannel_pixel_residual_flag:
        all_residual_ = np.zeros((residual_dataset.size, residual_dataset.height * residual_dataset.width, 3))
        
        
      for i in range(residual_dataset.size):
      # for i in range(6):
        train_case = next(residual_dataset)
        rendering = models.render_image(
          functools.partial(render_eval_pfn, eval_variables, train_frac),
          train_case.rays, rngs[0], config)
        
        #####################################################################
        # 统计方法1：统计所有的residual 这是拉直的，不是patch
        if pixel_residual_flag:
          residual = pixel_residual(postprocess_fn, rendering, train_case)
          
        
        #####################################################################
        # 统计方法2：采用patch计算residual
        elif patch_residual_flag:
          residual = patch_residual(postprocess_fn, rendering, train_case)
        
        #####################################################################
        # 统计方法3：统计所有的residual,但是分rgb3个通道
        if rgbchannel_pixel_residual_flag:
          residual = rgbchannel_pixel_residual(postprocess_fn, rendering, train_case)
          
        
          
        ######################################################################
        
        # 比例处理方法1：
        if False:
          '''
          不同的residual计算方法可能会有不同的效果, 因为clean和noisy 像素数量差别太大了。
          '''
          residual = -jnp.log(1-residual)*residual
        ######################################
        
        # all_residual.append(residual)
        all_residual_[train_case.cam_idx] = residual
        print(f'calculate No.{train_case.cam_idx} residual!')
        
      all_residual = jnp.concatenate(all_residual_)
      all_interval_sums_file_name = f'all_training_{step}_steps_all_residual.npy'
      np.save(os.path.join(all_residual_dir, all_interval_sums_file_name), all_residual)
      # assert all_residual.shape[-1]==1 , "all_residual shape is wrong!!!!"
      print(f'all_residual shape:', all_residual.shape)
      print(f'Statistics computed in {(time.time() - statistic_start_time):0.3f}s')
      
      gmm_start_time = time.time()
      
      ##############################################################################
      # 拟合GMM模型 方法1：直接使用所有pixel的residual：
      if direct_gmm_fitting_flag:
        gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type = direct_gmm_fitting(all_residual, gmm_start_time)
      
      ##############################################################################
      # 拟合GMM模型 方法2：根据residual在不同小区间内的累加和，转换成数量：
      elif accumulate_gmm_fitting_flag:
        gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type = accumulate_gmm_fitting(all_residual, gmm_start_time)
      
      ##############################################################################
      # 拟合GMM模型 方法3：根据每张图片的 residual_ 分别对每张图估计GMM模型：
      elif image_direct_gmm_fitting_flag:
        gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type = image_direct_gmm_fitting(all_residual_, gmm_start_time)
      
      # 拟合GMM模型 方法4：根据每张图片的 residual_ ,先计算统计量，然后分别对每张图估计GMM模型：
      elif image_accumulate_gmm_fitting_flag:
        gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type = image_accumulate_gmm_fitting(all_residual_, gmm_start_time)
      
      # 拟合GMM模型 方法5：根据每张图片的 rgb 3通道的residual 分别对每张图估计GMM模型：
      elif rgbchannel_direct_gmm_fitting_flag:
        gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type = rgbchannel_direct_gmm_fitting(all_residual_, gmm_start_time)
      
      # 拟合GMM模型 方法6：根据每张图片的 rgb 3通道的residual 先计算统计量，然后分别对每张图估计GMM模型：
      elif rgbchannel_accumulate_gmm_fitting_flag:
        gmm_means, gmm_covariances, gmm_weights, gmm_precisions_cholesky, gmm_covariance_type = rgbchannel_accumulate_gmm_fitting(all_residual_, gmm_start_time)
        
        
      ##############################################################################

  if jax.host_id() == 0 and config.max_steps % config.checkpoint_every != 0:
    state = jax.device_get(flax.jax_utils.unreplicate(state))
    checkpoints.save_checkpoint(
        config.checkpoint_dir, state, int(config.max_steps), keep=100)
    ####
    gmm_coef_dir = config.checkpoint_dir + '/gmm_coef/train'
    if not os.path.exists(gmm_coef_dir):
      os.makedirs(gmm_coef_dir)
      
    json_dict = {}
    json_dict['gmm_means_patch16'] = gmm_means_patch16.tolist()
    json_dict['gmm_covariances_patch16'] = gmm_covariances_patch16.tolist()
    json_dict['gmm_weights_patch16'] = gmm_weights_patch16.tolist()
    json_dict['gmm_precisions_cholesky_patch16'] = gmm_precisions_cholesky_patch16.tolist()
    json_dict['gmm_covariance_type_patch16'] = gmm_covariance_type_patch16.tolist()
    
    json_dict['gmm_means_patch8'] = gmm_means_patch8.tolist()
    json_dict['gmm_covariances_patch8'] = gmm_covariances_patch8.tolist()
    json_dict['gmm_weights_patch8'] = gmm_weights_patch8.tolist()
    json_dict['gmm_precisions_cholesky_patch8'] = gmm_precisions_cholesky_patch8.tolist()
    json_dict['gmm_covariance_type_patch8'] = gmm_covariance_type_patch8.tolist()
    
    json_dict['gmm_means_patch4'] = gmm_means_patch4.tolist()
    json_dict['gmm_covariances_patch4'] = gmm_covariances_patch4.tolist()
    json_dict['gmm_weights_patch4'] = gmm_weights_patch4.tolist()
    json_dict['gmm_precisions_cholesky_patch4'] = gmm_precisions_cholesky_patch4.tolist()
    json_dict['gmm_covariance_type_patch4'] = gmm_covariance_type_patch4.tolist()
    
    json_dict['gmm_means_pixel'] = gmm_means_pixel.tolist()
    json_dict['gmm_covariances_pixel'] = gmm_covariances_pixel.tolist()
    json_dict['gmm_weights_pixel'] = gmm_weights_pixel.tolist()
    json_dict['gmm_precisions_cholesky_pixel'] = gmm_precisions_cholesky_pixel.tolist()
    json_dict['gmm_covariance_type_pixel'] = gmm_covariance_type_pixel.tolist()
    
    gmm_coef_name = gmm_coef_dir + '/' + f'all_training_{step}_steps_gmm_coefficient.json'
    with open(gmm_coef_name, 'w') as json_file:
      json.dump(json_dict, json_file)


if __name__ == '__main__':
  with gin.config_scope('train'):
    app.run(main)
