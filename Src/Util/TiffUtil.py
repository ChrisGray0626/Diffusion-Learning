#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2025/11/12
"""
from typing import Tuple

import numpy as np
import rasterio
from affine import Affine
from matplotlib import pyplot as plt
from pyproj import Transformer, CRS
from rasterio.warp import transform_bounds


def read_tiff_data(file_path: str):
    with rasterio.open(file_path) as src:
        data = src.read(1)

    return data


def read_tiff(file_path: str, dst_epsg_code: int = 4326):
    with rasterio.open(file_path) as src:
        data = src.read(1)
        transform_affine = src.transform
        src_crs = src.crs  # 源投影 CRS
        width = src.width
        height = src.height

    # 构建行列索引网格
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))

    # 使用仿射变换将行列号转换为原始投影下的坐标
    xs, ys = rasterio.transform.xy(transform_affine, rows, cols, offset='center')

    # Reshape 为二维数组
    xs = np.array(xs).reshape((height, width))
    ys = np.array(ys).reshape((height, width))

    # 投影转换（原投影 -> EPSG:4326）
    if dst_epsg_code != src_crs.to_epsg():
        transformer = Transformer.from_crs(src_crs, CRS.from_epsg(dst_epsg_code), always_xy=True)
        xs, ys = transformer.transform(xs, ys)

    return data, xs, ys


def write_tiff(data,
               dst_file_path: str,
               transform: rasterio.Affine,
               epsg_code: int = None,
               crs: CRS = None,
               nodata: float = np.nan,
               dtype=None,
               ):
    data = np.asarray(data)

    if data.ndim != 2 and data.ndim != 3:
        raise ValueError(f"Expected 2-D or 3-D data, got shape {data.shape}")

    # Ensure data is 3-D for consistent processing
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    band, height, width = data.shape

    if epsg_code is not None:
        crs = CRS.from_epsg(epsg_code)
    profile = {
        'driver': 'GTiff',
        'width': width,
        'height': height,
        'count': band,
        'crs': crs,
        'transform': transform,
        'dtype': dtype or data.dtype,
        'nodata': nodata,
    }
    with rasterio.open(dst_file_path, 'w', **profile) as dst:
        dst.write(data)


def show_tiff(file_path: str, dst_epsg_code: int = 4326):
    with rasterio.open(file_path) as dataset:
        # 读取数据（第1波段）
        data = dataset.read(1)
        # 获取仿射变换（地理坐标到像素坐标转换）
        transform = dataset.transform
        # 获取坐标参考系统（CRS）
        crs = dataset.crs
        # 获取边界（范围）
        bounds = dataset.bounds
        # 获取像素大小
        res = dataset.res
    if crs is None:
        crs = CRS.from_epsg(dst_epsg_code)
    # 转换为 EPSG:4326
    if dst_epsg_code != crs.to_epsg():
        bounds = transform_bounds(crs, CRS.from_epsg(dst_epsg_code), *bounds)
    data = np.ma.masked_invalid(data)
    d = data[~np.isnan(data)]
    print("Transform: ", transform)
    print("Bounds: ", bounds)
    print("Resolution (pixel size): ", res)
    print("Data shape: ", data.shape)
    # 显示高程图像
    plt.imshow(data, cmap='terrain')
    plt.colorbar()
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()


def read_tiff_meta(grid_path: str) -> Tuple[Affine, CRS, int, int]:
    with rasterio.open(grid_path) as src:
        transform = src.transform
        crs = src.crs
        height = src.height
        width = src.width

    return transform, crs, height, width
