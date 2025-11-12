#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2025/11/12
"""
import numpy as np
import rasterio
from pyproj import Transformer, CRS


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
