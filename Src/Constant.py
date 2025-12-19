#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2025/11/6
"""
import os

# 基础路径
PROJ_PATH = os.getenv("PROJ_PATH") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR_PATH = os.path.join(PROJ_PATH, "Checkpoint")
DATA_DIR_PATH = os.getenv("DATA_DIR_PATH", "/Volumes/Elements SE/Data/Downscale-DM")
RAW_DIR_PATH: str = os.path.join(DATA_DIR_PATH, "Raw/2016-2020")
PROCESSED_DIR_PATH = os.path.join(DATA_DIR_PATH, "Processed")
INFERENCE_DIR_PATH = os.path.join(DATA_DIR_PATH, "Inference")

VALID_DATE_FILE_PATH = os.path.join(DATA_DIR_PATH, "ValidDate.txt")

DATE_NAME = "Date"
LONGITUDE_NAME = "Lon"
LATITUDE_NAME = "Lat"
PROJ_X_NAME = "ProjX"
PROJ_Y_NAME = "ProjY"
ROW_NAME = "Row"
COL_NAME = "Col"
DATA_NAME = "Data"
NDVI_NAME = "NDVI"
LST_NAME = "LST"
SM_NAME = "SM"
ALBEDO_NAME = "Albedo"
PRECIPITATION_NAME = "Precipitation"
DEM_NAME = "DEM"

RESOLUTION_36KM = "36km"
RESOLUTION_1KM = "1km"

REF_GRID_36KM_PATH = os.path.join(DATA_DIR_PATH, "Standard_Grid_36km.tif")
REF_GRID_1KM_PATH = os.path.join(DATA_DIR_PATH, "Standard_Grid_1km.tif")

X_COLUMN = [
    NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME
]
# 因变量列名
Y_COLUMN = [SM_NAME]

TIFF_SUFFIX = ".tif"

# 空间范围：Left Bottom Right Top
RANGE = [-120, 35, -104, 49]
