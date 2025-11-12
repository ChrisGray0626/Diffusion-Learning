#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2025/11/6
"""
import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_PATH = os.path.join(ROOT_PATH, "Data")
TRAIN_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, "Train")

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

STANDARD_GRID_36KM_PATH = os.path.join(DATA_DIR_PATH, "Standard_Grid_36km.tif")

X_COLUMN = [
    NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME
]
# 因变量列名
Y_COLUMN = [SM_NAME]

TIFF_SUFFIX = ".tif"
