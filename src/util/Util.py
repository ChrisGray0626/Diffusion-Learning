#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2025/12/1
"""
import torch

from Constant import VALID_DATE_FILE_PATH


def get_valid_dates():
    tgt_dates = read_txt(VALID_DATE_FILE_PATH)
    return tgt_dates


def read_txt(src_path):
    with open(src_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def build_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.mps.is_available():
        device = "mps"

    return device
