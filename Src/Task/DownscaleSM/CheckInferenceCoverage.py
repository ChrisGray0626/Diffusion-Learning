#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Check inference evaluation coverage by date
@Author Chris
@Date 2025/12/12
"""
import os

import pandas as pd

from Constant import RESULT_DIR_PATH, RESOLUTION_36KM
from Task.DownscaleSM.Dataset import InferenceEvaluationDataset


def main():
    dataset = InferenceEvaluationDataset(resolution=RESOLUTION_36KM)
    pred_map, insitu_map, valid_masks, dates, rows, cols = dataset.get_all()

    df = pd.DataFrame({
        'Date': dates,
        'Row': rows.flatten(),
        'Col': cols.flatten(),
        'Pred': pred_map.flatten(),
        'InSitu': insitu_map.flatten(),
        'Valid': valid_masks.flatten()
    })

    os.makedirs(RESULT_DIR_PATH, exist_ok=True)
    df.to_csv(os.path.join(RESULT_DIR_PATH, 'Coverage_Inference.csv'), index=False)


if __name__ == "__main__":
    main()
