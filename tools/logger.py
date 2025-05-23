from copy import Error
import logging
import os
import sys
from collections import Counter
from typing import Tuple

import mmcv
import numpy as np
import torch
from mmcv.runner.dist_utils import get_dist_info
from mmcv.utils import get_logger
from mmdet.core.visualization import imshow_det_bboxes
import wandb

_log_counter = Counter()

def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name="mmdet.ssod", log_file=log_file, log_level=log_level)
    logger.propagate = False
    return logger

def _find_caller():
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = r"ssod"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back

def log_every_n(msg: str, n: int = 50, level: int = logging.DEBUG, backend="auto"):
    """
    Args:
        msg (Any):
        n (int):
        level (int):
        name (str):
    """
    caller_module, key = _find_caller()
    _log_counter[key] += 1
    if n == 1 or _log_counter[key] % n == 1:
        if isinstance(msg, dict) and (wandb is not None) and (wandb.run is not None):
            wandb.log(msg, commit=False)
        else:
            get_root_logger().log(level, msg)

