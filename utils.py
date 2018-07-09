#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/07/01
Email       : yukunfg@gmail.com
Description : Misc utils
"""

import logging
import torch


def get_logger(log_file=None):
    """
    Logger from opennmt
    """

    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger


def word_ids_to_sentence(id_tensor, vocab, join=None):
    """Converts a sequence of word ids to a sentence
    id_tensor: torch-based tensor
    vocab: torchtext vocab
    """
    #  if isinstance(id_tensor, torch.LongTensor):
    ids = id_tensor.transpose(0, 1).contiguous().view(-1)
    #  elif isinstance(id_tensor, np.ndarray):
        #  ids = id_tensor.transpose().reshape(-1)
    batch = [vocab.itos[ind] for ind in ids]  # denumericalize
    if join is None:
        return batch
    return join.join(batch)
