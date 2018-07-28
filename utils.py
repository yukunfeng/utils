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


def word_ids_to_sentence(id_tensor, vocab):
    """Converts a sequence of word ids to a sentence
    id_tensor: torch-based tensor
    vocab: torchtext vocab
    """
    id_tensor = id_tensor.transpose(0, 1)
    symbols = []
    for row in id_tensor:
        row_symbols = []
        for col in row:
            row_symbols.append(vocab.itos[col])
        symbols.append(row_symbols)
    return symbols


def save_word_embedding(vocab, emb, file_name):
    """Saving word emb"""
    print(emb)
    with open(file_name, 'x') as fh:
        fh.write(f"{emb.size(0)} {emb.size(1)}\n")
        for word, vec in zip(vocab, emb):
            str_vec = [f"{x.item():5.4f}" for x in vec]
            line = word + " " + " ".join(str_vec) + "\n"
            fh.write(line)


def save_word_embedding_test():
    vocab = ["a", "b", "c"]
    emb = torch.rand(len(vocab), 5)
    save_word_embedding(vocab, emb, "vec.txt")


if __name__ == "__main__":
    # Unit test
    save_word_embedding_test()
