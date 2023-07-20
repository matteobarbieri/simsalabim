#!/usr/bin/env python
# coding: utf-8

"""
Performs some sanity checks on the dataset
"""

import os, sys

sys.path.append(os.path.dirname(__file__) + "/../src")

import pandas as pd

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_path", type=str, help="Path to the dataset to check")

    parser.add_argument(
        "--metadata-filename",
        type=str,
        default="metadata.csv",
        help="Name of the file containing the dataset's metadata",
    )

    parser.add_argument(
        "--subset-column",
        type=str,
        default="subset",
        help="Name of the column used for the subset",
    )

    parser.add_argument(
        "--original-column",
        type=str,
        default="original",
        help="Name of the column containing the original name of the file",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(os.path.join(args.dataset_path, args.metadata_filename))

    subsets = df[args.subset_column].unique()

    for s1 in subsets:
        for s2 in subsets:
            if s1 == s2:
                continue

            files_1 = set(df[df[args.subset_column] == s1][args.original_column])
            files_2 = set(df[df[args.subset_column] == s2][args.original_column])

            assert files_1.intersection(files_2) == set()

    print("All folds are non-overlapping with each other 🍰")


if __name__ == "__main__":
    main()
