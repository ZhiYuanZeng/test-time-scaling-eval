---
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: problem
    dtype: string
  - name: solution
    dtype: string
  - name: answer
    dtype: string
  - name: url
    dtype: string
  splits:
  - name: train
    num_bytes: 520431
    num_examples: 90
  download_size: 261038
  dataset_size: 520431
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
# Dataset Card for AIMO Validation AIME

All 90 problems come from AIME 22, AIME 23, and AIME 24, and have been extracted directly from the AOPS wiki page https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions

This dataset serves as an internal validation set during our participation in the AIMO progress prize competition. Using data after 2021 is to avoid potential overlap with the MATH training set.

Here are the different columns in the dataset:

- problem: the original problem statement from the website
- solution: one of the solutions proposed in the forum with \boxed answer
- url: url to the problem page in the website
