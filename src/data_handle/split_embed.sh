#!/bin/bash

python make_embed_splits.py \
  --data_dir /vol/biodata/data/Mammo/EMBED/pngs/1024x768 \
  --csv_filepath /vol/biomedic3/tx1215/mamo-flow/assets/EMBED_meta.csv \
  --out_dir /vol/biomedic3/tx1215/mamo-flow/assets/embed_splits_v1 \
  --exclude_cviews 1 \
  --hold_out_model_5 1 \
  --prop_train 1.0 \
  --valid_frac 0.075 \
  --test_frac 0.125 \
  --split_seed 33