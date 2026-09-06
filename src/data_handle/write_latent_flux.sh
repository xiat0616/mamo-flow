#!/bin/bash

source /vol/biomedic3/tx1215/mamo-flow/.venv/bin/activate

python write_latent_cache.py \
    --split_dir /vol/biomedic3/tx1215/mamo-flow/assets/embed_splits_v1 \
    --data_dir /vol/biodata/data/Mammo/EMBED/pngs/1024x768 \
    --out_dir /vol/biomedic3/tx1215/mamo-flow/cache/flux2_vae_512x384 \
    --device cuda:0 \
    --batch_size 16 \
    --num_workers 8 \
    --img_height 512 \
    --img_width 384