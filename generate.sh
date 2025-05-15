#!/bin/bash

python3 create_dataset.py \
    --background /mnt/cdisk/dewil/HW-SR24/datasets/datasets_bokeh/background/ \
    --foreground /mnt/cdisk/dewil/HW-SR24/datasets/datasets_bokeh/P3M-10k/validation/P3M-500-NP/original_image/ \
    --mask /mnt/cdisk/dewil/HW-SR24/datasets/datasets_bokeh/P3M-10k/validation/P3M-500-NP/mask/ \
    --output_all_in_focus /mnt/cdisk/dewil/HW-SR24/datasets/datasets_bokeh/our_videos/all_in_focus/ \
    --output_bokeh_masks /mnt/cdisk/dewil/HW-SR24/datasets/datasets_bokeh/our_videos/mask/ \
    --output_bokeh_frames /mnt/cdisk/dewil/HW-SR24/datasets/datasets_bokeh/our_videos/with_bokeh/ \
    --output_flows /mnt/cdisk/dewil/HW-SR24/datasets/datasets_bokeh/our_videos/flows/ \
    --nb_frames 10 --nb_videos 1000

#    --foreground /mnt/cdisk/dewil/HW-SR24/datasets/our_videos_with_bokeh/foreground_objects/ \
#    --mask /mnt/cdisk/dewil/HW-SR24/datasets/our_videos_with_bokeh/foreground_masks/ \
