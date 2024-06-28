import torch
import argparse
import numpy as np
import os
import rerun as rr

from unimatch.unimatch import UniMatch

from evaluate_flow import (validate_chairs, validate_things, validate_sintel, validate_kitti,
                           create_kitti_submission, create_sintel_submission,
                           inference_flow,
                           )

from utils import misc
from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed
from huggingface_hub import hf_hub_download

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = UniMatch(feature_channels=128,
                    num_scales=2,
                    upsample_factor=4,
                    num_head=1,
                    ffn_dim_expansion=4,
                    num_transformer_layers=6,
                    reg_refine=True,
                    task='flow').to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f'Number of parameters: {num_params}')

print("Load pretrained model")
checkpoint = hf_hub_download('pranay-ar/unimatch', 'gmflow-scale2-mixdata-train320x576-9ff1c094.pth')
checkpoint = torch.load(checkpoint, map_location='cuda')
model.load_state_dict(checkpoint['model'], strict=False) 

print("Model loaded")
flow, vis = inference_flow(model,
                inference_dir=None,
                inference_video='/home/pranay/test/unimatch/demo/kitti.mp4',
                output_path='output/gmflow-scale2-davis/test',
                padding_factor=32,
                attn_type='swin',
                attn_splits_list=[2,8],
                corr_radius_list=[-1,4],
                prop_radius_list=[-1,1],
                num_reg_refine=1,
                concat_flow_img=False,
                save_video=False,
                )

print(len(flow), len(vis))