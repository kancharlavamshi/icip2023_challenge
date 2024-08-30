# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import mmcv
import mmcv_custom  # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403
import os.path as osp
import numpy as np
import torch
from torchvision.ops import nms

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    
    parser.add_argument('--input_path', type=str, default="demo", help='out dir')
    parser.add_argument('--output_path', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args

import glob
from torchvision.ops import nms



import torch
from mmcv.ops.nms import batched_nms

from mmdet.core.bbox.iou_calculators import bbox_overlaps






def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    path_in = args.input_path
    path_out=args.output_path
   
    ##imgs = sorted(glob.glob("/content/d1/*.jpg"))
    imgs = sorted(glob.glob(str(path_in)+"*.jpg"))
    Bb=[]
    Score=[]
    filename=[]
    class_name=[]
    vamm=0
    for img_name in imgs:
        print(img_name)
        model.test_cfg.rcnn.score_thr=0.3
        result = inference_detector(model, img_name)
        #cfg=model.cfg

        mmcv.mkdir_or_exist(args.out)
        out_file = osp.join(args.out, osp.basename(img_name))
        bbox, segm = result

        import pandas as pd

        for i, b in enumerate(bbox):

          if len(b) == 0:
            pass
          else:
            vx = torch.Tensor(b[:,:4])
            print('tensor box',vx)
            print(len(b[:,4:5]))
            score_v=list(np.concatenate(b[:,4:5]))
            vs = torch.Tensor(score_v)
            ##print('tensor box, score',vx,vs)
            v1=nms(boxes = vx, scores = vs, iou_threshold=0.5)
            indexx=v1.tolist()
            vx_2=vx.tolist()
            vs_2=vs.tolist()

            for k in range(len(indexx)):
              Bb.append(vx_2[indexx[k]])
              filename.append(img_name)
              Score.append(vs_2[indexx[k]])
              class_name.append(i+1)
              print('BB',vx_2[indexx[k]])
              print('SCORE',vs_2[indexx[k]])
              print('class',i+1)
        vamm=vamm+1
        print('Images_completed',vamm)
            #print("list is not empty")
        # show the results
        """
        model.show_result(
            img_name,
            result,
            score_thr=args.score_thr,
            show=False,
            bbox_color=args.palette,
            text_color=(200, 200, 200),
            mask_color=args.palette,
            out_file=out_file
        )
        print(f"Result is save at {out_file}")
       """
    import pandas as pd
    v1=pd.DataFrame(Bb,columns=['x1', 'y1', 'x2', 'y2'])
    v2=pd.DataFrame(Score,columns=['Score'])    
    v3=pd.DataFrame(filename,columns=['filename'])
    v4=pd.DataFrame(class_name,columns=['class_name'])  
    final=pd.concat([v4,v3,v1,v2],axis=1) 
    final.to_csv(str(path_out)+'final.csv',index=False)
    #print(v1.head(4)) 

if __name__ == '__main__':
    args = parse_args()
    main(args)