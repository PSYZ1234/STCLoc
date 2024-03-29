# STCLoc
STCLoc: Deep LiDAR Localization with Spatio-Temporal Constraints

## Environment

- python 3.6.13

- pytorch 1.7.0


## Data

We support the Oxford Radar RobotCar and vReLoc datasets right now.
```
Oxford data_root
├── 2019-01-11-14-02-26-radar-oxford-10k
│   ├── velodyne_left
│       ├── xxx.bin
├── pose_stats.txt
├── pose_max_min.txt
├── train_split.txt
├── test_split.txt
```

## Run
### Oxford

- train -- 2 GPUs
```
python train.py --gpu_id 0 --batch_size 80 --val_batch_size 80 --decay_step 500 --log_dir log-oxford/ --dataset Oxford --num_loc 10 --num_ang 10 --skip 2
```

- test  -- 1 GPU
```
python eval.py --gpu_id 0 --val_batch_size 40 --log_dir log-oxford/ --dataset Oxford --num_loc 10 --num_ang 10 --skip 2 --resume_model checkpoint_epochxx.tar
```

### vReLoc

- train  -- 1 GPU
```
python train.py --gpu_id 0 --batch_size 40 --val_batch_size 40 --decay_step 25 --log_dir log-vreloc/ --dataset vReLoc --num_loc 2 --num_ang 10 --skip 0
 ```
- test  -- 1 GPU
```
python eval.py --gpu_id 0 --val_batch_size 40 --log_dir log-vreloc/ --dataset vReLoc --num_loc 2 --num_ang 10 --skip 0 --resume_model checkpoint_epochxx.tar
```

## Acknowledgement

 We appreciate the code of PointNet++ and AtLoc they shared.

## Citation

```
@ARTICLE{9928031,
  author={Yu, Shangshu and Wang, Cheng and Lin, Yitai and Wen, Chenglu and Cheng, Ming and Hu, Guosheng},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={STCLoc: Deep LiDAR Localization With Spatio-Temporal Constraints}, 
  year={2023},
  volume={24},
  number={1},
  pages={489-500},
  doi={10.1109/TITS.2022.3213311}}
```
