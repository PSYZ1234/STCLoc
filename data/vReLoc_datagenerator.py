import os
import numpy as np
import pickle
import os.path as osp
from utils.pose_util import process_poses, ds_pc, filter_overflow_ts, position_classification, orientation_classification
from torch.utils import data
from data.robotcar_sdk.python.velodyne import load_velodyne_binary
import json


BASE_DIR = osp.dirname(osp.abspath(__file__))


class vReLoc(data.Dataset):
    def __init__(self, data_path, train, valid=False, augmentation=[], num_points=4096, real=False,
                 skip_pcs=False, vo_lib='orbslam', num_loc=10, num_ang=10):
        self.skip_pcs = skip_pcs
        # directories
        data_dir = osp.join(data_path, 'vReLoc')

        # decide which sequences to use
        if train:
            split_file = osp.join(data_dir, 'TrainSplit.txt')
        elif valid:
            split_file = osp.join(data_dir, 'ValidSplit.txt')
        else:
            split_file = osp.join(data_dir, 'TestSplit.txt')
        with open(split_file, 'r') as f:
            seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]

        # read poses and collect image names
        self.c_imgs = []
        self.d_imgs = []
        self.gt_idx = np.empty((0,), dtype=np.int)
        ps = {}
        vo_stats = {}
        gt_offset = int(0)
        for seq in seqs:
            seq_dir = osp.join(data_dir, 'seq-{:02d}'.format(seq))
            p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if n.find('pose') >= 0]
            if real:
                pose_file = osp.join(data_dir, '{:s}_poses'.format(vo_lib), 'seq-{:02d}.txt'.format(seq))
                pss = np.loadtxt(pose_file)
                frame_idx = pss[:, 0].astype(np.int)
                if vo_lib == 'libviso2':
                    frame_idx -= 1
                ps[seq] = pss[:, 1:13]
                vo_stats_filename = osp.join(seq_dir, '{:s}_vo_stats.pkl'.format(vo_lib))
                with open(vo_stats_filename, 'rb') as f:
                    vo_stats[seq] = pickle.load(f)
            else:
                frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
                pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.
                format(i)), delimiter=',').flatten()[:12] for i in frame_idx]
                ps[seq] = np.asarray(pss)
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            self.gt_idx = np.hstack((self.gt_idx, gt_offset+frame_idx))
            gt_offset += len(p_filenames)
            c_imgs = [osp.join(seq_dir, 'frame-{:06d}.bin'.format(i))
                      for i in frame_idx]
            d_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i))
                      for i in frame_idx]
            self.c_imgs.extend(c_imgs)
            self.d_imgs.extend(d_imgs)

        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')

        if train:
            mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
            std_t  = np.ones(3)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # convert pose to translation + log quaternion
        self.poses     = np.empty((0, 6))
        self.poses_max = np.empty((0, 2))
        self.poses_min = np.empty((0, 2))  
        pose_max_min_filename = osp.join(data_dir, 'pose_max_min.txt')

        for seq in seqs:
            pss, pss_max, pss_min = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                                                  align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                                  align_s=vo_stats[seq]['s'])
            self.poses     = np.vstack((self.poses, pss)) 
            self.poses_max = np.vstack((self.poses_max, [pss_max]))
            self.poses_min = np.vstack((self.poses_min, [pss_min]))

        if train:
            self.poses_max = np.max(self.poses_max, axis=0)  # (2,)
            self.poses_min = np.min(self.poses_min, axis=0)  # (2,)
            center_point   = list((np.array(list(self.poses_min)) + np.array(list(self.poses_max))) / 2) 
            np.savetxt(pose_max_min_filename, np.vstack((self.poses_max, self.poses_min)), fmt='%8.7f')
        else:
            self.poses_max, self.poses_min = np.loadtxt(pose_max_min_filename)
            center_point = list((np.array(list(self.poses_min)) + np.array(list(self.poses_max))) / 2) 

        self.augmentation = augmentation
        self.num_points   = num_points
        self.num_loc = num_loc
        self.num_ang = num_ang
        
        if train:
            print("train data num:" + str(len(self.poses)))
            print("train position classification num:" + str(self.num_loc * self.num_loc))
            print("train orientation classification num:" + str(self.num_ang))
        else:
            print("valid data num:" + str(len(self.poses)))
            print("valid position classification num:" + str(self.num_loc * self.num_loc))
            print("valid orientation classification num:" + str(self.num_ang))

    def __getitem__(self, index):
        scan_path = self.c_imgs[index]
        ptcld = load_velodyne_binary(scan_path)  # (4, N)
        scan  = ptcld[:3].transpose()  # (N, 3)
        scan  = ds_pc(scan, self.num_points)

        for a in self.augmentation:
            scan = a.apply(scan) 
            
        pose = self.poses[index]  # (6,)  
        loc  = position_classification(pose, self.poses_max, self.poses_min, self.num_loc)  # (1, )
        ang  = orientation_classification(pose, self.num_ang)  # (1, )

        return scan, pose, loc, ang

    def __len__(self):
        return self.poses.shape[0]


if __name__ == '__main__':
    vReLoc_dataset = vReLoc(data_path='/home/yss/Data/vReLoc', train=True, valid=True)
    print("fiished")