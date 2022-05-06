# pylint: disable=no-member
import argparse
import os
import sys
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data.OxfordVelodyne_datagenerator import RobotCar
from data.vReLoc_datagenerator import vReLoc
from data.composition import MF
from data.augment import get_augmentations_from_list, Normalize
from models.model import STCLoc
from models.loss import Criterion
from utils.pose_util import val_translation, val_rotation, val_classification, qexp
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader
from os import path as osp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
cudnn.enabled = True

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0,
                    help='gpu id for network, only effective when multi_gpus is false')
parser.add_argument('--batch_size', type=int, default=80,
                    help='Batch Size during training [default: 80]')
parser.add_argument('--val_batch_size', type=int, default=80,
                    help='Batch Size during validating [default: 80]')
parser.add_argument('--max_epoch', type=int, default=999,
                    help='Epoch to run [default: 100]')
parser.add_argument('--init_learning_rate', type=float, default=0.001, 
                    help='Initial learning rate [default: 0.001]')
parser.add_argument("--decay_step", type=float, default=500,
                    help="decay step for learning rate, default: 100 for 7scenes / 500 for Oxford")
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--seed', type=int, default=20, metavar='S',
                    help='random seed (default: 20)')
parser.add_argument('--log_dir', default='log-oxford/',
                    help='Log dir [default: log]')
parser.add_argument('--dataset_folder', default='/home/data',
                    help='Our Dataset Folder')
parser.add_argument('--dataset', default='Oxford', 
                    help='Oxford or vReLoc')
parser.add_argument('--num_workers', type=int, default=8, 
                    help='num workers for dataloader, default:4')
parser.add_argument('--num_points', type=int, default=4096,
                    help='Number of points to downsample model to')
parser.add_argument('--augmentation', type=str, nargs='+', default=[],
                    choices=['Jitter', 'RotateSmall', 'Scale', 'Shift', 'Rotate1D', 'Rotate3D'],
                    help='Data augmentation settings to use during training')
parser.add_argument('--upright_axis', type=int, default=2,
                    help='Will learn invariance along this axis')
parser.add_argument('--num_loc', type=int, default=10, 
                    help='position classification, default: 10')
parser.add_argument('--num_ang', type=int, default=10, 
                    help='orientation classification, default: 10')
parser.add_argument('--skip', type=int, default=2, 
                    help='Number of frames to skip')
parser.add_argument('--steps', type=int, default=3, 
                    help='Number of frames to return on every call')
parser.add_argument('--normalize', action='store_true', default=False,
                    help='use normalize or not, default not')
parser.add_argument('--real', action='store_true', default=False, 
                    help='if True, load poses from SLAM / integration of VO')
parser.add_argument('--variable_skip', action='store_true', default=False, 
                    help='If True, skip = [1, ..., skip]')
parser.add_argument('--multi_gpus', action='store_true', default=True, 
                    help='if use multi_gpus, default false')
parser.add_argument('--skip_val', action='store_true', default=False,
                    help='if skip validation during training, default False')
parser.add_argument('--resume_model', type=str, default='',
                    help='If present, restore checkpoint and resume training')


FLAGS = parser.parse_args()
args = vars(FLAGS)
for (k, v) in args.items():
    print('%s: %s' % (str(k), str(v)))
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

LOG_FOUT = open(os.path.join(FLAGS.log_dir, 'log.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')
TOTAL_ITERATIONS = 0
if not FLAGS.multi_gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise ValueError("GPU not found!")
else:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise ValueError("GPU not found!")

train_augmentations = get_augmentations_from_list(FLAGS.augmentation, upright_axis=FLAGS.upright_axis)
valid_augmentations = []
if FLAGS.normalize:
    if FLAGS.dataset == 'vReLoc':
        stats_file = osp.join(FLAGS.dataset_folder, FLAGS.dataset, 'stats.txt')
    elif FLAGS.dataset == 'Oxford':
        stats_file = osp.join(FLAGS.dataset_folder, FLAGS.dataset, 'stats.txt')
    else:
        raise ValueError("dataset error!")
    stats = np.loadtxt(stats_file, dtype=np.float32)
    normalize_aug = Normalize(mean=stats[0], std=np.sqrt(stats[1]))
    train_augmentations.append(normalize_aug)
    valid_augmentations.append(normalize_aug)

train_kwargs = dict(data_path=FLAGS.dataset_folder, 
                    augmentation=train_augmentations, 
                    num_points=FLAGS.num_points, 
                    train=True, 
                    valid=False, 
                    num_loc=FLAGS.num_loc, 
                    num_ang=FLAGS.num_ang)
valid_kwargs = dict(data_path=FLAGS.dataset_folder, 
                    augmentation=valid_augmentations, 
                    num_points=FLAGS.num_points, 
                    train=False, 
                    valid=True, 
                    num_loc=FLAGS.num_loc, 
                    num_ang=FLAGS.num_ang)
pose_stats_file = os.path.join(FLAGS.dataset_folder, FLAGS.dataset, 'pose_stats.txt')
pose_m, pose_s  = np.loadtxt(pose_stats_file) 
Plus_kwargs  = dict(dataset=FLAGS.dataset, 
                    skip=FLAGS.skip, 
                    steps=FLAGS.steps, 
                    variable_skip=FLAGS.variable_skip, 
                    real=FLAGS.real)
train_kwargs = dict(**train_kwargs, **Plus_kwargs)
valid_kwargs = dict(**valid_kwargs, **Plus_kwargs)
train_set = MF(**train_kwargs)
val_set   = MF(**valid_kwargs)
train_loader = DataLoader(train_set, 
                        batch_size=FLAGS.batch_size, 
                        shuffle=True, 
                        num_workers=FLAGS.num_workers, 
                        pin_memory=True)
val_loader = DataLoader(val_set, 
                        batch_size=FLAGS.val_batch_size, 
                        shuffle=False, 
                        num_workers=FLAGS.num_workers, 
                        pin_memory=True)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train():
    global TOTAL_ITERATIONS
    setup_seed(FLAGS.seed)
    train_writer = SummaryWriter(os.path.join(FLAGS.log_dir, 'train'))
    val_writer   = SummaryWriter(os.path.join(FLAGS.log_dir, 'valid'))
    model        = STCLoc(FLAGS.steps, FLAGS.num_loc*FLAGS.num_loc, FLAGS.num_ang)
    loss         = Criterion()
    model        = model.to(device)
    loss         = loss.to(device)

    if FLAGS.optimizer == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), FLAGS.init_learning_rate)
    elif FLAGS.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), FLAGS.init_learning_rate)
    else:
        optimizer = None
        exit(0)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, FLAGS.decay_step, gamma=0.95)

    if len(FLAGS.resume_model) > 0:
        resume_filename  = FLAGS.log_dir + FLAGS.resume_model
        print("Resuming From ", resume_filename)
        checkpoint       = torch.load(resume_filename)
        saved_state_dict = checkpoint['state_dict']
        starting_epoch   = checkpoint['epoch'] + 1
        TOTAL_ITERATIONS = starting_epoch * len(train_set)
        model.load_state_dict(saved_state_dict)
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        starting_epoch = 0

    if FLAGS.multi_gpus:
        model = nn.DataParallel(model)

    LOG_FOUT.write("\n")
    LOG_FOUT.flush()

    for epoch in range(starting_epoch, FLAGS.max_epoch):
        log_string('**** EPOCH %03d ****' % epoch)
        sys.stdout.flush()
        if not FLAGS.skip_val and epoch % 5 == 0:
            valid_one_epoch(model, val_loader, val_writer, device)
            torch.cuda.empty_cache()
        train_one_epoch(model, train_loader, scheduler, epoch, train_writer, loss, device)
        torch.cuda.empty_cache()


def train_one_epoch(model, train_loader, scheduler, epoch, train_writer, loss, device):
    global TOTAL_ITERATIONS
    for step, (train_data, train_pose, train_loc, train_ori) in enumerate(train_loader):
        TOTAL_ITERATIONS += 1
        pcs_tensor     = train_data.to(device, dtype=torch.float32)  # [B, T, N, 3]
        s              = pcs_tensor.size()  # B, T, N, 3
        pcs_tensor     = pcs_tensor.view(-1, *s[2:])  # [B*T, N, 3]
        gt_t           = train_pose[..., :3].to(device, dtype=torch.float32)  # [B, T, 3]
        gt_q           = train_pose[..., 3:].to(device, dtype=torch.float32)  # [B, T, 3]
        gt_t           = gt_t.view(s[0] * s[1], 3)  # [B*T, 3]
        gt_q           = gt_q.view(s[0] * s[1], 3)  # [B*T, 3]
        gt_loc         = train_loc.to(device, dtype=torch.float32)  # [B, T]
        gt_ori         = train_ori.to(device, dtype=torch.float32)  # [B, T]
        gt_loc         = gt_loc.view(s[0] * s[1])   # [B*T]
        gt_ori         = gt_ori.view(s[0] * s[1])   # [B*T]

        scheduler.optimizer.zero_grad()
        pred_t, pred_q, pred_loc, pred_ori = run_model(model, pcs_tensor, validate=False)
        train_loss = loss(pred_t, pred_q, pred_loc, pred_ori, gt_t, gt_q, gt_loc, gt_ori)
        train_loss.backward()
        scheduler.optimizer.step()
        scheduler.step()
        log_string('Loss: %f' % train_loss)
        train_writer.add_scalar('Loss', train_loss.cpu().item(), TOTAL_ITERATIONS)

    if epoch % 1 == 0:
        if isinstance(model, nn.DataParallel):
            model_to_save = model.module
        else:
            model_to_save = model
        torch.save({
            'epoch': epoch,
            'iter': TOTAL_ITERATIONS,
            'state_dict': model_to_save.state_dict(),
            'scheduler': scheduler.state_dict(),
        },
            FLAGS.log_dir+'checkpoint_epoch{}.tar'.format(epoch))
        print("Model Saved As " + 'checkpoint_epoch{}.tar'.format(epoch))


def valid_one_epoch(model, val_loader, val_writer, device):
    gt_translation      = np.zeros((len(val_set), 3))
    pred_translation    = np.zeros((len(val_set), 3))
    gt_rotation         = np.zeros((len(val_set), 4))
    pred_rotation       = np.zeros((len(val_set), 4))
    error_t             = np.zeros(len(val_set))
    error_q             = np.zeros(len(val_set))
    correct_loc_results = []
    correct_ori_results = []
    time_results        = []
    for step, (val_data, val_pose, val_loc, val_ori) in enumerate(val_loader):
        start_idx                            = step * FLAGS.val_batch_size
        end_idx                              = min((step+1)*FLAGS.val_batch_size, len(val_set)) 
        val_pose                             = val_pose[:, -1, :]  
        gt_translation[start_idx:end_idx, :] = val_pose[:, :3].numpy() * pose_s + pose_m
        gt_rotation[start_idx:end_idx, :]    = np.asarray([qexp(q) for q in val_pose[:, 3:].numpy()]) 
        val_loc                              = val_loc[:, -1]  
        val_ori                              = val_ori[:, -1]  
        gt_loc                               = val_loc.to(device, dtype=torch.float32)  
        gt_ori                               = val_ori.to(device, dtype=torch.float32) 
        pcs_tensor                           = val_data.to(device)                  
        s                                    = pcs_tensor.size()  # B, T, N, 3
        pcs_tensor                           = pcs_tensor.view(-1, *s[2:])  # [B*T, N, 3]
        
        # run model and time eval
        start = time.time()         
        pred_t, pred_q, pred_loc, pred_ori = run_model(model, pcs_tensor, validate=True)
        end = time.time()
        cost_time = end - start
        time_results.append(cost_time)

        # last frame
        pred_t   = pred_t.view(s[0], s[1], 3)
        pred_q   = pred_q.view(s[0], s[1], 3)
        pred_t   = pred_t[:, -1, :] 
        pred_q   = pred_q[:, -1, :]     
        pred_loc = pred_loc.view(s[0], s[1], -1)
        pred_ori = pred_ori.view(s[0], s[1], -1)
        pred_loc = pred_loc[:, -1, :] 
        pred_ori = pred_ori[:, -1, :] 

        # RTE / RRE
        pred_translation[start_idx:end_idx, :] = pred_t.cpu().numpy() * pose_s + pose_m
        pred_rotation[start_idx:end_idx, :]    = np.asarray([qexp(q) for q in pred_q.cpu().numpy()])  
        error_t[start_idx:end_idx]             = np.asarray([val_translation(p, q) for p, q in zip(pred_translation[start_idx:end_idx, :], gt_translation[start_idx:end_idx, :])])
        error_q[start_idx:end_idx]             = np.asarray([val_rotation(p, q) for p, q in zip(pred_rotation[start_idx:end_idx, :], gt_rotation[start_idx:end_idx, :])])
        
        # classification loc and ang
        pred_loc_cls = val_classification(pred_loc, gt_loc)
        pred_ori_cls = val_classification(pred_ori, gt_ori)
        correct_loc_results.append(pred_loc_cls.item()/(end_idx - start_idx))
        correct_ori_results.append(pred_ori_cls.item()/(end_idx - start_idx))

        log_string('MeanTE(m): %f' % np.mean(error_t[start_idx:end_idx], axis=0))
        log_string('MeanRE(degrees): %f' % np.mean(error_q[start_idx:end_idx], axis=0))
        log_string('MedianTE(m): %f' % np.median(error_t[start_idx:end_idx], axis=0))
        log_string('MedianRE(degrees): %f' % np.median(error_q[start_idx:end_idx], axis=0))
        log_string('Cls_loc: %f' % np.mean(correct_loc_results[step]))
        log_string('Cls_ori: %f' % np.mean(correct_ori_results[step]))

    mean_time  = np.mean(time_results)
    mean_ATE   = np.mean(error_t)
    mean_ARE   = np.mean(error_q)
    median_ATE = np.median(error_t)
    median_ARE = np.median(error_q)
    mean_loc   = np.mean(correct_loc_results)
    mean_ori   = np.mean(correct_ori_results)

    log_string('Mean Cost Time(s): %f' % mean_time)
    log_string('Mean Position Error(m): %f' % mean_ATE)
    log_string('Mean Orientation Error(degrees): %f' % mean_ARE)
    log_string('Median Position Error(m): %f' % median_ATE)
    log_string('Median Orientation Error(degrees): %f' % median_ARE)
    log_string('Mean Loc Acc: %f' % mean_loc)
    log_string('Mean Ori Acc: %f' % mean_ori)
    
    val_writer.add_scalar('MeanTime', mean_time, TOTAL_ITERATIONS)
    val_writer.add_scalar('MeanATE', mean_ATE, TOTAL_ITERATIONS)
    val_writer.add_scalar('MeanARE', mean_ARE, TOTAL_ITERATIONS)
    val_writer.add_scalar('MedianATE', median_ATE, TOTAL_ITERATIONS)
    val_writer.add_scalar('MedianARE', median_ARE, TOTAL_ITERATIONS)
    val_writer.add_scalar('MeanLoc', mean_loc, TOTAL_ITERATIONS)
    val_writer.add_scalar('MeanOri', mean_ori, TOTAL_ITERATIONS)
    
    # trajectory
    fig = plt.figure()
    real_pose = pred_translation - pose_m
    gt_pose = gt_translation - pose_m
    plt.scatter(gt_pose[:, 1], gt_pose[:, 0], s=3, c='black')
    plt.scatter(real_pose[:, 1], real_pose[:, 0], s=3, c='red')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=10)
    image_filename = os.path.join(os.path.expanduser(FLAGS.log_dir), '{:s}.png'.format('trajectory'))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # save error and trajectory
    error_t_filename = osp.join(FLAGS.log_dir, 'error_t.txt')
    error_q_filename = osp.join(FLAGS.log_dir, 'error_q.txt')
    pred_t_filename  = osp.join(FLAGS.log_dir, 'pred_t.txt')
    gt_t_filename    = osp.join(FLAGS.log_dir, 'gt_t.txt')
    np.savetxt(error_t_filename, error_t, fmt='%8.7f')
    np.savetxt(error_q_filename, error_q, fmt='%8.7f')
    np.savetxt(pred_t_filename, real_pose, fmt='%8.7f')
    np.savetxt(gt_t_filename, gt_pose, fmt='%8.7f')


def run_model(model, PC, validate=False):
    if not validate:
        model.train()
        return model(PC)
    else:
        with torch.no_grad():
            model.eval()
            return model(PC)


if __name__ == "__main__":
    train()