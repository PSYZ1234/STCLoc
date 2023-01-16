import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # [B, N, M]
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    # dist = 2 * (1 - torch.sum(src * dst, -1))  # feature distance: (B, N)

    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1  # (1, S)
    # make batch_indeces have same dimensions as view_shape
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)  # [B, S]
    new_points = points[batch_indices, idx, :]

    return new_points


def farthest_point_sample(xyz, npoint, points):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # [B, ]
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # [B, ]

    # if points is not None:
    #     print(f'centroids:{centroids[0].sum()}, distance:{distance[0].sum()}, farthest:{farthest[0].sum()}, batch_indices:{batch_indices[0].sum()}')
    #     exit(-1)

    for i in range(npoint):
        centroids[:, i] = farthest  # [B, N]
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)  # [B, N]
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]  # [B, ]

    # if points is not None:
    #     print(f'centroids:{centroids[0].sum()}, distance:{distance[0].sum()}, farthest:{farthest[0].sum()}, batch_indices:{batch_indices[0].sum()}, centroid:{centroid[0].sum()}, dist:{dist[0].sum()}, mask:{mask[0].sum()}')
    #     exit(-1)

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # [B, S, N]
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # [B, S, nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])  # [B, S, nsample]
    mask = group_idx == N
    # temp1 = torch.sum(mask, -1) > 0
    # temp2 = torch.sum(temp1).to(torch.float32)
    # padding_rate = temp2 / (B * S)
    # print("padding rate is:", padding_rate.item())
    # if the point number in the sphere less than nsample, pad with group_first
    group_idx[mask] = group_first[mask]

    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, normalize_radius=False, returnfps=False):
    """
    Input:
        npoint: keyponts number to sample
        radius: sphere radius in a group
        nsample: how many points to group for a sphere
        xyz: input points position data, [B, N, C]
        points: additional input points data, [B, N, D]
        normalize_radius: scale normalization
        returnfps: whether return FPS result
    Return:
        new_xyz: sampled points position data, [B, npoint, C]
        new_points: sampled points data, [B, npoint, nsample, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint, points)  # [B, npoint]
    # torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, C]
    # torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample]
    # torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    # torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # [B, npoint, nsample, C] translation normalization
    # torch.cuda.empty_cache()

    if normalize_radius:
        grouped_xyz_norm /= radius

    if points is not None:
        grouped_points = index_points(points, idx)
        fps_points = index_points(points, fps_idx)
        fps_points = torch.cat([new_xyz, fps_points], dim=-1)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
        fps_points = new_xyz

    # if points is not None:
    #     print(f'fps_idx:{fps_idx[0].sum()}, new_xyz:{new_xyz[0].sum()}, idx:{idx[0].sum()}, grouped_xyz:{grouped_xyz[0].sum()}, grouped_xyz_norm:{grouped_xyz_norm[0].sum()}, grouped_points:{grouped_points[0].sum()}, new_points:{new_points[0].sum()}')
    #     exit(-1)

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_points
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)

    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points


def similarity(feat1, feat2, feat3):
    B, _, D  = feat1.size()
    feat1    = feat1.view(B, -1)  # [B, D]
    feat2    = feat2.view(B, -1)  # [B, D]
    feat3    = feat3.view(B, -1)  # [B, D]
    # Euclidean Distance
    # feat1, feat2, feat3 = map(lambda x: F.normalize(x, p=2), [feat1, feat2, feat3])
    # w1       = (feat1 * feat2).sum(dim=1)  # [B, 1]``
    # w2       = (feat1 * feat3).sum(dim=1)  # [B, 1]``
    w1       = F.pairwise_distance(feat1, feat2, p=2, keepdim=True)  # [B, 1]
    w2       = F.pairwise_distance(feat1, feat3, p=2, keepdim=True)  # [B, 1]
    w1       = 1 / (1 + w1)  # [B, 1] 
    w2       = 1 / (1 + w2)  # [B, 1]
    # cos
    w3       = torch.cosine_similarity(feat1, feat2, dim=1)  # [B, 1]
    w4       = torch.cosine_similarity(feat1, feat3, dim=1)  # [B, 1]
    w3       = w3.view(B, -1)  # [B, 1]
    w4       = w4.view(B, -1)  # [B, 1]
    w3       = 0.5 + 0.5 * w3  # [B, 1]
    w4       = 0.5 + 0.5 * w4  # [B, 1]

    # weighted 
    feat_out = feat1 + (w1 + w3) * feat2  + (w2 + w4) * feat3  # [B, D]
    feat_out = feat_out.view(B, 1, D)  # [B, 1, D]

    return feat_out
