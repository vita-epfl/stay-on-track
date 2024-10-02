import torch
import numpy as np


class ConsistencyMetric(torch.nn.Module):
    def __init__(self, distance_threshold=2.0, yaw_threshold=np.pi/3):
        super(ConsistencyMetric, self).__init__()
        self.distance_threshold = distance_threshold
        self.yaw_threshold = yaw_threshold

    def forward(self, preds, centerline_nodes, centerline_node_masks):
        """
        Calculates the Consistency Metric of the predicted trajectories.
        :param pred: Predicted trajectory in the shape (B, K, T, 2) in tensor
        :param centerline_nodes: Map node poses and yaws in the shape (B, N, n, 2) in tensor
        :param centerline_node_masks: Mask for node_feats in the shape (B, N, n) in tensor
        :return consistency: Consistency metric of predicted trajectories (the lower, the better) of shape (B, K)
        """
        B, K, T, _ = preds.shape

        # Compute yaw values for predicted trajectory
        direction_vectors = preds[:, :, 1:, :2] - preds[:, :, :-1, :2]
        pred_yaw = -torch.atan2(direction_vectors[..., 0], direction_vectors[..., 1])
        pred_yaw = torch.cat([torch.zeros(B, preds.shape[1], 1, device=pred_yaw.device), pred_yaw], dim=-1)  # add a 0 yaw for the first point

        # calculate yaw and flatten the lane and points in lane dimensions
        direction_vectors = centerline_nodes[:, :, 1:, :2] - centerline_nodes[:, :, :-1, :2]
        node_feats_yaw = -torch.atan2(direction_vectors[..., 0], direction_vectors[..., 1])
        node_feats_yaw = torch.cat([torch.zeros(B, centerline_nodes.shape[1], 1, device=node_feats_yaw.device), node_feats_yaw], dim=-1)  # add a 0 yaw for the first point
        centerline_nodes = centerline_nodes.reshape(B, -1, 2)
        centerline_node_masks = centerline_node_masks.reshape(B, -1)

        # Compute the distances and yaw differences
        dists = torch.cdist(centerline_nodes[:, :, :2], preds[:, :, :, :2].reshape(B, -1, 2), p=2)
        # dists[node_feats_masks == 1] = torch.tensor(float('inf'))
        inf_mask = (centerline_node_masks == 1).unsqueeze(-1).expand_as(dists)
        dists = torch.where(inf_mask, torch.full_like(dists, float('inf')), dists)

        yaw_diffs = node_feats_yaw.reshape(B, -1).unsqueeze(2) - pred_yaw.reshape(B, -1).unsqueeze(1)
        yaw_diffs = (yaw_diffs + torch.tensor(np.pi)) % (2 * torch.tensor(np.pi)) - torch.tensor(np.pi)

        # Compute the consistency scores
        dists_rectified = torch.functional.F.relu(dists - self.distance_threshold)
        yaw_diffs_rectified = torch.functional.F.relu(torch.abs(yaw_diffs) - self.yaw_threshold)
        consistency = torch.min(dists_rectified + yaw_diffs_rectified, dim=1).values
        # min_id = torch.min(dists_rectified + yaw_diffs_rectified, dim=1).indices
        # consistency is the yaw_diffs_rectified for the minimum distance
        # consistency = yaw_diffs_rectified.gather(1, min_id.unsqueeze(2)).squeeze(2)
        consistency = consistency.reshape(B, K, T).sum(dim=-1)

        return consistency
