import torch


class DiversityMetric(torch.nn.Module):
    def __init__(self, remove_offroads=True, offroad_threshold=2, all_points=False):
        super(DiversityMetric, self).__init__()
        self.remove_offroads = remove_offroads
        self.offroad_threshold = offroad_threshold
        self.all_points = all_points

    def forward(self, preds, offroads=None):
        """
        Calculates the Diversity Metric of the predicted trajectories.
        :param pred: Predicted trajectory in the shape (B, K, T, 2) in tensor
        :param offroads: Offroad scores of predicted trajectories in the shape (B, K) in tensor
        :return diversity: Diversity metric of predicted trajectory (the higher the better) of shape (B)
        """
        if self.remove_offroads:
            if offroads is None:
                raise ValueError("Offroad scores are required to remove offroad trajectories.")
            offroad_mask = offroads < self.offroad_threshold
        else:
            offroad_mask = torch.ones_like(preds[:, :, -1, 0], dtype=torch.bool)

        # Step 1: Compute pairwise distances for all points, then mask
        if not self.all_points:
            all_pairs_dist = torch.cdist(preds[:, :, -1, :], preds[:, :, -1, :], p=2)  # Compute all pairwise distances for final points
        else:
            all_pairs_dist = torch.norm(preds.unsqueeze(2) - preds.unsqueeze(1), p=2, dim=-1)  # Compute all pairwise distances for all points

        # Step 2: Apply mask to select relevant distances
        # Expand offroad_mask for broadcasting
        mask_expanded = offroad_mask[:, :, None] & offroad_mask[:, None, :]
        if self.all_points:
            mask_expanded = mask_expanded.unsqueeze(-1).expand(-1, -1, -1, preds.size(2))

        # Use the expanded mask to select distances, setting irrelevant distances to 0
        all_pairs_dist_masked = torch.where(mask_expanded, all_pairs_dist, torch.tensor(0.0))

        # Step 3: Sum the relevant distances
        # Since each distance is counted twice, divide by 2.
        if not self.all_points:
            diversity = all_pairs_dist_masked.sum(dim=(-2, -1)) / 2
        else:
            diversity = all_pairs_dist_masked.sum(dim=(-3, -2, -1)) / 2 / preds.size(2)

        return diversity