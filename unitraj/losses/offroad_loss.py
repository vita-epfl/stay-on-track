import torch


""" 
We have different versions of the OffroadLoss class. 
The first one is for the nuscenes dataset and the second one is for the general case.
The reason is that the nuscenes dataset has closed drivable area polygons, so we need to use ray casting to determine if 
the point is inside or outside the polygon.
For other datasets, we check if the line connecting the point to the closest centerline point intersects with road 
boundary to determine if the point is inside or outside the polygon.
"""

class OffroadLossPolygons(torch.nn.Module):
    """
    Loss function that penalizes points for being far from the drivable area.
    It accepts a batch of points and a drivable polygon and returns the sum of the distances of each point to the drivable area.
    It has a parameter threshold that is added to the distance to the drivable area, so the loss is zero if the point is inside the drivable area by a margin of threshold.

    The drivable_polygon should be a tensor of shape (B, K, N, 2). For each scene in the batch, there are K polygon, each having N points (the first and last should be the same).
    The points_tensor should be a tensor of shape (B, C, T, 2). B is the batch size, C is the number of prediction modes, and T is the number of time steps.

    You can get the drivable area from a NuScenes map and make a tensor of it like this:
    ```
    drivable_multipolygon = get_drivable_area(instance_token, sample_token, helper, maps)
    polygons = list(drivable_multipolygon)
    polygon_xys = []
    for polygon in polygons:
        # Process the exterior ring
        exterior_xy = np.array(polygon.exterior.coords)
        xy_fixed_size = np.zeros((num_max_points, 2))
        xy_fixed_size[:len(exterior_xy), :] = exterior_xy
        xy_fixed_size[len(exterior_xy):, :] = exterior_xy[-1]
        polygon_xys.append(xy_fixed_size)

        # Process each interior ring (hole)
        for interior in polygon.interiors:
            interior_xy = np.array(interior.coords)
            xy_fixed_size = np.zeros((num_max_points, 2))
            xy_fixed_size[:len(interior_xy), :] = interior_xy
            xy_fixed_size[len(interior_xy):, :] = 200
            polygon_xys.append(xy_fixed_size)
    polygon_xys = np.array(polygon_xys)
    ```
    """
    def __init__(self, threshold=0.5):
        super(OffroadLossPolygons, self).__init__()
        self.threshold = threshold
        self.relu = torch.nn.ReLU()

    def forward(self, points, drivable_multipolygon):
        B, C, T, _ = points.size()
        _, K, N, _ = drivable_multipolygon.size()

        # Expand drivable_multipolygon to match points' dimensions [B, K, 1, 1, N, 2] for broadcasting
        drivable_multipolygon_expanded = drivable_multipolygon.unsqueeze(2).unsqueeze(3).expand(B, K, C, T, N, 2)

        # Points need to be expanded to [B, K, C, T, N, 2] to compute distances to each polygon edge
        points_expanded = points.unsqueeze(1).unsqueeze(4).expand(-1, K, -1, -1, N - 1, -1)

        # Calculate vector from each point to both ends of each polygon edge
        v1 = points_expanded - drivable_multipolygon_expanded[:, :, :, :, :-1, :]

        # Edge vectors
        edge_vectors = drivable_multipolygon_expanded[:, :, :, :, 1:, :] - drivable_multipolygon_expanded[:, :, :, :, :-1, :]

        # Compute dot products
        dot_product = (v1 * edge_vectors).sum(dim=-1)
        edge_sq = (edge_vectors * edge_vectors).sum(dim=-1)
        epsilon = 1e-6
        proj = torch.clamp(dot_product / (edge_sq + epsilon), 0, 1)

        # Find the closest point on the edge to the point
        closest = edge_vectors * proj.unsqueeze(-1) + drivable_multipolygon_expanded[:, :, :, :, :-1, :]

        # Distance from the point to the closest point on the edge
        dist_sq = ((points_expanded - closest) ** 2).sum(dim=-1)
        dist_sq = torch.clamp(dist_sq, min=epsilon)

        # Minimum distance to any of the drivable areas (edges) for each point
        min_dist, _ = torch.sqrt(dist_sq).min(dim=-1)  # Minimum across N-1 edges
        min_dist, _ = min_dist.min(dim=1)  # Minimum across K polygons

        ##################################### Compute the inside mask for each point#####################################
        edge_start = drivable_multipolygon_expanded[:, :, :, :, :-1, :]
        edge_end = drivable_multipolygon_expanded[:, :, :, :, 1:, :]

        cond_y = (edge_start[..., 1] <= points_expanded[..., 1]) & (points_expanded[..., 1] < edge_end[..., 1]) | \
                 (edge_end[..., 1] <= points_expanded[..., 1]) & (points_expanded[..., 1] < edge_start[..., 1])

        edge_slope = (edge_end[..., 1] - edge_start[..., 1]) / (edge_end[..., 0] - edge_start[..., 0] + epsilon)
        intersect_x = edge_start[..., 0] + (points_expanded[..., 1] - edge_start[..., 1]) / (edge_slope + epsilon)

        is_left = intersect_x > points_expanded[..., 0]

        intersection = cond_y & is_left

        # Sum intersections across all polygons before determining if inside/outside
        inside = intersection.sum(dim=-1).sum(dim=1) % 2 == 1

        # Apply the mask to the minimum distances
        # min_dist *= 1 - inside.float()
        min_dist_inside = min_dist * -1

        # Use where to select negative distances for inside points and positive for outside
        min_dist = torch.where(inside, min_dist_inside, min_dist)

        # Apply the threshold
        min_dist = self.relu(min_dist + self.threshold)

        # Here you could sum or mean these distances depending on your loss calculation needs
        # For example, you might want the mean distance per batch or the total sum.
        return min_dist.sum(-1)


class OffroadLossCenterlines(torch.nn.Module):
    """
    Loss function that penalizes points for being far from the drivable area.
    It accepts a batch of points, a road boundary and centerlines, and returns the sum of the distances of each point to the drivable area.
    It has a parameter threshold that is added to the distance to the drivable area, so the loss is zero if the point is inside the drivable area by a margin of threshold.

    The road_boundary should be a tensor of shape (B, K, N, 2). For each scene in the batch, there are K boundary lines, each having N points.
    The centerline should be of the same shape as the drivable_polygon, (B, S, P, 2) and should contain the centerlines of the roads.
    The points_tensor should be a tensor of shape (B, C, T, 2). B is the batch size, C is the number of prediction modes, and T is the number of time steps.
    """

    def __init__(self, threshold=0.5):
        super(OffroadLossCenterlines, self).__init__()
        self.threshold = threshold
        self.relu = torch.nn.ReLU()

    def forward(self, points, road_boundary, centerlines):
        B, C, T, _ = points.size()
        _, K, N, _ = road_boundary.size()

        # Expand drivable_multipolygon to match points' dimensions [B, K, 1, 1, N, 2] for broadcasting
        road_boundary_expanded = road_boundary.unsqueeze(2).unsqueeze(3).expand(B, K, C, T, N, 2)

        # Points need to be expanded to [B, K, C, T, N, 2] to compute distances to each polygon edge
        points_expanded = points.unsqueeze(1).unsqueeze(4).expand(-1, K, -1, -1, N - 1, -1)

        # Calculate vector from each point to both ends of each polygon edge
        v1 = points_expanded - road_boundary_expanded[:, :, :, :, :-1, :]

        # Edge vectors
        edge_vectors = road_boundary_expanded[:, :, :, :, 1:, :] - road_boundary_expanded[:, :, :, :, :-1, :]

        # Compute dot products
        dot_product = (v1 * edge_vectors).sum(dim=-1)
        edge_sq = (edge_vectors * edge_vectors).sum(dim=-1)

        epsilon = 1e-6
        proj = torch.clamp(dot_product / (edge_sq + epsilon), 0, 1)

        # Find the closest point on the edge to the point
        closest = edge_vectors * proj.unsqueeze(-1) + road_boundary_expanded[:, :, :, :, :-1, :]

        # Distance from the point to the closest point on the edge
        dist_sq = ((points_expanded - closest) ** 2).sum(dim=-1)

        # Minimum distance to any of the drivable areas (edges) for each point
        min_dist, _ = torch.sqrt(dist_sq).min(dim=-1)  # Minimum across N-1 edges
        min_dist, _ = min_dist.min(dim=1)  # Minimum across K polygons

        ##################################### Compute the inside mask for each point#####################################
        # Now find the closest centerline point for each point
        _, S, P, _ = centerlines.size()
        centerlines = centerlines.reshape(B, S * P, 2)
        centerlines_expanded = centerlines.unsqueeze(2).unsqueeze(3).expand(B, S * P, C, T, 2)
        v2 = points.unsqueeze(1).expand(-1, S * P, -1, -1, -1) - centerlines_expanded
        centerline_dist_sq = (v2 ** 2).sum(dim=-1)
        closest_centerline_dist, closest_centerline_idx = torch.min(centerline_dist_sq, dim=1)  # Minimum over P points

        # Find the closest centerline points (shape is B, C, T, 2)
        closest_centerline_points = torch.gather(centerlines_expanded, 1, closest_centerline_idx.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, -1, 2)).squeeze(1)

        def check_intersection(a1, a2, b1, b2):
            """ Vectorized check if line segment (a1, a2) intersects with (b1, b2) """
            def cross(v1, v2):
                return v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
            da = a2 - a1
            db = b2 - b1
            dp = a1 - b1
            dap = cross(da, dp)
            ddp = cross(db, dp)
            dadb = cross(da, db)
            t = dap / (dadb + epsilon)
            u = ddp / (dadb + epsilon)
            ep = 0
            intersect = ((t >= -ep) & (t <= 1 + ep) & (u >= 0) & (u <= 1))
            return intersect

        # expand points and closest_centerline_points to B, K, N-1, C, T, 2
        a1 = points.unsqueeze(1).unsqueeze(1).expand(-1, K, N - 1, -1, -1, -1)
        a2 = closest_centerline_points.unsqueeze(1).unsqueeze(1).expand(-1, K, N - 1, -1, -1, -1)

        # expand drivable_multipolygon to B, K, N-1, C, T, 2
        b1 = road_boundary[:, :, :-1].unsqueeze(3).unsqueeze(3).expand(-1, -1, -1, C, T, -1)
        b2 = road_boundary[:, :, 1:].unsqueeze(3).unsqueeze(3).expand(-1, -1, -1, C, T, -1)

        # check intersection
        intersection = check_intersection(a1, a2, b1, b2)

        # create inside mask
        inside = torch.logical_not(intersection.any(dim=2).any(dim=1))

        # Apply the mask to the minimum distances
        min_dist_inside = min_dist * -1

        # Use where to select negative distances for inside points and positive for outside
        min_dist = torch.where(inside, min_dist_inside, min_dist)

        # Apply the threshold
        min_dist = self.relu(min_dist + self.threshold)

        # Here you could sum or mean these distances depending on your loss calculation needs
        # For example, you might want the mean distance per batch or the total sum.
        return min_dist.sum(-1)
# class OffroadLoss(torch.nn.Module):
#     """
#     Loss function that penalizes points for being far from the drivable area.
#     It accepts a batch of points, a drivable polygon and centerlines and returns the sum of the distances of each point to the drivable area.
#     It has a parameter threshold that is added to the distance to the drivable area, so the loss is zero if the point is inside the drivable area by a margin of threshold.
#
#     The drivable_polygon should be a tensor of shape (B, K, N, 2). For each scene in the batch, there are K polygon, each having N points (the first and last should be the same).
#     The centerline should be of the same shape as the drivable_polygon, (B, S, P, 2) and should contain the centerlines of the roads.
#     The points_tensor should be a tensor of shape (B, C, T, 2). B is the batch size, C is the number of prediction modes, and T is the number of time steps.
#     nuscenes_mask is a tensor of shape (B) that is 1 if the point is from the nuscenes dataset and 0 otherwise. It is used to calculate the inside mask for each point differently than other datasets.
#     """
#
#     def __init__(self, threshold=0.5):
#         super(OffroadLoss, self).__init__()
#         self.threshold = threshold
#         self.relu = torch.nn.ReLU()
#
#     def forward(self, points, drivable_multipolygon, centerlines, nuscenes_mask):
#         # This function consiste of multiple steps:
#         ##################################### Compute the distance to closest point road boundary #####################################
#         B, C, T, _ = points.size()
#         _, K, N, _ = drivable_multipolygon.size()
#
#         # Expand drivable_multipolygon to match points' dimensions [B, K, 1, 1, N, 2] for broadcasting
#         drivable_multipolygon_expanded = drivable_multipolygon.unsqueeze(2).unsqueeze(3).expand(B, K, C, T, N, 2)
#
#         # Points need to be expanded to [B, K, C, T, N, 2] to compute distances to each polygon edge
#         points_expanded = points.unsqueeze(1).unsqueeze(4).expand(-1, K, -1, -1, N - 1, -1)
#
#         # Calculate vector from each point to both ends of each polygon edge
#         v1 = points_expanded - drivable_multipolygon_expanded[:, :, :, :, :-1, :]
#
#         # Edge vectors
#         edge_vectors = drivable_multipolygon_expanded[:, :, :, :, 1:, :] - drivable_multipolygon_expanded[:, :, :, :, :-1, :]
#
#         # Compute dot products
#         dot_product = (v1 * edge_vectors).sum(dim=-1)
#         edge_sq = (edge_vectors * edge_vectors).sum(dim=-1)
#
#         epsilon = 1e-6
#         proj = torch.clamp(dot_product / (edge_sq + epsilon), 0, 1)
#
#         # Find the closest point on the edge to the point
#         closest = edge_vectors * proj.unsqueeze(-1) + drivable_multipolygon_expanded[:, :, :, :, :-1, :]
#
#         # Distance from the point to the closest point on the edge
#         dist_sq = ((points_expanded - closest) ** 2).sum(dim=-1)
#
#         # Minimum distance to any of the drivable areas (edges) for each point
#         min_dist, _ = torch.sqrt(dist_sq).min(dim=-1)  # Minimum across N-1 edges
#         min_dist, _ = min_dist.min(dim=1)  # Minimum across K polygons
#
#         ##################################### Compute the inside mask for each point in non-nuscenes datasets#####################################
#         # We will see if the point is inside the polygon by checking if the line connecting it to the closes centerline intersects with any of the edges
#         # Now find the closest centerline point for each point
#         B, S, P, _ = centerlines[nuscenes_mask == 0].size()
#         centerlines = centerlines.reshape(B, S * P, 2)
#         centerlines_expanded = centerlines.unsqueeze(2).unsqueeze(3).expand(B, S * P, C, T, 2)
#         v2 = points[nuscenes_mask == 0].unsqueeze(1).expand(-1, S * P, -1, -1, -1) - centerlines_expanded
#         centerline_dist_sq = (v2 ** 2).sum(dim=-1)
#         closest_centerline_dist, closest_centerline_idx = torch.min(centerline_dist_sq, dim=1)  # Minimum over P points
#
#         # Find the closest centerline points (shape is B, C, T, 2)
#         closest_centerline_points = torch.gather(centerlines_expanded, 1, closest_centerline_idx.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, -1, 2)).squeeze(1)
#
#         def check_intersection(a1, a2, b1, b2):
#             """ Vectorized check if line segment (a1, a2) intersects with (b1, b2) """
#             def cross(v1, v2):
#                 return v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
#             da = a2 - a1
#             db = b2 - b1
#             dp = a1 - b1
#             dap = cross(da, dp)
#             ddp = cross(db, dp)
#             dadb = cross(da, db)
#             t = dap / (dadb + epsilon)
#             u = ddp / (dadb + epsilon)
#             ep = 3e-1
#             intersect = ((t + ep >= 0) & (t - ep <= 1) & (u >= 0) & (u <= 1))
#             return intersect
#
#         # expand points and closest_centerline_points to B, K, N-1, C, T, 2
#         a1 = points.unsqueeze(1).unsqueeze(1).expand(-1, K, N - 1, -1, -1, -1)
#         a2 = closest_centerline_points.unsqueeze(1).unsqueeze(1).expand(-1, K, N - 1, -1, -1, -1)
#
#         # expand drivable_multipolygon to B, K, N-1, C, T, 2
#         b1 = drivable_multipolygon[nuscenes_mask == 0][:, :, :-1].unsqueeze(3).unsqueeze(3).expand(-1, -1, -1, C, T, -1)
#         b2 = drivable_multipolygon[nuscenes_mask == 0][:, :, 1:].unsqueeze(3).unsqueeze(3).expand(-1, -1, -1, C, T, -1)
#
#         # check intersection
#         intersection = check_intersection(a1, a2, b1, b2)
#
#         # create inside mask
#         inside_non_nuscenes = torch.logical_not(intersection.any(dim=(1, 2)))
#
#         ##################################### Compute the inside mask for each point in nuscenes datasets#####################################
#         # For nuscenes we have closed drivable area polygons, so we will use ray casting to determine if the point is inside or outside the polygon
#         B = (nuscenes_mask == 1).sum()
#         drivable_multipolygon_expanded = drivable_multipolygon[nuscenes_mask == 1].unsqueeze(2).unsqueeze(3).expand(B, K, C, T, N, 2)
#         points_expanded = points[nuscenes_mask == 1].unsqueeze(1).unsqueeze(4).expand(-1, K, -1, -1, N - 1, -1)
#         edge_start = drivable_multipolygon_expanded[:, :, :, :, :-1, :]
#         edge_end = drivable_multipolygon_expanded[:, :, :, :, 1:, :]
#
#         cond_y = (edge_start[..., 1] <= points_expanded[..., 1]) & (points_expanded[..., 1] < edge_end[..., 1]) | \
#                  (edge_end[..., 1] <= points_expanded[..., 1]) & (points_expanded[..., 1] < edge_start[..., 1])
#
#         edge_slope = (edge_end[..., 1] - edge_start[..., 1]) / (edge_end[..., 0] - edge_start[..., 0] + epsilon)
#         intersect_x = edge_start[..., 0] + (points_expanded[..., 1] - edge_start[..., 1]) / (edge_slope + epsilon)
#
#         is_left = intersect_x > points_expanded[..., 0]
#
#         intersection = cond_y & is_left
#
#         # Sum intersections across all polygons before determining if inside/outside
#         inside_nuscenes = intersection.sum(dim=-1).sum(dim=1) % 2 == 1
#
#         ##################################### Combine the inside masks and return the final loss #####################################
#         inside = torch.zeros(len(nuscenes_mask), C, T, dtype=torch.bool, device=points.device)
#         inside[nuscenes_mask == 0] = inside_non_nuscenes
#         inside[nuscenes_mask == 1] = inside_nuscenes
#
#         # Use where to select negative distances for inside points and positive for outside
#         min_dist_inside = min_dist * -1
#         min_dist = torch.where(inside, min_dist_inside, min_dist)
#
#         # Apply the threshold
#         min_dist = self.relu(min_dist + self.threshold)
#
#         # Here you could sum or mean these distances depending on your loss calculation needs
#         # For example, you might want the mean distance per batch or the total sum.
#         return min_dist.sum(-1)