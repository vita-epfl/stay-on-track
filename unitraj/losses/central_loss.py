from unitraj.losses.offroad_loss import OffroadLossPolygons, OffroadLossCenterlines
from unitraj.losses.direction_consistency_loss import ConsistencyMetric
from unitraj.losses.diversity_loss import DiversityMetric
import torch.nn as nn
import torch
from typing import Union
import numpy as np
from itertools import compress


class CentralLoss(nn.Module):
    def __init__(self, config):
        super(CentralLoss, self).__init__()
        self.config = config
        self.offroad_waymo = OffroadLossCenterlines(config['offroad_margin'])
        self.offroad_others = OffroadLossPolygons(config['offroad_margin'])
        self.consistency = ConsistencyMetric(config['consistency_distance_threshold'], config['consistency_yaw_threshold'])
        self.diversity = DiversityMetric(config['diversity_remove_offroads'], config['diversity_offroad_threshold'], config['diversity_all_points'])

    def forward(self, pred, inputs, return_all_losses=False, calculate_all_losses=False):
        """
        Calculate the loss of the model
        :param pred: Predicted trajectory in the shape (B, C, T, 2) in tensor
        :param inputs: Dictionary containing the inputs to the model
        :param return_all_losses: Whether to return all the losses or just the total loss
        :return loss: Loss of the model
        """
        loss, offroad_loss, consistency_loss, diversity_loss = 0, torch.zeros(1), torch.zeros(1), torch.zeros(1)
        if self.config['aux_loss_type'] in ['offroad', 'combination'] or (self.config["aux_loss_type"] == "diversity" and self.config["diversity_remove_offroads"]) or calculate_all_losses:
            pred['predicted_trajectory'] = pred['predicted_trajectory'][..., :2]
            dataset_name = np.array(inputs['dataset_name'])
            waymo_mask = dataset_name == 'waymo'

            # calculate loss for waymo's scenarios. We use a different approach for waymo because it doesn't have drivable area polygons
            pred_waymo = pred['predicted_trajectory'][waymo_mask]
            map_lanes = inputs['map_polylines'][waymo_mask]
            map_mask = inputs['map_polylines_mask'][waymo_mask]
            map_type = torch.argmax(map_lanes[..., 0, -20:], axis=-1)
            map_lanes = map_lanes[..., :2]

            def fill_invalid_nodes_torch(map_mask, map_lanes):
                # Get the indices for each node along the N axis
                indices = torch.arange(map_lanes.size(2), device=map_lanes.device).view(1, 1, -1)

                # Set invalid indices to a large negative value (-1) so they aren't selected
                valid_indices = torch.where(map_mask, indices, torch.tensor(-1, device=map_lanes.device))

                # Get the last valid indices for each node
                last_valid_indices = torch.cummax(valid_indices, dim=2).values
                last_valid_indices[last_valid_indices == -1] = 0

                # Gather the last valid values for each node
                last_valid_values = torch.gather(map_lanes, 2, last_valid_indices.unsqueeze(-1).expand(-1, -1, -1,
                                                                                                       map_lanes.size(
                                                                                                           -1)))

                # Fill the invalid nodes with the last valid values
                filled_map_lanes = torch.where(map_mask.unsqueeze(-1), map_lanes, last_valid_values)

                return filled_map_lanes

            filled_map_lanes = fill_invalid_nodes_torch(map_mask, map_lanes)
            centerlines = torch.ones_like(map_lanes) * 2000
            centerlines[map_type == 2] = filled_map_lanes[map_type == 2]
            road_boundary = torch.ones_like(map_lanes) * 2000
            # road_boundary_mask = torch.logical_or(map_type == 7, map_type == 15)
            road_boundary_mask = map_type == 15
            road_boundary[road_boundary_mask] = filled_map_lanes[road_boundary_mask]
            offroad_loss_waymo = self.offroad_waymo(pred_waymo, road_boundary, centerlines)
            # set the loss to 0 for scenarios where there is no road boundary or centerlines
            offroad_loss_waymo[torch.where(torch.logical_or((map_type == 2).sum(axis=1) == 0, road_boundary_mask.sum(axis=1) == 0))] = 0
            # calculate loss for other datasets' scenarios in which we use drivable area polygons
            pred_others = pred['predicted_trajectory'][~waymo_mask]
            if not np.any(waymo_mask):
                drivable_multipolygon = inputs['boundary_polygon_pts']
                offroad_loss_others = self.offroad_others(pred_others, drivable_multipolygon)

            elif not np.all(waymo_mask):
                drivable_multipolygon = torch.from_numpy(np.stack(list(compress(inputs['boundary_polygon_pts'], ~waymo_mask)), axis=0)).to(pred_others.device)
                offroad_loss_others = self.offroad_others(pred_others, drivable_multipolygon)
            else:
                offroad_loss_others = torch.tensor(0.0)

            # combine the losses
            offroad_loss = torch.empty_like(pred['predicted_trajectory'][:, :, 0, 0])
            offroad_loss[waymo_mask] = offroad_loss_waymo
            offroad_loss[~waymo_mask] = offroad_loss_others
            loss = loss + offroad_loss.mean(dim=-1) * self.config['offroad_loss_weight']

        if self.config['aux_loss_type'] in ['consistency', 'combination'] or calculate_all_losses:
            dataset_name = np.array(inputs['dataset_name'])
            nuscenes_mask = dataset_name == 'nuscenes'

            # calculate loss for nuscenes' scenarios. We use a different approach for nuscenes because centerlines available in scenarionet are not good for this dataset
            pred_nuscenes = pred['predicted_trajectory'][nuscenes_mask]
            if np.all(nuscenes_mask):
                centerline_nodes = inputs["node_feats"]
                centerline_nodes_masks = inputs["node_feats_masks"]
            elif np.any(nuscenes_mask):
                centerline_nodes = torch.from_numpy(np.stack(list(compress(inputs["node_feats"], nuscenes_mask)), axis=0)).to(pred_nuscenes.device)
                centerline_nodes_masks = torch.from_numpy(np.stack(list(compress(inputs["node_feats_masks"], nuscenes_mask)), axis=0)).to(pred_nuscenes.device)

            if np.any(nuscenes_mask):
                centerline_nodes = centerline_nodes[..., :2].float()
                pred_nuscenes = pred_nuscenes[..., :2]
                consistency_loss_nuscenes = self.consistency(pred_nuscenes, centerline_nodes, centerline_nodes_masks)
            else:
                consistency_loss_nuscenes = torch.tensor(0.0)

            # calculate loss for other datasets for which we can use the normal centerlines available in scenarionet
            if not np.all(nuscenes_mask):
                pred_others = pred['predicted_trajectory'][~nuscenes_mask]
                map_lanes = inputs['map_polylines'][~nuscenes_mask]
                map_mask = inputs['map_polylines_mask'][~nuscenes_mask]
                map_type = torch.argmax(map_lanes[..., 0, -20:], dim=-1)
                map_lanes = map_lanes[..., :2]
                centerline_nodes = torch.zeros_like(map_lanes)
                centerline_nodes_masks = torch.zeros_like(map_mask)
                centerline_nodes[map_type == 2] = map_lanes[map_type == 2]
                centerline_nodes_masks[map_type == 2] = map_mask[map_type == 2]
                # reverse the centerline_nodes_mask to match the description of mask in nuscenes (1 for invalid, 0 for valid)
                centerline_nodes_masks = ~centerline_nodes_masks
                pred_others = pred_others[..., :2]
                consistency_loss_others = self.consistency(pred_others, centerline_nodes, centerline_nodes_masks)
                # set the loss to 0 for scenarios where there are no centerlines
                consistency_loss_others[torch.where((map_type == 2).sum(axis=1)==0)] = 0
            else:
                consistency_loss_others = torch.tensor(0.0)

            # combine the losses
            consistency_loss = torch.empty_like(pred['predicted_trajectory'][:, :, 0, 0])
            consistency_loss[nuscenes_mask] = consistency_loss_nuscenes
            consistency_loss[~nuscenes_mask] = consistency_loss_others
            # remove inf from consistency_loss
            consistency_loss[torch.isinf(consistency_loss)] = 0

            loss = loss + consistency_loss.mean(dim=-1) * self.config['consistency_loss_weight']

        if self.config['aux_loss_type'] in ['diversity', 'combination'] or calculate_all_losses:
            if not self.config["diversity_remove_offroads"]:
                offroad_loss = None
            diversity_loss = self.diversity(pred['predicted_trajectory'][..., :2], offroad_loss)

            loss = loss - diversity_loss * self.config['diversity_loss_weight']

        if not return_all_losses:
            return loss.mean()
        else:
            return {"loss": loss, "offroad_loss": offroad_loss.mean(dim=-1), "consistency_loss": consistency_loss.mean(dim=-1), "diversity_loss": diversity_loss}
