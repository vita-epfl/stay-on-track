import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.utils import set_seed
from models import build_model
from datasets import build_dataset
import hydra
from omegaconf import OmegaConf
import pickle
from unitraj.losses.central_loss import CentralLoss
from scene_attack import attack_batch
from copy import deepcopy
import numpy as np


def to_device(input, device):
    if isinstance(input, torch.Tensor):
        return input.to(device)
    elif isinstance(input, dict):
        return {k: to_device(v, device) for k, v in input.items()}
    elif isinstance(input, list):
        return [to_device(x, device) for x in input]
    else:
        return input


@hydra.main(version_base=None, config_path="configs", config_name="config")
def evaluate_scene_attack(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    print(f"attacked_offroads_{cfg['method']['model_name']}_{str(cfg['val_data_path']).split('/')[-2]}_{'baseline' if 'baseline' in cfg['ckpt_path'] else 'offroad'}.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.ckpt_path)["state_dict"])
    model.to(device)
    model.eval()

    # Load the test dataset
    test_dataset = build_dataset(cfg,val=True)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, collate_fn=test_dataset.collate_fn)  # Batch size can be adjusted

    # Initialize the loss function and evaluate the offroad loss for the original and attacked scenarios
    central_loss = CentralLoss(cfg)
    baseline_offroads, attacked_offroads = [], {"smooth-turn": {}, "double-turn": {}, "ripple-road": {}}
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            batch = to_device(batch, device)
            # Calculate the offroad loss for the original scenario
            output = model.forward(batch)
            loss = central_loss(output[0], batch['input_dict'], calculate_all_losses=True, return_all_losses=True)
            baseline_offroads += loss['offroad_loss'].cpu().numpy().tolist()

            # Calculate the offroad loss for the attacked scenarios
            default_params = {"smooth-turn": {"attack_power": 0, "pow": 3, "border": 5},
                              "double-turn": {"attack_power": 0, "pow": 3, "l": 10, "border": 5},
                              "ripple-road": {"attack_power": 0, "l": 60, "border": 5}}
            original_batch = deepcopy(batch)
            for attack_type in ["smooth-turn", "double-turn", "ripple-road"]:
                for attack_power in range(-4, 5):
                    if attack_power == 0:
                        continue
                    default_params[attack_type]["attack_power"] = attack_power
                    batch = deepcopy(original_batch)
                    attack_batch(batch, default_params)
                    attacked_output = model.forward(batch)
                    attacked_loss = central_loss(attacked_output[0], batch['input_dict'], calculate_all_losses=True, return_all_losses=True)
                    if attack_power not in attacked_offroads[attack_type]:
                        attacked_offroads[attack_type][attack_power] = []
                    attacked_offroads[attack_type][attack_power] += attacked_loss['offroad_loss'].cpu().numpy().tolist()
                default_params[attack_type]["attack_power"] = 0

    if 'baseline' in cfg['ckpt_path']:
        type = 'baseline'
    elif 'offroad' in cfg['ckpt_path']:
        type = 'offroad'
    elif 'combination' in cfg['ckpt_path']:
        type = 'combination'
    with open(f"attacked_offroads_{cfg['method']['model_name']}_{str(cfg['val_data_path']).split('/')[-2]}_{type}.pkl", "wb") as f:
        pickle.dump((baseline_offroads, attacked_offroads), f)


def create_table2():
    # Once all 12 models have been evaluated, call this function to print the table for the offroad loss on scene attack benchmark
    def do_print(offroads):
        original_offroads, attacked_offroads = offroads
        smooth_offroads = np.stack([attacked_offroads['smooth-turn'][i] for i in [-3, -2, -1, 1, 2, 3]])
        double_offroads = np.stack([attacked_offroads['double-turn'][i] for i in [-3, -2, -1, 1, 2, 3]])
        ripple_offroads = np.stack([attacked_offroads['ripple-road'][i] for i in [-3, -2, -1, 1, 2, 3]])
        print(f"{np.mean(original_offroads):.2f} & "
              f"{np.mean([smooth_offroads.mean(), double_offroads.mean(), ripple_offroads.mean()]):.2f}"
              , end="")

    for model, print_name in [("autobot", "Autobots"), ("wayformer", "Wayformer")]:
        print(print_name, end=" & ")
        for dataset in ["nuscenes", "av2"]:
            with open(f"unitraj/attacked_offroads_{model}_{dataset}_baseline.pkl", "rb") as f:
                baseline = pickle.load(f)
            do_print(baseline)
            if dataset == "av2":
                print("\\\\")
            else:
                print(" & ", end="")
        print("+ Offroad", end=" & ")
        for dataset in ["nuscenes", "av2"]:
            with open(f"unitraj/attacked_offroads_{model}_{dataset}_offroad.pkl", "rb") as f:
                offroad = pickle.load(f)
            do_print(offroad)
            if dataset == "av2":
                print("\\\\")
            else:
                print(" & ", end="")
        print("+ All", end=" & ")
        for dataset in ["nuscenes", "av2"]:
            with open(f"unitraj/attacked_offroads_{model}_{dataset}_combination.pkl", "rb") as f:
                all = pickle.load(f)
            do_print(all)
            if dataset == "av2":
                print("\\\\")
            else:
                print(" & ", end="")
        print("\\midrule")


if __name__ == "__main__":
    evaluate_scene_attack()
    # create_table2()  # Uncomment this line to print the table for the offroad loss on scene attack benchmark
