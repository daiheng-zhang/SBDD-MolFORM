import argparse
import os
import shutil
import numpy as np
import torch

from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_diffusion import sample_diffusion_ligand


def _strip_confidence_state_dict(state_dict):
    prefixes = ("cfd_node_head.", "cfd_pos_head.")
    return {k: v for k, v in state_dict.items() if not k.startswith(prefixes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--logdir", type=str, default="./outputs")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--name", type=str, default="diffusion_infer")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--data_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--split_path", type=str, default="")
    args = parser.parse_args()

    branch, version = misc.get_version()
    version_short = "%s-%s" % (branch, version[:7])

    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[: os.path.basename(args.config).rfind(".")]

    log_dir = misc.get_new_log_dir(
        args.logdir, prefix=f"{config_name}_{version_short}_{args.tag}"
    )
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "commit.txt"), "w") as f:
        f.write(branch + "\n")
        f.write(version + "\n")

    logger = misc.get_logger("infer", log_dir)
    logger.info(args)
    logger.info(config)

    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copyfile("./scripts/infer_diffusion.py", os.path.join(log_dir, "infer_diffusion.py"))

    misc.seed_all(config.sample.seed)

    ckpt_path = args.checkpoint if args.checkpoint else config.model.checkpoint
    ckpt = misc.load_checkpoint(ckpt_path, map_location=args.device)
    train_config = misc.to_easydict(ckpt["config"])
    dataset_config = misc.resolve_dataset_config(
        config,
        train_config,
        data_path=args.data_path or None,
        split_path=args.split_path or None,
    )
    logger.info(f"Training Config: {train_config}")
    logger.info(f"Resolved dataset config: {dataset_config}")

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = dataset_config.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    bond_featureizer = trans.FeaturizeLigandBond()

    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        bond_featureizer,
    ])

    dataset, subsets = get_dataset(
        config=dataset_config,
        transform=transform,
    )
    split_set = subsets[args.split]
    logger.info(f"Successfully load the dataset ({args.split}, size: {len(split_set)})!")

    new_state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    new_state_dict = _strip_confidence_state_dict(new_state_dict)

    model = ScorePosNet3D(
        train_config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(args.device)
    model.load_state_dict(new_state_dict)

    if args.data_id < 0 or args.data_id >= len(split_set):
        raise ValueError(f"data_id {args.data_id} out of range (0, {len(split_set) - 1})")

    data = split_set[args.data_id]

    num_samples = args.num_samples if args.num_samples is not None else config.sample.num_samples
    num_steps = args.num_steps if args.num_steps is not None else config.sample.num_steps

    sample_out = sample_diffusion_ligand(
        model=model,
        data=data,
        num_samples=num_samples,
        num_step=num_steps,
        batch_size=args.batch_size,
        ligand_v_temp=config.sample.ligand_v_temp,
        ligand_v_noise=config.sample.ligand_v_noise,
        sample_time_schedule=config.sample.sample_time_schedule,
        device=args.device,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
        return_confidence=False,
    )
    (
        pred_pos,
        pred_v,
        pred_pos_traj,
        pred_v_traj,
    ) = sample_out

    max_indices = [np.argmax(array, axis=1) for array in pred_v]
    max_indices_traj = [np.argmax(array, axis=-1) for array in pred_v_traj]

    result = {
        "data": data,
        "pred_ligand_pos": pred_pos,
        "pred_ligand_v": max_indices,
        "pred_ligand_pos_traj": pred_pos_traj,
        "pred_ligand_v_traj": max_indices_traj,
        "checkpoint": ckpt_path,
    }

    torch.save(result, os.path.join(log_dir, f"result_{args.data_id}.pt"))
    logger.info("Inference done!")


if __name__ == "__main__":
    main()
