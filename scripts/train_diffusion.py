import argparse
import os
import shutil
import socket
import pickle
import random
from scripts.evaluate_diffusion import run_eval_process
from scripts.sample_diffusion import sample_diffusion_ligand
import wandb
import numpy as np
import math

import torch
import torch.distributed as distrib
# import torch.utils.tensorboard
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_mean
from tqdm.auto import tqdm
from torch.utils.data import Subset

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
import utils.reward_qed_sa as reward_qed_sa
from utils.evaluation import atom_num
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from datasets.merge_dataset import MergedProteinLigandData, FOLLOW_BATCH2
from models.molopt_score_model import ScorePosNet3D
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from rdkit import RDLogger
from easydict import EasyDict


def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.0
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            "basic": trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            "add_aromatic": trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            "full": trans.MAP_INDEX_TO_ATOM_TYPE_FULL,
        }
        print(f"atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}")
    return avg_auroc / len(y_true)


def is_port_available(port, host="localhost"):
    """
    Check if a given port is available.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        s.close()
        return True
    except OSError:
        s.close()
        return False


def _strip_confidence_state_dict(state_dict):
    prefixes = ("cfd_node_head.", "cfd_pos_head.")
    stripped = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith(prefixes):
            continue
        stripped[k] = v
    return stripped


def _regularize_step_probs(step_probs, predict_ligand_v):
    num_atoms, num_classes = step_probs.shape
    device = step_probs.device
    assert predict_ligand_v.shape == (num_atoms,)

    step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
    step_probs[
        torch.arange(num_atoms, device=device),
        predict_ligand_v.long().flatten()
    ] = 0.0
    step_probs[
        torch.arange(num_atoms, device=device),
        predict_ligand_v.long().flatten()
    ] = 1.0 - torch.sum(step_probs, dim=-1).flatten()
    step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
    return step_probs


def main(rank, num_gpus):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--logdir", type=str, default="./logs_diffusion")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--train_report_iter", type=int, default=100)
    parser.add_argument(
        "--is_debug", action="store_true", help="Enable debug mode", default=False
    )
    parser.add_argument("--name", type=str, default="flow matching")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to a previous checkpoint",
    )
    parser.add_argument(
        "--no_optimizer_state",
        action="store_true",
        help="Do not load optimizer/scheduler state from checkpoint",
    )
    parser.add_argument(
        "--non_strict_load",
        action="store_true",
        help="Load model weights with strict=False",
    )
    parser.add_argument(
        "--dpo_data",
        type=str,
        default="",
        help="Path to DPO preference data file (pkl format). If provided, enables DPO training."
    )
    parser.add_argument(
        "--reuse_logdir",
        action="store_true",
        help="Reuse --logdir as the active run directory instead of creating a new subdirectory.",
    )
    parser.add_argument(
        "--wandb_id",
        type=str,
        default="",
        help="W&B run ID for resuming the same run.",
    )
    parser.add_argument(
        "--reset_iteration",
        action="store_true",
        help="Reset the loaded checkpoint iteration to 1 before training.",
    )
    args = parser.parse_args()


    # Version control
    branch, version = misc.get_version()
    version_short = "%s-%s" % (branch, version[:7])

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[
        : os.path.basename(args.config).rfind(".")
    ]
    misc.seed_all(config.train.seed)

    # Logging 
    if args.is_debug and rank == 0:
        logger = misc.get_logger("train", None)
    elif rank == 0:
        RDLogger.DisableLog('rdApp.*')
        wandb_kwargs = {
            "project": args.name,
            "config": config,
            "name": f"{config_name}[{args.tag}]",
        }
        if args.wandb_id:
            wandb_kwargs["id"] = args.wandb_id
            wandb_kwargs["resume"] = "must"
        run = wandb.init(**wandb_kwargs)

        if args.reuse_logdir:
            log_dir = os.path.realpath(args.logdir)
            os.makedirs(log_dir, exist_ok=True)
        else:
            log_dir = misc.get_new_log_dir(
                args.logdir, prefix=f"{config_name}_{version_short}_{args.tag}"
            )
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        with open(os.path.join(log_dir, "commit.txt"), "w") as f:
            f.write(branch + "\n")
            f.write(version + "\n")
        logger = misc.get_logger("train", log_dir)
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        cfg_dst = os.path.join(log_dir, os.path.basename(args.config))
        if not os.path.exists(cfg_dst):
            shutil.copyfile(args.config, cfg_dst)
        models_dst = os.path.join(log_dir, "models")
        if not os.path.exists(models_dst):
            shutil.copytree("./models", models_dst)

        logger.info(args)
        logger.info(config)

    torch.cuda.set_device(rank)
    distrib.init_process_group(
        backend="nccl", rank=rank, world_size=num_gpus, init_method="env://"
    )

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(
        config.data.transform.ligand_atom_mode
    )
    bond_featureizer = trans.FeaturizeLigandBond()
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        bond_featureizer,
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info("Loading dataset...") if rank == 0 else None
    dataset, subsets = get_dataset(config=config.data, transform=transform)
    train_set, val_set = subsets["train"], subsets["test"]

    use_dpo = getattr(config.train, "use_dpo", False)
    use_nft = getattr(config.train, "use_nft", False)
    if use_dpo and use_nft:
        raise ValueError("use_dpo and use_nft cannot both be True.")

    # Check if DPO training is enabled
    if use_dpo:
        logger.info("DPO training enabled. Loading preference data...") if rank == 0 else None

        def get_dpo_data(subset, split='train', fullset=dataset):
            logger.info('Getting DPO {} data...'.format(split)) if rank == 0 else None

            with open(args.dpo_data, 'rb') as file:
                dpo_idx = pickle.load(file)

            train_id = subset.indices
            train_id_set = set(train_id)
            cleaned_dpo_idx = {k: [item for item in v if item in train_id_set] for k, v in dpo_idx.items() if v}
            cleaned_dpo_idx = {k: v for k, v in cleaned_dpo_idx.items() if v}  # Remove any keys with empty lists

            new_pair = {}
            for idx in tqdm(subset.indices, 'Creating Preference Train Set'):
                if idx in cleaned_dpo_idx.keys():
                    losing_id = cleaned_dpo_idx[idx]
                    assert losing_id
                    new_pair[idx] = losing_id[0]


            winning_idx = list(new_pair.keys())
            losing_idx = list(new_pair.values())
            subset_1 = Subset(fullset, winning_idx)
            subset_2 = Subset(fullset, losing_idx)

            dpo_subset = MergedProteinLigandData(subset_1, subset_2)
            return dpo_subset

        dpo_train_set = get_dpo_data(train_set, split='train', fullset=dataset)

        train_sampler = DistributedSampler(
            dpo_train_set, num_replicas=num_gpus, rank=rank, shuffle=False, drop_last=False
        )

        logger.info(f"DPO Training: {len(dpo_train_set)} Validation: {len(val_set)}") if rank == 0 else None

        collate_exclude_keys = ['ligand_nbh_list', 'ligand_nbh_list2']
        follow_batch = FOLLOW_BATCH2
        actual_train_set = dpo_train_set
    else:
        logger.info("Standard training enabled.") if rank == 0 else None
        train_sampler = DistributedSampler(
            train_set, num_replicas=num_gpus, rank=rank, shuffle=False, drop_last=False
        )
        logger.info(f"Training: {len(train_set)} Validation: {len(val_set)}") if rank == 0 else None
        collate_exclude_keys = ["ligand_nbh_list"]
        follow_batch = FOLLOW_BATCH
        actual_train_set = train_set

    train_iterator = utils_train.inf_iterator(
        DataLoader(
            actual_train_set,
            batch_size=config.train.batch_size,
            sampler=train_sampler,
            num_workers=config.train.num_workers,
            follow_batch=follow_batch,
            exclude_keys=collate_exclude_keys,
        )
    )
    rollout_iterator = None
    if use_nft:
        nft_cfg = getattr(config.train, "nft", EasyDict({}))
        rollout_bs = getattr(nft_cfg, "rollout_batch_size", config.train.batch_size)
        rollout_sampler = DistributedSampler(
            train_set, num_replicas=num_gpus, rank=rank, shuffle=False, drop_last=False
        )
        rollout_iterator = utils_train.inf_iterator(
            DataLoader(
                train_set,
                batch_size=rollout_bs,
                sampler=rollout_sampler,
                num_workers=config.train.num_workers,
                follow_batch=FOLLOW_BATCH,
                exclude_keys=["ligand_nbh_list"],
            )
        )

    if rank == 0:
        val_loader = DataLoader(
            val_set,
            config.train.batch_size,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            exclude_keys=collate_exclude_keys,
        )

    # Model
    logger.info("Building model...") if rank == 0 else None

    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).cuda(rank)

    # Load reference model for DPO training if needed
    ref_model = None
    if use_dpo:
        if hasattr(config.model, 'ref_model_checkpoint') and config.model.ref_model_checkpoint:
            logger.info("Loading reference model for DPO training...") if rank == 0 else None
            ckpt = misc.load_checkpoint(config.model.ref_model_checkpoint, map_location=f"cuda:{rank}")
            new_state_dict = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
            new_state_dict = _strip_confidence_state_dict(new_state_dict)

            # Load weights into main model
            model.load_state_dict(new_state_dict)

            # Create reference model
            ref_model = ScorePosNet3D(
                config.model,
                protein_atom_feature_dim=protein_featurizer.feature_dim,
                ligand_atom_feature_dim=ligand_featurizer.feature_dim
            ).cuda(rank)
            ref_model.load_state_dict(new_state_dict)
            ref_model.eval()

            # Freeze reference model parameters
            for param in ref_model.parameters():
                param.requires_grad = False
            logger.info("Reference model loaded and frozen.") if rank == 0 else None
        else:
            raise ValueError("DPO training requires ref_model_checkpoint in config.model")

    model = DDP(model, device_ids=[rank], output_device=rank)
    logger.info(f"protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}") if rank == 0 else None
    logger.info(f"Model has {misc.count_parameters(model) / 1e6:.4f} M parameters.") if rank == 0 else None

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    # Loading from checkpoint if provided
    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            logger.info(f"Loading from checkpoint {args.checkpoint}") if rank == 0 else None
            checkpoint = misc.load_checkpoint(
                args.checkpoint, map_location=f"cuda:{rank}"
            )
            start_iter = 1 if args.reset_iteration else checkpoint["iteration"]
            state_dict = _strip_confidence_state_dict(checkpoint["model"])
            model.load_state_dict(state_dict, strict=not args.non_strict_load)
            if not args.no_optimizer_state:
                if "optimizer" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if "scheduler" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler"])
            if rank == 0:
                if args.reset_iteration:
                    logger.info(
                        f"=> loaded checkpoint {args.checkpoint} "
                        f"(checkpoint iteration {checkpoint['iteration']} -> reset to 1)"
                    )
                else:
                    logger.info(
                        f"=> loaded checkpoint {args.checkpoint} "
                        f"(iteration {checkpoint['iteration']})"
                    )
        else:
            logger.info(f"=> no checkpoint found at {args.checkpoint}") if rank == 0 else None
            start_iter = 1
    else:
        start_iter = 1

    model_old = None
    if use_nft:
        model_old = ScorePosNet3D(
            config.model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        ).cuda(rank)
        model_old.load_state_dict(model.module.state_dict())
        model_old.eval()
        for p in model_old.parameters():
            p.requires_grad = False

        nft_cfg = getattr(config.train, "nft", EasyDict({}))
        rollout_num_samples = int(getattr(nft_cfg, "rollout_num_samples", 8))
        rollout_chunk_size = int(getattr(nft_cfg, "rollout_chunk_size", 0))
        train_batch_size = int(getattr(nft_cfg, "train_batch_size", config.train.batch_size))
        num_inner_epochs = int(getattr(nft_cfg, "num_inner_epochs", 1))
        beta_nft = float(getattr(nft_cfg, "beta", 0.3))
        beta_nft_v = float(getattr(nft_cfg, "beta_discrete", beta_nft))
        ema_decay = float(getattr(nft_cfg, "ema_decay", 0.995))

        reward_cfg = getattr(nft_cfg, "reward", EasyDict({}))
        reward_type = str(getattr(reward_cfg, "type", "qed_sa")).lower()
        invalid_reward = float(getattr(reward_cfg, "invalid_reward", 0.0))
        norm_ema_decay = float(getattr(reward_cfg, "norm_ema_decay", 0.95))
        min_std = float(getattr(reward_cfg, "min_std", 1e-3))

        use_vina_reward = reward_type in ("vina", "vina_score", "vina-score")
        use_vina_sa_reward = reward_type in ("vina_sa", "vina+sa", "vina_sa_reward")
        use_vina_qed_reward = reward_type in (
            "vina_qed",
            "vina+qed",
            "qed_vina",
            "qed+vina",
            "vina_qed_reward",
        )
        use_vina_qed_sa_reward = reward_type in (
            "vina_qed_sa",
            "vina+qed+sa",
            "vina+sa+qed",
            "vina_qedsa",
            "vina_qed_sa_reward",
        )
        use_vina_based_reward = (
            use_vina_reward
            or use_vina_sa_reward
            or use_vina_qed_reward
            or use_vina_qed_sa_reward
        )
        reward_vina_cfg = None
        reward_vina_sa_cfg = None
        reward_vina_qed_cfg = None
        reward_vina_qed_sa_cfg = None
        if reward_type in ("qed_sa", "qed+sa", "qed_sa_reward"):
            w_qed = float(getattr(reward_cfg, "w_qed", 1.0))
            w_sa = float(getattr(reward_cfg, "w_sa", 1.0))
        elif use_vina_based_reward:
            reward_vina_cfg = EasyDict(
                protein_root=getattr(reward_cfg, "protein_root", getattr(config.data, "path", None)),
                exhaustiveness=int(getattr(reward_cfg, "exhaustiveness", 8)),
                size_factor=float(getattr(reward_cfg, "size_factor", 1.0)),
                buffer=float(getattr(reward_cfg, "buffer", 5.0)),
                tmp_dir=getattr(reward_cfg, "tmp_dir", "./tmp_vina"),
                score_sign=float(getattr(reward_cfg, "score_sign", -1.0)),
                log_vina=bool(getattr(reward_cfg, "log_vina", False)),
                cleanup=bool(getattr(reward_cfg, "cleanup", True)),
            )
            if use_vina_reward:
                import utils.reward_vina as reward_vina
            elif use_vina_sa_reward:
                reward_vina_sa_cfg = EasyDict(
                    vina_clip_low=float(getattr(reward_cfg, "vina_clip_low", -16.0)),
                    vina_clip_high=float(getattr(reward_cfg, "vina_clip_high", -1.0)),
                    vina_offset=float(getattr(reward_cfg, "vina_offset", 1.0)),
                    vina_divisor=float(getattr(reward_cfg, "vina_divisor", 15.0)),
                    sa_shift=float(getattr(reward_cfg, "sa_shift", 0.17)),
                    sa_scale=float(getattr(reward_cfg, "sa_scale", 0.83)),
                )
                import utils.reward_vina_sa as reward_vina_sa
            elif use_vina_qed_reward:
                reward_vina_qed_cfg = EasyDict(
                    vina_clip_low=float(getattr(reward_cfg, "vina_clip_low", -16.0)),
                    vina_clip_high=float(getattr(reward_cfg, "vina_clip_high", -1.0)),
                    vina_offset=float(getattr(reward_cfg, "vina_offset", 1.0)),
                    vina_divisor=float(getattr(reward_cfg, "vina_divisor", 15.0)),
                    qed_shift=float(getattr(reward_cfg, "qed_shift", 0.17)),
                    qed_scale=float(getattr(reward_cfg, "qed_scale", 0.83)),
                )
                import utils.reward_vina_qed as reward_vina_qed
            else:
                reward_vina_qed_sa_cfg = EasyDict(
                    vina_clip_low=float(getattr(reward_cfg, "vina_clip_low", -16.0)),
                    vina_clip_high=float(getattr(reward_cfg, "vina_clip_high", -1.0)),
                    vina_offset=float(getattr(reward_cfg, "vina_offset", 1.0)),
                    vina_divisor=float(getattr(reward_cfg, "vina_divisor", 15.0)),
                    qed_shift=float(getattr(reward_cfg, "qed_shift", 0.17)),
                    qed_scale=float(getattr(reward_cfg, "qed_scale", 0.83)),
                    sa_shift=float(getattr(reward_cfg, "sa_shift", 0.17)),
                    sa_scale=float(getattr(reward_cfg, "sa_scale", 0.83)),
                )
                import utils.reward_vina_qed_sa as reward_vina_qed_sa
        else:
            raise ValueError(f"Unsupported NFT reward type: {reward_type}")

        sample_num_steps = int(getattr(nft_cfg, "sample_num_steps", 100))
        sample_time_schedule = getattr(nft_cfg, "sample_time_schedule", "log")
        sample_num_atoms = getattr(nft_cfg, "sample_num_atoms", "prior")
        ligand_v_temp = float(getattr(nft_cfg, "ligand_v_temp", 0.01))
        ligand_v_noise = float(getattr(nft_cfg, "ligand_v_noise", 1.0))
        center_pos_mode = getattr(nft_cfg, "center_pos_mode", config.model.center_pos_mode)

        atom_mode = config.data.transform.ligand_atom_mode

        reward_ema_mean = None
        reward_ema_var = None
        last_rollout_stats = {}

        def _update_reward_stats(raw_rewards):
            nonlocal reward_ema_mean, reward_ema_var
            if len(raw_rewards) == 0:
                return
            batch_mean = float(np.mean(raw_rewards))
            batch_var = float(np.var(raw_rewards))
            if reward_ema_mean is None:
                reward_ema_mean = batch_mean
                reward_ema_var = batch_var
            else:
                reward_ema_mean = norm_ema_decay * reward_ema_mean + (1 - norm_ema_decay) * batch_mean
                reward_ema_var = norm_ema_decay * reward_ema_var + (1 - norm_ema_decay) * batch_var

        def _reward_std():
            if reward_ema_var is None:
                return 1.0
            return max(float(np.sqrt(reward_ema_var)), min_std)

        def _collate_samples(samples, device):
            protein_pos = torch.cat([s["protein_pos"] for s in samples], dim=0).to(device)
            protein_v = torch.cat([s["protein_v"] for s in samples], dim=0).to(device)
            ligand_pos = torch.cat([s["ligand_pos"] for s in samples], dim=0).to(device)
            ligand_v = torch.cat([s["ligand_v"] for s in samples], dim=0).to(device)

            batch_protein = []
            batch_ligand = []
            for i, s in enumerate(samples):
                batch_protein.append(torch.full((s["protein_pos"].shape[0],), i, dtype=torch.long))
                batch_ligand.append(torch.full((s["ligand_pos"].shape[0],), i, dtype=torch.long))
            batch_protein = torch.cat(batch_protein, dim=0).to(device)
            batch_ligand = torch.cat(batch_ligand, dim=0).to(device)
            reward = torch.tensor([s["reward"] for s in samples], dtype=torch.float32, device=device)

            return protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, reward

        def _ema_update(target, source, decay):
            with torch.no_grad():
                for p_t, p_s in zip(target.parameters(), source.parameters()):
                    p_t.data.mul_(decay).add_(p_s.data, alpha=1.0 - decay)

        def _sample_diffusion_multi_pocket(model, pocket_data_list):
            if len(pocket_data_list) == 0:
                return [], []

            if sample_time_schedule == 'log':
                time_points = (1 - np.geomspace(0.01, 1.0, sample_num_steps + 1)).tolist()
            elif sample_time_schedule == 'linear':
                time_points = np.linspace(0.01, 1.0, sample_num_steps + 1).tolist()
            else:
                raise ValueError(f"Invalid sample_time_schedule: {sample_time_schedule}")
            time_points.reverse()

            pocket_sizes = None
            if sample_num_atoms == 'prior':
                pocket_sizes = [
                    atom_num.get_space_size(d.protein_pos.detach().cpu().numpy())
                    for d in pocket_data_list
                ]

            sample_entries = []
            for p_idx, data in enumerate(pocket_data_list):
                for _ in range(rollout_num_samples):
                    sample_entries.append((p_idx, data.clone()))

            if rollout_chunk_size is None or rollout_chunk_size <= 0:
                chunk_size = len(sample_entries)
            else:
                chunk_size = rollout_chunk_size

            all_pred_pos = [[] for _ in pocket_data_list]
            all_pred_v = [[] for _ in pocket_data_list]

            for start in range(0, len(sample_entries), chunk_size):
                chunk = sample_entries[start:start + chunk_size]
                chunk_pocket_idx = [p for p, _ in chunk]
                chunk_data_list = [d for _, d in chunk]
                n_samples = len(chunk_data_list)

                batch = Batch.from_data_list(chunk_data_list, follow_batch=FOLLOW_BATCH).to(f"cuda:{rank}")
                batch_protein = batch.protein_element_batch

                if sample_num_atoms == 'prior':
                    ligand_num_atoms = [
                        int(atom_num.sample_atom_num(pocket_sizes[p_idx]))
                        for p_idx in chunk_pocket_idx
                    ]
                elif sample_num_atoms == 'ref':
                    ligand_num_atoms = [
                        int(d.ligand_element.shape[0]) for d in chunk_data_list
                    ]
                else:
                    raise ValueError(f"Unsupported sample_num_atoms in NFT rollout: {sample_num_atoms}")

                batch_ligand = torch.repeat_interleave(
                    torch.arange(n_samples, device=f"cuda:{rank}"),
                    torch.tensor(ligand_num_atoms, device=f"cuda:{rank}"),
                )

                center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
                init_ligand_pos = torch.randn(center_pos[batch_ligand].shape, device=f"cuda:{rank}")
                init_ligand_v = torch.randint(0, model.num_classes, (batch_ligand.shape[0],), device=f"cuda:{rank}")
                init_ligand_v_onehot = F.one_hot(init_ligand_v, num_classes=model.num_classes).to(f"cuda:{rank}")

                batch.protein_pos = batch.protein_pos - center_pos[batch_protein]

                ligand_pos = init_ligand_pos.clone()
                ligand_v_onehot = init_ligand_v_onehot.clone().float()
                ligand_v_t = init_ligand_v.clone()

                with torch.no_grad():
                    for i, t in enumerate(time_points):
                        if i == 0:
                            continue
                        dt = time_points[i] - time_points[i - 1]
                        r = model.sample_diffusion(
                            protein_pos=batch.protein_pos,
                            protein_v=batch.protein_atom_feature.float(),
                            batch_protein=batch_protein,
                            ligand_pos=ligand_pos,
                            ligand_v=ligand_v_onehot,
                            batch_ligand=batch_ligand,
                            t=t,
                            device=f"cuda:{rank}",
                            center_pos_mode=center_pos_mode,
                        )
                        pred_pos, pred_v = r['pred_pos'], r['pred_v']

                        velocity_pos = (pred_pos - ligand_pos) / (1 - t)
                        ligand_pos = ligand_pos + velocity_pos * dt

                        pred_v_probs = F.softmax(pred_v / ligand_v_temp, dim=-1)
                        pt_x1_eq_xt_prob = torch.gather(
                            pred_v_probs, dim=-1, index=ligand_v_t.long().unsqueeze(-1)
                        )
                        N = ligand_v_noise
                        step_probs = dt * (
                            pred_v_probs * ((1 + N + N * (model.num_classes - 1) * t) / (1 - t))
                            + N * pt_x1_eq_xt_prob
                        )
                        step_probs = _regularize_step_probs(step_probs, ligand_v_t)

                        ligand_v_t = torch.multinomial(
                            step_probs.view(-1, model.num_classes), num_samples=1
                        ).view(step_probs.shape[0],)
                        ligand_v_onehot = F.one_hot(ligand_v_t, num_classes=model.num_classes).to(f"cuda:{rank}")

                    r = model.sample_diffusion(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch_protein,
                        ligand_pos=ligand_pos,
                        ligand_v=ligand_v_onehot,
                        batch_ligand=batch_ligand,
                        t=1.0,
                        device=f"cuda:{rank}",
                        center_pos_mode=center_pos_mode,
                    )
                    ligand_pos, ligand_v = r['pred_pos'], r['pred_v']
                    ligand_pos = ligand_pos + center_pos[batch_ligand]

                ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
                ligand_pos_array = ligand_pos.detach().cpu().numpy().astype(np.float64)
                ligand_v_array = ligand_v.detach().cpu().numpy()
                for i in range(n_samples):
                    start_idx = ligand_cum_atoms[i]
                    end_idx = ligand_cum_atoms[i + 1]
                    p_idx = chunk_pocket_idx[i]
                    all_pred_pos[p_idx].append(ligand_pos_array[start_idx:end_idx])
                    all_pred_v[p_idx].append(ligand_v_array[start_idx:end_idx])

            return all_pred_pos, all_pred_v

        def _collect_rollout_data():
            nonlocal last_rollout_stats
            batch = next(rollout_iterator)
            data_list = batch.to_data_list()

            all_raw_rewards = []
            collected_samples = []
            invalid_count = 0
            total_count = 0

            pred_pos_grouped, pred_v_grouped = _sample_diffusion_multi_pocket(
                model_old, data_list
            )

            pocket_samples = []
            for data, pred_pos_list, pred_v_list in zip(
                data_list, pred_pos_grouped, pred_v_grouped
            ):
                raw_rewards = []
                cur_samples = []
                for pred_pos, pred_v in zip(pred_pos_list, pred_v_list):
                    pred_v_idx = np.argmax(pred_v, axis=1)
                    if use_vina_reward:
                        reward_raw, _, _ = reward_vina.reward_from_generated(
                            pred_pos,
                            pred_v_idx,
                            atom_mode=atom_mode,
                            data=data,
                            protein_root=reward_vina_cfg.protein_root,
                            exhaustiveness=reward_vina_cfg.exhaustiveness,
                            size_factor=reward_vina_cfg.size_factor,
                            buffer=reward_vina_cfg.buffer,
                            tmp_dir=reward_vina_cfg.tmp_dir,
                            score_sign=reward_vina_cfg.score_sign,
                            log_vina=reward_vina_cfg.log_vina,
                            cleanup=reward_vina_cfg.cleanup,
                        )
                    elif use_vina_sa_reward:
                        reward_raw, _, _ = reward_vina_sa.reward_from_generated(
                            pred_pos,
                            pred_v_idx,
                            atom_mode=atom_mode,
                            data=data,
                            protein_root=reward_vina_cfg.protein_root,
                            exhaustiveness=reward_vina_cfg.exhaustiveness,
                            size_factor=reward_vina_cfg.size_factor,
                            buffer=reward_vina_cfg.buffer,
                            tmp_dir=reward_vina_cfg.tmp_dir,
                            score_sign=reward_vina_cfg.score_sign,
                            log_vina=reward_vina_cfg.log_vina,
                            cleanup=reward_vina_cfg.cleanup,
                            vina_clip_low=reward_vina_sa_cfg.vina_clip_low,
                            vina_clip_high=reward_vina_sa_cfg.vina_clip_high,
                            vina_offset=reward_vina_sa_cfg.vina_offset,
                            vina_divisor=reward_vina_sa_cfg.vina_divisor,
                            sa_shift=reward_vina_sa_cfg.sa_shift,
                            sa_scale=reward_vina_sa_cfg.sa_scale,
                        )
                    elif use_vina_qed_reward:
                        reward_raw, _, _ = reward_vina_qed.reward_from_generated(
                            pred_pos,
                            pred_v_idx,
                            atom_mode=atom_mode,
                            data=data,
                            protein_root=reward_vina_cfg.protein_root,
                            exhaustiveness=reward_vina_cfg.exhaustiveness,
                            size_factor=reward_vina_cfg.size_factor,
                            buffer=reward_vina_cfg.buffer,
                            tmp_dir=reward_vina_cfg.tmp_dir,
                            score_sign=reward_vina_cfg.score_sign,
                            log_vina=reward_vina_cfg.log_vina,
                            cleanup=reward_vina_cfg.cleanup,
                            vina_clip_low=reward_vina_qed_cfg.vina_clip_low,
                            vina_clip_high=reward_vina_qed_cfg.vina_clip_high,
                            vina_offset=reward_vina_qed_cfg.vina_offset,
                            vina_divisor=reward_vina_qed_cfg.vina_divisor,
                            qed_shift=reward_vina_qed_cfg.qed_shift,
                            qed_scale=reward_vina_qed_cfg.qed_scale,
                        )
                    elif use_vina_qed_sa_reward:
                        reward_raw, _, _ = reward_vina_qed_sa.reward_from_generated(
                            pred_pos,
                            pred_v_idx,
                            atom_mode=atom_mode,
                            data=data,
                            protein_root=reward_vina_cfg.protein_root,
                            exhaustiveness=reward_vina_cfg.exhaustiveness,
                            size_factor=reward_vina_cfg.size_factor,
                            buffer=reward_vina_cfg.buffer,
                            tmp_dir=reward_vina_cfg.tmp_dir,
                            score_sign=reward_vina_cfg.score_sign,
                            log_vina=reward_vina_cfg.log_vina,
                            cleanup=reward_vina_cfg.cleanup,
                            vina_clip_low=reward_vina_qed_sa_cfg.vina_clip_low,
                            vina_clip_high=reward_vina_qed_sa_cfg.vina_clip_high,
                            vina_offset=reward_vina_qed_sa_cfg.vina_offset,
                            vina_divisor=reward_vina_qed_sa_cfg.vina_divisor,
                            qed_shift=reward_vina_qed_sa_cfg.qed_shift,
                            qed_scale=reward_vina_qed_sa_cfg.qed_scale,
                            sa_shift=reward_vina_qed_sa_cfg.sa_shift,
                            sa_scale=reward_vina_qed_sa_cfg.sa_scale,
                        )
                    else:
                        reward_raw, _, _ = reward_qed_sa.reward_from_generated(
                            pred_pos,
                            pred_v_idx,
                            atom_mode=atom_mode,
                            w_qed=w_qed,
                            w_sa=w_sa,
                        )

                    total_count += 1
                    if reward_raw is None:
                        invalid_count += 1
                        # For Vina-based rewards, skip invalid molecules instead of assigning 0.
                        if use_vina_based_reward:
                            continue
                        reward_raw = invalid_reward
                    raw_rewards.append(float(reward_raw))
                    cur_samples.append(
                        {
                            "protein_pos": data.protein_pos.detach().cpu(),
                            "protein_v": data.protein_atom_feature.float().detach().cpu(),
                            "ligand_pos": torch.tensor(pred_pos, dtype=torch.float32),
                            "ligand_v": torch.tensor(pred_v_idx, dtype=torch.long),
                        }
                    )

                all_raw_rewards.extend(raw_rewards)
                pocket_samples.append((cur_samples, raw_rewards))

            _update_reward_stats(all_raw_rewards)
            std = _reward_std()

            for cur_samples, raw_rewards in pocket_samples:
                if len(raw_rewards) == 0:
                    continue
                mean_c = float(np.mean(raw_rewards))
                std_c = float(np.std(raw_rewards))
                z_c = max(std_c, min_std)
                for s, r_raw in zip(cur_samples, raw_rewards):
                    r_norm = float(r_raw) - mean_c
                    r_clip = max(-1.0, min(1.0, r_norm / z_c))
                    r = 0.5 + 0.5 * r_clip
                    s["reward"] = r
                    collected_samples.append(s)

            last_rollout_stats = {
                "num_samples": len(collected_samples),
                "reward_mean": float(np.mean(all_raw_rewards)) if all_raw_rewards else 0.0,
                "reward_std": float(np.std(all_raw_rewards)) if all_raw_rewards else 0.0,
                "reward_std_ema": std,
                "invalid_ratio": (invalid_count / max(total_count, 1)),
            }
            return collected_samples

        def train(it):
            model.train()
            samples = _collect_rollout_data()
            if len(samples) == 0:
                if rank == 0 and it % args.train_report_iter == 0:
                    logger.info(f"[NFT] Iter {it} | No valid samples collected.")
                return

            dataset_size = len(samples)
            indices = list(range(dataset_size))
            steps_per_ep = int(np.ceil(dataset_size / float(train_batch_size)))
            total_batches = max(1, steps_per_ep * num_inner_epochs)

            total_loss = 0.0
            total_loss_pos = 0.0
            total_loss_v = 0.0
            total_chamfer = 0.0
            batch_count = 0
            last_grad_norm = 0.0

            optimizer.zero_grad()

            for inner_ep in range(num_inner_epochs):
                random.shuffle(indices)
                for start_idx in range(0, dataset_size, train_batch_size):
                    end_idx = min(start_idx + train_batch_size, dataset_size)
                    batch_indices = indices[start_idx:end_idx]
                    batch_samples = [samples[i] for i in batch_indices]

                    (
                        protein_pos,
                        protein_v,
                        batch_protein,
                        ligand_pos,
                        ligand_v,
                        batch_ligand,
                        reward,
                    ) = _collate_samples(batch_samples, device=f"cuda:{rank}")

                    results = model.module.get_diffusion_loss_nft(
                        beta_nft=beta_nft,
                        beta_nft_v=beta_nft_v,
                        reward=reward,
                        old_model=model_old,
                        protein_pos=protein_pos,
                        protein_v=protein_v,
                        batch_protein=batch_protein,
                        ligand_pos=ligand_pos,
                        ligand_v=ligand_v,
                        batch_ligand=batch_ligand,
                    )

                    loss, loss_pos, loss_v, chamfer_loss = (
                        results["loss"],
                        results["loss_pos"],
                        results["loss_v"],
                        results["loss_chamfer"],
                    )

                    (loss / total_batches).backward()

                    total_loss += float(loss)
                    total_loss_pos += float(loss_pos)
                    total_loss_v += float(loss_v)
                    total_chamfer += float(chamfer_loss)
                    batch_count += 1

            if batch_count > 0:
                last_grad_norm = clip_grad_norm_(
                    model.parameters(), config.train.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
            _ema_update(model_old, model.module, ema_decay)

            if rank == 0 and it % args.train_report_iter == 0:
                denom = max(1, batch_count)
                logger.info(
                    "[NFT] Iter %d | Loss %.6f (pos %.6f | v %.6f | chamfer %.6f) | "
                    "Samples %d | InnerEp %d | Rmean %.4f | Rstd %.4f | Rstd_ema %.4f | invalid %.2f%% | Lr %.6f | Grad %.6f"
                    % (
                        it,
                        total_loss / denom,
                        total_loss_pos / denom,
                        total_loss_v / denom,
                        total_chamfer / denom,
                        last_rollout_stats.get("num_samples", len(samples)),
                        num_inner_epochs,
                        last_rollout_stats.get("reward_mean", 0.0),
                        last_rollout_stats.get("reward_std", 0.0),
                        last_rollout_stats.get("reward_std_ema", 0.0),
                        100.0 * last_rollout_stats.get("invalid_ratio", 0.0),
                        optimizer.param_groups[0]["lr"],
                        last_grad_norm,
                    )
                )
                if not args.is_debug and rank == 0:
                    wandb.log(
                        {
                            "train/loss": total_loss / denom,
                            "train/loss_pos": total_loss_pos / denom,
                            "train/loss_v": total_loss_v / denom,
                            "train/chamfer_loss": total_chamfer / denom,
                            "train/nft_num_samples": last_rollout_stats.get(
                                "num_samples", len(samples)
                            ),
                            "train/nft_buffer": last_rollout_stats.get(
                                "num_samples", len(samples)
                            ),
                            "train/nft_inner_epochs": num_inner_epochs,
                            "train/nft_updates_per_iter": 1,
                            "train/nft_total_batches": total_batches,
                            "train/nft_reward_mean": last_rollout_stats.get("reward_mean", 0.0),
                            "train/nft_reward_std": last_rollout_stats.get("reward_std", 0.0),
                            "train/nft_reward_std_ema": last_rollout_stats.get("reward_std_ema", 0.0),
                            "train/nft_invalid_ratio": last_rollout_stats.get("invalid_ratio", 0.0),
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/grad_norm": last_grad_norm,
                            "iteration": it,
                        }
                    )

    else:
        def train(it):
            model.train()
            optimizer.zero_grad()
            for _ in range(config.train.n_acc_batch):

                batch = next(train_iterator).cuda(rank)

                if use_dpo:
                    # DPO training
                    protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
                    protein_noise2 = torch.randn_like(batch.protein_pos2) * config.train.pos_noise_std

                    gt_protein_pos = batch.protein_pos + protein_noise
                    gt_protein_pos2 = batch.protein_pos2 + protein_noise2

                    results = model.module.get_diffusion_loss_dpo(
                        beta_dpo=config.train.beta_dpo,
                        discete_beta_dpo=config.train.discete_beta_dpo,
                        ref_model=ref_model,

                        protein_pos_w=gt_protein_pos,
                        protein_v_w=batch.protein_atom_feature.float(),
                        batch_protein_w=batch.protein_element_batch,
                        ligand_pos_w=batch.ligand_pos,
                        ligand_v_w=batch.ligand_atom_feature_full,
                        batch_ligand_w=batch.ligand_element_batch,

                        protein_pos_l=gt_protein_pos2,
                        protein_v_l=batch.protein_atom_feature2.float(),
                        batch_protein_l=batch.protein_element2_batch,
                        ligand_pos_l=batch.ligand_pos2,
                        ligand_v_l=batch.ligand_atom_feature_full2,
                        batch_ligand_l=batch.ligand_element2_batch,
                    )

                    loss, loss_pos, loss_v, chamfer_loss = (
                        results["loss"],
                        results["loss_pos"],
                        results["loss_v"],
                        results["loss_chamfer"],
                    )
                else:
                    # Standard training
                    protein_noise = (
                        torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
                    )
                    gt_protein_pos = batch.protein_pos + protein_noise
                    results = model.module.get_diffusion_loss(
                        protein_pos=gt_protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,
                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                    )

                    loss, loss_pos, loss_v, chamfer_loss = (
                        results["loss"],
                        results["loss_pos"],
                        results["loss_v"],
                        results["chamfer_loss"],
                    )

                loss = loss / config.train.n_acc_batch
                loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()

            if rank==0 and it % args.train_report_iter == 0:
                logger.info(
                    "[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f | chamfer %.6f) | Lr: %.6f | Grad Norm: %.6f"
                    % (
                        it,
                        loss,
                        loss_pos,
                        loss_v,
                        chamfer_loss,
                        optimizer.param_groups[0]["lr"],
                        orig_grad_norm,
                    )
                )
                # Add wandb logging
                if not args.is_debug and rank == 0:
                    wandb.log(
                        {
                            "train/loss": loss,
                            "train/loss_pos": loss_pos,
                            "train/loss_v": loss_v,
                            "train/chamfer_loss": chamfer_loss,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/grad_norm": orig_grad_norm,
                            "iteration": it,
                        }
                    )
    def validate(it):
        # fix time steps
        sum_loss, sum_loss_pos, sum_loss_v, sum_chamfer_loss, sum_n = 0, 0, 0, 0, 0
        all_pred_ligand_typ = []
        all_pred_v, all_true_v = [], []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc="Validate"):

                batch = batch.cuda(rank)
                batch_size = batch.num_graphs
                results = model.module.get_diffusion_loss(
                    protein_pos=batch.protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch.protein_element_batch,
                    ligand_pos=batch.ligand_pos,
                    ligand_v=batch.ligand_atom_feature_full,
                    batch_ligand=batch.ligand_element_batch,
                )
                loss, loss_pos, loss_v, chamfer_loss = (
                    results["loss"],
                    results["loss_pos"],
                    results["loss_v"],
                    results["chamfer_loss"],
                )
                sum_loss += float(loss) * batch_size
                sum_loss_pos += float(loss_pos) * batch_size
                sum_loss_v += float(loss_v) * batch_size
                sum_chamfer_loss += float(chamfer_loss) * batch_size
                sum_n += batch_size
                all_pred_v.append(results["ligand_v_recon"].detach().cpu().numpy())
                all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())
                all_pred_ligand_typ.append(
                    results["pred_ligand_v"].detach().cpu().numpy()
                )

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        avg_chamfer_loss = sum_chamfer_loss / sum_n

        atom_auroc = None  # Initialize atom_auroc
        try:
            atom_auroc = get_auroc(
                np.concatenate(all_true_v),
                np.concatenate(all_pred_v, axis=0),
                feat_mode=config.data.transform.ligand_atom_mode,
            )
        except Exception as e:
            logger.info(f"An error occurred while calculating AUROC: {e}")
            logger.info("pred_v has Nan")
            logger.info(np.concatenate(all_pred_ligand_typ, axis=0))
        # Add wandb logging
        if not args.is_debug:
            log_dict = {
                "val/loss": avg_loss,
                "val/loss_pos": avg_loss_pos,
                "val/loss_v": avg_loss_v,
                "val/chamfer_loss": avg_chamfer_loss,
                "iteration": it,
            }
            if atom_auroc is not None:
                log_dict["val/atom_auroc"] = atom_auroc
            wandb.log(log_dict)
        # Ensure that atom_auroc is not None before logging it
        if atom_auroc is not None:
            logger.info(
                "[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f | chamfer %.6f |Avg atom auroc %.6f"
                % (it, avg_loss, avg_loss_pos, avg_loss_v, avg_chamfer_loss, atom_auroc)
            )
        else:
            logger.info(
                "[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f | chamfer %.6f | Avg atom auroc calculation failed"
                % (it, avg_loss, avg_loss_pos, avg_chamfer_loss, avg_loss_v)
            )
        return avg_loss

    try:
        best_loss, best_iter = None, None
        best_reward_mean, best_reward_iter = None, None
        best_vina_score = None
        best_vina_iter = None
        for it in range(start_iter, config.train.max_iters + 1):
            train(it)
            if (rank == 0) and (it % 100 == 0 or it == config.train.max_iters):
                val_loss = validate(it)
                if not args.is_debug:
                    final_ckpt = os.path.join(ckpt_dir, "final.pt")
                    torch.save(
                            {
                                "config": config,
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "iteration": it,
                            },
                            final_ckpt,
                        )
                if use_nft:
                    current_reward_mean = last_rollout_stats.get("reward_mean", None)
                    if current_reward_mean is None:
                        logger.info("[NFT] Reward stats unavailable; skip best-reward checkpoint.")
                    elif best_reward_mean is None or current_reward_mean > best_reward_mean:
                        logger.info(f"[NFT] Best reward mean achieved: {current_reward_mean:.6f}")
                        best_reward_mean, best_reward_iter = current_reward_mean, it
                        if not args.is_debug:
                            ckpt_path = os.path.join(ckpt_dir, "%d.pt" % it)
                            torch.save(
                                {
                                    "config": config,
                                    "model": model.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "scheduler": scheduler.state_dict(),
                                    "iteration": it,
                                },
                                ckpt_path,
                            )
                    else:
                        logger.info(
                            f"[NFT] Reward mean is not improved. "
                            f"Best reward mean: {best_reward_mean:.6f} at iter {best_reward_iter}"
                        )
                else:
                    if best_loss is None or val_loss < best_loss:
                        logger.info(f"[Validate] Best val loss achieved: {val_loss:.6f}")
                        best_loss, best_iter = val_loss, it
                        if args.is_debug:
                            continue
                        ckpt_path = os.path.join(ckpt_dir, "%d.pt" % it)
                        torch.save(
                            {
                                "config": config,
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "iteration": it,
                            },
                            ckpt_path,
                        )
                    else:
                        logger.info(
                            f"[Validate] Val loss is not improved. "
                            f"Best val loss: {best_loss:.6f} at iter {best_iter}"
                        )
            
            if config.train.use_dpo and (rank == 0) and (it % config.train.eval_freq == 0 or it == config.train.max_iters):
                num_eval_pockets = config.train.num_eval_pockets
                num_samples_per_pocket = config.train.num_eval_samples_per_pocket
                protein_root = config.train.protein_root
                eval_results = []
                for eval_idx in range(num_eval_pockets):
                    pred_pos, pred_v, pred_pos_traj, pred_v_traj = sample_diffusion_ligand(
                        model=model.module, 
                        data=val_set[eval_idx], 
                        num_samples=num_samples_per_pocket,
                        batch_size=num_samples_per_pocket, 
                        device=f"cuda:{rank}",
                        num_step=100,
                        ligand_v_temp=0.1,
                        ligand_v_noise=1.0,
                        sample_time_schedule='log',
                        center_pos_mode='protein',
                        sample_num_atoms='prior'
                    )
                    max_indices = [np.argmax(array, axis=1) for array in pred_v]
                    result = {
                        'data': val_set[eval_idx],
                        'pred_ligand_pos': pred_pos,
                        'pred_ligand_v': max_indices,
                        "pred_ligand_pos_traj": pred_pos_traj,
                        "pred_ligand_v_traj": pred_v_traj,
                    }
                    eval_results.append(result)
                eval_result = run_eval_process(
                    results_list=eval_results,
                    atom_enc_mode='add_aromatic',
                    protein_root=protein_root,
                    exhaustiveness=16,
                    docking_mode="vina_score",
                    logger=logger,
                    multiprocess=False
                )
                vina_score = eval_result[-1]
                print(vina_score)
                if not math.isnan(vina_score) and (best_vina_score is None or math.isnan(best_vina_score) or vina_score < best_vina_score):
                    best_vina_score = vina_score
                    best_vina_iter = it
                    logger.info(f"[Validate] Best vina score achieved: {vina_score:.6f}")
                    if not args.is_debug:
                        ckpt_path = os.path.join(ckpt_dir, f"best_vina_{it}.pt")
                        torch.save(
                            {
                                "config": config,
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "iteration": it,
                            },
                            ckpt_path,
                        )
                else:
                    if best_vina_score is None or best_vina_iter is None:
                        best_vina_msg = "not available yet"
                    else:
                        best_vina_msg = f"{best_vina_score:.6f} at iter {best_vina_iter}"
                    logger.info(
                        f"[Validate] Vina score is not improved. "
                        f"Best vina score: {best_vina_msg}"
                    )
                if not args.is_debug:
                    log_dict = {
                    "vina_score": vina_score,
                    "iteration": it,
                    }
                    wandb.log(log_dict)
            
    except KeyboardInterrupt:
        logger.info("Terminating...")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    os.environ["MASTER_ADDR"] = "localhost"

    # Start from port 12355 and find an available port
    port = 12355
    while not is_port_available(port) and port < 13000:
        port += 1

    if port >= 13000:
        raise ValueError("Could not find an available port between 12355 and 13000")

    os.environ["MASTER_PORT"] = str(port)
    num_gpus = torch.cuda.device_count()
    print("num_gpus: ", num_gpus)
    if num_gpus < 1:
        raise SystemExit(
            "ERROR: scripts.train_diffusion requires at least one visible CUDA GPU. "
            "Set CUDA_VISIBLE_DEVICES to a valid GPU and retry."
        )
    torch.multiprocessing.spawn(main, args=(num_gpus,), nprocs=num_gpus, join=True)
