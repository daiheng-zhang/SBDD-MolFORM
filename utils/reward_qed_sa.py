import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED

from utils import reconstruct, transforms
from utils.evaluation.sascorer import compute_sa_score


def qed_sa_from_mol(mol, w_qed: float = 1.0, w_sa: float = 1.0):
    if mol is None:
        return None, None
    try:
        qed = float(QED.qed(mol))
        sa = float(compute_sa_score(mol))
    except Exception:
        return None, None
    reward = w_qed * qed + w_sa * sa
    info = {
        "qed": qed,
        "sa": sa,
    }
    return reward, info


def reward_from_generated(
    pred_pos,
    pred_v_idx,
    atom_mode: str = "add_aromatic",
    w_qed: float = 1.0,
    w_sa: float = 1.0,
):
    """
    Compute QED+SA reward from generated geometry and atom types.

    Args:
        pred_pos: (N, 3) array-like, ligand atom positions
        pred_v_idx: (N,) array-like, atom type indices
        atom_mode: ligand atom encoding mode
        w_qed, w_sa: weights for QED and SA components
    Returns:
        (reward, info, mol) where reward is float or None if invalid.
    """
    try:
        idx = np.asarray(pred_v_idx).astype(int)
        atomic_nums = transforms.get_atomic_number_from_index(idx, atom_mode)
        aromatic = transforms.is_aromatic_from_index(idx, atom_mode)
        mol = reconstruct.reconstruct_from_generated(pred_pos, atomic_nums, aromatic)
        # Sanitize via round-trip to SMILES for stability
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    except Exception:
        return None, None, None

    reward, info = qed_sa_from_mol(mol, w_qed=w_qed, w_sa=w_sa)
    if reward is None or (isinstance(reward, float) and (math.isnan(reward) or math.isinf(reward))):
        return None, None, None
    return reward, info, mol
