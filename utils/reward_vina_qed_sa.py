import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED

import utils.reward_vina as reward_vina
from utils.evaluation.sascorer import compute_sa_score


def reward_from_generated(
    pred_pos,
    pred_v_idx,
    atom_mode: str = "add_aromatic",
    data: Optional[Any] = None,
    protein_root: Optional[str] = None,
    ligand_filename: Optional[str] = None,
    exhaustiveness: int = 8,
    size_factor: float = 1.0,
    buffer: float = 5.0,
    tmp_dir: str = "./tmp_vina",
    score_sign: float = -1.0,
    log_vina: bool = False,
    cleanup: bool = True,
    vina_clip_low: float = -16.0,
    vina_clip_high: float = -1.0,
    vina_offset: float = 1.0,
    vina_divisor: float = 15.0,
    qed_shift: float = 0.17,
    qed_scale: float = 0.83,
    sa_shift: float = 0.17,
    sa_scale: float = 0.83,
) -> Tuple[Optional[float], Optional[Dict[str, Any]], Optional[Chem.Mol]]:
    """
    Compute mixed reward with normalized Vina + normalized QED + normalized SA:
        r(m) = (-clip(vina(m), -16, -1) + 1) / 15
             + (QED(m) - 0.17) / 0.83
             + (SA(m) - 0.17) / 0.83
    """
    if vina_clip_high <= vina_clip_low:
        raise ValueError(
            f"Invalid vina clipping range: [{vina_clip_low}, {vina_clip_high}]"
        )
    if abs(vina_divisor) < 1e-12:
        raise ValueError("vina_divisor must be non-zero.")
    if abs(qed_scale) < 1e-12:
        raise ValueError("qed_scale must be non-zero.")
    if abs(sa_scale) < 1e-12:
        raise ValueError("sa_scale must be non-zero.")

    _, vina_info, mol = reward_vina.reward_from_generated(
        pred_pos,
        pred_v_idx,
        atom_mode=atom_mode,
        data=data,
        protein_root=protein_root,
        ligand_filename=ligand_filename,
        exhaustiveness=exhaustiveness,
        size_factor=size_factor,
        buffer=buffer,
        tmp_dir=tmp_dir,
        score_sign=score_sign,
        log_vina=log_vina,
        cleanup=cleanup,
    )
    if vina_info is None or mol is None:
        return None, None, mol

    vina_score = vina_info.get("vina_score", None)
    if vina_score is None:
        return None, None, mol
    try:
        vina_score = float(vina_score)
    except Exception:
        return None, None, mol
    if math.isnan(vina_score) or math.isinf(vina_score):
        return None, None, mol

    try:
        qed = float(QED.qed(mol))
    except Exception:
        return None, None, mol
    if math.isnan(qed) or math.isinf(qed):
        return None, None, mol

    try:
        sa = float(compute_sa_score(mol))
    except Exception:
        return None, None, mol
    if math.isnan(sa) or math.isinf(sa):
        return None, None, mol

    vina_score_clip = float(np.clip(vina_score, vina_clip_low, vina_clip_high))
    vina_term = (-vina_score_clip + vina_offset) / vina_divisor
    qed_term = (qed - qed_shift) / qed_scale
    sa_term = (sa - sa_shift) / sa_scale
    reward = vina_term + qed_term + sa_term
    if math.isnan(reward) or math.isinf(reward):
        return None, None, mol

    info = {
        "vina_score": vina_score,
        "vina_score_clip": vina_score_clip,
        "vina_term": float(vina_term),
        "qed": qed,
        "qed_term": float(qed_term),
        "sa": sa,
        "sa_term": float(sa_term),
        "reward_vina_qed_sa": float(reward),
        "fail_summary": vina_info.get("fail_summary", "none"),
    }
    return float(reward), info, mol
