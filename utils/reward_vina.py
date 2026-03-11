import contextlib
import io
import math
import os
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import numpy as np
from rdkit import Chem

from utils import reconstruct, transforms
from utils.evaluation.docking_vina import VinaDockingTask

_FAIL_COUNTS = Counter()


def _record_failure(reason: str, detail: Optional[str] = None, log: bool = False) -> None:
    _FAIL_COUNTS[reason] += 1
    if log:
        if detail:
            print(f"[reward_vina] fail: {reason} | {detail}")
        else:
            print(f"[reward_vina] fail: {reason}")


def _summarize_failures() -> str:
    if not _FAIL_COUNTS:
        return "none"
    return ", ".join([f"{k}={v}" for k, v in _FAIL_COUNTS.most_common()])


def _reconstruct_mol(pred_pos, pred_v_idx, atom_mode: str):
    idx = np.asarray(pred_v_idx).astype(int)
    atomic_nums = transforms.get_atomic_number_from_index(idx, atom_mode)
    aromatic = transforms.is_aromatic_from_index(idx, atom_mode)
    mol = reconstruct.reconstruct_from_generated(pred_pos, atomic_nums, aromatic)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            mol = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol), removeHs=False, sanitize=True)
        except Exception:
            return None
    return mol


def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


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
) -> Tuple[Optional[float], Optional[Dict[str, Any]], Optional[Chem.Mol]]:
    """
    Compute Vina score-only reward from generated geometry and atom types.

    Returns:
        (reward, info, mol) where reward is float or None if invalid.
    """
    try:
        mol = _reconstruct_mol(pred_pos, pred_v_idx, atom_mode)
    except Exception:
        _record_failure("reconstruct_exception", log=log_vina)
        return None, None, None

    if mol is None:
        _record_failure("reconstruct_failed", log=log_vina)
        return None, None, None

    if ligand_filename is None and data is not None:
        ligand_filename = getattr(data, "ligand_filename", None)

    protein_path = None
    if data is not None:
        protein_filename = getattr(data, "protein_filename", None)
        if protein_filename is not None:
            if os.path.isabs(protein_filename):
                protein_path = protein_filename
            elif protein_root is not None:
                protein_path = os.path.join(protein_root, protein_filename)
            else:
                protein_path = protein_filename

    # Prefer evaluate_diffusion_multiprocess.py style resolution if possible
    if ligand_filename is not None and protein_root is not None:
        candidate = os.path.join(
            protein_root,
            os.path.dirname(ligand_filename),
            os.path.basename(ligand_filename)[:10] + ".pdb",
        )
        if os.path.exists(candidate):
            protein_path = candidate
    elif ligand_filename is not None and os.path.isabs(ligand_filename):
        candidate = os.path.join(
            os.path.dirname(ligand_filename),
            os.path.basename(ligand_filename)[:10] + ".pdb",
        )
        if os.path.exists(candidate):
            protein_path = candidate

    task = None
    try:
        if protein_path is not None and os.path.exists(protein_path):
            task = VinaDockingTask(
                protein_path,
                mol,
                tmp_dir=tmp_dir,
                size_factor=size_factor,
                buffer=buffer,
            )
        else:
            if ligand_filename is None or protein_root is None:
                _record_failure(
                    "missing_protein_path",
                    detail=f"protein_path=None ligand_filename={ligand_filename} protein_root={protein_root}",
                    log=log_vina,
                )
                return None, None, mol
            task = VinaDockingTask.from_generated_mol(
                mol,
                ligand_filename,
                protein_root=protein_root,
                tmp_dir=tmp_dir,
                size_factor=size_factor,
                buffer=buffer,
            )

        if log_vina:
            results = task.run(mode="score_only", exhaustiveness=exhaustiveness)
        else:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                results = task.run(mode="score_only", exhaustiveness=exhaustiveness)

        if not results or results[0].get("affinity", None) is None:
            _record_failure("vina_no_result", log=log_vina)
            return None, None, mol

        affinity = float(results[0]["affinity"])
    except Exception as e:
        _record_failure(
            "vina_exception",
            detail=f"protein_path={protein_path} ligand_filename={ligand_filename} err={type(e).__name__}: {e}",
            log=log_vina,
        )
        return None, None, mol
    finally:
        if cleanup and task is not None:
            _safe_remove(getattr(task, "ligand_path", None))
            ligand_pdbqt = getattr(task, "ligand_path", "")
            if ligand_pdbqt:
                _safe_remove(ligand_pdbqt[:-4] + ".pdbqt")

    if math.isnan(affinity) or math.isinf(affinity):
        _record_failure("vina_nan", log=log_vina)
        return None, None, mol

    # Clip extreme docking scores and map to [0, 1] to avoid reward hacking.
    # More negative affinity is better, so map [-16, -1] -> [1, 0] by default.
    affinity_clip = float(np.clip(affinity, -16.0, -1.0))
    if score_sign < 0:
        reward = (-1.0 - affinity_clip) / 15.0
    else:
        reward = (affinity_clip + 16.0) / 15.0
    info = {
        "vina_score": affinity,
        "vina_score_clip": affinity_clip,
        "fail_summary": _summarize_failures(),
    }
    return reward, info, mol
