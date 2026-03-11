import argparse
from multiprocessing import Pool
from multiprocessing import Pool
import os
import pickle

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from sympy.sets.sets import true
from scripts.eval_steric_clash import eval_steric_clash, parse_sdf_file
from scripts.evaluate_diffusion import SimilarityAnalysis
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter
# from posecheck import PoseCheck
# pc = PoseCheck()

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length, eval_bond_angle
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask


def tanimoto_sim(mol, ref):
    fp1 = Chem.RDKFingerprint(ref)
    fp2 = Chem.RDKFingerprint(mol)
    return DataStructs.TanimotoSimilarity(fp1,fp2)

def tanimoto_dis(mol, ref):
    return 1 - tanimoto_sim(mol, ref)

def tanimoto_dis_N_to_1(mols, ref):
    sim = [tanimoto_dis(m, ref) for m in mols]
    return sim

def compute_diversity(results):
    diversity = []
    for res in tqdm(results, desc='pocket'):
        pocket_results = [r for r in res if r['mol'] is not None]

        mols = [r['mol'] for r in pocket_results]
        for j in range(len(mols)):
            tmp = tanimoto_dis_N_to_1(mols, mols[j])
            tmp.pop(j)
            diversity += tmp
    print_stats("diversity", diversity,logger)
    return diversity


def eval_sample(example_idx, sample_idx, pred_pos, pred_v, ligand_filename,
                atom_enc_mode, protein_root, exhaustiveness, docking_mode, logger,
                mol_conf=None):
    
    protein_fn = os.path.join(
                    protein_root,
                    os.path.dirname(ligand_filename),
                    os.path.basename(ligand_filename)[:10] + '.pdb'
                        )
    # stability check
    recon_success = 0
    complete = 0
    eval_success = 0
    clash_detected = 0
    # clash = None
    # strain = None
    vina_results = None
    vina_score = None
    bond_dist = None
    bond_angle = None
    dic = None

    pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=atom_enc_mode)
    r_stable = analyze.check_stability(pred_pos, pred_atom_type)

    pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
    # reconstruction
    try:
        pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=atom_enc_mode)
        mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
        # sdf_path = os.path.join('/u/dzhang5/sbdd/data_for_paper/final_dpo_model', f'molecule_{example_idx}_{sample_idx}.sdf')
        # writer = Chem.SDWriter(sdf_path)
        # writer.write(mol)
        # writer.close()
        smiles = Chem.MolToSmiles(mol)
    except reconstruct.MolReconsError:
        logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
        return pred_atom_type, r_stable, pair_dist, recon_success, complete, eval_success, clash_detected, bond_dist, bond_angle, dic
    recon_success = 1

    
    if '.' in smiles:
        return pred_atom_type, r_stable, pair_dist, recon_success, complete, eval_success, clash_detected, bond_dist, bond_angle, dic
    complete = 1

    # # posecheck
    # try:
    #     pc.load_protein_from_pdb(protein_fn)
    #     pc.load_ligands_from_mols([mol])
    #     clash = pc.calculate_clashes()[0]
    #     strain = pc.calculate_strain_energy()[0]
    
    #     if strain != strain:
    #         strain = 1e10

    # except Exception as error:
    #     logger.warning(error)
    #     logger.warning('Posecheck failed %s' % f'{example_idx}_{sample_idx}')

    # steric clash check
    try:
        clash_detected, additional_info = eval_steric_clash(mol, protein_fn)
        if clash_detected:
            logger.info('Steric clash detected %s' % f'{example_idx}_{sample_idx}')
        clash_detected = 1 if clash_detected else 0
    except Exception as error:
        logger.warning(error)
        logger.warning('Steric clash check failed %s' % f'{example_idx}_{sample_idx}')
    
    # chemical and docking check
    try:
        chem_results = scoring_func.get_chem(mol)
        if docking_mode == 'qvina':
            vina_task = QVinaDockingTask.from_generated_mol(
                mol, ligand_filename, protein_root=protein_root)
            vina_results = vina_task.run_sync()
            if vina_results and len(vina_results) > 0:
                vina_score = vina_results[0].get('affinity', None)
        elif docking_mode in ['vina_score', 'vina_dock']:
            vina_task = VinaDockingTask.from_generated_mol(
                mol, ligand_filename, protein_root=protein_root)
            score_only_results = vina_task.run(mode='score_only', exhaustiveness=exhaustiveness)
            minimize_results = vina_task.run(mode='minimize', exhaustiveness=exhaustiveness)
            vina_results = {
                'score_only': score_only_results,
                'minimize': minimize_results
            }
            if score_only_results and len(score_only_results) > 0:
                vina_score = score_only_results[0].get('affinity', None)
            if docking_mode == 'vina_dock':
                docking_results = vina_task.run(mode='dock', exhaustiveness=exhaustiveness)
                vina_results['dock'] = docking_results
        else:
            vina_results = None

        eval_success = 1
    except Exception as error:
        
        logger.warning(error)
        logger.warning('Evaluation failed for %s' % f'{example_idx}_{sample_idx}')
        return pred_atom_type, r_stable, pair_dist, recon_success, complete, eval_success, clash_detected, bond_dist, bond_angle, dic

    
    # now we only consider complete molecules as success
    bond_dist = eval_bond_length.bond_distance_from_mol(mol)
    bond_angle = eval_bond_angle.bond_angle_from_mol(mol)
    
    dic = {
        'mol': mol,
        'smiles': smiles,
        'ligand_filename': ligand_filename,
        'pred_pos': pred_pos,
        'pred_v': pred_v,
        'chem_results': chem_results,
        'vina': vina_results,
        'vina_score': vina_score,
        'mol_conf': mol_conf,
        'example_idx': example_idx,
        'sample_idx': sample_idx,
        # 'clash': clash,
        # 'strain': strain,
        'steric_clash': additional_info
    }
        
        
    return pred_atom_type, r_stable, pair_dist, recon_success, complete, eval_success, clash_detected, bond_dist, bond_angle, dic

def eval_process_wrapper(inputs):
    data, args = inputs
    example_idx, sample_idx, pred_pos, pred_v, ligand_filename, mol_conf = data
    atom_enc_mode, protein_root, exhaustiveness, docking_mode, logger = args
    return eval_sample(example_idx, sample_idx, pred_pos, pred_v, ligand_filename,
                       atom_enc_mode, protein_root, exhaustiveness, docking_mode, logger,
                       mol_conf=mol_conf)

def save_result_to_file(results, save_path:str=None):
    result_dict = {}
    for idx, result in enumerate(results):
        if idx not in result_dict:
            result_dict[idx] = {}
        pred_atom_type, r_stable, pair_dist, recon_success, complete, eval_success, clash_detected, bond_dist, bond_angle, dic = result
        
        result_dict[idx]['pred_atom_type'] = pred_atom_type
        result_dict[idx]['r_stable'] = r_stable
        result_dict[idx]['pair_dist'] = pair_dist
        result_dict[idx]['recon_success'] = recon_success
        result_dict[idx]['complete'] = complete
        result_dict[idx]['eval_success'] = eval_success
        result_dict[idx]['clash_detected'] = clash_detected
        result_dict[idx]['bond_dist'] = bond_dist
        result_dict[idx]['bond_angle'] = bond_angle
        result_dict[idx]['dic'] = dic
    
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(result_dict, f)

    return result_dict

def run_eval_process(results_list, atom_enc_mode, protein_root, 
                     exhaustiveness, docking_mode, logger, multiprocess=True, save_pickle=False, result_path=None):
    results = [] #list of list of dicts
    num_samples, all_mol_stable, all_atom_stable, all_n_atom = 0,0,0,0
    n_recon_success, n_eval_success, n_complete, n_clash_detected = 0,0,0,0
    all_pair_dist, all_bond_dist, all_bond_angle, success_pair_dist = [], [], [], []
    all_atom_types = Counter()
    success_atom_types = Counter()

    
    for example_idx, sample_fn in tqdm(enumerate(results_list), total=len(results_list),desc='Evaluating'):
        results_per_sample = []
        if type(sample_fn) == str:
            sample = torch.load(sample_fn)
        else:
            sample = sample_fn
       
        pred_ligand_pos = sample['pred_ligand_pos']
        pred_ligand_v = sample['pred_ligand_v']
        pred_ligand_pos_traj = sample['pred_ligand_pos_traj']
        pred_ligand_v_traj = sample['pred_ligand_v_traj']
        ligand_filename = sample['data'].ligand_filename
        num_samples += len(pred_ligand_pos)

        
        mol_conf_list = None
        if isinstance(sample, dict):
            mol_conf_list = sample.get('mol_conf', None)

        sample_data = []
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(pred_ligand_pos, pred_ligand_v)):
            mol_conf = None
            if mol_conf_list is not None and sample_idx < len(mol_conf_list):
                mol_conf = mol_conf_list[sample_idx]
            sample_data.append((example_idx, sample_idx, pred_pos, pred_v, ligand_filename, mol_conf))
    
        input_args = (atom_enc_mode, protein_root, exhaustiveness, docking_mode, logger)
        if multiprocess:
            partitioned_data = [(data, input_args) for data in sample_data]
            with Pool(min(len(partitioned_data), 10)) as pool:
                print(f"Pool is using {pool._processes} processes")
                results_data = list(tqdm(pool.imap(eval_process_wrapper, partitioned_data), total=len(partitioned_data)))
        else:
            results_data = [eval_process_wrapper((data, input_args)) for data in sample_data]

        if save_pickle and result_path is not None:
            pickle_save_path = os.path.join(result_path, f'example_results/example_{example_idx}_results.pkl')
            save_result_to_file(results_data, pickle_save_path)
            print(f"save result {example_idx} to path {pickle_save_path}")

        for result in results_data:
    
            pred_atom_type, r_stable, pair_dist, recon_success, complete, eval_success, clash_detected, bond_dist, bond_angle, dic = result
            all_atom_types += Counter(pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]
            all_pair_dist += pair_dist
            
            n_clash_detected += clash_detected
            n_recon_success += recon_success
            n_complete += complete
            n_eval_success += eval_success
            
            if eval_success:
                all_bond_dist += bond_dist
                all_bond_angle += bond_angle
                success_pair_dist += pair_dist
                success_atom_types += Counter(pred_atom_type)
                results_per_sample.append(dic)

        results.append(results_per_sample)

    logger.info(f'Evaluate done! {n_eval_success} samples in total.')
    if n_eval_success == 0:
        logger.warning('No molecules passed evaluation; reporting NaN validity metrics.')

    fraction_mol_stable = (all_mol_stable / n_eval_success) if n_eval_success > 0 else float('nan')
    fraction_atm_stable = (all_atom_stable / all_n_atom) if all_n_atom > 0 else float('nan')

    fraction_recon = (n_recon_success / num_samples) if num_samples > 0 else float('nan')
    fraction_complete = (n_complete / n_eval_success) if n_eval_success > 0 else float('nan')
    fraction_clash_detected = (n_clash_detected / n_eval_success) if n_eval_success > 0 else float('nan')


    intra_clash_atom_num = [r['steric_clash']['lig_lig_clash']['clash_atom_num'] for sample_results in results for r in sample_results]
    inter_clash_atom_num = [r['steric_clash']['lig_pro_clash']['clash_atom_num'] for sample_results in results for r in sample_results]
    intra_clash_atom_ratio = (np.sum(intra_clash_atom_num) / all_n_atom) if all_n_atom > 0 else float('nan')
    inter_clash_atom_ratio = (np.sum(inter_clash_atom_num) / all_n_atom) if all_n_atom > 0 else float('nan')

    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'complete': fraction_complete,
        'clash_detected': fraction_clash_detected,
        'intra_clash_atom_ratio': intra_clash_atom_ratio,
        'inter_clash_atom_ratio': inter_clash_atom_ratio
    }

    vina, vina_score_only, vina_min, vina_dock, pocket_top1_stats = print_results(n_recon_success=n_recon_success,
                n_complete=n_complete,
                results=results,
                logger=logger,
                docking_mode=docking_mode)

    return results, all_bond_dist, all_bond_angle, success_pair_dist, success_atom_types, validity_dict, vina_score_only, pocket_top1_stats

def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')

def print_results(n_recon_success:float, n_complete:float, results:list, logger, docking_mode:str):

    logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
    n_recon_success, n_complete, sum(len(r) for r in results)))

    vina = None
    vina_score_only = None
    vina_min = None
    vina_dock = None

    qed = []
    sa = []
    vina_list = []
    vina_score_list = []
    vina_min_list = []
    vina_dock_list = []

    # Lists to store top1 (best) vina scores for each pocket
    pocket_top1_vina = []
    pocket_top1_vina_score = []
    pocket_top1_vina_min = []
    pocket_top1_vina_dock = []

    for sample_results in results:
        # Collect vina scores for current pocket
        pocket_vina_list = []
        pocket_vina_score_list = []
        pocket_vina_min_list = []
        pocket_vina_dock_list = []

        for result in sample_results:
            if result['chem_results']['qed'] is not None:
                qed.append(result['chem_results']['qed'])
            if result['chem_results']['sa'] is not None:
                sa.append(result['chem_results']['sa'])
            # if result['clash'] is not None:
            #     clash.append(result['clash'])
            # if result['strain'] is not None:
            #     strain.append(result['strain'])
            if result['vina'] is not None:
                if docking_mode == 'qvina':
                    score = result['vina'][0]['affinity']
                    vina_list.append(score)
                    pocket_vina_list.append(score)
                elif docking_mode in ['vina_dock', 'vina_score']:
                    score_only = result['vina']['score_only'][0]['affinity']
                    minimize = result['vina']['minimize'][0]['affinity']
                    vina_score_list.append(score_only)
                    vina_min_list.append(minimize)
                    pocket_vina_score_list.append(score_only)
                    pocket_vina_min_list.append(minimize)
                    if docking_mode == 'vina_dock':
                        dock_score = result['vina']['dock'][0]['affinity']
                        vina_dock_list.append(dock_score)
                        pocket_vina_dock_list.append(dock_score)

        # Find top1 (minimum/best) score for current pocket
        if pocket_vina_list:
            pocket_top1_vina.append(min(pocket_vina_list))
        if pocket_vina_score_list:
            pocket_top1_vina_score.append(min(pocket_vina_score_list))
        if pocket_vina_min_list:
            pocket_top1_vina_min.append(min(pocket_vina_min_list))
        if pocket_vina_dock_list:
            pocket_top1_vina_dock.append(min(pocket_vina_dock_list))

    print_stats('QED', qed, logger)
    print_stats('SA', sa, logger)
    # print_stats('Clash', clash, logger)
    # print_stats('Strain', strain, logger)
    if docking_mode == 'qvina':
        vina = print_stats('Vina', vina_list, logger)
        # Print top1 statistics
        if pocket_top1_vina:
            logger.info(f'Pocket Top1 Vina: Mean: {np.mean(pocket_top1_vina):.3f}, '
                       f'Median: {np.median(pocket_top1_vina):.3f}, '
                       f'Std: {np.std(pocket_top1_vina):.3f}, '
                       f'Count: {len(pocket_top1_vina)}')
    elif docking_mode in ['vina_dock', 'vina_score']:
        vina_score_only = print_stats('Vina Score', vina_score_list, logger)
        vina_min = print_stats('Vina Min', vina_min_list, logger)
        # Print top1 statistics for score_only
        if pocket_top1_vina_score:
            logger.info(f'Pocket Top1 Vina Score: Mean: {np.mean(pocket_top1_vina_score):.3f}, '
                       f'Median: {np.median(pocket_top1_vina_score):.3f}, '
                       f'Std: {np.std(pocket_top1_vina_score):.3f}, '
                       f'Count: {len(pocket_top1_vina_score)}')
        # Print top1 statistics for minimize
        if pocket_top1_vina_min:
            logger.info(f'Pocket Top1 Vina Min: Mean: {np.mean(pocket_top1_vina_min):.3f}, '
                       f'Median: {np.median(pocket_top1_vina_min):.3f}, '
                       f'Std: {np.std(pocket_top1_vina_min):.3f}, '
                       f'Count: {len(pocket_top1_vina_min)}')
        if docking_mode == 'vina_dock':
            vina_dock = print_stats('Vina Dock', vina_dock_list, logger)
            # Print top1 statistics for dock
            if pocket_top1_vina_dock:
                logger.info(f'Pocket Top1 Vina Dock: Mean: {np.mean(pocket_top1_vina_dock):.3f}, '
                           f'Median: {np.median(pocket_top1_vina_dock):.3f}, '
                           f'Std: {np.std(pocket_top1_vina_dock):.3f}, '
                           f'Count: {len(pocket_top1_vina_dock)}')

    # Prepare pocket top1 statistics to return
    pocket_top1_stats = {
        'pocket_top1_vina': pocket_top1_vina if pocket_top1_vina else None,
        'pocket_top1_vina_score': pocket_top1_vina_score if pocket_top1_vina_score else None,
        'pocket_top1_vina_min': pocket_top1_vina_min if pocket_top1_vina_min else None,
        'pocket_top1_vina_dock': pocket_top1_vina_dock if pocket_top1_vina_dock else None
    }

    return vina, vina_score_only, vina_min, vina_dock, pocket_top1_stats

def calculate_stats(values):
    if not values:
        return None
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'q25': np.quantile(values, 0.25),
        'q75': np.quantile(values, 0.75)
    }

def _as_scalar_conf(conf_value):
    if conf_value is None:
        return None
    if isinstance(conf_value, torch.Tensor):
        conf_value = conf_value.detach().cpu().numpy()
    if isinstance(conf_value, (list, tuple, np.ndarray)):
        if len(conf_value) == 0:
            return None
        return float(np.mean(conf_value))
    try:
        return float(conf_value)
    except Exception:
        return None

def _pearson_corr(x, y):
    if len(x) < 2:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])

def _rankdata(a):
    a = np.asarray(a)
    sorter = np.argsort(a, kind='mergesort')
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1], [True]))
    idx = np.flatnonzero(obs)
    ranks = np.zeros(len(a), dtype=float)
    for i in range(len(idx) - 1):
        start = idx[i]
        end = idx[i + 1]
        rank = 0.5 * (start + end - 1)
        ranks[start:end] = rank
    return ranks[inv]

def _spearman_corr(x, y):
    if len(x) < 2:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson_corr(rx, ry)

def report_confidence_vina(results, docking_mode, logger, result_path):
    rows = []
    for sample_results in results:
        for res in sample_results:
            if not isinstance(res, dict):
                continue
            conf = _as_scalar_conf(res.get('mol_conf', None))
            vina = res.get('vina_score', None)
            if conf is None or vina is None:
                continue
            try:
                vina = float(vina)
            except Exception:
                continue
            rows.append((
                res.get('example_idx', None),
                res.get('sample_idx', None),
                conf,
                vina
            ))

    if len(rows) == 0:
        logger.info('Confidence-Vina: no valid pairs found (missing mol_conf or vina_score).')
        return None

    conf_scores = [r[2] for r in rows]
    vina_scores = [r[3] for r in rows]

    pearson = _pearson_corr(conf_scores, vina_scores)
    spearman = _spearman_corr(conf_scores, vina_scores)

    save_path = os.path.join(result_path, 'confidence_vina.tsv')
    with open(save_path, 'w') as f:
        f.write('example_idx\tsample_idx\tmol_conf\tvina_score\n')
        for ex_idx, smp_idx, conf, vina in rows:
            f.write(f'{ex_idx}\t{smp_idx}\t{conf:.6f}\t{vina:.6f}\n')

    logger.info(f'Confidence-Vina pairs: {len(rows)} (saved to {save_path})')
    for i, (ex_idx, smp_idx, conf, vina) in enumerate(rows[:5]):
        logger.info(f'Conf-Vina sample[{i}]: example={ex_idx}, sample={smp_idx}, conf={conf:.6f}, vina={vina:.6f}')
    if pearson is not None:
        logger.info(f'Confidence-Vina Pearson: {pearson:.4f}')
    if spearman is not None:
        logger.info(f'Confidence-Vina Spearman: {spearman:.4f}')
    return save_path

def print_stats(name, values, logger):

    stats = calculate_stats(values)
    if stats:
        logger.info(f'{name}: Mean: {stats["mean"]:.3f} Median: {stats["median"]:.3f}, '
                   f'std: {stats["std"]:.3f}, q25: {stats["q25"]:.3f}, q75: {stats["q75"]:.3f}')
    return stats['mean'] if stats else None

def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default='./data/test_set')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none'])
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--multiprocess', type=eval, default=True)
    parser.add_argument('--save_pickle', type=eval, default=True, help='是否保存详细结果到pickle文件')
    parser.add_argument('--report_confidence_vina', type=eval, default=False,
                        help='是否输出confidence与vina score对应结果')
    args = parser.parse_args()

    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    if args.save_pickle:
        eval_result_path = os.path.join(result_path, "example_results")
        print(f"create path: {eval_result_path}")
        os.makedirs(eval_result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')
    

    # Load generated data
    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)
    logger.info(f'Load generated data done! {num_examples} examples in total.')


    results, all_bond_dist,all_bond_angle, success_pair_dist, success_atom_types, validity_dict, vina_score_only, pocket_top1_stats \
            = run_eval_process(results_fn_list, args.atom_enc_mode, args.protein_root, \
                            args.exhaustiveness, args.docking_mode, logger, multiprocess=args.multiprocess, 
                            save_pickle=args.save_pickle, result_path=result_path)

    has_eval_data = any(len(sample_results) > 0 for sample_results in results)
    if has_eval_data:
        c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
        c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
        success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
        success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
        atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
        c_bond_angle_profile = eval_bond_angle.get_bond_angle_profile(all_bond_angle)
        c_bond_angle_dict = eval_bond_angle.eval_bond_angle_profile(c_bond_angle_profile)
    else:
        logger.warning('No successfully evaluated samples; skip bond/atom/diversity metrics.')
        c_bond_length_dict = {'JSD_BL': float('nan')}
        c_bond_angle_dict = {'JSD_BA': float('nan')}
        success_pair_length_profile = {}
        success_js_metrics = {}
        atom_type_js = float('nan')
   

    if has_eval_data:
        filtered_c_bond_length_dict = {k: round(v, 4) for k, v in c_bond_length_dict.items() if v is not None}
        filtered_c_bond_angle_dict = {k: round(v, 4) for k, v in c_bond_angle_dict.items() if v is not None}
        c_bond_length_dict['JSD_BL'] = round(np.mean([v for _, v in filtered_c_bond_length_dict.items()]), 4)
        c_bond_angle_dict['JSD_BA'] = round(np.mean([v for _, v in filtered_c_bond_angle_dict.items()]), 4)
   
    

    print_dict(validity_dict, logger)

    logger.info('JS bond distances of complete mols: ')
    print_dict(c_bond_length_dict, logger)
    logger.info('JS bond angles of complete mols: ')
    print_dict(c_bond_angle_dict, logger)
    print_dict(success_js_metrics, logger)

    logger.info('Atom type JS: %.4f' % atom_type_js)

    # check ring distribution
    if has_eval_data:
        print_ring_ratio([r['chem_results']['ring_size'] for sample_results in results for r in sample_results], logger)
        # Begin Tanimoto similarity analysis
        compute_diversity(results)
    
    if args.report_confidence_vina:
        report_confidence_vina(results, args.docking_mode, logger, result_path)

    if args.save:
        if has_eval_data:
            eval_bond_length.plot_distance_hist(success_pair_length_profile,
                                                metrics=success_js_metrics,
                                                save_path=os.path.join(result_path, f'pair_dist_hist_{args.eval_step}.png'))
        torch.save({
            'stability': validity_dict,
            'bond_length': all_bond_dist,
            'all_results': results,
            'pocket_top1_stats': pocket_top1_stats
        }, os.path.join(result_path, f'metrics_{args.eval_step}.pt'))
