#!/usr/bin/env python3
"""Split a .pt result file into N chunks and evaluate them in parallel.

Usage:
    python split_and_eval_parallel.py result.pt --protein_root /data/ye/protein-ligand \
        --output_dir ./eval_output --exhaustiveness 8 --num_workers 8
"""
import argparse
import os, sys, json, shutil, time, tempfile
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _slice_pt_molecules(data: dict, start: int, end: int):
    """Keep only molecules in [start, end) for list-valued pred_* / time fields.

    Returns:
        (sliced_data, n_kept)
    """
    out = {k: v for k, v in data.items()}
    pos_list = list(data.get('pred_ligand_pos', []) or [])
    n_total = len(pos_list)
    start = max(0, int(start))
    end = min(n_total, int(end))
    if start >= end:
        raise ValueError(f"empty slice start={start} end={end} n_total={n_total}")
    out['pred_ligand_pos'] = pos_list[start:end]
    v_list = data.get('pred_ligand_v', []) or []
    if len(v_list) == n_total:
        out['pred_ligand_v'] = v_list[start:end]
    for traj_key in ['pred_ligand_pos_traj', 'pred_ligand_v_traj', 'pred_ligand_log_v_traj']:
        if traj_key in data and len(data[traj_key]) == n_total:
            out[traj_key] = data[traj_key][start:end]
    if 'time' in data and len(data['time']) == n_total:
        out['time'] = data['time'][start:end]
    return out, end - start


def resolve_eval_slice(data: dict, *, prudent_final_only: bool = False, max_samples: int = None):
    """Return (start, end, n_total, note) for molecules to evaluate."""
    pos_list = data.get('pred_ligand_pos', []) or []
    n_total = len(pos_list)
    start, end = 0, n_total
    note = f'all {n_total}'

    if prudent_final_only:
        meta = data.get('meta') or {}
        extra = data.get('extra_info') or {}
        n_final = meta.get('n_final_pool', extra.get('n_final_pool'))
        fps = meta.get('final_pool_start', extra.get('final_pool_start'))
        if n_final is not None and fps is not None:
            start = int(fps)
            end = min(n_total, start + int(n_final))
            note = f'prudent_final_only meta start={start} n={end - start}'
        elif n_final is not None:
            n_final = int(n_final)
            start = max(0, n_total - n_final)
            end = n_total
            note = f'prudent_final_only last {n_final}'
        else:
            stats = meta.get('generation_stats') or []
            if stats and isinstance(stats[-1], dict) and 'pool_size' in stats[-1]:
                n_final = int(stats[-1]['pool_size'])
                start = max(0, n_total - n_final)
                end = n_total
                note = f'prudent_final_only generation_stats last pool_size={n_final}'
            else:
                note = 'prudent_final_only requested but no meta; evaluating all'

    if max_samples is not None and max_samples > 0 and (end - start) > max_samples:
        end = start + int(max_samples)
        note = f'{note}; max_samples={max_samples}'

    return start, end, n_total, note


def split_pt(pt_path: str, num_chunks: int, tmp_dir: str,
             *, prudent_final_only: bool = False, max_samples: int = None):
    """Load .pt and split pred_ligand_pos / pred_ligand_v into num_chunks."""
    data = torch.load(pt_path, map_location='cpu')
    pos_list = data.get('pred_ligand_pos', [])
    n_total = len(pos_list)

    if n_total == 0:
        raise ValueError(f"No molecules found in {pt_path} (pred_ligand_pos is empty)")

    start, end, n_total, note = resolve_eval_slice(
        data, prudent_final_only=prudent_final_only, max_samples=max_samples,
    )
    if start != 0 or end != n_total:
        data, n_eval = _slice_pt_molecules(data, start, end)
        print(f"Slice molecules [{start}:{end}] → {n_eval} to evaluate ({note})")
    else:
        n_eval = n_total
        print(f"Evaluating all {n_eval} molecules ({note})")

    pos_list = data.get('pred_ligand_pos', [])
    v_list = data.get('pred_ligand_v', [])
    n_eval = len(pos_list)

    chunk_size = max(1, int(np.ceil(n_eval / num_chunks)))
    chunks = []
    for i in range(0, n_eval, chunk_size):
        chunk_data = {k: v for k, v in data.items()}
        chunk_data['pred_ligand_pos'] = pos_list[i:i + chunk_size]
        chunk_data['pred_ligand_v'] = v_list[i:i + chunk_size] if len(v_list) == n_eval else []
        for traj_key in ['pred_ligand_pos_traj', 'pred_ligand_v_traj', 'pred_ligand_log_v_traj']:
            if traj_key in data and len(data[traj_key]) == n_eval:
                chunk_data[traj_key] = data[traj_key][i:i + chunk_size]
        if 'time' in data and len(data['time']) == n_eval:
            chunk_data['time'] = data['time'][i:i + chunk_size]

        chunk_path = os.path.join(tmp_dir, f"chunk_{len(chunks):03d}.pt")
        torch.save(chunk_data, chunk_path)
        chunks.append((chunk_path, len(chunk_data['pred_ligand_pos'])))

    print(f"Split {n_eval} molecules into {len(chunks)} chunks (chunk_size={chunk_size})")
    return chunks, n_eval, n_total


def eval_chunk(args_tuple):
    """Run evaluate_pt_with_correct_reconstruct.py on a single chunk."""
    idx, chunk_path, n_mols, eval_script, output_dir, protein_root, exhaustiveness, extra_args = args_tuple
    chunk_out = os.path.join(output_dir, f"chunk_{idx:03d}")
    os.makedirs(chunk_out, exist_ok=True)

    tmp_dock = os.path.join(output_dir, f"tmp_dock_{idx:03d}")
    os.makedirs(tmp_dock, exist_ok=True)

    cmd = [
        sys.executable, '-u', eval_script,
        chunk_path,
        '--protein_root', protein_root,
        '--output_dir', chunk_out,
        '--exhaustiveness', str(exhaustiveness),
        '--tmp_dir', tmp_dock,
        '--save_intermediate_interval', '0',  # Save only at end
    ] + extra_args

    print(f"[chunk {idx:03d}] Starting {n_mols} molecules: {' '.join(cmd[:8])}...")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"[chunk {idx:03d}] FAILED after {elapsed:.0f}s: {result.stderr[-500:]}")
            return {'chunk_idx': idx, 'success': False, 'xlsx_path': None, 'n_mols': n_mols, 'elapsed': elapsed}
        print(f"[chunk {idx:03d}] Done in {elapsed:.0f}s ({n_mols / elapsed:.1f} mol/s)")
        # Find the xlsx file in chunk_out
        xlsx_files = list(Path(chunk_out).rglob("*.xlsx"))
        xlsx_path = str(xlsx_files[0]) if xlsx_files else None
        if not xlsx_path:
            print(f"[chunk {idx:03d}] WARNING: No xlsx found in {chunk_out}")
        return {'chunk_idx': idx, 'success': True, 'xlsx_path': xlsx_path, 'n_mols': n_mols, 'elapsed': elapsed}
    except subprocess.TimeoutExpired:
        print(f"[chunk {idx:03d}] TIMEOUT after 7200s")
        return {'chunk_idx': idx, 'success': False, 'xlsx_path': None, 'n_mols': n_mols, 'elapsed': 7200}


def merge_xlsx_results(xlsx_paths, output_xlsx):
    """Merge multiple evaluation xlsx files into one."""
    dfs = []
    for path in xlsx_paths:
        if path and os.path.exists(path):
            try:
                df = pd.read_excel(path, sheet_name='评估结果')
                dfs.append(df)
            except Exception as e:
                print(f"WARNING: Could not read {path}: {e}")
                # Try reading any sheet
                try:
                    xl = pd.ExcelFile(path)
                    for sheet in xl.sheet_names:
                        df = pd.read_excel(path, sheet_name=sheet)
                        dfs.append(df)
                        break
                except Exception:
                    pass

    if not dfs:
        print("ERROR: No valid xlsx files to merge!")
        return False

    merged = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(output_xlsx) if os.path.dirname(output_xlsx) else '.', exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
        merged.to_excel(writer, sheet_name='评估结果', index=False)
    print(f"Merged {len(dfs)} sheets → {output_xlsx} ({len(merged)} rows)")
    return True


def main():
    parser = argparse.ArgumentParser(description='Split .pt and evaluate in parallel')
    parser.add_argument('--pt_file', type=str, required=True, help='Path to result .pt file')
    parser.add_argument('--protein_root', type=str, default='/data/ye/protein-ligand')
    parser.add_argument('--output_dir', type=str, required=True, help='Base output directory')
    parser.add_argument('--exhaustiveness', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--max_samples', type=int, default=None, help='Max total samples to evaluate')
    parser.add_argument('--prudent_final_only', action='store_true',
                        help='Only evaluate last-round final pool (scaffold_prudent)')
    parser.add_argument('--vina_timeout', type=int, default=None, help='Vina timeout per molecule (seconds)')
    parser.add_argument('--keep_chunks', action='store_true', help='Keep intermediate chunk outputs')
    args = parser.parse_args()

    pt_file = os.path.abspath(args.pt_file)
    if not os.path.exists(pt_file):
        print(f"ERROR: .pt file not found: {pt_file}")
        sys.exit(1)

    eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'evaluate_pt_with_correct_reconstruct.py')
    if not os.path.exists(eval_script):
        print(f"ERROR: evaluate_pt_with_correct_reconstruct.py not found at {eval_script}")
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create temp dir for chunk .pt files
    tmp_dir = tempfile.mkdtemp(prefix='split_eval_', dir=output_dir)
    print(f"Temporary dir: {tmp_dir}")

    try:
        # Step 1: Split (optionally slice to final pool / max_samples first)
        print(f"\n{'='*60}\nStep 1: Loading & splitting {pt_file}")
        chunks, n_eval, n_total = split_pt(
            pt_file, args.num_workers, tmp_dir,
            prudent_final_only=bool(args.prudent_final_only),
            max_samples=args.max_samples,
        )

        # Step 2: Evaluate chunks in parallel
        print(f"\n{'='*60}\nStep 2: Evaluating {len(chunks)} chunks with {args.num_workers} workers")
        print(f"Will evaluate {n_eval}/{n_total} molecules from .pt")
        extra_args = []
        if args.vina_timeout:
            extra_args.extend(['--vina-timeout-seconds', str(args.vina_timeout)])
        # Do NOT pass max_samples to each chunk (already sliced)

        tasks = [(i, cp, nm, eval_script, output_dir, args.protein_root, args.exhaustiveness, extra_args)
                 for i, (cp, nm) in enumerate(chunks)]

        t0 = time.time()
        results = []
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(eval_chunk, t): t for t in tasks}
            for future in as_completed(futures):
                r = future.result()
                if r:
                    results.append(r)
                else:
                    print(f"WARNING: A chunk returned None")

        results.sort(key=lambda r: r['chunk_idx'])
        elapsed_total = time.time() - t0
        n_success = sum(1 for r in results if r['success'])
        print(f"\nEvaluation complete: {n_success}/{len(results)} chunks succeeded in {elapsed_total:.0f}s")

        # Step 3: Merge results
        print(f"\n{'='*60}\nStep 3: Merging results")
        xlsx_paths = [r['xlsx_path'] for r in results if r['success'] and r['xlsx_path']]
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        merged_xlsx = os.path.join(output_dir, f'evaluation_results_{timestamp}.xlsx')
        merge_xlsx_results(xlsx_paths, merged_xlsx)

        # Step 4: Collect all reconstructed SDFs into one directory
        sdf_dir = os.path.join(output_dir, f'reconstructed_molecules')
        os.makedirs(sdf_dir, exist_ok=True)
        for r in results:
            if r['success']:
                chunk_dir = os.path.join(output_dir, f"chunk_{r['chunk_idx']:03d}")
                for sdf in Path(chunk_dir).rglob("*.sdf"):
                    dest = os.path.join(sdf_dir, sdf.name)
                    if not os.path.exists(dest):
                        shutil.copy2(sdf, dest)

        # Step 5: Cleanup
        if not args.keep_chunks:
            print(f"\n{'='*60}\nStep 5: Cleaning up chunks")
            for r in results:
                chunk_dir = os.path.join(output_dir, f"chunk_{r['chunk_idx']:03d}")
                if os.path.exists(chunk_dir):
                    shutil.rmtree(chunk_dir, ignore_errors=True)
                tmp_dock = os.path.join(output_dir, f"tmp_dock_{r['chunk_idx']:03d}")
                if os.path.exists(tmp_dock):
                    shutil.rmtree(tmp_dock, ignore_errors=True)

        print(f"\n{'='*60}")
        print(f"DONE: {n_eval}/{n_total} molecules evaluated in {elapsed_total:.0f}s")
        print(f"  Results: {merged_xlsx}")
        print(f"  SDFs: {sdf_dir}/")
        print(f"  Parallel speedup: ~{args.num_workers}x")

    finally:
        # Clean up temp chunk .pt files
        if not args.keep_chunks:
            for f in Path(tmp_dir).glob("chunk_*.pt"):
                f.unlink()
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass


if __name__ == '__main__':
    main()
