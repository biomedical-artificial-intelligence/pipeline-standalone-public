import argparse
import asyncio
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

from services.bandit.service import BanditStreamService, JobMeta
from services.bandit.utils import generate_random_sequence
from services.dock.service import DockService
from services.fold.service import FoldService
from utils.job_utils import JobFileUtils
import db

from config import OUTPUT_BASE_DIR

# ZDOCK concurrency = (GPU workers) x DOCK_WORKERS_PER_GPU.
# AF feeds one fold per GPU, so ZDOCK parallelism scales with the GPU count.
DOCK_WORKERS_PER_GPU = 10


async def run_pipeline(
    pdb_name: str,
    target_residue: str,
    fixed_chain_sequence: str,
    prefix: str,
    suffix: str,
    initial_cdr: str,
    cdr_chain_id: str,
    gpu_ids: list[int],
    *,
    mode: str = "antibody",
    max_iter: int,
    alpha: float = 0.1,
    dock_workers_per_gpu: int = DOCK_WORKERS_PER_GPU,
    warmup_size: int | None = None,
    early_stop: bool = True,
    warmup_steps: int | None = None,
    es_window_size: int = 100,
    es_slope_eps: float = 0.01,
    es_patience_windows: int = 5,
    distance_processes: int = 2,
) -> dict:
    n_workers = len(gpu_ids)
    n_dock_workers = n_workers * dock_workers_per_gpu

    is_peptide = (mode == "peptide")
    cdr_length = len(initial_cdr)            # peptide: 펩타이드 길이 / antibody: CDR 길이
    if is_peptide:
        # 펩타이드는 CDR 범위 없음 → filtering이 distance(chain P, 1..N) 경로로 분기.
        cdr_start = None
        cdr_end = None
        dist_thr = 15.0
    else:
        cdr_start = len(prefix) + 1
        cdr_end = len(prefix) + cdr_length
        dist_thr = 50.0

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    await db.init_db()
    pipeline_id = await db.create_pipeline(
        pdb_name=pdb_name,
        length=cdr_length,
        max_iter=max_iter,
        alpha=alpha,
        batch_size=n_dock_workers,
    )

    output_dir = os.path.join(OUTPUT_BASE_DIR, str(pipeline_id))
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "pipeline.log")),
            logging.StreamHandler(),
        ]
    )

    logger.info(
        f"{mode} pipeline start - ID: {pipeline_id}, "
        f"length: {cdr_length}, cdr_pos={cdr_start}:{cdr_end}, cdr_chain={cdr_chain_id}, "
        f"seed={initial_cdr}, dist_max={dist_thr}, Max iter: {max_iter}, GPUs: {gpu_ids}, "
        f"ZDOCK workers: {n_dock_workers}"
    )

    fold_service = FoldService()
    # Dedicated thread pool for ZDOCK offload, sized to the semaphore so the
    # default ThreadPoolExecutor(min(32, cpu+4)) does not silently cap parallelism.
    dock_executor = ThreadPoolExecutor(max_workers=n_dock_workers)
    dock_service = DockService(executor=dock_executor, n_processes=distance_processes)
    # The ZDOCK binaries run with cwd=DOCK_DIR, so a relative path would resolve
    # against DOCK_DIR instead of the caller's directory.
    receptor_path = os.path.abspath(pdb_name)
    if not os.path.isfile(receptor_path):
        raise FileNotFoundError(f"--pdb_name not found: {pdb_name} (resolved to {receptor_path})")

    bandit = BanditStreamService(
        length=cdr_length,
        seed_seq=initial_cdr,
        n_gpu_workers=n_workers,
        n_dock_workers=n_dock_workers,
        early_stop=early_stop,
        warmup_steps=warmup_steps,
        es_window_size=es_window_size,
        es_slope_eps=es_slope_eps,
        es_patience_windows=es_patience_windows,
    )

    seed_job = bandit.register_seed()
    n_warmup = warmup_size if warmup_size is not None else 3 * cdr_length
    warmup_seqs = [generate_random_sequence(cdr_length) for _ in range(n_warmup)]
    warmup_jobs = bandit.register_initial_seqs(warmup_seqs)
    warmup_jobs_all = [seed_job] + warmup_jobs

    JobFileUtils.save_warmup_sequences(
        output_dir=output_dir,
        seed_seq=initial_cdr,
        initial_seqs=warmup_seqs,
    )

    logger.info(
        f"Warm-up {len(warmup_jobs_all)} seqs (seed 1 + initial {len(warmup_jobs)}), "
        f"AF workers {n_workers}, ZDOCK workers {n_dock_workers}"
    )

    af_queue: asyncio.Queue[JobMeta | None] = asyncio.Queue()
    result_queue: asyncio.Queue[tuple] = asyncio.Queue()
    dock_sem = asyncio.Semaphore(n_dock_workers)
    dock_tasks: set[asyncio.Task] = set()
    stop_event = asyncio.Event()

    jobs_submitted = len(warmup_jobs_all)
    jobs_completed = 0
    early_stopped = False
    dock_stats = {"ok": 0, "fail": 0}

    for job in warmup_jobs_all:
        await af_queue.put(job)

    # ----------------------------------------------------------------
    # AF Worker
    # ----------------------------------------------------------------
    async def af_worker(worker_gpu_id: int):
        while not stop_event.is_set():
            try:
                job = await asyncio.wait_for(af_queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                continue

            if job is None:
                break

            job_dir = os.path.join(output_dir, f"tmp_{str(job.job_id).zfill(5)}")
            os.makedirs(job_dir, exist_ok=True)

            try:
                if is_peptide:
                    await fold_service.run_single_fold_peptide(
                        job_id=job.job_id,
                        sequence=job.seq,
                        pipeline_id=pipeline_id,
                        iter_dir=job_dir,
                        gpu_id=worker_gpu_id,
                    )
                else:
                    await fold_service.run_single_fold(
                        job_id=job.job_id,
                        fixed_chain_sequence=fixed_chain_sequence,
                        modified_chain_sequence=prefix + job.seq + suffix,
                        cdr_chain_id=cdr_chain_id,
                        pipeline_id=pipeline_id,
                        iter_dir=job_dir,
                        gpu_id=worker_gpu_id,
                    )
                # Keep a strong reference to the fire-and-forget task (GC could
                # otherwise cancel it mid-run); drop it on completion.
                t = asyncio.create_task(dock_task(job, job_dir))
                dock_tasks.add(t)
                t.add_done_callback(dock_tasks.discard)
            except Exception as e:
                logger.error(f"AF failed - job {job.job_id}, gpu {worker_gpu_id}: {e}")
                await result_queue.put((job, job_dir, 0.1, None, None))

    # ----------------------------------------------------------------
    # ZDOCK Task
    # ----------------------------------------------------------------
    async def dock_task(job: JobMeta, job_dir: str):
        async with dock_sem:
            try:
                result = await dock_service.run_single_dock(
                    job_id=job.job_id,
                    iter_dir=job_dir,
                    receptor_pdb=receptor_path,
                    target_residue=target_residue,
                    pipeline_id=pipeline_id,
                    cdr_start_position=cdr_start,
                    cdr_end_position=cdr_end,
                    cdr_chain_id=cdr_chain_id,
                    peptide_length=cdr_length,
                    distance_threshold=dist_thr,
                )
                score = result.get("score", 0.1)
                ranking = result.get("ranking", None)
                dock_stats["ok"] += 1
            except Exception as e:
                logger.error(f"ZDOCK failed - job {job.job_id}: {e}")
                score = 0.1
                ranking = None
                dock_stats["fail"] += 1

            await result_queue.put((job, job_dir, score, ranking, None))

    # ----------------------------------------------------------------
    # Updater
    # ----------------------------------------------------------------
    async def updater():
        nonlocal jobs_completed, early_stopped, jobs_submitted
        seq_to_iter = {}
        total_af_jobs = 0

        while True:
            if bandit.iter_count >= max_iter:
                break
            if early_stopped and jobs_completed >= jobs_submitted:
                break

            try:
                job, job_dir, score, ranking, source_iter = await asyncio.wait_for(
                    result_queue.get(), timeout=2.0
                )
            except asyncio.TimeoutError:
                if early_stopped and jobs_completed >= jobs_submitted:
                    break
                continue

            jobs_completed += 1
            next_jobs = bandit.on_result(job, score)

            # ── Warm-up: skip DB, drop temp dir ──────────────────────────
            if job.is_warmup:
                if job_dir and os.path.exists(job_dir):
                    shutil.rmtree(job_dir, ignore_errors=True)

                for next_job in next_jobs:
                    await af_queue.put(next_job)
                    jobs_submitted += 1
                    total_af_jobs += 1

                continue

            # ── Search phase ─────────────────────────────────────────────
            current_iter = bandit.iter_count
            iter_dir = os.path.join(output_dir, str(current_iter).zfill(5))

            if source_iter is not None:
                source_dir = os.path.join(output_dir, str(source_iter).zfill(5))
                if os.path.exists(source_dir) and not os.path.exists(iter_dir):
                    os.symlink(source_dir, iter_dir)
            elif job_dir and os.path.exists(job_dir):
                os.rename(job_dir, iter_dir)

            await db.create_result(pipeline_id, current_iter, job.seq)
            await db.update_score(pipeline_id, current_iter, score, ranking)

            seq_to_iter[job.seq] = current_iter
            parent_iter = seq_to_iter.get(job.parent_seq) if job.parent_seq else None

            JobFileUtils.save_mutation_history(
                output_dir=output_dir,
                iter=current_iter,
                sequence=job.seq,
                score=score,
                parent_seq=job.parent_seq,
                parent_iter=parent_iter,
            )

            if bandit.is_early_stopped and not early_stopped:
                early_stopped = True
                logger.info(
                    f"Early stop triggered at iter {bandit.iter_count} - "
                    f"draining {jobs_submitted - jobs_completed} remaining jobs"
                )

            if not early_stopped and total_af_jobs < max_iter:
                for next_job in next_jobs:
                    cached = await db.get_score_by_sequence(pipeline_id, next_job.seq)
                    if cached:
                        logger.info(
                            f"Cache hit - seq: {next_job.seq}, score: {cached['score']:.4f}"
                        )
                        await result_queue.put(
                            (next_job, None, cached["score"], cached["ranking"], cached["iter"])
                        )
                        jobs_submitted += 1
                    else:
                        await af_queue.put(next_job)
                        jobs_submitted += 1
                    total_af_jobs += 1

        stop_event.set()
        for _ in range(n_workers):
            await af_queue.put(None)

    # ----------------------------------------------------------------
    # Run
    # ----------------------------------------------------------------
    try:
        await asyncio.gather(
            *[af_worker(gpu_id) for gpu_id in gpu_ids],
            updater(),
        )

        # In-flight zdock tasks may still be running after the updater exits;
        # wait for them before cleaning tmp dirs to avoid an rmtree race.
        if dock_tasks:
            await asyncio.gather(*dock_tasks, return_exceptions=True)

        for item in os.listdir(output_dir):
            if item.startswith("tmp_"):
                shutil.rmtree(os.path.join(output_dir, item), ignore_errors=True)

        await db.finish_pipeline(pipeline_id)

        if dock_stats["ok"] == 0 and dock_stats["fail"]:
            raise RuntimeError(
                f"every docking attempt failed ({dock_stats['fail']}); "
                "scores are meaningless. Check DOCK_DIR and the log above."
            )

        stop_mode = "EARLY_STOP" if early_stopped else "MAX_ITER"
        logger.info(
            f"Pipeline complete [{stop_mode}] - "
            f"Best score: {bandit.best_score:.4f}, Best seq: {bandit.best_seq}"
        )

        return {
            "pipeline_id": pipeline_id,
            "best_score": bandit.best_score,
            "best_sequence": bandit.best_seq,
            "total_iter": bandit.iter_count,
            "stop_mode": stop_mode,
        }

    finally:
        dock_executor.shutdown(wait=False)


def main():
    parser = argparse.ArgumentParser(
        description="Antibody CDR / Peptide design pipeline (UCB Bandit + AlphaFold3 + ZDOCK)"
    )
    parser.add_argument("--mode", choices=["antibody", "peptide"], default="antibody",
                        help="Design mode (default: antibody). peptide = single-chain free peptide.")
    parser.add_argument("--pdb_name",             required=True, help="Receptor (antigen/target) PDB file path")
    parser.add_argument("--target_residue",       required=True, help="Target residue on the receptor (e.g. A:100)")
    parser.add_argument("--initial_cdr",          required=True, help="Seed sequence (length defines design length). antibody=CDR, peptide=peptide")
    parser.add_argument("--fixed_chain_sequence", default=None,  help="[antibody] Fixed antibody chain sequence (chain NOT being designed)")
    parser.add_argument("--prefix",               default="",    help="[antibody] Framework before the CDR on the designed chain")
    parser.add_argument("--suffix",               default="",    help="[antibody] Framework after the CDR on the designed chain")
    parser.add_argument("--selected_chain_type",  default=None,  help="[antibody] Designed chain type: 'Light' or 'Heavy'")
    parser.add_argument("--gpu_ids",              required=True, help="GPU IDs comma-separated (e.g. 0,1,2)")
    parser.add_argument("--max_iter",             type=int, required=True, help="Max search iterations")
    parser.add_argument("--alpha",                type=float, default=0.1, help="UCB alpha (default: 0.1)")
    parser.add_argument("--dock_workers_per_gpu", type=int, default=DOCK_WORKERS_PER_GPU,
                        help=f"ZDOCK workers per GPU (default: {DOCK_WORKERS_PER_GPU})")
    parser.add_argument("--warmup_size", type=int, default=None,
                        help="Number of random warm-up sequences (default: 3 x CDR length). "
                             "Set small (e.g. 2) for a quick end-to-end smoke test.")
    # ── Search / bandit tuning ───────────────────────────────────────────
    # NOTE: UCB random→guided 전환 시점은 이제 내부 고정값 random_phase_steps=10*N.
    parser.add_argument("--early_stop", dest="early_stop", action="store_true", default=True,
                        help="Enable early stopping on score plateau (default: on).")
    parser.add_argument("--no_early_stop", dest="early_stop", action="store_false",
                        help="Disable early stopping (run the full --max_iter).")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Search steps excluded from early-stop detection "
                             "(initial random-exploration region; default: 10 x CDR length).")
    parser.add_argument("--es_window_size", type=int, default=100,
                        help="Early-stop score-averaging window size (default: 100).")
    parser.add_argument("--es_slope_eps", type=float, default=0.01,
                        help="Early-stop plateau slope threshold (default: 0.01). Larger = stop sooner.")
    parser.add_argument("--es_patience_windows", type=int, default=5,
                        help="Early-stop patience in windows (default: 5). Smaller = stop sooner.")
    # ── Docking parallelism ──────────────────────────────────────────────
    parser.add_argument("--distance_processes", type=int, default=2,
                        help="Worker processes for the distance/pose-ranking pool (default: 2).")

    args = parser.parse_args()
    gpu_id_list = [int(g.strip()) for g in args.gpu_ids.split(",")]

    if args.mode == "antibody":
        if not args.fixed_chain_sequence or not args.selected_chain_type:
            parser.error("--fixed_chain_sequence 와 --selected_chain_type 는 antibody 모드에서 필수입니다.")
        cdr_chain_id = "L" if "Light" in args.selected_chain_type else "H"
        fixed_chain_sequence = args.fixed_chain_sequence
    else:  # peptide: 단일 체인, fixed chain/prefix/suffix 미사용. ligand 체인은 내부적으로 "P".
        cdr_chain_id = "P"
        fixed_chain_sequence = ""

    result = asyncio.run(run_pipeline(
        pdb_name=args.pdb_name,
        target_residue=args.target_residue,
        fixed_chain_sequence=fixed_chain_sequence,
        prefix=args.prefix,
        suffix=args.suffix,
        initial_cdr=args.initial_cdr,
        cdr_chain_id=cdr_chain_id,
        gpu_ids=gpu_id_list,
        mode=args.mode,
        max_iter=args.max_iter,
        alpha=args.alpha,
        dock_workers_per_gpu=args.dock_workers_per_gpu,
        warmup_size=args.warmup_size,
        early_stop=args.early_stop,
        warmup_steps=args.warmup_steps,
        es_window_size=args.es_window_size,
        es_slope_eps=args.es_slope_eps,
        es_patience_windows=args.es_patience_windows,
        distance_processes=args.distance_processes,
    ))

    print("\n=== Pipeline Result ===")
    print(f"Pipeline ID  : {result['pipeline_id']}")
    print(f"Output dir   : {os.path.join(OUTPUT_BASE_DIR, str(result['pipeline_id']))}")
    print(f"Stop mode    : {result['stop_mode']}")
    print(f"Total iter   : {result['total_iter']}")
    print(f"Best score   : {result['best_score']:.4f}")
    print(f"Best seq     : {result['best_sequence']}")


if __name__ == "__main__":
    main()
