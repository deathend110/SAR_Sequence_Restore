"""顺序启动短程作业，记录 GPU 状态并限制单作业及总体运行时间。"""

import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/sar_short_20261028'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES='0,1',
        PYTHONPATH=str(ROOT),
        OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1',
        PYTHONUNBUFFERED='1')
    jobs = [
        ('baseline_swinir', ['--phase', 'baseline', '--model', 'swinir']),
        ('baseline_hint', ['--phase', 'baseline', '--model', 'hint']),
        ('finetune_swinir', ['--phase', 'finetune', '--model', 'swinir', '--steps', '400']),
        ('finetune_hint', ['--phase', 'finetune', '--model', 'hint', '--steps', '200']),
    ]
    for model, batches in [('swinir', (3, 4, 6)), ('hint', (6, 8, 10))]:
        jobs.extend(
            (f'batch_{model}_b{batch}', ['--phase', 'batch', '--model', model, '--batch',
                                         str(batch), '--steps', '50']) for batch in batches)
    jobs.extend([
        ('warmup_hint_w-1', ['--phase', 'warmup', '--model', 'hint', '--warmup', '-1', '--steps', '400']),
        ('warmup_hint_w3000', ['--phase', 'warmup', '--model', 'hint', '--warmup', '3000', '--steps', '400']),
        ('baseline_hint_repeat', ['--phase', 'baseline', '--model', 'hint', '--label', 'baseline_hint_repeat']),
    ])
    for deterministic in (False, True):
        for warmup in (-1, 3000):
            name = f'audit_{"deterministic" if deterministic else "warmup"}_hint_w{warmup}'
            args = [
                '--phase', 'warmup', '--model', 'hint', '--warmup',
                str(warmup), '--steps', '4', '--burnin', '0', '--audit-first', '--label', name
            ]
            if deterministic:
                args.append('--deterministic')
            jobs.append((name, args))
    jobs.extend([
        ('controlled_warmup_hint_w-1', [
            '--phase', 'warmup', '--model', 'hint', '--warmup', '-1', '--steps', '400', '--deterministic', '--label',
            'controlled_warmup_hint_w-1'
        ]),
        ('controlled_warmup_hint_w3000', [
            '--phase', 'warmup', '--model', 'hint', '--warmup', '3000', '--steps', '400', '--deterministic', '--label',
            'controlled_warmup_hint_w3000'
        ]),
    ])
    ledger = []
    provenance = {
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'checkpoint_sha256': {}
    }
    for name in ('experiments/swinir_sar_single/models/net_g_493358.pth',
                 'experiments/hint_sar_single_archived_20261021_151605/models/net_g_813496.pth'):
        with (ROOT / name).open('rb') as stream:
            provenance['checkpoint_sha256'][name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    start = time.monotonic()
    with (OUT / 'gpu_telemetry.csv').open('w', encoding='utf-8') as telemetry:
        monitor = subprocess.Popen([
            'nvidia-smi', '--id=0,1',
            '--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu', '--format=csv', '-l',
            '5'
        ],
                                   stdout=telemetry,
                                   stderr=subprocess.STDOUT)
        try:
            for name, args in jobs:
                if time.monotonic() - start > 2400:
                    raise RuntimeError('总体短程实验达到 40 分钟运行上限。')
                command = [
                    sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                    str(Path(__file__).with_name('run_check.py')), *args
                ]
                print(f'START {name}', flush=True)
                tick = time.monotonic()
                with (OUT / f'{name}.log').open('w', encoding='utf-8') as log:
                    job_env = dict(env)
                    if '--deterministic' in args:
                        job_env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                    process = subprocess.Popen(
                        command, cwd=ROOT, env=job_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = process.wait(timeout=min(600, max(1, 2400 - (time.monotonic() - start))))
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        code = -124
                    except BaseException:
                        # 外部中断调度器时，一并结束属于本次实验的训练进程组。
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                        raise
                entry = dict(
                    name=name,
                    returncode=code,
                    seconds=time.monotonic() - tick,
                    command=command,
                    ended=datetime.datetime.now().astimezone().isoformat())
                ledger.append(entry)
                (OUT / 'jobs.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding='utf-8')
                if name == 'baseline_swinir' and code == 0:
                    provenance['subsets'] = {
                        split: (OUT / f'{split}_subset.txt').read_text(encoding='utf-8').split()
                        for split in ('val', 'test')
                    }
                    (OUT / 'provenance.json').write_text(
                        json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f'END {name}: returncode={code}, seconds={entry["seconds"]:.1f}', flush=True)
                if code and not name.startswith('batch_'):
                    raise RuntimeError(f'作业失败，需要检查后继续：{name}')
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)


if __name__ == '__main__':
    main()
