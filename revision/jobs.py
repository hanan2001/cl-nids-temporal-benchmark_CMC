"""
Job list of the revision experiments and a small parallel runner.

Each job is one (experiment, method, seed) run of run_paper_v2.py writing its
own JSON file results/rev/<experiment>/<method>_s<seed>.json, so jobs can run
in parallel and an interrupted queue can simply be restarted (finished jobs are
skipped).

Usage:
  python revision/jobs.py --list
  python revision/jobs.py --workers 4 --threads 2 [--only main sev_cap ...]
"""
import os, sys, json, time, argparse, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S5 = [42, 123, 456, 789, 1011]
S_EXTRA = [2024, 3141, 5555, 7777, 9999]
S3 = [42, 123, 456]
S2 = [42, 123]
MAIN_SUBMITTED = ['CB_BETA=0.999', 'ES_CRITERION=plain']   # configuration of the submitted Table 4

# experiment -> (protocol args, --set overrides, [(method, seeds)])
EXPERIMENTS = {
    # 0. reproducibility of the submitted main experiment (file order, submitted config)
    'verify':     (['--order', 'file'], MAIN_SUBMITTED, [('ReplayUniform', [42])]),
    # 1. revised main protocol: timestamp order, class-stratified chronological split
    'main':       (['--order', 'time'], [], [('Majority', [42]), ('ReplayUniform', S5), ('ReplayRisk', S5),
                                             ('FineTuning', S5), ('EWC', S5), ('Joint', S5)]),
    'main_extra': (['--order', 'time'], [], [('ReplayUniform', S_EXTRA), ('ReplayRisk', S_EXTRA)]),
    # 2. severity decomposition and capacity controls (revised protocol)
    'sev_cap':    (['--order', 'time'], ['RISK_ON_SAMPLING=False', 'RISK_ON_LOSS=False'], [('ReplayRisk', S3)]),
    'sev_samp':   (['--order', 'time'], ['RISK_ON_CAPACITY=False', 'RISK_ON_LOSS=False'], [('ReplayRisk', S3)]),
    'sev_loss':   (['--order', 'time'], ['RISK_ON_CAPACITY=False', 'RISK_ON_SAMPLING=False'], [('ReplayRisk', S3)]),
    'sev_budget': (['--order', 'time'], ['RISK_BUDGET_MATCHED=True'], [('ReplayRisk', S5)]),
    'sev_boost':  (['--order', 'time'], ['AIM_MODE=uniform', 'RISK_ON_SAMPLING=False', 'RISK_ON_LOSS=False'],
                   [('ReplayRisk', S5)]),
    # 3. matched reference of Table 7 (submitted file-order protocol, supplementary config)
    'ref_file':   (['--order', 'file'], [], [('ReplayUniform', S3), ('ReplayRisk', S3)]),
    # 4. duplicate-aware evaluation
    'dedup':      (['--order', 'time', '--dedup', 'eval'], [], [('ReplayUniform', S3), ('ReplayRisk', S3), ('Joint', S2)]),
    # 5. strictly chronological shared cutoffs
    'timecut':    (['--order', 'time', '--split', 'time_cut'], [],
                   [('Majority', [42]), ('ReplayUniform', S3), ('ReplayRisk', S3), ('Joint', S2)]),
    # 6. second dataset
    'c2018':      (['--dataset', 'cic2018'], [], [('Majority', [42]), ('ReplayUniform', S3), ('ReplayRisk', S3),
                                                   ('FineTuning', S3), ('EWC', S3), ('Joint', S3)]),
    # 7. retention-aware checkpoint selection
    'retsel':     (['--order', 'time'], ['SELECT_CRITERION=retention'],
                   [('EWC', S3), ('ReplayUniform', S3)]),
    # 8. P1 decomposed (file order, supplementary config, matched with Table 7)
    'p1a':        (['--order', 'file'], ['PITFALL_PERSISTENT_ADAM=True'], [('ReplayUniform', S3)]),
    'p1b':        (['--order', 'file'], ['PITFALL_PERSISTENT_SCHED=True'], [('ReplayUniform', S3)]),
    # 9. EWC diagnostics
    'ewcdiag_ft':   (['--order', 'time'], ['EWC_DIAG=True'], [('FineTuning', [42])]),
    'ewcdiag_base': (['--order', 'time'], ['EWC_DIAG=True'], [('EWC', [42])]),
    'ewcdiag_e2':   (['--order', 'time'], ['EWC_DIAG=True', 'EWC_FISHER_MODE=sampled_example',
                                           'EWC_FISHER_BALANCED=True', 'EWC_LAMBDA=100'], [('EWC', S3)]),
    'ewcdiag_e3':   (['--order', 'time'], ['EWC_DIAG=True', 'EWC_FISHER_MODE=sampled_example',
                                           'EWC_FISHER_BALANCED=True', 'EWC_LAMBDA=1000'], [('EWC', S2)]),
    'ewcdiag_e4':   (['--order', 'time'], ['EWC_DIAG=True', 'EWC_FISHER_MODE=sampled_example',
                                           'EWC_FISHER_BALANCED=True', 'EWC_LAMBDA=10000'], [('EWC', [42])]),
    # 10. offline reference with a longer optimisation budget
    'jointopt':   (['--order', 'time'], ['LR=0.001', 'PATIENCE=20'], [('Joint', S2)]),
}
ORDER = ['verify', 'main', 'sev_budget', 'sev_boost', 'c2018', 'sev_cap', 'sev_samp', 'sev_loss',
         'ref_file', 'dedup', 'timecut', 'main_extra', 'retsel', 'p1a', 'p1b',
         'ewcdiag_ft', 'ewcdiag_base', 'ewcdiag_e3', 'ewcdiag_e4', 'jointopt']


def jobs(only=None):
    out = []
    exps = [e for e in ORDER if not only or e in only]
    # interleave seeds: first seed of every method, then the second, ...
    for e in exps:
        proto, sets, ms = EXPERIMENTS[e]
        n = max(len(s) for _, s in ms)
        for i in range(n):
            for m, seeds in ms:
                if i < len(seeds):
                    s = seeds[i]
                    out.append({'id': f'{e}/{m}_s{s}', 'exp': e, 'method': m, 'seed': s,
                                'out': f'results/rev/{e}/{m}_s{s}.json',
                                'args': proto + ['--methods', m, '--seeds', str(s)]
                                        + (['--set'] + sets if sets else [])})
    return out


def _alive(pid):
    try:
        out = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'], capture_output=True,
                             text=True, encoding='latin-1', errors='replace').stdout
        return f'"{pid}"' in out and 'python' in out.lower()   # guard against PID reuse
    except Exception:
        return False


def lock_path(job):
    return os.path.join(ROOT, 'results/rev/locks', job['id'].replace('/', '__') + '.pid')


def locked(job):
    p = lock_path(job)
    if not os.path.exists(p):
        return False
    try:
        return _alive(int(open(p).read().strip()))
    except Exception:
        return False


def done(job):
    p = os.path.join(ROOT, job['out'])
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            return any(r['method'] == job['method'] and r['seed'] == job['seed'] for r in json.load(f)['runs'])
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    todo = [j for j in jobs(a.only) if not done(j) and not locked(j)]
    os.makedirs(os.path.join(ROOT, 'results/rev/locks'), exist_ok=True)
    if a.list:
        for j in todo:
            print(j['id'], ' '.join(j['args']))
        print(len(todo), 'jobs to run')
        return
    os.makedirs(os.path.join(ROOT, 'results/rev/logs'), exist_ok=True)
    running = []
    def external():   # jobs started by an earlier runner that are still alive
        ids = {r[0]['id'] for r in running}
        return [j for j in jobs(a.only) if j['id'] not in ids and locked(j)]
    while todo or running or external():
        if os.path.exists(os.path.join(ROOT, 'results/rev/STOP')):
            todo = []                         # finish running jobs, start no new one
        while todo and len(running) + len(external()) < a.workers:
            j = todo.pop(0)
            # the reproducibility check keeps PyTorch's default thread count
            thr = [] if j['exp'] == 'verify' else ['--threads', str(a.threads)]
            cmd = [sys.executable, '-u', 'run_paper_v2.py', '--quiet', '--out', j['out']] + thr + j['args']
            log = open(os.path.join(ROOT, 'results/rev/logs', j['id'].replace('/', '__') + '.log'), 'w')
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
            with open(lock_path(j), 'w') as fh:
                fh.write(str(p.pid))
            running.append((j, p, log, time.time()))
            print(time.strftime('%H:%M:%S'), 'START', j['id'], flush=True)
        time.sleep(60)
        for item in running[:]:
            j, p, log, t0 = item
            if p.poll() is not None:
                log.close(); running.remove(item)
                print(time.strftime('%H:%M:%S'), 'DONE ' if p.returncode == 0 else f'FAIL({p.returncode})',
                      j['id'], f'{(time.time() - t0) / 60:.0f} min', flush=True)


if __name__ == '__main__':
    main()
