"""
Aggregates the revision runs (results/rev/<experiment>/*.json) and the submitted
runs (results/*.json) into results/rev/summary.json, used by the manuscript
tables and figures. Prints every table.

Statistics: paired two-sided t-tests across seeds with Holm correction for the
planned comparisons; 95 % CI of the mean paired difference; equivalence by two
one-sided tests (TOST) with margin EQ_MARGIN (90 % CI inside +/- margin).
"""
import os, sys, json, glob
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
EQ_MARGIN = 0.02
METHODS = ['Majority', 'FineTuning', 'EWC', 'ReplayUniform', 'ReplayRisk', 'Joint']
KEYS = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted', 'mcc',
        'detection_rate', 'false_positive_rate']
S5 = [42, 123, 456, 789, 1011]


def load_exp(exp):
    runs = []
    for f in sorted(glob.glob(f'results/rev/{exp}/*.json')):
        with open(f) as fh:
            d = json.load(fh)
        for r in d['runs']:
            r['_counts'] = d.get('counts'); r['_info'] = d.get('protocol_info')
            runs.append(r)
    return runs


def load_old(fn):
    with open(fn) as f:
        return json.load(f)['runs']


def sel(runs, method, seeds=None):
    return {r['seed']: r for r in runs if r['method'] == method and (seeds is None or r['seed'] in seeds)}


def get(r, key):
    m = r['metrics']
    if key == 'forgetting':
        return r.get('forgetting_macro_recall')
    if key.startswith('fixed.'):
        return (m.get('at_fixed_fpr') or {}).get(key[6:])
    return m.get(key)


def msd(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 'n': len(v)}


def paired(ra, rb, key='mcc'):
    seeds = sorted(set(ra) & set(rb))
    a = np.array([get(ra[s], key) for s in seeds], float)
    b = np.array([get(rb[s], key) for s in seeds], float)
    d = a - b
    n = len(d)
    if n < 2:
        return None
    md, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    if sd == 0:
        return {'n': n, 'diff': float(md), 'ci95': [float(md), float(md)], 'p': 1.0 if md == 0 else 0.0}
    t95, t90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    p = float(stats.ttest_rel(a, b).pvalue)
    # TOST: H0 |delta| >= margin
    p_low = 1 - stats.t.cdf((md + EQ_MARGIN) / se, n - 1)
    p_up = stats.t.cdf((md - EQ_MARGIN) / se, n - 1)
    return {'n': n, 'diff': float(md), 'sd_diff': float(sd), 'ci95': [float(md - t95 * se), float(md + t95 * se)],
            'ci90': [float(md - t90 * se), float(md + t90 * se)], 'p': p,
            'p_tost': float(max(p_low, p_up)), 'margin': EQ_MARGIN,
            'equivalent': bool(max(p_low, p_up) < 0.05)}


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj.tolist()


def table(runs, methods=METHODS, seeds=None, keys=KEYS + ['forgetting', 'sw_detection_rate',
                                                           'sw_missed_attack_rate', 'fixed.detection_rate',
                                                           'fixed.sw_detection_rate', 'fixed.test_fpr']):
    out = {}
    for m in methods:
        rr = sel(runs, m, seeds)
        if rr:
            out[m] = {k: msd([get(r, k) for r in rr.values()]) for k in keys}
            out[m]['seeds'] = sorted(rr)
    return out


def per_class(runs, methods, seeds=None):
    out = {}
    for m in methods:
        rr = sel(runs, m, seeds)
        for r in rr.values():
            for c, v in r['metrics']['per_class'].items():
                out.setdefault(c, {'support': v['support']}).setdefault(m, []).append(v['recall'])
    return {c: {k: (float(np.mean(v)) if isinstance(v, list) else v) for k, v in d.items()} for c, d in out.items()}


def retention(runs, method, seeds=None):
    Rs = [r['R_macro_recall'] for r in sel(runs, method, seeds).values() if r.get('R_macro_recall')]
    if not Rs:
        return None
    T = len(Rs[0])
    return [[float(np.mean([R[t][i] for R in Rs])) for i in range(t + 1)] for t in range(T)]


def fmt(x, d=3):
    if x is None:
        return '-'
    return f"{x['mean']:.{d}f}+/-{x['sd']:.{d}f}" if isinstance(x, dict) else f"{x:.{d}f}"


def show_table(name, t, keys=('mcc', 'balanced_accuracy', 'f1_macro', 'detection_rate',
                              'false_positive_rate', 'forgetting', 'sw_detection_rate', 'fixed.sw_detection_rate')):
    print(f"\n=== {name}")
    print(f"{'method':<15}" + ''.join(f"{k[-14:]:>18}" for k in keys) + '  seeds')
    for m, v in t.items():
        print(f"{m:<15}" + ''.join(f"{fmt(v.get(k)):>18}" for k in keys) + f"  {len(v['seeds'])}")


def show_paired(name, res):
    print(f"\n--- {name}")
    for k, r in res.items():
        if r is None:
            print(f"   {k}: n/a"); continue
        print(f"   {k:<40} n={r['n']:<3} diff={r['diff']:+.4f} CI95=[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] "
              f"p={r['p']:.4f}" + (f" p_holm={r['p_holm']:.4f}" if 'p_holm' in r else '')
              + (f" TOST(+/-{r['margin']}) p={r['p_tost']:.4f}" if 'p_tost' in r else ''))


S = {}

# ---------------------------------------------------------------- reproducibility
main_old = load_old('results/main.json')
ver = load_exp('verify')
if ver:
    a = sel(main_old, 'ReplayUniform')[42]; b = ver[0]
    S['verify'] = {'submitted_mcc': a['metrics']['mcc'], 'rerun_mcc': b['metrics']['mcc'],
                   'identical_R': a['R_macro_recall'] == b['R_macro_recall'],
                   'identical_per_class': all(a['metrics']['per_class'][c]['recall'] == b['metrics']['per_class'][c]['recall']
                                              for c in a['metrics']['per_class'])}
    print('\nREPRODUCIBILITY', S['verify'])

# ---------------------------------------------------------------- revised main
main = load_exp('main') + load_exp('main_extra')
S['main'] = table(main, seeds=S5)
S['main_replay10'] = table(main, ['ReplayUniform', 'ReplayRisk'])
show_table('Revised main protocol (timestamp order), 5 seeds', S['main'])
show_table('Replay pair, all seeds', S['main_replay10'])
U, R, E, F, J = (sel(main, m) for m in ('ReplayUniform', 'ReplayRisk', 'EWC', 'FineTuning', 'Joint'))
U5, E5, F5 = sel(main, 'ReplayUniform', S5), sel(main, 'EWC', S5), sel(main, 'FineTuning', S5)
planned = {'ReplayRisk - ReplayUniform': paired(R, U), 'ReplayUniform - EWC': paired(U5, E5),
           'ReplayUniform - FineTuning': paired(U5, F5), 'EWC - FineTuning': paired(E5, F5)}
ok = {k: v for k, v in planned.items() if v}
if ok:
    for (k, v), pa in zip(ok.items(), holm([v['p'] for v in ok.values()])):
        v['p_holm'] = pa
S['main_tests'] = planned
S['main_tests_5seeds_risk'] = paired(sel(main, 'ReplayRisk', S5), U5)
show_paired('Planned comparisons on MCC (revised main)', planned)
sev_out = {k: paired(R, U, k) for k in ('sw_detection_rate', 'sw_missed_attack_rate', 'fixed.sw_detection_rate',
                                        'fixed.detection_rate', 'detection_rate', 'false_positive_rate')}
S['main_severity_outcomes'] = sev_out
show_paired('Severity-sensitive outcomes, ReplayRisk - ReplayUniform', sev_out)
S['main_per_class'] = per_class(main, METHODS[1:], S5)
S['main_retention'] = {m: retention(main, m, S5) for m in ('FineTuning', 'EWC', 'ReplayUniform', 'ReplayRisk')}
S['main_counts'] = next((r['_counts'] for r in main if r.get('_counts')), None)
S['main_protocol_info'] = next((r['_info'] for r in main if r.get('_info')), None)
S['main_task_info'] = {m: [r['task_info'] for r in sel(main, m, S5).values()] for m in ('ReplayUniform', 'ReplayRisk', 'FineTuning', 'EWC')}
S['main_buffer'] = {m: [r['buffer_sizes'] for r in sel(main, m, S5).values()] for m in ('ReplayUniform', 'ReplayRisk')}
S['main_per_seed'] = {m: {s: {k: get(r, k) for k in ('mcc', 'forgetting')} for s, r in sel(main, m).items()}
                      for m in METHODS}

# resources
res = {}
for m in METHODS[1:]:
    rr = sel(main, m, S5).values()
    if rr:
        res[m] = {'memory_bytes': msd([r.get('resources', {}).get('memory_bytes') for r in rr]),
                  'total_minutes': msd([r.get('resources', {}).get('total_seconds', 0) / 60 for r in rr]),
                  'ms_per_update': msd([1000 * np.mean([t['seconds_per_update'] for t in r['task_info']]) for r in rr]),
                  'updates': msd([r.get('resources', {}).get('total_updates') or r['task_info'][0]['n_updates'] for r in rr]),
                  'buffer_flows': msd([r.get('resources', {}).get('buffer_flows') for r in rr]),
                  'n_params': next(iter(rr)).get('resources', {}).get('n_params')}
S['resources'] = res
print('\n=== Resources'); [print(f"   {m:<14} {v}") for m, v in res.items()]

# ---------------------------------------------------------------- severity decomposition
dec = {}
for exp, label in (('sev_cap', 'capacity only'), ('sev_samp', 'sampling only'), ('sev_loss', 'loss weight only'),
                   ('sev_budget', 'all three, total memory matched'),
                   ('sev_boost', 'capacity only, uniform attack scores')):
    rr = sel(load_exp(exp), 'ReplayRisk')
    if not rr:
        continue
    U_s = {s: U[s] for s in rr if s in U}
    dec[exp] = {'label': label, 'stats': {k: msd([get(r, k) for r in rr.values()]) for k in
                                          ('mcc', 'false_positive_rate', 'forgetting', 'sw_detection_rate', 'fixed.sw_detection_rate')},
                'buffer_flows': msd([sum(r['buffer_sizes'].values()) for r in rr.values()]),
                'vs_uniform_mcc': paired(rr, U_s), 'vs_uniform_swdr': paired(rr, U_s, 'sw_detection_rate'),
                'vs_uniform_fixed_swdr': paired(rr, U_s, 'fixed.sw_detection_rate'),
                'per_class': per_class(list(rr.values()), ['ReplayRisk'])}
if dec:
    ps = [v['vs_uniform_mcc']['p'] for v in dec.values() if v['vs_uniform_mcc']]
    for v, pa in zip([v for v in dec.values() if v['vs_uniform_mcc']], holm(ps)):
        v['vs_uniform_mcc']['p_holm'] = pa
S['severity_decomposition'] = dec
S['uniform_buffer_flows'] = msd([sum(r['buffer_sizes'].values()) for r in sel(main, 'ReplayUniform', S5).values()])
S['risk_buffer_flows'] = msd([sum(r['buffer_sizes'].values()) for r in sel(main, 'ReplayRisk', S5).values()])
show_paired('Severity decomposition vs uniform replay (MCC)', {v['label']: v['vs_uniform_mcc'] for v in dec.values()})
show_paired('Severity decomposition vs uniform replay (SW-DR)', {v['label']: v['vs_uniform_swdr'] for v in dec.values()})

# ---------------------------------------------------------------- Table 7 (file order, supplementary config)
ref = load_exp('ref_file')
t7 = {'reference': table(ref, ['ReplayUniform', 'ReplayRisk'])}
for name, fn, m in (('aim_uniform', 'abl_aim_uniform', 'ReplayRisk'), ('aim_inverted', 'abl_aim_inverted', 'ReplayRisk'),
                    ('aim_permuted', 'abl_aim_permuted', 'ReplayRisk'), ('lam025', 'abl_lam025', None),
                    ('lam1', 'abl_lam1', None), ('q100', 'abl_q100', None), ('P1', 'pit_opt', 'ReplayUniform'),
                    ('P2', 'pit_cb', 'ReplayUniform'), ('P3', 'pit_replay', 'ReplayUniform'), ('P4', 'pit_split', 'ReplayUniform')):
    runs = load_old(f'results/{fn}.json')
    t7[name] = {mm: {'stats': table(runs, [mm], keys=('mcc', 'false_positive_rate', 'forgetting'))[mm],
                     'vs_ref': paired(sel(runs, mm), sel(ref, mm)),
                     'vs_ref_uniform': paired(sel(runs, mm), sel(ref, 'ReplayUniform'))}
                for mm in ([m] if m else ['ReplayUniform', 'ReplayRisk']) if sel(runs, mm)}
for name in ('p1a', 'p1b'):
    runs = load_exp(name)
    if runs:
        t7[name] = {'ReplayUniform': {'stats': table(runs, ['ReplayUniform'], keys=('mcc', 'false_positive_rate', 'forgetting'))['ReplayUniform'],
                                      'vs_ref': paired(sel(runs, 'ReplayUniform'), sel(ref, 'ReplayUniform'))}}
if ref:
    t7['reference_risk_vs_uniform'] = paired(sel(ref, 'ReplayRisk'), sel(ref, 'ReplayUniform'))
S['table7'] = t7
print('\n=== Table 7 (file-order protocol, supplementary configuration)')
for k, v in t7.items():
    if k == 'reference':
        for m, st in v.items():
            print(f"   reference {m:<14} MCC {fmt(st['mcc'])}  FPR {fmt(st['false_positive_rate'])}  F {fmt(st['forgetting'])}")
    elif k == 'reference_risk_vs_uniform':
        print('   reference risk-uniform', v and f"{v['diff']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] p={v['p']:.3f}")
    else:
        for m, st in v.items():
            d = st.get('vs_ref')
            print(f"   {k:<13} {m:<14} MCC {fmt(st['stats']['mcc'])}  vs ref: " +
                  (f"{d['diff']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] p={d['p']:.3f}" if d else '-'))

# ---------------------------------------------------------------- dedup / timecut / retsel / jointopt
for exp, methods in (('dedup', ['ReplayUniform', 'ReplayRisk', 'Joint']),
                     ('timecut', ['Majority', 'ReplayUniform', 'ReplayRisk', 'Joint']),
                     ('retsel', ['EWC', 'ReplayUniform', 'ReplayRisk']),
                     ('jointopt', ['Joint'])):
    runs = load_exp(exp)
    if not runs:
        continue
    S[exp] = {'table': table(runs, methods), 'per_class': per_class(runs, methods),
              'counts': next((r['_counts'] for r in runs if r.get('_counts')), None),
              'info': next((r['_info'] for r in runs if r.get('_info')), None),
              'vs_main': {m: paired(sel(runs, m), sel(main, m)) for m in methods if m != 'Majority'},
              'risk_vs_uniform': paired(sel(runs, 'ReplayRisk'), sel(runs, 'ReplayUniform'))}
    if exp == 'retsel':
        S[exp]['retention'] = {m: retention(runs, m) for m in methods}
    show_table(exp, S[exp]['table'])
    show_paired(f'{exp}: vs revised main (MCC)', S[exp]['vs_main'])
    show_paired(f'{exp}: risk - uniform', {'MCC': S[exp]['risk_vs_uniform']})

# ---------------------------------------------------------------- CSE-CIC-IDS2018
c18 = load_exp('c2018')
if c18:
    S['c2018'] = {'table': table(c18), 'per_class': per_class(c18, METHODS[1:]),
                  'counts': next((r['_counts'] for r in c18 if r.get('_counts')), None),
                  'retention': {m: retention(c18, m) for m in ('FineTuning', 'EWC', 'ReplayUniform', 'ReplayRisk')}}
    U18, R18, E18, F18 = (sel(c18, m) for m in ('ReplayUniform', 'ReplayRisk', 'EWC', 'FineTuning'))
    t = {'ReplayRisk - ReplayUniform': paired(R18, U18), 'ReplayUniform - EWC': paired(U18, E18),
         'ReplayUniform - FineTuning': paired(U18, F18), 'EWC - FineTuning': paired(E18, F18)}
    ok = {k: v for k, v in t.items() if v}
    for (k, v), pa in zip(ok.items(), holm([v['p'] for v in ok.values()])):
        v['p_holm'] = pa
    S['c2018']['tests'] = t
    S['c2018']['severity_outcomes'] = {k: paired(R18, U18, k) for k in ('sw_detection_rate', 'fixed.sw_detection_rate')}
    show_table('CSE-CIC-IDS2018', S['c2018']['table'])
    show_paired('CSE-CIC-IDS2018 planned comparisons', t)

# ---------------------------------------------------------------- EWC diagnostics
diag = {}
for exp in ('ewcdiag_ft', 'ewcdiag_base', 'ewcdiag_e2', 'ewcdiag_e3', 'ewcdiag_e4'):
    runs = load_exp(exp)
    if not runs:
        continue
    rows = []
    for r in runs:
        for t, ti in enumerate(r['task_info']):
            if 'diag' in ti:
                rows.append({'seed': r['seed'], 'task': t + 1, **ti['diag'], 'train_macro_recall': ti['train_macro_recall'],
                             'pen_ratio_last': (ti.get('ewc_penalty_over_loss_by_epoch') or [None])[-1]})
    agg = {k: float(np.nanmean([x[k] for x in rows if x[k] is not None])) for k in
           ('drift_top1pct_fisher', 'drift_bottom50pct_fisher', 'drift_all', 'fisher_weighted_drift', 'train_macro_recall')} if rows else {}
    if rows:
        agg['ratio_top_over_bottom'] = float(np.nanmean([x['drift_top1pct_fisher'] / max(x['drift_bottom50pct_fisher'], 1e-30) for x in rows]))
        pr = [x['pen_ratio_last'] for x in rows if x['pen_ratio_last'] is not None]
        agg['penalty_over_loss'] = float(np.mean(pr)) if pr else None
    diag[exp] = {'stats': table(runs, [runs[0]['method']], keys=('mcc', 'balanced_accuracy', 'forgetting', 'detection_rate'))[runs[0]['method']],
                 'diag': agg, 'rows': rows,
                 'new_task_learned': float(np.mean([ti['train_macro_recall'] >= 0.5 for r in runs for ti in r['task_info'][1:]]))}
S['ewc_diag'] = diag
print('\n=== EWC diagnostics')
for k, v in diag.items():
    print(f"   {k:<14} MCC {fmt(v['stats']['mcc'])} F {fmt(v['stats']['forgetting'])} new tasks learned {v['new_task_learned']:.2f} diag {v['diag']}")

# ---------------------------------------------------------------- other classifiers
for fn in ('results/other_classifiers.json', 'results/other_classifiers_dedup.json'):
    if os.path.exists(fn):
        d = load_old(fn)
        key = os.path.basename(fn)[:-5]
        S[key] = {'table': table(d, ['RandomForest', 'HistGradientBoosting'], keys=KEYS),
                  'per_class': per_class(d, ['RandomForest', 'HistGradientBoosting'])}
        show_table(key, S[key]['table'], keys=('mcc', 'balanced_accuracy', 'false_positive_rate'))

# ---------------------------------------------------------------- audit
if os.path.exists('results/audit_protocol.json'):
    with open('results/audit_protocol.json') as f:
        S['audit'] = json.load(f)
if os.path.exists('results/audit_protocol_time.json'):
    with open('results/audit_protocol_time.json') as f:
        S['audit_time'] = json.load(f)
    t = S['audit_time']
    S['audit_time_dup'] = {'test_dup': t['total_test_dup_raw']['test_dup'], 'test': t['total_test_dup_raw']['test'],
                           'pct': 100 * t['total_test_dup_raw']['test_dup'] / t['total_test_dup_raw']['test'],
                           'pct_pre': 100 * t['total_test_dup_pre']['test_dup'] / t['total_test_dup_pre']['test'],
                           'other_label': sum(d['dup_with_other_label'] for d in t['duplicates']
                                              if d['definition'] == 'raw' and d['partition'] == 'test')}
if S.get('dedup', {}).get('info'):
    S['dedup_removed_test'] = sum(sum((ti.get('removed_duplicates') or {}).get('test', {}).values())
                                  for ti in S['dedup']['info'])

with open('results/rev/summary.json', 'w') as f:
    json.dump(S, f, indent=1, default=float)
print('\nwritten results/rev/summary.json')
