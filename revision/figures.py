"""
Figures of the revised manuscript, drawn from results/rev/summary.json and the
per-seed files (Figs. 2-4), plus the protocol schematic (Fig. 1).
Usage: python revision/figures.py  -> results/rev/figures/fig{1,2,3,4,5}.png
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = 'results/rev/figures'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial', 'DejaVu Sans'],
                     'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03})
COL = {'Majority': '#999999', 'FineTuning': '#E6A817', 'EWC': '#CC79A7', 'ReplayUniform': '#0072B2',
       'ReplayRisk': '#D55E00', 'Joint': '#009E73'}
LAB = {'Majority': 'Majority', 'FineTuning': 'Fine-tuning', 'EWC': 'EWC', 'ReplayUniform': 'Uniform replay',
       'ReplayRisk': 'Severity replay', 'Joint': 'Joint (offline ref.)'}
SHORT = {'Web Attack - Brute Force': 'Web: Brute Force', 'Web Attack - XSS': 'Web: XSS',
         'Web Attack - Sql Injection': 'Web: SQL Injection'}
CLASS_ORDER = ['BENIGN', 'FTP-Patator', 'SSH-Patator', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
               'DoS slowloris', 'Heartbleed', 'Web Attack - Brute Force', 'Web Attack - XSS',
               'Web Attack - Sql Injection', 'Infiltration', 'Bot', 'PortScan', 'DDoS']


def runs_of(exp):
    out = []
    for f in sorted(glob.glob(f'results/rev/{exp}/*.json')):
        out += json.load(open(f))['runs']
    return out


def fig1():
    fig = plt.figure(figsize=(6.4, 4.86))
    ax = fig.add_axes([0, 0.47, 1, 0.53]); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis('off')
    tasks = [('T1', 'Mon', 'BENIGN\nonly'), ('T2', 'Tue', 'FTP-,\nSSH-Patator'), ('T3', 'Wed', '4 DoS,\nHeartbleed'),
             ('T4', 'Thu AM', 'Web: BF,\nXSS, SQLi'), ('T5', 'Thu PM', 'Infiltration'), ('T6', 'Fri AM', 'Bot'),
             ('T7', 'Fri PM', 'PortScan'), ('T8', 'Fri PM', 'DDoS')]
    ax.text(0.5, 49, '(a)', fontsize=9, weight='bold', va='top')
    w, g = 11.2, 1.3
    for i, (t, d, c) in enumerate(tasks):
        x = 1 + i * (w + g)
        ax.add_patch(FancyBboxPatch((x, 19), w, 23, boxstyle='round,pad=0,rounding_size=1.2', fc='#F2F2F2', ec='k', lw=0.8))
        ax.text(x + w / 2, 39.5, t, ha='center', va='center', weight='bold', fontsize=9)
        ax.text(x + w / 2, 35.5, d, ha='center', va='center', fontsize=8.5)
        ax.text(x + w / 2, 28, c, ha='center', va='center', fontsize=8.5)
        for (a, b, col) in ((0, 0.6, '#0072B2'), (0.6, 0.8, '#E69F00'), (0.8, 1.0, '#009E73')):
            ax.add_patch(Rectangle((x + 0.8 + a * (w - 1.6), 20.3), (b - a) * (w - 1.6), 2.2, fc=col, ec='none'))
        if i < 7:
            ax.annotate('', xy=(x + w + g, 30.5), xytext=(x + w, 30.5), arrowprops=dict(arrowstyle='-|>', lw=0.8))
    ax.annotate('', xy=(99, 15.5), xytext=(1, 15.5), arrowprops=dict(arrowstyle='-|>', lw=1))
    ax.text(50, 12.5, 'Capture time (one task per CIC-IDS2017 file; BENIGN present in every task)', ha='center', fontsize=8.5)
    for k, (lab, col) in enumerate((('Train 60%', '#0072B2'), ('Validation 20%', '#E69F00'), ('Test 20%', '#009E73'))):
        ax.add_patch(Rectangle((1 + k * 17, 6.3), 2.4, 2.2, fc=col)); ax.text(4.3 + k * 17, 7.4, lab, va='center', fontsize=8.5)
    ax.text(52, 7.4, 'Rows sorted by flow timestamp; chronological split\nwithin each class and day', va='center', fontsize=8)
    ax.text(1, 1.5, 'Scaling: signed log + StandardScaler fitted on T1 train only; evaluation after T8 on all test partitions',
            fontsize=8)
    bx = fig.add_axes([0, 0, 1, 0.46]); bx.set_xlim(0, 100); bx.set_ylim(0, 44); bx.axis('off')
    bx.text(0.5, 43, '(b)', fontsize=9, weight='bold', va='top')
    bx.text(99, 40.5, 'Shared by all methods; replay batch, severity weights and EWC penalty are method-specific',
            ha='right', style='italic', fontsize=8)
    def box(x, y, w, h, txt, fc='white'):
        bx.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.8', fc=fc, ec='k', lw=0.8))
        bx.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=8)
    box(1, 26, 19, 9, 'Current-task batch\n$B_t$ (512 flows)')
    box(1, 6, 19, 10, 'Replay batch $B_r$ (256)\nfrom class-balanced\nreservoir (≤ 500/class)', '#DCE9F5')
    box(23.5, 13, 20, 10, 'MLP\n78–512–256–128–15\nReLU, dropout 0.3')
    box(47, 13, 24, 10, r'$\mathcal{L}_{\mathrm{cb{-}focal}}(B_t)$' + '\n+0.5 CE($B_r$) (replay)\n+ EWC penalty (EWC)')
    box(76, 13, 23, 10, 'Adam, LR = $10^{-4}$\n(new optimizer per task)\ngradient clipping 1.0')
    box(74, 0.5, 25, 7, 'Best epoch: class-averaged\nvalidation CE (patience 10)', '#F5F5F5')
    box(44, 0.5, 25, 7, 'After the task: update reservoir\n(replay) or Fisher (EWC)', '#F2E3EF')
    for a, b in (((20, 30), (23.5, 20)), ((20, 12), (23.5, 16)), ((43.5, 18), (47, 18)), ((71, 18), (76, 18)),
                 ((87.5, 13), (87.5, 7.5)), ((74, 4), (69, 4))):
        bx.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', lw=0.8))
    bx.plot([44, 10.5], [4, 4], 'k', lw=0.8); bx.annotate('', xy=(10.5, 6), xytext=(10.5, 3.9), arrowprops=dict(arrowstyle='-|>', lw=0.8))
    bx.text(27, 5, 'Next task', style='italic', fontsize=8)
    fig.savefig(f'{OUT}/fig1.png'); plt.close(fig)


def fig2(main, S5):
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 3.9), gridspec_kw={'width_ratios': [2.4, 1, 1.6]})
    ms = ['Majority', 'FineTuning', 'EWC', 'ReplayUniform', 'ReplayRisk', 'Joint']
    rng = np.random.default_rng(0)
    for i, m in enumerate(ms):
        v = [r['metrics']['mcc'] for r in main if r['method'] == m]
        if not v:
            continue
        axs[0].scatter(i - 0.12 + rng.uniform(-0.06, 0.06, len(v)), v, color=COL[m], alpha=0.8, s=12, lw=0)
        mu = np.mean(v)
        if len(v) > 1:
            from scipy import stats
            h = stats.t.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v))
            axs[0].errorbar(i + 0.15, mu, yerr=h, fmt='o', mfc='white', mec='k', ecolor='k', capsize=2, ms=4)
        else:
            axs[0].scatter(i + 0.15, mu, facecolor='white', edgecolor='k')
    axs[0].set_xticks(range(len(ms))); axs[0].set_xticklabels([LAB[m] for m in ms], rotation=40, ha='right')
    axs[0].set_ylabel('MCC (aggregated test set)'); axs[0].set_title('(a)', loc='left', weight='bold')
    U = {r['seed']: r['metrics']['mcc'] for r in main if r['method'] == 'ReplayUniform'}
    R = {r['seed']: r['metrics']['mcc'] for r in main if r['method'] == 'ReplayRisk'}
    for s in sorted(set(U) & set(R)):
        axs[1].plot([0, 1], [U[s], R[s]], color='#555', lw=1)
        axs[1].scatter([0], [U[s]], color=COL['ReplayUniform'], zorder=3, s=12)
        axs[1].scatter([1], [R[s]], color=COL['ReplayRisk'], zorder=3, s=12)
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(['Uniform replay', 'Severity replay'], rotation=40, ha='right')
    axs[1].set_xlim(-0.4, 1.4); axs[1].set_ylabel('MCC'); axs[1].set_title('(b)', loc='left', weight='bold')
    mf = ['FineTuning', 'EWC', 'ReplayUniform', 'ReplayRisk']
    for i, m in enumerate(mf):
        v = [r['forgetting_macro_recall'] for r in main if r['method'] == m and r.get('forgetting_macro_recall') is not None]
        if v:
            axs[2].scatter(i + rng.uniform(-0.08, 0.08, len(v)), v, color=COL[m], alpha=0.8, s=12, lw=0)
            axs[2].plot([i - 0.2, i + 0.2], [np.mean(v)] * 2, 'k', lw=2)
    axs[2].axhline(0, color='grey', ls=':', lw=1)
    axs[2].set_xticks(range(4)); axs[2].set_xticklabels([LAB[m] for m in mf], rotation=40, ha='right')
    axs[2].set_ylabel('Forgetting (macro-recall)'); axs[2].set_title('(c)', loc='left', weight='bold')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig2.png'); plt.close(fig)


def fig3(main, counts_support, fn='fig3.png', methods=('ReplayUniform', 'ReplayRisk', 'Joint'), order=CLASS_ORDER):
    classes = [c for c in order if c in counts_support]
    fig, ax = plt.subplots(figsize=(5.5, 0.27 * len(classes) + 0.9))
    off = {'ReplayUniform': -0.22, 'ReplayRisk': 0.0, 'Joint': 0.22, 'FineTuning': 0.33, 'EWC': -0.33}
    mk = {'ReplayUniform': 'o', 'ReplayRisk': 'o', 'Joint': 'D', 'FineTuning': 's', 'EWC': '^'}
    for k, c in enumerate(classes):
        y = len(classes) - 1 - k
        if counts_support[c] < 10:
            ax.axhspan(y - 0.5, y + 0.5, color='#EDEDED', zorder=0)
        for m in methods:
            v = [r['metrics']['per_class'][c]['recall'] for r in main if r['method'] == m and c in r['metrics']['per_class']]
            if not v:
                continue
            ax.scatter(v, [y - off[m]] * len(v), color=COL[m], s=6, alpha=0.6, lw=0)
            ax.scatter([np.mean(v)], [y - off[m]], color=COL[m], s=26, edgecolor='k', linewidth=0.6, marker=mk[m], zorder=3,
                       label=LAB[m] if k == 0 else None)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([f"{SHORT.get(c, c)} (n = {counts_support[c]:,})" for c in classes][::-1])
    ax.axvline(0.5, color='grey', ls=':', lw=0.8); ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel('Test recall after the final task')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(methods), frameon=False)
    fig.tight_layout(); fig.savefig(f'{OUT}/{fn}'); plt.close(fig)


def fig4(main, fn='fig4.png', methods=('FineTuning', 'EWC', 'ReplayUniform', 'ReplayRisk')):
    fig, axs = plt.subplots(2, 2, figsize=(5.0, 4.2), gridspec_kw={'hspace': 0.3, 'wspace': 0.12})
    for ax, m, tag in zip(axs.flat, methods, 'abcd'):
        Rs = [r['R_macro_recall'] for r in main if r['method'] == m and r.get('R_macro_recall')]
        ax.set_title(f'({tag}) {LAB[m]}', loc='left', weight='bold')
        if not Rs:
            ax.axis('off'); continue
        T = len(Rs[0]); M = np.full((T, T), np.nan)
        for t in range(T):
            for i in range(t + 1):
                M[t, i] = np.mean([R[t][i] for R in Rs])
        im = ax.imshow(M, vmin=0, vmax=1, cmap='viridis')
        for t in range(T):
            for i in range(t + 1):
                v = M[t, i]; s = '1' if v >= 0.995 else f'{v:.2f}'.lstrip('0')
                ax.text(i, t, s, ha='center', va='center', color='k' if v > 0.55 else 'w', fontsize=8)
        ax.set_xticks(range(T)); ax.set_yticks(range(T))
        ax.set_xticklabels([f'T{i+1}' for i in range(T)]); ax.set_yticklabels([f'T{i+1}' for i in range(T)])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        if tag in 'ac': ax.set_ylabel('After training on task $t$')
        if tag in 'cd': ax.set_xlabel('Evaluated task $i$')
    cb = fig.colorbar(im, ax=axs, shrink=0.5); cb.set_label('Test macro-recall $R_{t,i}$')
    fig.savefig(f'{OUT}/{fn}'); plt.close(fig)


if __name__ == '__main__':
    fig1()
    main = runs_of('main') + runs_of('main_extra')
    S5 = [42, 123, 456, 789, 1011]
    main5 = [r for r in main if r['seed'] in S5 or r['method'] in ('ReplayUniform', 'ReplayRisk')]
    if main:
        fig2(main5, S5)
        sup = {}
        for r in main:
            for c, v in r['metrics']['per_class'].items():
                sup[c] = v['support']
        fig3([r for r in main if r['seed'] in S5], sup)
        fig4([r for r in main if r['seed'] in S5])
    c18 = runs_of('c2018')
    if c18:
        sup = {}
        for r in c18:
            for c, v in r['metrics']['per_class'].items():
                sup[c] = v['support']
        order = sorted(sup, key=lambda c: (c != 'Benign', c))
        fig3(c18, sup, fn='fig5_2018_perclass.png', methods=('ReplayUniform', 'ReplayRisk', 'Joint'), order=order)
    print('figures written to', OUT)
