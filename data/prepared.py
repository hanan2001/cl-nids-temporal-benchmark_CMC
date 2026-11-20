"""
Disk cache of the prepared task arrays, loaded with memory mapping so that
several training processes share one copy of the data in the OS page cache.
Values are identical to those returned by the task constructors.
"""
import os, json, hashlib, warnings
import numpy as np
import torch
from torch.utils.data import TensorDataset

PREP_DIR = './data/prepared/'


def _key(**kw):
    s = json.dumps(kw, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12], s


def load_or_prepare(dataset, data_path, split='per_class', max_per_day=None,
                    order='file', dedup=None):
    key, desc = _key(dataset=dataset, split=split, max_per_day=max_per_day,
                     order=order, dedup=dedup)
    d = os.path.join(PREP_DIR, key)
    meta_f = os.path.join(d, 'meta.json')
    if not os.path.exists(meta_f):
        if dataset == 'cic2018':
            from data.cic2018 import create_tasks_2018
            tasks, names, counts, info = create_tasks_2018(split=split, dedup=dedup,
                                                           return_info=True)
        else:
            from data.generator import create_tasks_temporal
            tasks, names, counts, info = create_tasks_temporal(
                data_path, split=split, max_per_day=max_per_day, order=order,
                dedup=dedup, return_info=True)
        os.makedirs(d, exist_ok=True)
        for i, t in enumerate(tasks):
            for part in ('train', 'val', 'test'):
                np.save(os.path.join(d, f't{i}_{part}_X.npy'), t[part].tensors[0].numpy())
                np.save(os.path.join(d, f't{i}_{part}_y.npy'), t[part].tensors[1].numpy())
        tmp = meta_f + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'desc': desc, 'names': [t['name'] for t in tasks], 'class_names': names,
                       'counts': counts, 'info': info}, f)
        os.replace(tmp, meta_f)
    with open(meta_f) as f:
        meta = json.load(f)
    tasks = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')        # read-only memory map
        for i, name in enumerate(meta['names']):
            t = {'name': name}
            for part in ('train', 'val', 'test'):
                X = np.load(os.path.join(d, f't{i}_{part}_X.npy'), mmap_mode='r')
                y = np.load(os.path.join(d, f't{i}_{part}_y.npy'))
                t[part] = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
            tasks.append(t)
    return tasks, meta['class_names'], meta['counts'], meta['info']
