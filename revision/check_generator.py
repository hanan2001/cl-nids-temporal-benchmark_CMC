"""The default protocol of the new generator must give exactly the arrays of the submitted one."""
import os, sys, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from data.generator import create_tasks_temporal as new
spec = importlib.util.spec_from_file_location('old', os.path.join(os.path.dirname(__file__), '_gen_old.py'))
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
P = './data/CIC-IDS2017/MachineLearningCSV/'
for split in ('per_class', 'rows'):
    a, ca, _ = old.create_tasks_temporal(P, split=split, verbose=False)
    b, cb, _ = new(P, split=split, verbose=False)
    assert ca == cb
    for ta, tb in zip(a, b):
        for part in ('train', 'val', 'test'):
            for i in (0, 1):
                assert torch.equal(ta[part].tensors[i], tb[part].tensors[i]), (split, ta['name'], part, i)
    print(split, 'IDENTICAL')
