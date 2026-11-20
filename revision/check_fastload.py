"""New cl_methods (FastLoader, refactored options) vs submitted cl_methods (DataLoader):
same seed, same data subset -> identical metrics and retention matrices."""
import os, sys, time, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generator import create_tasks_temporal
import cl_methods as new
spec = importlib.util.spec_from_file_location('old', os.path.join(os.path.dirname(__file__), '_cl_old.py'))
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
tasks, names, _ = create_tasks_temporal('./data/CIC-IDS2017/MachineLearningCSV/', max_per_day=int(sys.argv[1]), verbose=False)
for C in (old.CLConfig, new.CLConfig):
    C.VERBOSE = False; C.MAX_EPOCHS = 6; C.PATIENCE = 3; C.EWC_FISHER_BATCHES = 20
for m in sys.argv[2:]:
    t0 = time.time(); a = old.run_method(m, tasks, names, old.CLConfig, 42); t1 = time.time()
    b = new.run_method(m, tasks, names, new.CLConfig, 42); t2 = time.time()
    same = (a['metrics']['mcc'] == b['metrics']['mcc'] and a.get('R_macro_recall') == b.get('R_macro_recall')
            and all(a['metrics']['per_class'][c]['recall'] == b['metrics']['per_class'][c]['recall'] for c in a['metrics']['per_class']))
    print(f"{m:<14} MCC old={a['metrics']['mcc']:.6f} new={b['metrics']['mcc']:.6f} identical={same} "
          f"time old={t1-t0:.1f}s new={t2-t1:.1f}s")
