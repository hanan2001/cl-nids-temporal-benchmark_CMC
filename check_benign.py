from data.generator import create_tasks

tasks, names = create_tasks(mode='temporal')
b = names.index('BENIGN')

for i, t in enumerate(tasks, 1):
    v = (t['val'].tensors[1].numpy() == b).mean()
    te = (t['test'].tensors[1].numpy() == b).mean()
    tr = (t['train'].tensors[1].numpy() == b).mean()  # <-- AJOUTEZ CETTE LIGNE
    print(f"T{i}  BENIGN  train={tr:.3f}  val={v:.3f}  test={te:.3f}")