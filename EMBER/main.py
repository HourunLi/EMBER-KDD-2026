import json
import os

import numpy as np

from config import args
from dataset import get_dataset, process_dataset
from runner import train_and_adapt
from utils import seed_everything

if __name__ == '__main__':
    seed_everything(args.seed)

    source_data = get_dataset(args, args.inid)
    target_data = get_dataset(args, args.outid)

    print("********************process source data********************")
    source_data = process_dataset(args, source_data)
    print("********************process target data********************")
    target_data = process_dataset(args, target_data, is_target=True)

    (src_acc, src_auc_roc, src_parity, src_equality,
     tgt_acc, tgt_auc_roc, tgt_parity, tgt_equality,
     ada_acc, ada_auc_roc, ada_parity, ada_equality) = train_and_adapt(args, source_data, target_data)

    print(f"=========== {args.inid} (Source) ===========")
    print(f"Acc:      {np.mean(src_acc):.2f} ± {np.std(src_acc):.2f}")
    print(f"AUC-ROC:  {np.mean(src_auc_roc):.2f} ± {np.std(src_auc_roc):.2f}")
    print(f"Parity:   {np.mean(src_parity):.2f} ± {np.std(src_parity):.2f}")
    print(f"Equality: {np.mean(src_equality):.2f} ± {np.std(src_equality):.2f}")

    print(f"\n=========== {args.outid} (Target, before adapt) ===========")
    print(f"Acc:      {np.mean(tgt_acc):.2f} ± {np.std(tgt_acc):.2f}")
    print(f"AUC-ROC:  {np.mean(tgt_auc_roc):.2f} ± {np.std(tgt_auc_roc):.2f}")
    print(f"Parity:   {np.mean(tgt_parity):.2f} ± {np.std(tgt_parity):.2f}")
    print(f"Equality: {np.mean(tgt_equality):.2f} ± {np.std(tgt_equality):.2f}")

    print(f"\n=========== {args.outid} (Target, after SFDA adapt) ===========")
    print(f"Acc:      {np.mean(ada_acc):.2f} ± {np.std(ada_acc):.2f}")
    print(f"AUC-ROC:  {np.mean(ada_auc_roc):.2f} ± {np.std(ada_auc_roc):.2f}")
    print(f"Parity:   {np.mean(ada_parity):.2f} ± {np.std(ada_parity):.2f}")
    print(f"Equality: {np.mean(ada_equality):.2f} ± {np.std(ada_equality):.2f}")

    if args.result_path:
        def values(array):
            return np.asarray(array).reshape(-1).astype(float).tolist()

        payload = {
            'dataset': args.dataset,
            'inid': args.inid,
            'outid': args.outid,
            'seed': args.seed,
            'target_seed': args.target_seed,
            'ablation': args.ablation,
            'metrics': {
                'source': {
                    'acc': values(src_acc), 'auc': values(src_auc_roc),
                    'dp': values(src_parity), 'eo': values(src_equality),
                },
                'target_before': {
                    'acc': values(tgt_acc), 'auc': values(tgt_auc_roc),
                    'dp': values(tgt_parity), 'eo': values(tgt_equality),
                },
                'target_after': {
                    'acc': values(ada_acc), 'auc': values(ada_auc_roc),
                    'dp': values(ada_parity), 'eo': values(ada_equality),
                },
            },
        }
        result_dir = os.path.dirname(os.path.abspath(args.result_path))
        os.makedirs(result_dir, exist_ok=True)
        with open(args.result_path, 'w', encoding='utf-8') as result_file:
            json.dump(payload, result_file, ensure_ascii=False, indent=2)
