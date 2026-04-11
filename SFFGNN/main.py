from dataset import process_dataset, get_dataset
from models import *
from utils import *
from runner import *
from config import args
import numpy as np
import torch

if __name__ == '__main__':
    seed_everything(args.seed)

    source_data = get_dataset(args, args.inid)
    target_data = get_dataset(args, args.outid)

    print("********************process source data********************")
    source_data = process_dataset(args, source_data)
    print("********************process target data********************")
    target_data = process_dataset(args, target_data)

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
