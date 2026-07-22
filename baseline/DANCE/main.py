from dataset import process_dataset, get_data2, get_dataset
from model import *
from utils import *
from learn import *
from tqdm import tqdm
from runner import *
from config import args

if __name__ == '__main__':
    seed_everything(args.seed)
    data, args.sens_idx, args.corr_sens, args.corr_idx, args.x_min, args.x_max = get_dataset(args, args.inid)
    data2 = get_data2(args, data)
    process_dataset(args, data, data2)
    acc, f1, auc_roc, parity, equality = train(args, data, data2)

    for i in range(len(args.strlist)):
        print("==========={}============".format(args.outid+args.strlist[i]))
        print('Acc: {:.2f} ± {:.2f}'.format(np.mean(acc.T[i]), np.std(acc.T[i])))
        print('auc_roc: {:.2f} ± {:.2f}'.format(np.mean(auc_roc.T[i]), np.std(auc_roc.T[i])))
        print('parity: {:.2f} ± {:.2f}'.format(np.mean(parity.T[i]), np.std(parity.T[i])))
        print('equality: {:.2f} ± {:.2f}'.format(np.mean(equality.T[i]), np.std(equality.T[i])))
