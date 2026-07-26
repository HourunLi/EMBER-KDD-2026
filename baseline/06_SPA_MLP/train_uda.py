import argparse
import random
import os
import os.path as osp
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

import loss
import utils


FAIR_DATASETS = {'bailA', 'germanA', 'pokec', 'syn'}


def gauss_(v1, v2, sigma):
    norm_ = torch.norm(v1 - v2, p=2, dim=0)
    return torch.exp(-0.5 * norm_ / sigma**2)

def euc_(v1, v2):
    return torch.norm(v1 - v2, p=2, dim=0)

def adj_(s, t, ap='cos'):
    # s, t [bsize, dim], [bsize, dim] -> [bsize, bsize]
    if ap == 'cos':
        s_norm = s / torch.norm(s, p=2, dim=1, keepdim=True)
        t_norm = t / torch.norm(t, p=2, dim=1, keepdim=True)
        adj = s_norm @ t_norm.T
    elif ap == 'gauss':
        sigma_ = 1.5
        M, N = s.shape[0], t.shape[0]
        adj = torch.zeros([M, N], dtype=s.dtype, device=s.device)
        for i in range(M):
            for j in range(i):
                adj[i][j] = adj[j][i] = gauss_(s[i], t[j], sigma_)
    elif ap == 'euc':
        M, N = s.shape[0], t.shape[0]
        adj = torch.zeros([M, N], dtype=s.dtype, device=s.device)
        for i in range(M):
            for j in range(i):
                adj[i][j] = adj[j][i] = euc_(s[i], t[j])
    return adj
    
def laplacian_(A, ltype='laplac1'):
    v = torch.sum(A, dim=1)
    if ltype == 'laplac1':
        v_inv = 1 / v
        D_inv = torch.diag(v_inv).cuda()
        return -D_inv @ A
    elif ltype == 'laplac2':
        D = torch.diag(v).cuda()
        return D - A
    elif ltype == 'laplac3':
        v_sqrt = 1 / torch.sqrt(v)
        D_sqrt = torch.diag(v_sqrt).cuda()
        I = torch.eye(A.shape[0]).cuda()
        return I - D_sqrt @ A @ D_sqrt

def svd_loss_(s, t):
    # s, t [bsize, dim], [bsize, dim]
    s_matrix = adj_(s, s, args.ap)
    t_matrix = adj_(t, t, args.ap)
    s_matrix = laplacian_(s_matrix, args.laplac)
    t_matrix = laplacian_(t_matrix, args.laplac)
    _, s_v, _ = torch.svd(s_matrix)
    _, t_v, _ = torch.svd(t_matrix)
    svd_loss = torch.norm(s_v - t_v, p=2)
    return svd_loss

def lr_scheduler(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def data_load(args, labels=None):
    if args.dset == 'syn':
        source_set = utils.ObjectSynthetic(
            args.s_dset_path, args.s_label_path, y=labels)
        source_mean = source_set.features.mean(dim=0)
        source_std = source_set.features.std(
            dim=0, unbiased=False).clamp_min(1e-6)
        source_set.standardize(source_mean, source_std)

        target_set = utils.ObjectSynthetic(
            args.t_dset_path, args.t_label_path, ridx=True)
        if target_set.in_features != source_set.in_features:
            raise ValueError(
                'Source Syn features contain {} columns, but target Syn features '
                'contain {}'.format(
                    source_set.in_features, target_set.in_features))
        target_set.standardize(source_mean, source_std)

        test_set = utils.ObjectSynthetic(
            args.test_dset_path,
            args.test_label_path,
            sensitive_path=args.test_sensitive_path,
            mean=source_mean,
            std=source_std,
            return_sensitive=True)
    elif args.dset in FAIR_DATASETS:
        tabular_kwargs = {
            'label_name': args.label_name,
            'sensitive_name': args.sensitive_name,
            'excluded_feature_names': args.excluded_feature_names,
            'label_mapping': args.label_mapping,
            'label_positive_threshold': args.label_positive_threshold,
            'sensitive_mapping': args.sensitive_mapping,
            'invalid_label_values': args.invalid_label_values
        }
        feature_names = None
        if args.dset == 'pokec':
            feature_names = utils.common_tabular_features(
                args.s_dset_path,
                args.t_dset_path,
                args.label_name,
                args.sensitive_name,
                excluded_feature_names=args.excluded_feature_names)
        source_set = utils.ObjectTabular(
            args.s_dset_path, feature_names=feature_names, y=labels,
            **tabular_kwargs)
        source_mean = source_set.features.mean(dim=0)
        source_std = source_set.features.std(dim=0, unbiased=False).clamp_min(1e-6)
        source_set.standardize(source_mean, source_std)

        target_set = utils.ObjectTabular(
            args.t_dset_path, feature_names=source_set.feature_names,
            mean=source_mean, std=source_std, ridx=True, **tabular_kwargs)
        test_set = utils.ObjectTabular(
            args.test_dset_path, feature_names=source_set.feature_names,
            mean=source_mean, std=source_std, return_sensitive=True,
            **tabular_kwargs)
    else:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.RandomCrop((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        source_set = utils.ObjectImage(args.data_root, args.s_dset_path, train_transform, y=labels)
        target_set = utils.ObjectImage(args.data_root, args.t_dset_path, train_transform, ridx=True)
        test_set = utils.ObjectImage(args.data_root, args.test_dset_path, test_transform)

    dset_loaders = {}
    dset_loaders["source"] = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.worker, drop_last=True)
    dset_loaders["target"] = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.worker, drop_last=True)
    dset_loaders["test"] = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size*3,
        shuffle=False, num_workers=args.worker, drop_last=False)
    return dset_loaders


def train(args, validate=False, label=None):
    ## set pre-process
    dset_loaders = data_load(args, label)
    class_num = args.class_num
    class_weight_src = torch.ones(class_num, ).cuda()
    ##################################################################################################
    ## set base network
    if args.net == 'resnet101':
        netG = utils.ResBase101().cuda()
    elif args.net == 'resnet50':
        netG = utils.ResBase50().cuda()
    elif args.net == 'mlp':
        input_dim = dset_loaders["source"].dataset.in_features
        netG = utils.MLPBase(input_dim, args.mlp_hidden_dim, args.mlp_dropout).cuda()

    netF = utils.ResClassifier(class_num=class_num, feature_dim=netG.in_features, 
        bottleneck_dim=args.bottleneck_dim).cuda()

    max_len = max(len(dset_loaders["source"]), len(dset_loaders["target"]))
    args.max_iter = args.max_epoch * max_len

    ad_flag = False
    if args.method in {'DANN', 'DANNE'}:
        ad_net = utils.AdversarialNetwork(args.bottleneck_dim, 1024, max_iter=args.max_iter).cuda()
        ad_flag = True
    if args.method in {'CDAN', 'CDANE'}:
        ad_net = utils.AdversarialNetwork(args.bottleneck_dim*class_num, 1024, max_iter=args.max_iter).cuda() 
        random_layer = None
        ad_flag = True  

    optimizer_g = optim.SGD(netG.parameters(), lr = args.lr * 0.1)
    optimizer_f = optim.SGD(netF.parameters(), lr = args.lr)
    if ad_flag:
        optimizer_d = optim.SGD(ad_net.parameters(), lr = args.lr)
   
    base_network = nn.Sequential(netG, netF)

    mem_fea = torch.rand(len(dset_loaders["target"].dataset), args.bottleneck_dim).cuda()
    mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
    mem_cls = torch.ones(len(dset_loaders["target"].dataset), class_num).cuda() / class_num

    source_loader_iter = iter(dset_loaders["source"])
    target_loader_iter = iter(dset_loaders["target"])
    ####
    list_acc = []
    best_ent = 100
    for iter_num in range(1, args.max_iter + 1):
        base_network.train()
        lr_scheduler(optimizer_g, init_lr=args.lr * 0.1, iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler(optimizer_f, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)
        if ad_flag:
            lr_scheduler(optimizer_d, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)

        try:
            inputs_source, labels_source = next(source_loader_iter)
        except:
            source_loader_iter = iter(dset_loaders["source"])
            inputs_source, labels_source = next(source_loader_iter)
        try:
            inputs_target, _, idx = next(target_loader_iter)
        except:
            target_loader_iter = iter(dset_loaders["target"])
            inputs_target, _, idx = next(target_loader_iter)
        
        inputs_source = inputs_source.cuda()
        inputs_target = inputs_target.cuda()
        labels_source = labels_source.cuda()
        idx = idx.cuda()

        if args.method == 'srconly' and args.pl == 'none':
            features_source, outputs_source = base_network(inputs_source)
        else:
            features_source, outputs_source = base_network(inputs_source)
            features_target, outputs_target = base_network(inputs_target)
            features = torch.cat((features_source, features_target), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)

        eff = utils.calc_coeff(iter_num, max_iter=args.max_iter)
        if args.method[-1] == 'E':
            entropy = loss.Entropy(softmax_out)
        else:
            entropy = None

        if args.method in {'CDAN', 'CDANE'}:           
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, eff, random_layer)
        elif args.method in {'DANN', 'DANNE'}:  
            transfer_loss = loss.DANN(features, ad_net, entropy, eff)
        elif args.method == 'srconly':
            transfer_loss = torch.tensor(0.0).cuda()
        else:
            raise ValueError('Method cannot be recognized.')

        src_ = loss.CrossEntropyLabelSmooth(reduction='none',num_classes=class_num, epsilon=args.smooth)(outputs_source, labels_source)
        weight_src = class_weight_src[labels_source].unsqueeze(0)
        classifier_loss = torch.sum(weight_src * src_) / torch.sum(weight_src)
        total_loss = transfer_loss + classifier_loss

        eff = iter_num / args.max_iter

        if args.ifcorrect:
            features_target = features_target / torch.norm(features_target, p=2, dim=1, keepdim=True)
        dis = -torch.mm(features_target.detach(), mem_fea.t())
        for di in range(dis.size(0)):
            dis[di, idx[di]] = torch.max(dis)
        _, p1 = torch.sort(dis, dim=1)

        w = torch.zeros(
            features_target.size(0), mem_fea.size(0),
            dtype=features_target.dtype, device=features_target.device)
        for wi in range(w.size(0)):
            for wj in range(args.K):
                w[wi][p1[wi, wj]] = 1/ args.K
        weight_, pred = torch.max(w.mm(mem_cls), 1)

        loss_ = nn.CrossEntropyLoss(reduction='none')(outputs_target, pred)
        classifier_loss = torch.sum(weight_ * loss_) / torch.sum(weight_)
        pl_loss = args.tar_par * eff * classifier_loss
        if args.pl != 'none':
            total_loss += pl_loss

        if args.ifsvd:
            # svd loss
            f_s = features_source
            f_t = features_target
            # svd_loss = args.svd_par * eff * svd_loss_(f_s, f_t)
            svd_loss = args.svd_par * svd_loss_(f_s, f_t)
            total_loss += svd_loss
            
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        if ad_flag:
            optimizer_d.zero_grad()
        total_loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        if ad_flag:
            optimizer_d.step()

        base_network.eval() 
        with torch.no_grad():
            features_target, outputs_target = base_network(inputs_target)
            features_target = features_target / torch.norm(features_target, p=2, dim=1, keepdim=True)
            softmax_out = nn.Softmax(dim=1)(outputs_target)
            outputs_target = softmax_out**2 / ((softmax_out**2).sum(dim=0))

            mem_fea[idx] = (1.0 - args.momentum) * mem_fea[idx] + args.momentum * features_target.clone()
            mem_cls[idx] = (1.0 - args.momentum) * mem_cls[idx] + args.momentum * outputs_target.clone()


        if iter_num % 10 == 0:
            epoch_num = (iter_num - 1) // int(max_len) + 1
            batch_num = (iter_num - 1) % int(max_len) + 1
            iter_str = 'Epoch:{}/{}, Batch:{}/{}, total:{:.5f}, trans: {:.5f}, cls: {:.5f}'.format(
                epoch_num, args.max_epoch, batch_num, int(max_len),
                total_loss.item(), transfer_loss.item(), classifier_loss.item())
            if args.pl != 'none':
                iter_str += ', pl:{:.5f}'.format(pl_loss.item())
            if args.ifsvd:
                iter_str += ', svd:{:.5f}'.format(svd_loss.item())
            print(iter_str)
            
        if iter_num % int(max_len) == 0:
            base_network.eval()
            if args.dset == 'visda2017':
                acc, py, score, y, tacc = utils.cal_acc_visda(dset_loaders["test"], base_network)
                args.out_file.write(tacc + '\n')
                args.out_file.flush()
                print(tacc)

                _ent = loss.Entropy(score)
                mean_ent = 0
                for ci in range(args.class_num):
                    mean_ent += _ent[py==ci].mean()
                mean_ent /= args.class_num
            elif args.dset in FAIR_DATASETS:
                final_target_test = iter_num == args.max_iter
                test_result = utils.cal_acc(
                    dset_loaders["test"], base_network,
                    return_fairness=True,
                    return_features=final_target_test)
                if final_target_test:
                    (acc, py, score, y, roc_auc, parity, equality,
                     test_representations) = test_result
                else:
                    (acc, py, score, y, roc_auc,
                     parity, equality) = test_result
                mean_ent = torch.mean(loss.Entropy(score))
            else:
                acc, py, score, y = utils.cal_acc(dset_loaders["test"], base_network)
                mean_ent = torch.mean(loss.Entropy(score))

            list_acc.append(acc * 100)
            if best_ent > mean_ent:
                best_ent = mean_ent
                val_acc = acc * 100
                best_y = y
                best_py = py
                best_score = score
                if args.dset in FAIR_DATASETS:
                    best_roc_auc = roc_auc
                    best_parity = parity
                    best_equality = equality

            epoch_num = iter_num // int(max_len)
            log_str = 'Task: {}, Epoch:{}/{}, Batch:{}/{}, Iter:{}/{}; Accuracy = {:.2f}%; Mean Ent = {:.4f}'.format(
                args.name, epoch_num, args.max_epoch, int(max_len), int(max_len),
                iter_num, args.max_iter, acc*100, mean_ent)
            if args.dset in FAIR_DATASETS:
                log_str += '; ROC_AUC = {:.2f}%; Parity = {:.2f}%; Equality = {:.2f}%'.format(
                    roc_auc * 100, parity * 100, equality * 100)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')            

    idx = np.argmax(np.array(list_acc))
    max_acc = list_acc[idx]
    final_acc = list_acc[-1]

    log_str = '\n==========================================\n'
    log_str += '\nVal Acc = {:.2f}%\nMax Acc = {:.2f}%\nFin Acc = {:.2f}%\n'.format(
        val_acc, max_acc, final_acc)
    if args.dset in FAIR_DATASETS:
        log_str += 'Val ROC_AUC = {:.2f}%\n'.format(best_roc_auc * 100)
        log_str += 'Val Parity = {:.2f}%\nVal Equality = {:.2f}%\n'.format(
            best_parity * 100, best_equality * 100)
        log_str += 'Fin ROC_AUC = {:.2f}%\n'.format(roc_auc * 100)
        log_str += 'Fin Parity = {:.2f}%\nFin Equality = {:.2f}%\n'.format(
            parity * 100, equality * 100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')  
    
    best_labels = best_y.cpu().numpy().astype(np.int64)
    if args.dset in FAIR_DATASETS:
        if not validate:
            test_dataset = dset_loaders["test"].dataset
            visualization_dir = osp.join(
                os.getcwd(), '{}_seed{}'.format(args.dset, args.seed))
            feat_path, labels_path = utils.save_visualization_embeddings(
                visualization_dir,
                test_representations,
                py,
                test_dataset.sensitive,
                valid_mask=test_dataset.valid_mask)
            valid_target_count = int(test_dataset.valid_mask.sum())
            export_str = (
                'Visualization embeddings saved: shape=[{}, {}], feat={}, '
                'labels={}'.format(
                    valid_target_count, test_representations.size(1),
                    feat_path, labels_path))
            args.out_file.write(export_str + '\n')
            args.out_file.flush()
            print(export_str)
        seed_metrics = {
            'accuracy': final_acc / 100.0,
            'roc_auc': roc_auc,
            'parity': parity,
            'equality': equality
        }
        return best_labels, seed_metrics
    return best_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation Methods')
    parser.add_argument('--method', type=str, default='srconly', choices=['srconly', 'CDAN', 'CDANE', 'DANN', 'DANNE'])
    parser.add_argument('--pl', type=str, default='none', choices=['none', 'spa', 'npl', 'bsp'])

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help="run multiple random seeds sequentially")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--bottleneck_dim', type=int, default=256)

    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--momentum', type=float, default=1.0)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--tar_par', type=float, default=1.0)
    parser.add_argument('--validate', action='store_true')
    
    parser.add_argument('--net', type=str, default='mlp', choices=["resnet50", "resnet101", "mlp"])
    parser.add_argument('--mlp_hidden_dim', type=int, default=64)
    parser.add_argument('--mlp_dropout', type=float, default=0.5)
    parser.add_argument('--dset', type=str, default='bailA', choices=['domain_net', 'multi', 'visda2017', 'office31', 'office_home', 'bailA', 'germanA', 'pokec', 'syn'], help="dataset used")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")

    parser.add_argument('--ifcorrect', action='store_true')
    parser.add_argument('--ifsvd', action='store_true')
    parser.add_argument('--svd_par', type=float, default=1.0)
    parser.add_argument('--laplac', type=str, default='laplac1', choices=["laplac1", "laplac2", "laplac3"])
    parser.add_argument('--ap', type=str, default='euc', choices=['cos', 'gauss', 'euc'])
    args = parser.parse_args()
    args.output = args.output.strip()

    args.eval_epoch = args.max_epoch / 10

    if args.dset == 'office_home':
        names = ['Art', 'Clipart', 'Product', 'Real']
        args.class_num = 65 
        args.data_root = ''
    if args.dset == 'office31':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
        args.data_root = ''
    if args.dset == 'visda2017':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'multi': # DomainNet-126
        names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 126
        args.data_root = '/data/domain_net/'
    if args.dset == 'domain_net':
        names = ['clipart_train', 'painting_train', 'real_train', 'sketch_train']
        tests = ['clipart_test', 'painting_test', 'real_test', 'sketch_test']
        args.class_num = 345  
        args.data_root = ''
    if args.dset == 'bailA':
        names = ['bailA_2', 'bailA_1']
        args.class_num = 2
        args.data_root = ''
        args.label_name = 'RECID'
        args.sensitive_name = 'WHITE'
        args.excluded_feature_names = []
        args.label_mapping = None
        args.label_positive_threshold = None
        args.invalid_label_values = None
        args.sensitive_mapping = None
        args.s_dset_path = './dataset/bailA/bailA_2.csv'
        args.t_dset_path = './dataset/bailA/bailA_1.csv'
        args.test_dset_path = args.t_dset_path
    if args.dset == 'germanA':
        names = ['germanA_2', 'germanA_1']
        args.class_num = 2
        args.data_root = ''
        args.label_name = 'GoodCustomer'
        args.sensitive_name = 'Gender'
        args.excluded_feature_names = ['PurposeOfLoan', 'OtherLoansAtStore']
        args.label_mapping = {-1: 0, 1: 1}
        args.label_positive_threshold = None
        args.invalid_label_values = None
        args.sensitive_mapping = {'Male': 0, 'Female': 1}
        args.s_dset_path = './dataset/germanA/germanA_2.csv'
        args.t_dset_path = './dataset/germanA/germanA_1.csv'
        args.test_dset_path = args.t_dset_path
    if args.dset == 'pokec':
        names = ['pokec_z', 'pokec_n']
        args.class_num = 2
        args.data_root = ''
        args.label_name = 'I_am_working_in_field'
        args.sensitive_name = 'region'
        args.excluded_feature_names = []
        args.label_mapping = None
        args.label_positive_threshold = 0
        args.invalid_label_values = [-1]
        args.sensitive_mapping = None
        args.s_dset_path = './dataset/pokec_z/pokec_z.csv'
        args.t_dset_path = './dataset/pokec_n/pokec_n.csv'
        args.test_dset_path = args.t_dset_path
    if args.dset == 'syn':
        names = ['syn-2', 'syn-1']
        args.class_num = 2
        args.data_root = ''
        args.label_name = 'label'
        args.sensitive_name = 'sens'
        args.excluded_feature_names = []
        args.label_mapping = None
        args.label_positive_threshold = None
        args.invalid_label_values = None
        args.sensitive_mapping = None
        args.s_dset_path = './dataset/syn/syn-2_feat.csv'
        args.s_label_path = './dataset/syn/syn-2_label.txt'
        args.t_dset_path = './dataset/syn/syn-1_feat.csv'
        args.t_label_path = './dataset/syn/syn-1_label.txt'
        args.test_dset_path = args.t_dset_path
        args.test_label_path = args.t_label_path
        args.test_sensitive_path = './dataset/syn/syn-1_sens.txt'

    if args.dset in FAIR_DATASETS:
        if args.net != 'mlp':
            parser.error('--dset {} requires --net mlp'.format(args.dset))
        if args.s != 0 or args.t != 1:
            parser.error(
                '{} is configured for source {} (--s 0) and target {} (--t 1)'.format(
                    args.dset, names[0], names[1]))
    else:
        args.s_dset_path = './data/uda/' + args.dset + '/' + names[args.s] + '.txt'
        args.t_dset_path = './data/uda/' + args.dset + '/' + names[args.t] + '.txt'
        if args.dset == 'domain_net':
            args.test_dset_path = './data/' + args.dset + '/' + tests[args.t] + '.txt'
        else:
            args.test_dset_path = args.t_dset_path

    if args.pl == 'none':
        args.output_dir = osp.join(args.output, args.pl, args.dset, 
            names[args.s][0].upper() + names[args.t][0].upper())
    else:
        args.output_dir = osp.join(args.output, args.pl + '_' + str(args.tar_par), args.dset, 
            names[args.s][0].upper() + names[args.t][0].upper())

    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    torch.backends.cudnn.deterministic = True

    seeds = args.seeds if args.seeds is not None else [args.seed]
    if args.validate and len(seeds) > 1:
        parser.error('--validate is only supported for a single-seed run')

    seed_results = []
    seed_metric_history = {
        'accuracy': [],
        'roc_auc': [],
        'parity': [],
        'equality': []
    }

    for current_seed in seeds:
        args.seed = current_seed
        if len(seeds) > 1:
            args.log = '{}_seed{}'.format(args.method, current_seed)
        else:
            args.log = args.method
        args.out_file = open(
            osp.join(args.output_dir, "{:}.txt".format(args.log)), "w")

        torch.manual_seed(current_seed)
        torch.cuda.manual_seed(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)

        print('\n========== Running seed {} =========='.format(current_seed))
        utils.print_args(args)
        train_result = train(args)

        if args.dset in FAIR_DATASETS:
            label, seed_metrics = train_result
            seed_results.append((current_seed, seed_metrics))
            for metric_name in seed_metric_history:
                seed_metric_history[metric_name].append(seed_metrics[metric_name])
        else:
            label = train_result

        if args.validate:
            train(args, validate=True, label=label)

        args.out_file.close()
        torch.cuda.empty_cache()

    if args.dset in FAIR_DATASETS and len(seeds) > 1:
        seed_summary = utils.metric_mean_std(
            seed_metric_history, expected_count=len(seeds))
        seed_text = '\n==========================================\n'
        seed_text += 'Cross-Seed Summary ({} seeds: {}; final-epoch metrics; population std)\n'.format(
            len(seeds), ','.join(str(seed) for seed in seeds))
        seed_text += '==========================================\n'

        for current_seed, seed_metrics in seed_results:
            seed_text += (
                'Seed {}: Accuracy = {:.2f}%; ROC_AUC = {:.2f}%; '
                'Parity = {:.2f}%; Equality = {:.2f}%\n'
            ).format(
                current_seed,
                seed_metrics['accuracy'] * 100,
                seed_metrics['roc_auc'] * 100,
                seed_metrics['parity'] * 100,
                seed_metrics['equality'] * 100)

        seed_text += '\nFive-Seed Mean +/- Std:\n'
        seed_text += 'Accuracy = {:.2f}% +/- {:.2f}%\n'.format(
            seed_summary['accuracy']['mean'] * 100,
            seed_summary['accuracy']['std'] * 100)
        seed_text += 'ROC_AUC = {:.2f}% +/- {:.2f}%\n'.format(
            seed_summary['roc_auc']['mean'] * 100,
            seed_summary['roc_auc']['std'] * 100)
        seed_text += 'Parity = {:.2f}% +/- {:.2f}%\n'.format(
            seed_summary['parity']['mean'] * 100,
            seed_summary['parity']['std'] * 100)
        seed_text += 'Equality = {:.2f}% +/- {:.2f}%\n'.format(
            seed_summary['equality']['mean'] * 100,
            seed_summary['equality']['std'] * 100)

        summary_name = '{}_seeds_{}_summary.txt'.format(
            args.method, '-'.join(str(seed) for seed in seeds))
        summary_path = osp.join(args.output_dir, summary_name)
        with open(summary_path, 'w') as summary_file:
            summary_file.write(seed_text)
        print(seed_text)
        print('Cross-seed summary saved to: {}'.format(summary_path))
