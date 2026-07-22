from tqdm import tqdm
from dataset import get_dataset
import numpy as np
from model import *
from config import mprint
from utils import *
from learn import *
from mixup import *
from torch import nn
from torch.nn import functional as F
from torch import optim
from nets import create_gcn, create_gat, create_sage, create_encoder
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(args, data, data2):
    pbar = tqdm(range(args.runs), unit='run')
    if args.ood == 2:
        acc, f1, auc_roc, parity, equality = np.zeros([args.runs,len(data2)]), np.zeros([args.runs,len(data2)]), np.zeros([args.runs,len(data2)]), np.zeros([args.runs,len(data2)]), np.zeros([args.runs, len(data2)])
    elif args.ood == 1:
        acc, f1, auc_roc, parity, equality = np.zeros([args.runs,len(args.strlist)]), np.zeros([args.runs,len(args.strlist)]), np.zeros([args.runs,len(args.strlist)]), np.zeros([args.runs,len(args.strlist)]), np.zeros([args.runs, len(args.strlist)])
    else:
        acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)

    encoder, optimizer_e = create_encoder(args)
    # encoder = MLP_encoder(args).to(args.device)
    # optimizer_e = torch.optim.Adam(params=encoder.parameters(), weight_decay=args.e_wd, lr=args.e_lr)
    generator = channel_masker(args).to(args.device)
    optimizer_g = torch.optim.Adam([dict(params = generator.parameters(), weight_decay = args.g_wd)], lr = args.g_lr)
    classifier = MLP_classifier(args).to(args.device)
    optimizer_c = torch.optim.Adam([dict(params = classifier.parameters(), weight_decay = args.c_wd)], lr = args.c_lr)
    discriminator = MLP_discriminator(args).to(args.device)
    optimizer_d = torch.optim.Adam([dict(params = discriminator.parameters(), weight_decay = args.d_wd)], lr = args.d_lr)
    projector = MLP_projector(args).to(args.device)
    optimizer_p = torch.optim.Adam([dict(params = projector.parameters(), weight_decay = args.g_wd)], lr = args.p_lr)
    classify_criterion = nn.MSELoss()
    # classify_criterion = nn.L1Loss()
    # classify_criterion = nn.SmoothL1Loss()
    # classify_criterion = nn.HuberLoss()
    # discriminate_criterion = nn.BCELoss()
    # discriminate_criterion = nn.L1Loss()
    discriminate_criterion = nn.MSELoss()
    data = data.to(args.device)
    if args.ood == 2:
        for i in range(len(data2)):
            data2[i] = data2[i].to(args.device)
            data2[i].test_mask = data2[i].test_mask | data2[i].val_mask | data2[i].test_mask
    elif data2 != None:
        data2 = data2.to(args.device)
    else:
        data2 = data2

    labels = data.y[data.train_mask]
    labels_view = data.y.view(-1, 1)[data.train_mask]
    batch_size = data.y.view(-1, 1)[data.train_mask].size(0)
    
    # 处理整个数据集
    t_idx_s0 = data.sens[data.train_mask] == 0
    t_idx_s1 = data.sens[data.train_mask] == 1
    t_idx_s0_y1 = torch.logical_and(t_idx_s0, labels == 1)
    t_idx_s1_y1 = torch.logical_and(t_idx_s1, labels == 1)
    t_idx_s0_y0 = torch.logical_and(t_idx_s0, labels == 0)
    t_idx_s1_y0 = torch.logical_and(t_idx_s1, labels == 0)
    num_t_s0_y1, num_t_s1_y1, num_t_s0_y0, num_t_s1_y0 = sum(t_idx_s0_y1), sum(t_idx_s1_y1), sum(t_idx_s0_y0), sum(t_idx_s1_y0)


    idx_s0 = data.sens == 0
    idx_s1 = data.sens == 1
    idx_s0_y1 = torch.logical_and(idx_s0, data.y == 1)
    idx_s1_y1 = torch.logical_and(idx_s1, data.y == 1)
    idx_s0_y0 = torch.logical_and(idx_s0, data.y == 0)
    idx_s1_y0 = torch.logical_and(idx_s1, data.y == 0)
    num_s0_y1, num_s1_y1, num_s0_y0, num_s1_y0, = sum(idx_s0_y1), sum(idx_s1_y1), sum(idx_s0_y0), sum(idx_s1_y0)
    
    mask = torch.eq(labels_view, labels_view.T).to(args.device)  # 类别标签掩码
    sens = data.sens.view(-1, 1)[data.train_mask]
    sensitive_mask = torch.eq(sens, sens.T).to(args.device)  # 敏感属性掩码
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(args.device), 0)  # 用于排除自对比样本的掩码 
    # print(torch.arange(batch_size).view(-1, 1)) # [100,1]
    # print(logits_mask) # 对角线为0，排除自身
    # 同类
    
    intra_group = (logits_mask & mask & sensitive_mask).float() # 同类同敏
    target_inter_group = ((~mask) & sensitive_mask).float() # 不同类同敏
    sensitve_inter_group = (mask & (~sensitive_mask)).float() # 同类不同敏
    tar_sens_inter_group = ((~mask) & (~sensitive_mask)).float() # 不同类不同敏
    numerator_mask = (logits_mask & mask).float()
    # print(intra_group.device)
    
    # # 这种计算二阶邻接矩阵的操作通常用于图神经网络（GNN）中，特别是在需要捕捉更远距离节点之间的关系时。
    # # 二阶邻接矩阵能够帮助模型识别图中二阶邻居的特征交互，从而增强信息的传播和聚合效果。
    eweight = torch.ones(data.edge_index.shape[1]).to(data.x.device)
    adj = torch.sparse_coo_tensor(data.edge_index, eweight, [data.x.shape[0], data.x.shape[0]])
    # 该矩阵表示图中二阶邻居（即通过两条边可达的节点）。
    # A2 = torch.spmm(adj, adj)
    # prevout = None

    # print(sum(data.y[data.train_mask] == 0))
    # print(sum(data.y[data.train_mask] == 1))
    for count in pbar:
        seed_everything(count + args.seed)
        
        encoder.reset_parameters()
        classifier.reset_parameters()
        projector.reset_parameters()
        discriminator.reset_parameters()
        generator.reset_parameters()

        for epoch in range(0, args.epochs):
            print(f"=======epoch: {epoch}=======")
            encoder.train()
            classifier.train()
            projector.eval()
            discriminator.eval()
            for _ in range(0, args.cla_epochs):
                optimizer_e.zero_grad()
                optimizer_c.zero_grad()
                feat = encoder(data.x, data.edge_index)
                output = classifier(feat)
                # loss_c = F.binary_cross_entropy_with_logits(output[data.train_mask], data.y[data.train_mask].float().unsqueeze(1)).to(args.device)
                loss_c = classify_criterion(output[data.train_mask], data.y[data.train_mask].float().unsqueeze(1)).to(args.device)
                # pred = (output[data.train_mask].squeeze() > 0.5).type_as(data.y)
                # print(sum(pred == 0), sum(data.y[data.train_mask]==0))
                # print(output[0:50])
                # if epoch > 10:
                #     exit(0)
                # acc_train = pred.eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
                # auc_rocs_train = roc_auc_score(data.y[data.train_mask].cpu().numpy(), torch.sigmoid(output[data.train_mask]).detach().cpu().numpy())
                # print(f"acc: {acc_train}, acc_rocs: {auc_rocs_train}")
                # print(f"classify: {loss_c.item()}")
                loss_c_item = loss_c.item()
                # print(f"classify: {loss_c_item}")
                loss_c.backward()
                optimizer_e.step()
                optimizer_c.step()
            
            # 数据集中的敏感信息是完全可以获取的
            # 敏感信息其实挺多的，最难的其实是标签信息，标签信息的比例只有100个。
            # train discriminator to recognize the sensitive group / make group close
            # just training the discriminator
            encoder.eval()
            projector.eval()
            generator.eval()
            discriminator.train()
            for _ in range(0, args.dic_epochs):
                optimizer_d.zero_grad()
                h = encoder(data.x, data.edge_index)
                output = discriminator(h)
                # the discriminator should fitting the args sensitive label
                # it means the mixup up node should have the truth lable
                loss_d = discriminate_criterion(output.view(-1), data.x[:, args.sens_idx])
                loss_d.backward()
                # mprint(f"\t discriminator: {loss_d.item()}", end = "")
                optimizer_d.step()
            discriminator.eval()
            
            # """
            # 虽然在 args.f_mask == 'no' 的情况下没有显式生成器来修改 data.x，但编码器在这里可以被看作生成器的一部分。
            # 生成器通过修改 特征嵌入（embeddings） 而非输入数据，来进行对抗学习。
            # 因此，生成器的功能通过编码器实现，它试图生成难以被判别器识别的特征表示。
            # """
            # adversarial learning
            # train generator to fool discriminator
            
            generator.train()
            encoder.train()
            discriminator.eval()
            for _ in range(0, args.g_epochs):
                optimizer_g.zero_grad()
                optimizer_e.zero_grad()
                
                if (args.f_mask):
                    loss_g = 0
                    feature_weights = generator()
                    for _ in range(args.K):
                        mask = F.gumbel_softmax(feature_weights, tau = 1, hard = False)[:, 0]
                        adv_x = data.x * mask
                        h = encoder(adv_x, data.edge_index)
                        output = discriminator(h)
                        # fool discriminator to output 0.5 (uncertainty)
                        # make mask more like to be 1, to keep most features
                        loss_g += discriminate_criterion(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + args.ratio * F.mse_loss(mask.view(-1), torch.ones_like(mask.view(-1)))
                    loss_g = loss_g / args.K
                else:
                    h = encoder(data.x, data.edge_index)
                    output = discriminator(h)
                    loss_g = discriminate_criterion(output.view(-1), 0.5 * torch.ones_like(output.view(-1)))
                loss_g.backward()
                # print(f"\t adversarial: {loss_g.item()}", end = "")
                optimizer_g.step()
                optimizer_e.step()
            
            # add contrastive learning method
            encoder.train()
            projector.train()
            discriminator.eval()
            generator.eval()
            classifier.eval()
            for _ in range(0, args.con_epochs):
                optimizer_e.zero_grad()
                optimizer_p.zero_grad()
                feat = projector(encoder(data.x, data.edge_index))
                anchor_dot_contrast = torch.div(torch.matmul(feat[data.train_mask], feat[data.train_mask].T), args.tau)
                # 根据是否有归一化，如果有归一化，其实可以取消
                # need comparasion
                #logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                logits = anchor_dot_contrast # - logits_max.detach() # [b, b]

                exp_logits_fair = torch.exp(logits) * target_inter_group # 其实这里不会有自身的，因为不同类同敏已经将自身排除了
                exp_logits_sum = exp_logits_fair.sum(1, keepdim=True)
                log_prob = logits - torch.log(exp_logits_sum + (exp_logits_sum == 0) * 1)
                # [b]
                # 同类同敏分子 [b, b]
                mean_log_prob = (numerator_mask * log_prob).sum(1) / numerator_mask.sum(1)
                loss_e = -((t_idx_s0_y1 * mean_log_prob).sum() / num_t_s0_y1 + (t_idx_s1_y1 * mean_log_prob).sum() / num_t_s1_y1 + \
                    (t_idx_s0_y0 * mean_log_prob).sum() / num_t_s0_y0 + (t_idx_s1_y0 * mean_log_prob).sum()/ num_t_s1_y0)
                loss_e.backward()
                # mprint(f"\t contrastive: {loss_e.item()}")
                optimizer_e.step()
                optimizer_p.step()
            encoder.eval()
            projector.eval()
            
            # 没啥用，对最后结果不影响
            encoder.train()
            for _ in range(0, args.clo_epochs):
                optimizer_e.zero_grad()
                h = encoder(data.x, data.edge_index)[data.train_mask]
                h0 = F.normalize(h[t_idx_s0_y1])
                h1 = F.normalize(h[t_idx_s1_y1])
                h2 = F.normalize(h[t_idx_s0_y0])
                h3 = F.normalize(h[t_idx_s1_y0])
                loss_e = -torch.mean(torch.mm(h0, h1.T)) - torch.mean(torch.mm(h2, h3.T))
                loss_e.backward()
                optimizer_e.step()
            encoder.eval()

            # if epoch > args.start:
            #     graphEdit.train()
            #     encoder.eval()
            #     classifier.eval()
            #     if epoch % 10 == 0 and args.modiStru == 1:
            #         if epoch % 20 == 0:
            #             edge_index2 = graphEdit.modify_structure1(data.edge_index, adj, A2, data.sens, data.x.shape[0], args.drope_rate)
            #         else:
            #             edge_index2 = graphEdit.modify_structure2(data.edge_index, adj, A2, data.sens, data.x.shape[0], args.drope_rate)
            #     else:
            #         edge_index2 = data.edge_index
            #     for epoch_g in range(0, args.dtb_epochs):
            #         optimizer_gF.zero_grad()
            #         x2 = graphEdit(data.x)
            #         h2 = encoder(x2, edge_index2)
            #         h2 = F.normalize(h2)
            #         output2 = classifier(h2)
            #         loss_edit2 = -fair_metric2(output2[data.train_mask][:, 1], data.y[data.train_mask], t_idx_s0_y1, t_idx_s1_y1, num_t_s0_y1, num_t_s1_y1)
            #         # print(loss_edit2)
            #         loss_edit2.backward()
            #         optimizer_gF.step()

            # # shift align
            # if epoch > args.start:
            #     graphEdit.eval()
            #     encoder.train()
            #     classifier.train()
            #     x2 = graphEdit(data.x).detach()
            #     for epoch_a in range(0, args.a_epochs):
            #         optimizer_e.zero_grad()
            #         optimizer_c.zero_grad()
            #         h2 = encoder(x2, edge_index2)
            #         h1 = encoder(data.x, data.edge_index)
            #         # h2 = F.normalize(h2)
            #         # h1 = F.normalize(h1)

            #         loss_align = - (data.x.shape[0]) / (num_s0_y0)  * torch.mean(torch.mm(h1[idx_s0_y0], h2[idx_s0_y0].T)) \
            #                      - (data.x.shape[0]) / (num_s0_y1) * torch.mean(torch.mm(h1[idx_s0_y1], h2[idx_s0_y1].T)) \
            #                      - (data.x.shape[0]) / (num_s1_y0) * torch.mean(torch.mm(h1[idx_s1_y0], h2[idx_s1_y0].T)) \
            #                      - (data.x.shape[0]) / (num_s1_y1) * torch.mean(torch.mm(h1[idx_s1_y1], h2[idx_s1_y1].T))

            #         loss_align = loss_align * 0.01
            #         loss_align.backward(retain_graph=True)

            #         optimizer_e.step()
            #         optimizer_c.step()
                    
            "=====test======="
            if args.ood == 1:
                test_acc = [0 for n in range(len(args.strlist))]
                best_val_tradeoff = [0 for n in range(len(args.strlist))]
                test_auc_roc = [0 for n in range(len(args.strlist))]
                test_f1 = [0 for n in range(len(args.strlist))]
                test_parity = [0 for n in range(len(args.strlist))]
                test_equality = [0 for n in range(len(args.strlist))]
            elif args.ood == 2:
                test_acc = [0 for n in range(len(data2))]
                best_val_tradeoff = [0 for n in range(len(data2))]
                test_auc_roc = [0 for n in range(len(data2))]
                test_f1 = [0 for n in range(len(data2))]
                test_parity = [0 for n in range(len(data2))]
                test_equality = [0 for n in range(len(data2))]


            if args.ood == 2:
                for i in range(len(data2)):
                    accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged3(data2[i].x, classifier, discriminator, generator, encoder, data2[i], args)
                    if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff[i]:
                        test_acc[i] = accs['test']
                        test_auc_roc[i] = auc_rocs['test']
                        test_f1[i] = F1s['test']
                        test_parity[i], test_equality[i] = tmp_parity['test'], tmp_equality['test']
                        best_val_tradeoff[i] = auc_rocs['val'] + F1s['val'] + accs['val'] - (tmp_parity['val'] + tmp_equality['val'])
                        
            elif args.ood == 1:
                if epoch != (args.epochs - 1):
                    continue
                for i in range(len(args.strlist)):
                    datatmp, _, _, _, _, _ = get_dataset(args.dataset, args.outid + args.strlist[i], args.top_k)
                    datatmp = datatmp.to(args.device)
                    datatmp.test_mask = datatmp.test_mask | datatmp.val_mask | datatmp.test_mask
                    accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged3(datatmp.x, classifier, discriminator, generator, encoder, datatmp, args)

                    test_acc[i] = accs['test']
                    test_auc_roc[i] = auc_rocs['test']
                    test_f1[i] = F1s['test']
                    test_parity[i], test_equality[i] = tmp_parity['test'], tmp_equality['test']

            else:
                accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged3(data.x, classifier, discriminator, generator, encoder, data, args)
                if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
                    test_acc = accs['test']
                    test_auc_roc = auc_rocs['test']
                    test_f1 = F1s['test']
                    test_parity, test_equality = tmp_parity['test'], tmp_equality['test']

                    best_val_tradeoff = auc_rocs['val'] + F1s['val'] + accs['val'] - (tmp_parity['val'] + tmp_equality['val'])

            # print(test_acc[i], test_auc_roc[i], test_parity[i], test_equality[i])
        for i in range(len(args.strlist)):
            acc[count][i] = test_acc[i] * 100
            f1[count][i] = test_f1[i] * 100
            auc_roc[count][i] = test_auc_roc[i] * 100
            parity[count][i] = test_parity[i] * 100
            equality[count][i] = test_equality[i] * 100
            
    return acc, f1, auc_roc, parity, equality