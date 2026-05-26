from dataset import *
from model import *
from utils import *
from learn import *
import argparse
from tqdm import tqdm
from torch import tensor
import warnings
warnings.filterwarnings('ignore')
import math
from constraint import Constraint_SAGE
import scipy.optimize as sopt
from torch.optim.lr_scheduler import ExponentialLR
import time

cnst_config = {
    'lr_scheduler': True,
    'continue_training': False,
    # 'with_constraint': True,
    'nit': 100,
    'criterion': 100,  # 100
    'cnst': 0.01,
    'alpha': 2.1,
    'rho': 20 # 50, 100
}
def loss_cvar(loss_vector, alpha):
  batch_size = len(loss_vector)
  n = int(alpha * batch_size)
  rk = torch.argsort(loss_vector, descending=True)
  loss = loss_vector[rk[:n]].mean()  
  return loss


def loss_chisq(loss_vector, alpha):
  max_l = 10.
  C = math.sqrt(1 + (1 / alpha - 1) ** 2)
  foo = lambda eta: C * math.sqrt((F.relu(loss_vector - eta) ** 2).mean().item()) + eta
  opt_eta = sopt.brent(foo, brack=(0, max_l))
  loss = C * torch.sqrt((F.relu(loss_vector - opt_eta) ** 2).mean()) + opt_eta
  return loss


def cvar_doro(loss_vector, alpha, eps):
  gamma = eps + alpha * (1 - eps)
  batch_size = len(loss_vector)
  n1 = int(gamma * batch_size)
  n2 = int(eps * batch_size)
  rk = torch.argsort(loss_vector, descending=True)
  loss = loss_vector[rk[n2:n1]].sum() / alpha / (batch_size - n2)  
  return loss

def chisq_doro(loss_vector, alpha, eps):
  max_l = 10.
  batch_size = len(loss_vector)
  C = math.sqrt(1 + (1 / alpha - 1) ** 2)
  n = int(eps * batch_size)
  rk = torch.argsort(loss_vector, descending=True)
  l0 = loss_vector[rk[n:]]
  foo = lambda eta: C * math.sqrt((F.relu(l0 - eta) ** 2).mean().item()) + eta
  opt_eta = sopt.brent(foo, brack=(0, max_l))
  loss = C * torch.sqrt((F.relu(l0 - opt_eta) ** 2).mean()) + opt_eta
  return loss


def run(data, args):
    pbar = tqdm(range(args.runs), unit='run')
    criterion = nn.BCELoss()
    acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)

    data = data.to(args.device)

    generator = channel_masker(args).to(args.device)
    optimizer_g = torch.optim.Adam([
        dict(params=generator.weights, weight_decay=args.g_wd)], lr=args.g_lr)

    discriminator = MLP_discriminator(args).to(args.device)
    optimizer_d = torch.optim.Adam([
        dict(params=discriminator.lin.parameters(), weight_decay=args.d_wd)], lr=args.d_lr)

    classifier = MLP_classifier(args).to(args.device)
    optimizer_c = torch.optim.Adam([
        dict(params=classifier.lin.parameters(), weight_decay=args.c_wd)], lr=args.c_lr)

    if(args.encoder == 'MLP'):
        encoder = MLP_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.lin.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
    elif(args.encoder == 'GCN'):
        if args.prop == 'scatter':
            encoder = GCN_encoder_scatter(args).to(args.device)
        else:
            encoder = GCN_encoder_spmm(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.lin.parameters(), weight_decay=args.e_wd),
            dict(params=encoder.bias, weight_decay=args.e_wd)], lr=args.e_lr)
    elif(args.encoder == 'GIN'):
        encoder = GIN_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.conv.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
    elif(args.encoder == 'SAGE'):
        neurons_per_layer= [args.num_features, args.hidden, args.hidden]
        encoder = SAGE_encoder(args, neurons_per_layer).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)  

    ResLogFile = os.path.join('logs', args.dataset) + "_ResLog.txt"
    for count in pbar:
        seed_everything(count + args.seed)
        generator.reset_parameters()
        discriminator.reset_parameters()
        classifier.reset_parameters()
        encoder.reset_parameters()

        best_val_tradeoff = 0
        cnt = 0
        best_val_loss = math.inf
        for epoch in range(0, args.epochs):
            if(args.f_mask == 'yes'):
                generator.eval()
                feature_weights, masks, = generator(), []
                for k in range(args.K):
                    mask = F.gumbel_softmax(feature_weights, tau=1, hard=False)[:, 0]
                    masks.append(mask)

            # train discriminator to recognize the sensitive group
            discriminator.train()
            encoder.train()
            for epoch_d in range(0, args.d_epochs):
                optimizer_d.zero_grad()
                optimizer_e.zero_grad()

                if(args.f_mask == 'yes'):
                    loss_d = 0

                    for k in range(args.K):
                        x = data.x * masks[k].detach()
                        # h = encoder(x, data.edge_index, data.adj_norm_sp)
                        h = encoder(x, data.edge_index)
                        output = discriminator(h)

                        loss_d += criterion(output.view(-1), data.x[:, args.sens_idx])

                    loss_d = loss_d / args.K
                else:
                    # h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                    h = encoder(data.x, data.edge_index)
                    output = discriminator(h)
                    
                    loss_d = criterion(output.view(-1), data.x[:, args.sens_idx])

                loss_d.backward()
                optimizer_d.step()
                optimizer_e.step()

            # train classifier
            classifier.train()
            encoder.train()
            for epoch_c in range(0, args.c_epochs):
                optimizer_c.zero_grad()
                optimizer_e.zero_grad()

                if(args.f_mask == 'yes'):
                    # # Save previous weights
                    # num_layers = len(encoder.gcn_stack)
                    # old_weights = [None] * num_layers
                    # for i, gcn_block in enumerate(encoder.gcn_stack):
                    #     old_weights[i] = {'lin_r': torch.zeros_like(gcn_block.lin_r.weight.data),
                    #                     'lin_l': torch.zeros_like(gcn_block.lin_l.weight.data)}                          
                    loss_c = 0
                    for k in range(args.K):
                        x = data.x * masks[k].detach()
                        # h = encoder(x, data.edge_index, data.adj_norm_sp)
                        h = encoder(x, data.edge_index)
                        output = classifier(h)
                        
                        # for i, gcn_block in enumerate(encoder.gcn_stack):
                        #         w0 = torch.clone(gcn_block.lin_r.weight.detach().data)
                        #         w1 = torch.clone(gcn_block.lin_l.weight.detach().data)
                        #         old_weights[i]['lin_r'] += w0
                        #         old_weights[i]['lin_l'] += w1

                        loss_c += F.binary_cross_entropy_with_logits(
                            output[data.train_mask], data.y[data.train_mask].unsqueeze(1).to(args.device))

                    # # Average the accumulated weights
                    # for i in range(num_layers):
                    #         old_weights[i]['lin_r'] /= args.K
                    #         old_weights[i]['lin_l'] /= args.K
                    #         old_weights[i] = [old_weights[i]['lin_r'], old_weights[i]['lin_l']]
                            
                    loss_c = loss_c / args.K

                else:
                    # h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                    h = encoder(data.x, data.edge_index)
                    output = classifier(h)

                    loss_c = F.binary_cross_entropy_with_logits(
                        output[data.train_mask], data.y[data.train_mask].unsqueeze(1).to(args.device))

                loss_c.backward()

                optimizer_e.step()
                optimizer_c.step()  
                
                # # Apply constraint
                # cnst_config['num_layers'] = len(neurons_per_layer) - 1
                # cnst_config['neurons_per_layer'] = neurons_per_layer
                # constraint = Constraint_SAGE(encoder, parameters=cnst_config, device=args.device, with_constraint=args.with_constraint)
                # if epoch >= 100:
                #     constraint.on_batch_end(old_weights)    

            # train generator to fool discriminator
            generator.train()
            encoder.train()
            discriminator.eval()
            for epoch_g in range(0, args.g_epochs):
                optimizer_g.zero_grad()
                optimizer_e.zero_grad()

                if(args.f_mask == 'yes'):
                    # Save previous weights
                    num_layers = len(encoder.gcn_stack)
                    old_weights = [None] * num_layers
                    generator_weights = None
                    for i, gcn_block in enumerate(encoder.gcn_stack):
                        old_weights[i] = {'lin_r': torch.zeros_like(gcn_block.lin_r.weight.data),
                                        'lin_l': torch.zeros_like(gcn_block.lin_l.weight.data)}
                    
                    loss_g = 0
                    loss_g1 = 0
                    loss_g2 = 0
                    feature_weights = generator()
                    for k in range(args.K):
                        mask = F.gumbel_softmax(feature_weights, tau=1, hard=False)[:, 0]

                        x = data.x * mask
                        # h = encoder(x, data.edge_index, data.adj_norm_sp)
                        h = encoder(x, data.edge_index)
                        output = discriminator(h)
                        
                        
                        generator_weights = generator.weights[:, 0]
                        for i, gcn_block in enumerate(encoder.gcn_stack):
                                w0 = torch.clone(gcn_block.lin_r.weight.detach().data)
                                w1 = torch.clone(gcn_block.lin_l.weight.detach().data)
                                old_weights[i]['lin_r'] += w0
                                old_weights[i]['lin_l'] += w1

                        loss_g1 += F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1)), reduction='none') 
                        loss_g2 += args.ratio * F.mse_loss(mask.view(-1), torch.ones_like(mask.view(-1)))
                        loss_g = F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + \
                            args.ratio * F.mse_loss(mask.view(-1), torch.ones_like(mask.view(-1))) 
                    
                    # Average the accumulated weights
                    for i in range(num_layers):
                        old_weights[i]['lin_r'] /= args.K
                        old_weights[i]['lin_l'] /= args.K
                        old_weights[i] = [old_weights[i]['lin_r'], old_weights[i]['lin_l']]
                    
                    loss_g1 = loss_g1 / args.K
                    loss_g2 = loss_g2 / args.K
                    # loss_g = loss_g / args.K
                    # solve outlier
                    loss_g = loss_chisq(loss_g1, args.loss_alpha) + loss_g2
                    # loss_g = cvar_doro(loss_g1, args.loss_alpha, args.eps) + loss_g2
                    
                    
                else:
                    # h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                    h = encoder(data.x, data.edge_index)
                    output = discriminator(h)
                    
                    loss_g = F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1)))

                loss_g.backward()

                optimizer_g.step()
                optimizer_e.step()
                
                # Apply constraint
                cnst_config['num_layers'] = len(neurons_per_layer) - 1
                cnst_config['neurons_per_layer'] = neurons_per_layer
                cnst_config['rho'] = args.rho
                constraint = Constraint_SAGE(encoder, parameters=cnst_config, device=args.device, with_constraint=args.with_constraint)
                constraint.on_batch_end(old_weights, generator_weights)    
        
            
            if(args.weight_clip == 'yes'):
                if(args.f_mask == 'yes'):
                    weights = torch.stack(masks).mean(dim=0)
                else:
                    weights = torch.ones_like(data.x[0])

                encoder.clip_parameters(weights)

            accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged3(
                data.x, classifier, discriminator, generator, encoder, data, args)

            print(epoch, 'Acc:', accs['test'], 'AUC_ROC:', auc_rocs['test'], 'F1:', F1s['test'],
                  'Parity:', tmp_parity['test'], 'Equality:', tmp_equality['test'])
            with open(ResLogFile, "a+") as file:
                    now = time.localtime()
                    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S ", now)
                    strLog = formatted_time + "epoch {} Test -- ACC: {:.4f} AUC_ROC:{:.4f} F1:{:.4f} DP:{:.4f} EO:{:.4f}".format(epoch, accs['test'], auc_rocs['test'], F1s['test'], tmp_parity['test'], tmp_equality['test']) 
                    file.write(strLog +"\n")
            # if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
            #     test_acc = accs['test']
            #     test_auc_roc = auc_rocs['test']
            #     test_f1 = F1s['test']
            #     test_parity, test_equality = tmp_parity['test'], tmp_equality['test']

            #     best_val_tradeoff = auc_rocs['val'] + F1s['val'] + accs['val'] - (tmp_parity['val'] + tmp_equality['val'])


            # visualize mask weights
            # ResLogFile1 = os.path.join('logs', args.dataset) + "_ResLog_mask-origin.txt"
            # with open(ResLogFile1, "a+") as file:
            #     now = time.localtime()
            #     formatted_time = time.strftime("%Y-%m-%d %H:%M:%S ", now)
            #     strLog = formatted_time + str(generator.weights[:, 0])
            #     file.write(strLog +"\n") 
            
            # visualize gnn encoder weights
            # import seaborn as sns
            # sns.heatmap(encoder.gcn_stack[1].lin_r.weight.detach().data[3:13,3:13].cpu().detach().numpy(), annot=True, cmap='YlGnBu') 
            # plt.savefig("./maps/ct-heatmap-" + str(epoch) + ".png")
            # # plt.show()
            # plt.clf()  # 清除当前图形      
            
            # Early Stopping
            cur_val_tradeoff = auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val'])
            # cur_val_tradeoff = auc_rocs['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val'])
            if cur_val_tradeoff <= best_val_tradeoff:
                cnt += epoch >= args.pretrain
            else:
                cnt = (cnt + int(epoch >= args.pretrain)) if accs['test'] == 1.0 else 0
                test_acc = accs['test']
                test_auc_roc = auc_rocs['test']
                test_f1 = F1s['test']
                test_parity, test_equality = tmp_parity['test'], tmp_equality['test']

                best_val_tradeoff = auc_rocs['val'] + F1s['val'] + accs['val'] - (tmp_parity['val'] + tmp_equality['val'])

             

            if epoch >= args.pretrain and cnt >= args.early_stopping:
                print("Early Stopping")
                print('=====TEST=====', epoch)
                print('Acc:', test_acc, 'AUC_ROC:', test_auc_roc, 'F1:', test_f1,
                      'Parity:', test_parity, 'Equality:', test_equality)
                break
            
            
                # print('=====VALIDATION=====', epoch, epoch_g)
                # print('Utility:', auc_rocs['val'] + F1s['val'] + accs['val'],
                #       'Fairness:', tmp_parity['val'] + tmp_equality['val'])

                # print('=====VALIDATION-BEST=====', epoch, epoch_g)
                # print('Utility:', args.best_val_model_utility,
                #       'Fairness:', args.best_val_fair)

                # print('=====TEST=====', epoch)
                # print('Acc:', test_acc, 'AUC_ROC:', test_auc_roc, 'F1:', test_f1,
                #       'Parity:', test_parity, 'Equality:', test_equality)

                # print('=====epoch:{}====='.format(epoch))
                # print('sens_acc:', (((output.view(-1) > 0.5) & (data.x[:, args.sens_idx] == 1)).sum() + ((output.view(-1) < 0.5) &
                #                                                                                          (data.x[:, args.sens_idx] == 0)).sum()).item() / len(data.y))

        if args.with_constraint:
            theta_bar = get_Lips_constant_upper(encoder)
            print("######get_Lips_constant_upper", theta_bar)
        else:
            theta_bar = get_Lips_constant(encoder)
            print("######get_Lips_constant", theta_bar)
        
        acc[count] = test_acc
        f1[count] = test_f1
        auc_roc[count] = test_auc_roc
        parity[count] = test_parity
        equality[count] = test_equality

        # print('auc_roc:', np.mean(auc_roc[:(count + 1)]))
        # print('f1:', np.mean(f1[:(count + 1)]))
        # print('acc:', np.mean(acc[:(count + 1)]))
        # print('Statistical parity:', np.mean(parity[:(count + 1)]))
        # print('Equal Opportunity:', np.mean(equality[:(count + 1)]))
        with open(ResLogFile, "a+") as file:
                now = time.localtime()
                formatted_time = time.strftime("%Y-%m-%d %H:%M:%S ", now)
                strLog = formatted_time + "best Test -- ACC: {:.4f} AUC_ROC:{:.4f} F1:{:.4f} DP:{:.4f} EO:{:.4f}".format(acc[count], auc_roc[count], f1[count], parity[count], equality[count]) 
                file.write(strLog +"\n")

    return acc, f1, auc_roc, parity, equality


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='german')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--d_epochs', type=int, default=5)
    parser.add_argument('--g_epochs', type=int, default=5)
    parser.add_argument('--c_epochs', type=int, default=5)
    parser.add_argument('--g_lr', type=float, default=0.001)
    parser.add_argument('--g_wd', type=float, default=0)
    parser.add_argument('--d_lr', type=float, default=0.001)
    parser.add_argument('--d_wd', type=float, default=0)
    parser.add_argument('--c_lr', type=float, default=0.001)
    parser.add_argument('--c_wd', type=float, default=0)
    parser.add_argument('--e_lr', type=float, default=0.001)
    parser.add_argument('--e_wd', type=float, default=0)
    parser.add_argument('--early_stopping', type=int, default=5)
    parser.add_argument('--prop', type=str, default='scatter')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='GIN')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--clip_e', type=float, default=0.1)
    parser.add_argument('--f_mask', type=str, default='yes')
    parser.add_argument('--weight_clip', type=str, default='no')
    parser.add_argument('--ratio', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--with_constraint', action='store_true', default=False)
    parser.add_argument('--pretrain', type=int, default=200)
    parser.add_argument('--loss_alpha', type=float, default=1)
    parser.add_argument('--eps', type=float, default=0)
    parser.add_argument('--rho', type=float, default=100)
    

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, args.sens_idx, args.corr_sens, args.corr_idx, args.x_min, args.x_max = get_dataset(args.dataset, args.top_k)
    args.num_features, args.num_classes = data.x.shape[1], 1

    # print((data.y == 1).sum(), (data.y == 0).sum())
    # print((data.y[data.train_mask] == 1).sum(),
    #       (data.y[data.train_mask] == 0).sum())
    # print((data.y[data.val_mask] == 1).sum(),
    #       (data.y[data.val_mask] == 0).sum())
    # print((data.y[data.test_mask] == 1).sum(),
    #       (data.y[data.test_mask] == 0).sum())

    args.train_ratio, args.val_ratio = torch.tensor([
        (data.y[data.train_mask] == 0).sum(), (data.y[data.train_mask] == 1).sum()]), torch.tensor([
            (data.y[data.val_mask] == 0).sum(), (data.y[data.val_mask] == 1).sum()])
    args.train_ratio, args.val_ratio = torch.max(
        args.train_ratio) / args.train_ratio, torch.max(args.val_ratio) / args.val_ratio
    args.train_ratio, args.val_ratio = args.train_ratio[
        data.y[data.train_mask].long()], args.val_ratio[data.y[data.val_mask].long()]

    # print(args.val_ratio, data.y[data.val_mask])

    acc, f1, auc_roc, parity, equality = run(data, args)
    print('======' + args.dataset + args.encoder + '======')
    print('auc_roc:', np.mean(auc_roc) * 100, np.std(auc_roc) * 100)
    print('Acc:', np.mean(acc) * 100, np.std(acc) * 100)
    print('f1:', np.mean(f1) * 100, np.std(f1) * 100)
    print('parity:', np.mean(parity) * 100, np.std(parity) * 100)
    print('equality:', np.mean(equality) * 100, np.std(equality) * 100)
    ResLogFile = os.path.join('logs', args.dataset) + "_" + str(args.with_constraint) +  "_" + str(args.loss_alpha) +  "_" + str(args.eps) + "_ResLog.txt"
    with open(ResLogFile, "a+") as file:
        now = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S ", now)
        strLog = formatted_time + "final Test -- ACC: {:.2f}-+-{:.2f} AUC_ROC:{:.2f}-+-{:.2f} F1:{:.2f}-+-{:.2f} DP:{:.2f}-+-{:.2f} EO:{:.2f}-+-{:.2f}".format(
            np.mean(acc) * 100, np.std(acc) * 100,
            np.mean(auc_roc) * 100, np.std(auc_roc) * 100,
            np.mean(f1) * 100,  np.std(f1) * 100,
            np.mean(parity) * 100, np.std(parity) * 100, 
            np.mean(equality) * 100, np.std(equality) * 100) 
        file.write(strLog +"\n")
