import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class ResBase34(nn.Module):
    def __init__(self):
        super(ResBase34, self).__init__()
        model_resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResBase50(nn.Module):
    def __init__(self):
        super(ResBase50, self).__init__()
        model_resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.in_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResBase101(nn.Module):
    def __init__(self):
        super(ResBase101, self).__init__()
        model_resnet101 = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool
        self.in_features = model_resnet101.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class MLPBase(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.5):
        super(MLPBase, self).__init__()
        self.in_features = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)

class ResClassifier(nn.Module):
    def __init__(self, class_num, feature_dim, bottleneck_dim=256):
        super(ResClassifier, self).__init__()
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.bottleneck(x)
        y = self.fc(x)
        return x, y


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size, max_iter=10000):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = max_iter

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]



IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(root, label):
    images = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(' ')
        if is_image_file(data[0]):
            path = os.path.join(root, data[0])
        gt = int(data[1])
        item = (path, gt)
        images.append(item)
    return images

class ObjectImage(torch.utils.data.Dataset):
    def __init__(self, root, path, transform=None, y=None, ridx=False, loader=default_loader):
        imgs = make_dataset(root, path)
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.y = y
        self.ridx = ridx
        self.loader = loader
        
    def __getitem__(self, index):
        if self.y == None:
            path, target = self.imgs[index]
        else:
            path, _ = self.imgs[index]
            target = self.y[index]
        img = self.loader(path)
        if self.transform is not None:
            if type(self.transform).__name__=='list':
                    img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)
        if self.ridx:
            return img, target, index
        else:
            return img, target

    def __len__(self):
        return len(self.imgs)


class ObjectTabular(torch.utils.data.Dataset):
    def __init__(self, path, feature_names=None, mean=None, std=None, y=None, ridx=False,
                 label_name='RECID', id_name='user_id', sensitive_name='WHITE',
                 return_sensitive=False, excluded_feature_names=None,
                 label_mapping=None, label_positive_threshold=None,
                 sensitive_mapping=None, invalid_label_values=None):
        table = pd.read_csv(path)
        columns = list(table.columns)
        excluded_feature_names = set(excluded_feature_names or ())

        if label_name not in columns:
            raise ValueError('{} must contain the {} label column'.format(path, label_name))
        if feature_names is None:
            feature_names = [
                name for name in columns
                if name not in {label_name, id_name, sensitive_name}
                and name not in excluded_feature_names
            ]
        forbidden_features = {label_name, id_name, sensitive_name} | excluded_feature_names
        forbidden_used = forbidden_features.intersection(feature_names)
        if forbidden_used:
            raise ValueError(
                '{} cannot be used as model features'.format(sorted(forbidden_used)))

        missing = [name for name in feature_names if name not in columns]
        if missing:
            raise ValueError('{} is missing feature columns: {}'.format(path, missing))

        features = table.loc[:, feature_names].to_numpy(dtype=np.float32)
        self.features = torch.from_numpy(features)
        if mean is not None and std is not None:
            self.features = (self.features - mean) / std

        label_values = table[label_name]
        invalid_values = set(invalid_label_values or ())
        self.valid_mask = torch.from_numpy(
            (~label_values.isin(invalid_values)).to_numpy(dtype=np.bool_)
        )
        if label_mapping is not None:
            label_values = label_values.map(label_mapping)
        elif label_positive_threshold is not None:
            label_values = (label_values > label_positive_threshold).astype(np.int64)
        if label_values.isnull().any():
            raise ValueError('{} contains unmapped {} values'.format(path, label_name))
        self.labels = torch.from_numpy(label_values.to_numpy(dtype=np.int64))
        self.sensitive = None
        if return_sensitive:
            if sensitive_name not in columns:
                raise ValueError('{} must contain the {} sensitive column'.format(path, sensitive_name))
            sensitive_values = table[sensitive_name]
            if sensitive_mapping is not None:
                sensitive_values = sensitive_values.map(sensitive_mapping)
            if sensitive_values.isnull().any():
                raise ValueError(
                    '{} contains unmapped {} values'.format(path, sensitive_name))
            self.sensitive = torch.from_numpy(
                sensitive_values.to_numpy(dtype=np.int64)
            )
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)
        self.ridx = ridx
        self.return_sensitive = return_sensitive
        self.feature_names = list(feature_names)
        self.in_features = self.features.size(1)

    def standardize(self, mean, std):
        self.features = (self.features - mean) / std

    def __getitem__(self, index):
        target = self.labels[index] if self.y is None else self.y[index]
        if self.ridx:
            return self.features[index], target, index
        if self.return_sensitive:
            return self.features[index], target, self.sensitive[index]
        return self.features[index], target

    def __len__(self):
        return self.features.size(0)


class ObjectSynthetic(torch.utils.data.Dataset):
    def __init__(self, feature_path, label_path, sensitive_path=None,
                 mean=None, std=None, y=None, ridx=False,
                 return_sensitive=False):
        feature_table = pd.read_csv(feature_path, header=None)
        features = feature_table.to_numpy(dtype=np.float32)
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError('{} contains an empty feature matrix'.format(feature_path))
        if not np.isfinite(features).all():
            raise ValueError('{} contains NaN or infinite feature values'.format(feature_path))

        label_table = pd.read_csv(label_path, header=None)
        if label_table.shape[1] != 1:
            raise ValueError('{} must contain exactly one label column'.format(label_path))
        labels = label_table.iloc[:, 0].to_numpy(dtype=np.int64)
        if labels.shape[0] != features.shape[0]:
            raise ValueError(
                '{} has {} feature rows, but {} has {} labels'.format(
                    feature_path, features.shape[0], label_path, labels.shape[0]))
        if not np.isin(labels, (0, 1)).all():
            raise ValueError('{} labels must contain only 0 and 1'.format(label_path))

        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)
        self.valid_mask = torch.ones(self.features.size(0), dtype=torch.bool)
        self.in_features = self.features.size(1)
        self.feature_names = [
            'feature_{}'.format(index) for index in range(self.in_features)
        ]
        if mean is not None and std is not None:
            self.standardize(mean, std)

        self.sensitive = None
        if return_sensitive:
            if sensitive_path is None:
                raise ValueError('sensitive_path is required for fairness evaluation')
            sensitive_table = pd.read_csv(sensitive_path, header=None)
            if sensitive_table.shape[1] != 1:
                raise ValueError(
                    '{} must contain exactly one sensitive column'.format(
                        sensitive_path))
            sensitive = sensitive_table.iloc[:, 0].to_numpy(dtype=np.int64)
            if sensitive.shape[0] != features.shape[0]:
                raise ValueError(
                    '{} has {} feature rows, but {} has {} sensitive values'.format(
                        feature_path, features.shape[0],
                        sensitive_path, sensitive.shape[0]))
            if not np.isin(sensitive, (0, 1)).all():
                raise ValueError(
                    '{} sensitive values must contain only 0 and 1'.format(
                        sensitive_path))
            self.sensitive = torch.from_numpy(sensitive)

        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)
        if self.y is not None and self.y.numel() != self.features.size(0):
            raise ValueError(
                'Replacement labels contain {} values, expected {}'.format(
                    self.y.numel(), self.features.size(0)))
        self.ridx = ridx
        self.return_sensitive = return_sensitive

    def standardize(self, mean, std):
        if mean.numel() != self.in_features or std.numel() != self.in_features:
            raise ValueError(
                'Expected {} normalization values, received mean={} and std={}'.format(
                    self.in_features, mean.numel(), std.numel()))
        self.features = (self.features - mean) / std

    def __getitem__(self, index):
        target = self.labels[index] if self.y is None else self.y[index]
        if self.ridx:
            return self.features[index], target, index
        if self.return_sensitive:
            return self.features[index], target, self.sensitive[index]
        return self.features[index], target

    def __len__(self):
        return self.features.size(0)


def common_tabular_features(source_path, target_path, label_name,
                            sensitive_name, id_name='user_id',
                            excluded_feature_names=None):
    source_columns = pd.read_csv(source_path, nrows=0).columns
    target_columns = pd.read_csv(target_path, nrows=0).columns
    source_column_set = set(source_columns)
    excluded = {
        label_name,
        sensitive_name,
        id_name,
        *(excluded_feature_names or ())
    }
    feature_names = [
        name for name in target_columns
        if name in source_column_set and name not in excluded
    ]
    if not feature_names:
        raise ValueError(
            '{} and {} do not contain any common model feature columns'.format(
                source_path, target_path))
    return feature_names


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def encode_ys_groups(y, sensitive):
    """Encode predicted Y and sensitive S into the visualization's 0--3 groups."""
    y_array = _as_numpy(y).astype(np.int64).reshape(-1)
    sensitive_array = _as_numpy(sensitive).astype(np.int64).reshape(-1)
    if y_array.shape[0] != sensitive_array.shape[0]:
        raise ValueError(
            'Predicted labels and sensitive values must have the same length')
    if not np.isin(y_array, (0, 1)).all():
        raise ValueError('Predicted labels must contain only 0 and 1')
    if not np.isin(sensitive_array, (0, 1)).all():
        raise ValueError('Sensitive values must contain only 0 and 1')

    groups = np.full(y_array.shape[0], -1, dtype=np.int64)
    groups[(y_array == 1) & (sensitive_array == 0)] = 0
    groups[(y_array == 1) & (sensitive_array == 1)] = 1
    groups[(y_array == 0) & (sensitive_array == 0)] = 2
    groups[(y_array == 0) & (sensitive_array == 1)] = 3
    return groups


def save_visualization_embeddings(output_dir, representations, predicted_y,
                                  sensitive, valid_mask=None):
    """Save one seed's target representations in the zyt visualization format."""
    embeddings = _as_numpy(representations)
    if embeddings.ndim != 2:
        raise ValueError(
            'representations must have shape [num_nodes, feature_dim]')
    if not np.isfinite(embeddings).all():
        raise ValueError('representations contain NaN or Inf')

    predicted_y = _as_numpy(predicted_y).astype(np.int64).reshape(-1)
    sensitive = _as_numpy(sensitive).astype(np.int64).reshape(-1)
    if embeddings.shape[0] != predicted_y.shape[0]:
        raise ValueError(
            'representations and predicted labels length mismatch: {} vs {}'.format(
                embeddings.shape[0], predicted_y.shape[0]))
    if embeddings.shape[0] != sensitive.shape[0]:
        raise ValueError(
            'representations and sensitive values length mismatch: {} vs {}'.format(
                embeddings.shape[0], sensitive.shape[0]))

    if valid_mask is not None:
        valid_mask = _as_numpy(valid_mask).astype(bool).reshape(-1)
        if valid_mask.shape[0] != embeddings.shape[0]:
            raise ValueError(
                'valid_mask length mismatch: {} vs {}'.format(
                    valid_mask.shape[0], embeddings.shape[0]))
        embeddings = embeddings[valid_mask]
        predicted_y = predicted_y[valid_mask]
        sensitive = sensitive[valid_mask]

    if embeddings.shape[0] == 0:
        raise ValueError('No valid target nodes remain for visualization export')
    labels = encode_ys_groups(predicted_y, sensitive)
    if labels.shape[0] != embeddings.shape[0]:
        raise ValueError('Visualization representations and labels must align')
    if not np.isin(labels, (0, 1, 2, 3)).all():
        raise ValueError('Visualization labels must contain only 0, 1, 2, and 3')

    os.makedirs(output_dir, exist_ok=True)
    feat_path = os.path.join(output_dir, 'feat.npz')
    labels_path = os.path.join(output_dir, 'labels.npz')
    np.savez_compressed(feat_path, representations=embeddings)
    np.savez_compressed(labels_path, labels=labels)
    return feat_path, labels_path


def print_args(args):
    log_str = ("==========================================\n")
    log_str += ("==========       config      =============\n")
    log_str += ("==========================================\n")
    for arg, content in args.__dict__.items():
        log_str += ("{}:{}\n".format(arg, content))
    log_str += ("\n==========================================\n")
    print(log_str)
    args.out_file.write(log_str+'\n')
    args.out_file.flush()

def cal_fea(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            inputs, labels = next(iter_test)
            inputs = inputs.cuda()
            feas, outputs = model(inputs)
            if start_test:
                all_feas = feas.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_feas = torch.cat((all_feas, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    return all_feas, all_label

def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = idx_s0 & (labels == 1)
    idx_s1_y1 = idx_s1 & (labels == 1)

    def positive_rate(mask):
        if not torch.any(mask).item():
            return float('nan')
        return pred[mask].float().mean().item()

    parity = abs(positive_rate(idx_s0) - positive_rate(idx_s1))
    equality = abs(positive_rate(idx_s0_y1) - positive_rate(idx_s1_y1))
    return parity, equality


def cal_roc_auc(scores, labels):
    if scores.dim() != 2 or scores.size(1) != 2:
        raise ValueError('ROC-AUC requires class probabilities with shape [N, 2]')

    labels_np = labels.detach().long().cpu().numpy()
    positive_scores_np = scores[:, 1].detach().cpu().numpy()
    if np.unique(labels_np).size < 2:
        return float('nan')
    return float(roc_auc_score(labels_np, positive_scores_np))


def metric_mean_std(metric_history, expected_count=None):
    metric_names = ('accuracy', 'roc_auc', 'parity', 'equality')
    missing = [name for name in metric_names if name not in metric_history]
    if missing:
        raise ValueError('Missing metric histories: {}'.format(missing))

    lengths = {len(metric_history[name]) for name in metric_names}
    if len(lengths) != 1:
        raise ValueError('Metric histories must have the same number of values')
    test_count = next(iter(lengths))
    if test_count == 0:
        raise ValueError('Metric histories cannot be empty')
    if expected_count is not None and test_count != expected_count:
        raise ValueError(
            'Expected {} test results, but received {}'.format(expected_count, test_count))

    summary = {}
    for name in metric_names:
        values = np.asarray(metric_history[name], dtype=np.float64)
        summary[name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=0))
        }
    return summary


def cal_acc(loader, model, flag=True, fc=None, return_fairness=False,
            return_features=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            if return_fairness:
                sensitive = data[2]
            inputs = inputs.cuda()
            if flag:
                feas, outputs = model(inputs)
            else:
                if fc is not None:
                    feas, outputs = model(inputs)
                    outputs = fc(feas)
                else:
                    feas = None
                    outputs = model(inputs)
            if return_features and feas is None:
                raise ValueError('return_features requires a model that returns features')
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                if return_fairness:
                    all_sensitive = sensitive.long()
                if return_features:
                    all_features = feas.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                if return_fairness:
                    all_sensitive = torch.cat((all_sensitive, sensitive.long()), 0)
                if return_features:
                    all_features = torch.cat((all_features, feas.float().cpu()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if return_fairness:
        roc_auc = cal_roc_auc(all_output, all_label)
        parity, equality = fair_metric(predict, all_label, all_sensitive)
        if return_features:
            return (accuracy, predict, all_output, all_label, roc_auc,
                    parity, equality, all_features)
        return accuracy, predict, all_output, all_label, roc_auc, parity, equality
    if return_features:
        return accuracy, predict, all_output, all_label, all_features
    return accuracy, predict, all_output, all_label

def cal_acc_visda(loader, model, flag=True, fc=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    feas, outputs = model(inputs)
                    outputs = fc(feas)
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    aacc = acc.mean() / 100
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)
    print(acc)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    print(accuracy)
    return aacc, predict, all_output, all_label, acc

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
    
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, max_epochs=30, lambda_u=75):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, lambda_u, linear_rampup(epoch, max_epochs)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]
