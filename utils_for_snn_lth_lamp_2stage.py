import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from spikingjelly.activation_based.functional import reset_net
from spikingjelly.activation_based import layer


def _is_prunable_module(m):
    return (isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d) or isinstance(m,layer.Linear) or isinstance(m,layer.Conv2d))

def get_weights(model):
    weights = []
    for name,m in model.named_modules():
        if _is_prunable_module(m) and 'TA' not in name:
            print(name)
            weights.append(m.weight)
    return weights

def get_modules(model):
    modules = []
    for name,m in model.named_modules():
        if _is_prunable_module(m) and 'TA' not in name:
            print(name)
            modules.append(m)
    return modules

def _count_unmasked_weights(model):
    """
    Return a 1-dimensional tensor of #unmasked weights.
    """
    mlist = get_modules(model)
    unmaskeds = []
    for m in mlist:
        unmaskeds.append(torch.count_nonzero(m.weight))
    return torch.FloatTensor(unmaskeds)


def _compute_lamp_amounts(model, amount):
    """
    Compute normalization schemes.
    """
    unmaskeds = _count_unmasked_weights(model)
    num_surv = int(np.round(unmaskeds.sum() * (1.0 - amount)))

    flattened_scores = [_normalize_scores(w ** 2).view(-1) for w in get_weights(model)]
    concat_scores = torch.cat(flattened_scores, dim=0)
    topks, _ = torch.topk(concat_scores, num_surv) #分位数
    threshold = topks[-1]

    # We don't care much about tiebreakers, for now.
    final_survs = [torch.ge(score, threshold * torch.ones(score.size()).to(score.device)).sum() for score in
                   flattened_scores]
    amounts = []
    for idx, final_surv in enumerate(final_survs):
        amounts.append(1.0 - (final_surv / unmaskeds[idx]))

    return amounts

def _normalize_scores(scores):
    """
    Normalizing scheme for LAMP.
    """
    # sort scores in an ascending order
    sorted_scores, sorted_idx = scores.view(-1).sort(descending=False)
    # compute cumulative sum
    scores_cumsum_temp = sorted_scores.cumsum(dim=0)
    scores_cumsum = torch.zeros(scores_cumsum_temp.shape, device=scores.device)
    scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp) - 1]
    # normalize by cumulative sum
    sorted_scores /= (scores.sum() - scores_cumsum)
    # tidy up and output
    new_scores = torch.zeros(scores_cumsum.shape, device=scores.device)
    new_scores[sorted_idx] = sorted_scores

    return new_scores.view(scores.shape)



def make_mask(model):
    step = 0
    #model.named_parameters() name+param
    for name, param in model.named_parameters():
        if 'weight' in name and 'TA' not in name:
            step = step + 1
    mask = [None]* step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name and 'TA' not in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0

    return mask

# Prune by Percentile module
def prune_by_percentile_weight_trace(args, percent, mask, model,trace):
    model_step = 0
    amount_step = 0
    amounts = _compute_lamp_amounts(model,percent*0.01)
    for name, param in model.named_parameters():
        # We do not prune bias term
        if 'weight' in name and 'TA' not in name:
            tensor_re = param.data.cpu().numpy()
            if (len(tensor_re.shape)) == 1:  # We do not prune BN term
                model_step += 1
                continue
            tensor = np.abs(tensor_re)
            alive = tensor[np.nonzero(tensor)]
            percentile_value = np.sort(alive)[int(float(alive.shape[0]) / float(1. / amounts[amount_step]))]
            weight_dev = param.device
            new_mask = np.where(tensor < percentile_value, 0, mask[model_step])
            param.data = torch.from_numpy(tensor_re * new_mask).to(weight_dev)
            mask[model_step] = new_mask
            model_step+=1
            amount_step+=1
            # try:
            #
            # except:
            #     raise StopIteration
                # next(iter_trace).data.cpu().numpy()是一个(a,)的numpy的array，需要乘以tensor这个(a,b)或者(a,b,c,d)的numpy的array

    return model, mask


def get_pruning_maks(args, percent, mask, model):
    step = 0
    all_param = []
    #包含CONV 和 ACTF层
    for name, param in model.named_parameters():
        # We do not prune bias term
        if 'weight' in name and 'TA' not in name:
            tensor = param.data.cpu().numpy()
            if (len(tensor.shape)) == 1:  # We do not prune BN term
                continue
            alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
            all_param.append(list(abs(alive)))
    param_whole = np.concatenate(all_param)
    percentile_value = np.sort(param_whole)[int(float(param_whole.shape[0]) / float(100. / percent))]

    step = 0

    for name, param in model.named_parameters():
        # We do not prune bias term
        if 'weight' in name and 'TA' not in name:
            tensor = param.data.cpu().numpy()
            if (len(tensor.shape)) == 1:  # We do not prune BN term
                step += 1
                continue
            new_mask = np.where(abs(tensor) < percentile_value, 0, torch.FloatTensor([1]))#按数值大小置0
            mask[step] = new_mask
            step += 1
    step = 0

    return  mask


def original_initialization(mask_temp, initial_state_dict, model):

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name and 'TA' not in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name or 'TA' in name:
            param.data = initial_state_dict[name]
    step = 0

    return model

def original_initialization_nobias(mask_temp, initial_state_dict, model):

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name] +1

    step = 0

    return model


# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = sum(output)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            reset_net(model)

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy


def test_dvs(model, test_loader, criterion,timestep=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    test_samples = 0
    test_acc = 0
    with torch.no_grad():
        for data, label in test_loader:
            data = data[:, :timestep]
            data, label = data.to(device), label.to(device)
            data = data.transpose(0, 1)
            label_onehot = F.one_hot(label, 10).float()
            out_fr = model(data).mean(0)
            loss = F.mse_loss(out_fr, label_onehot)
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            reset_net(model)

        test_loss /= test_samples
        test_acc /= test_samples

    return test_acc*100


def test_ann(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            reset_net(model)

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy




def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose1d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)
        init.constant_(m.bias.data, 0)

