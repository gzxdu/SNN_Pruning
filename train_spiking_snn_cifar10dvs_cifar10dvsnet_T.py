import random
import time

import numpy as np
import torch

import utils
import config_lth
import torch.cuda.amp as amp
import torchvision
import os
import copy
import pickle
import math

from archs.cifar10dvs.DVScifar10_net import DVSCIFAR10NET


from utils_for_snn_lth_temp_lamp_2stage import *
from utils import data_transforms
from spikingjelly.activation_based import monitor,neuron,functional,surrogate
from spikingjelly.activation_based.functional import reset_net
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS


from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def find_decision(args, model, train_loader, epoch=300):
    model.eval()
    with torch.no_grad():
        candidate_t = [i for i in range(1, args.timestep+1)]
        KDdivs = []
        for temp_t in candidate_t:
            out_prob_t_temp_list = []
            targets_t_temp_list = []
            for inputs,targets in train_loader:
                inputs = inputs.cuda()
                targets = targets.cuda()
                inputs = inputs.transpose(0, 1)
                targets = F.one_hot(targets, 10).float()
                out_prob_t_temp_list.append(model(inputs[:temp_t]).mean(0))
                targets_t_temp_list.append(targets)
                reset_net(model)
            out_prob_t_temp = torch.cat(out_prob_t_temp_list, dim=0)
            targets_t_temp = torch.cat(targets_t_temp_list,dim=0)
            KDdiv = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(out_prob_t_temp, dim=1),
                                   F.softmax(targets_t_temp,dim=1))
            KDdivs.append(float(KDdiv.cpu().data.numpy()) * 1e4)
        print(KDdivs)
        threshold = 0.01
        new_timestep = args.timestep
        norm_KLdivs = ((KDdivs - np.min(KDdivs)) / (np.max(KDdivs) - np.min(KDdivs))).tolist()
        for i, n_kldiv in enumerate(norm_KLdivs):
            if n_kldiv < threshold:
                new_timestep = candidate_t[i]
                break

    print('epoch', epoch, ':', norm_KLdivs, '|Newtimesetp', new_timestep)
    return new_timestep


def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int,
                            random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


def main():
    args = config_lth.get_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    writer = SummaryWriter(f"./runs/lamp_KL_GT_NOTA_RNG/{args.dataset}_{args.arch}")
    ori_dataset = CIFAR10DVS(root='../datasets/cifar10dvs/', data_type='frame', frames_number=args.timestep,
                             split_by='number')

    train_datasets, test_datasets = split_to_train_test_set(0.9, ori_dataset, 10)
    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True, drop_last=True, num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True, drop_last=True, num_workers=4
    )
    rewinding_epoch = 0
    criterion = nn.MSELoss()
    model = DVSCIFAR10NET(channels=128, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(),
                          detach_reset=True).cuda()
    functional.set_step_mode(model, 'm')
    initial_state_dict = copy.deepcopy(model.state_dict())
    timestep = find_decision(args, model, train_loader)
    utils.checkdir(f"{os.getcwd()}/snn_laterewind_lth/lamp_KL_GT_NOTA_RNG/{args.arch}/{args.dataset}/init_rewind")
    mask = make_mask(model)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)
    all_loss = np.zeros(args.end_iter, float)
    all_accuracy = np.zeros(args.end_iter, float)
    Trace = {}

    for _ite in range(args.resume_iter, ITERATION):
        if not _ite == 0:
            rewinding_epoch = args.rewinding_epoch
            model, mask = prune_by_percentile_weight_trace(args, args.prune_percent, mask, model, Trace)
            model = original_initialization(mask, initial_state_dict, model)
            optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
        print(f"\n--- Pruning Level [round{args.round}:{_ite}/{ITERATION}]: ---")
        if args.scheduler is not None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=int(args.end_iter - rewinding_epoch),
                                                                   eta_min=0)
        comp1, param_each_layer = utils.print_nonzeros(model)
        comp[_ite] = comp1
        writer.add_scalar('global_rate', comp1, _ite)
        for key, value in param_each_layer.items():
            writer.add_scalar(key + str(value[1]), round(value[0] / value[1], 2), _ite)
        utils.checkdir(
            f"{os.getcwd()}/dumps/snn_laterewind_lth/lamp_KL_GT_NOTA_RNG/{args.arch}/{args.dataset}/itearation{_ite}")
        with open(
                f"{os.getcwd()}/dumps/snn_laterewind_lth/lamp_KL_GT_NOTA_RNG/{args.arch}/{args.dataset}/itearation{_ite}/pruned{comp1}.pkl",
                'wb') as fp:
            pickle.dump(mask, fp)
        loss = 0
        accuracy = 0
        for iter_ in range(args.resume_epoch + 1, args.end_iter - rewinding_epoch + 1):

            if (iter_) % args.valid_freq == 0 or iter_ == 1:
                accuracy = test_dvs(model, test_loader, criterion, timestep)
                writer.add_scalar('accuracy_' + f'{_ite}', accuracy, iter_ + rewinding_epoch)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.checkdir(
                        f"{os.getcwd()}/snn_laterewind_lth/lamp_KL_GT_NOTA_RNG/{args.arch}/{args.dataset}/itearation{_ite}")
                    torch.save(model,
                               f"{os.getcwd()}/snn_laterewind_lth/lamp_KL_GT_NOTA_RNG/{args.arch}/{args.dataset}/itearation{_ite}/pruned{1 - comp1}_acc{accuracy}_epoch{iter_ + rewinding_epoch}_model.pth.tar")
            loss, Trace = train(args, iter_ + rewinding_epoch, train_loader, model, criterion, optimizer, scheduler,
                                _ite, timestep)

            all_loss[iter_ - 1] = loss
            all_accuracy[iter_ - 1] = accuracy
            if _ite == 0 and iter_ == args.rewinding_epoch:
                print('find laterewinding weight--------')
                initial_state_dict = copy.deepcopy(model.state_dict())
                utils.checkdir(
                    f"{os.getcwd()}/snn_laterewind_lth/lamp_KL_GT_NOTA_RNG/{args.arch}/{args.dataset}/init_rewind")
                torch.save(initial_state_dict,
                           f"{os.getcwd()}/snn_laterewind_lth/lamp_KL_GT_NOTA_RNG/{args.arch}/{args.dataset}/init_rewind/rewind_state_dict.pth.tar")

            if (iter_) % args.print_freq == 0 or iter_ == 1:
                print(
                    f'Train Epoch: {iter_ + rewinding_epoch}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')
        bestacc[_ite] = best_accuracy
        print(bestacc)
        checkpoint = {
            "net": model.state_dict(),
            'mask': mask,
            'Trace': Trace,
            '_ite': _ite,
            'bestacc': bestacc,
            'comp': comp,
            'timestep': timestep,
            'rng':torch.get_rng_state()

        }
        utils.checkdir(f"{os.getcwd()}/snn_laterewind_lth/lamp_KL_GT_NOTA_RNG/{args.arch}/{args.dataset}/itearation{_ite}")
        torch.save(checkpoint,
                   f"{os.getcwd()}/snn_laterewind_lth/lamp_KL_GT_NOTA_RNG/{args.arch}/{args.dataset}/itearation{_ite}/ckpt.pth.tar")
        best_accuracy = 0
    writer.close()


def train(args, epoch, train_data, model, criterion, optimizer, scheduler=None, iteration=None, timestep=10):
    model.train()
    EPS = 1e-6
    train_loss = 0.0
    train_samples = 0
    for batch_idx, (imgs, labels) in enumerate(train_data):
        optimizer.zero_grad()
        imgs = imgs[:, :timestep, ...]
        imgs, labels = imgs.cuda(), labels.cuda()
        out_fr = model(imgs)
        loss = TET_loss(out_fr, labels)
        loss.backward()
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data
                if (len(tensor.size())) == 1:
                    continue
                grad_tensor = p.grad
                grad_tensor = torch.where(tensor.abs() < EPS, torch.zeros_like(grad_tensor), grad_tensor)
                p.grad.data = grad_tensor
        optimizer.step()
        reset_net(model)
        train_samples += labels.numel()
        train_loss += loss.item() * labels.numel()
    train_loss /= train_samples

    if scheduler is not None:
        scheduler.step()
    return train_loss


if __name__ == '__main__':
    main()
