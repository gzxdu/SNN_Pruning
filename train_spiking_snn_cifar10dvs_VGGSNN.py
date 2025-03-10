import numpy as np
import torch
import utils
import config_lth
import torch.cuda.amp as amp
from torchvision import transforms
import torchvision
import os
import copy
import pickle
import math
from torch.amp import autocast, GradScaler
from archs.cifar10dvs.VGGSNN import VGGSNN
from utils_for_snn_lth_lamp_VGGSNN import *
from utils import data_transforms
from spikingjelly.activation_based import monitor,neuron,functional,surrogate
from spikingjelly.activation_based.functional import reset_net
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
import random
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Augment:
    def __init__(self):
        pass

    class Cutout:
        """Randomly mask out one or more patches from an image.
        Args:
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        """
        def __init__(self, ratio):
            self.ratio = ratio

        def __call__(self, img):
            h = img.size(1)
            w = img.size(2)
            lenth_h = int(self.ratio * h)
            lenth_w = int(self.ratio * w)
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - lenth_h // 2, 0, h)
            y2 = np.clip(y + lenth_h // 2, 0, h)
            x1 = np.clip(x - lenth_w // 2, 0, w)
            x2 = np.clip(x + lenth_w // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask
            return img

    class Roll:
        def __init__(self, off):
            self.off = off

        def __call__(self, img):
            off1 = random.randint(-self.off, self.off)
            off2 = random.randint(-self.off, self.off)
            return torch.roll(img, shifts=(off1, off2), dims=(1, 2))

    def function_nda(self, data, M=1, N=2):
        c = 15 * N
        rotate = transforms.RandomRotation(degrees=c)
        e = N / 6
        cutout = self.Cutout(ratio=e)
        a = N * 2 + 1
        roll = self.Roll(off=a)

        transforms_list = [roll, rotate, cutout]
        sampled_ops = np.random.choice(transforms_list, M)
        for op in sampled_ops:
            data = op(data)
        return data

    def trans(self, data):
        flip = random.random() > 0.5
        if flip:
            data = torch.flip(data, dims=(2, ))
        data = self.function_nda(data)
        return data

    def __call__(self, img):
        return self.trans(img)


def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
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

class DVStransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = torch.from_numpy(img).float()
        shape = [img.shape[0], img.shape[1]]
        img = img.flatten(0, 1)
        img = self.transform(img)
        shape.extend(img.shape[1:])
        return img.view(shape)


class DatasetSplitter(torch.utils.data.Dataset):
    '''To split CIFAR10DVS into training dataset and test dataset'''
    def __init__(self, parent_dataset, rate=0.1, train=True):

        self.parent_dataset = parent_dataset
        self.rate = rate
        self.train = train
        self.it_of_original = len(parent_dataset) // 10
        self.it_of_split = int(self.it_of_original * rate)

    def __len__(self):
        return int(len(self.parent_dataset) * self.rate)

    def __getitem__(self, index):
        base = (index // self.it_of_split) * self.it_of_original
        off = index % self.it_of_split
        if not self.train:
            off = self.it_of_original - off - 1
        item = self.parent_dataset[base + off]

        return item

class DatasetWarpper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.trasnform = transform

    def __getitem__(self, index):
        return self.trasnform(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)
    
def load_data(dataset_dir,T: int, distributed=False):

    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
    transform_train = DVStransform(transform=transforms.Compose([
        transforms.Resize(size=(48, 48), antialias=True),
        Augment()]))
    transform_test = DVStransform(transform=transforms.Resize(size=(48, 48), antialias=True))
    dataset = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=T, split_by='number')
    dataset, dataset_test = DatasetSplitter(dataset, 0.9,
                                            True), DatasetSplitter(dataset, 0.1, False)

    dataset_train = DatasetWarpper(dataset, transform_train)
    dataset_test = DatasetWarpper(dataset_test, transform_test)


    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    return dataset_train, dataset_test, train_sampler, test_sampler



def main():
    args = config_lth.get_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    writer = SummaryWriter(f"./runs/lamp_VGGSNN_adam/{args.dataset}_{args.arch}")

    dataset_train, dataset_test, train_sampler, test_sampler = load_data(
        args.data_dir,args.timestep)



    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler,
        pin_memory=True, drop_last=False, num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler,
        pin_memory=True, drop_last=False, num_workers=4
    )
    rewinding_epoch = 0
    criterion = nn.CrossEntropyLoss()
    model = VGGSNN(spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True).cuda()
    functional.set_backend(model, 'cupy')
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{os.getcwd()}/snn_laterewind_lth/lamp_VGGSNN_adam/{args.arch}/{args.dataset}/init_rewind")
    torch.save(model.state_dict(), f"{os.getcwd()}/snn_laterewind_lth/lamp_VGGSNN_adam/{args.arch}/{args.dataset}/init_rewind/initial_state_dict.pth.tar")
    mask = make_mask(model)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)
    all_loss = np.zeros(args.end_iter, float)
    all_accuracy = np.zeros(args.end_iter, float)
    Trace={}
    if args.resume_iter!= 0:
        ckpt = torch.load(f"{os.getcwd()}/snn_laterewind_lth/lamp_VGGSNN_adam/{args.arch}/{args.dataset}/itearation{args.resume_iter-1}/ckpt.pth.tar")
        model = VGGSNN(spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True).cuda()
        functional.set_backend(model, 'cupy')
        model.load_state_dict(ckpt['net'])
        mask = ckpt['mask']
        comp = ckpt['comp']
        _ite = ckpt['_ite']
        initial_state_dict = torch.load(f"{os.getcwd()}/snn_laterewind_lth/lamp_VGGSNN_adam/{args.arch}/{args.dataset}/init_rewind/rewind_state_dict.pth.tar")

    for _ite in range(args.resume_iter,ITERATION):
        if not _ite == 0:
            rewinding_epoch = args.rewinding_epoch
            model, mask = prune_by_percentile_weight_trace(args, args.prune_percent, mask , model, Trace)
            model = original_initialization(mask, initial_state_dict, model)
            optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
        print(f"\n--- Pruning Level [round{args.round}:{_ite}/{ITERATION}]: ---")
        if args.scheduler is not None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.end_iter - rewinding_epoch),
                                                               eta_min=0)
        comp1,param_each_layer = utils.print_nonzeros(model)
        comp[_ite] = comp1
        writer.add_scalar('global_rate', comp1, _ite)
        for key,value in param_each_layer.items():
            writer.add_scalar(key+str(value[1]),round(value[0]/value[1],2),_ite)
        utils.checkdir(f"{os.getcwd()}/dumps/snn_laterewind_lth/lamp_VGGSNN_adam/{args.arch}/{args.dataset}/itearation{_ite}")
        with open(
                f"{os.getcwd()}/dumps/snn_laterewind_lth/lamp_VGGSNN_adam/{args.arch}/{args.dataset}/itearation{_ite}/pruned{comp1}.pkl",
                'wb') as fp:
            pickle.dump(mask, fp)
        loss = 0
        accuracy =0
        for iter_ in range(args.resume_epoch+1,args.end_iter-rewinding_epoch+1):
            if (iter_) % args.valid_freq == 0 or iter_ ==1:
                accuracy = test_dvs(model, test_loader, criterion)
                writer.add_scalar('accuracy_'+f'{_ite}', accuracy, iter_+rewinding_epoch)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.checkdir(f"{os.getcwd()}/snn_laterewind_lth/lamp_VGGSNN_adam/{args.arch}/{args.dataset}/itearation{_ite}")
                    torch.save(model,
                               f"{os.getcwd()}/snn_laterewind_lth/lamp_VGGSNN_adam/{args.arch}/{args.dataset}/itearation{_ite}/pruned{1-comp1}_acc{accuracy}_epoch{iter_+rewinding_epoch}_model.pth.tar")

            loss = train(args, iter_+rewinding_epoch, train_loader, model, criterion, optimizer,scheduler,_ite)

            all_loss[iter_-1] = loss
            all_accuracy[iter_-1] = accuracy
            if _ite == 0 and iter_ == args.rewinding_epoch:
                print ('find laterewinding weight--------')
                initial_state_dict = copy.deepcopy(model.state_dict())
                utils.checkdir(f"{os.getcwd()}/snn_laterewind_lth/lamp_VGGSNN_adam/{args.arch}/{args.dataset}/init_rewind")
                torch.save(initial_state_dict,
                           f"{os.getcwd()}/snn_laterewind_lth/lamp_VGGSNN_adam/{args.arch}/{args.dataset}/init_rewind/rewind_state_dict.pth.tar")

            if (iter_) % args.print_freq == 0 or iter_ ==1:
                print(
                    f'Train Epoch: {iter_+rewinding_epoch}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')
        bestacc[_ite]=best_accuracy
        print(bestacc)
        checkpoint = {
            "net": model.state_dict(),
            'mask': mask,
            '_ite': _ite,
            'bestacc': bestacc,
            'comp': comp
        }
        utils.checkdir(f"{os.getcwd()}/snn_laterewind_lth/lamp_VGGSNN_adam/{args.arch}/{args.dataset}/itearation{_ite}")
        torch.save(checkpoint,
                   f"{os.getcwd()}/snn_laterewind_lth/lamp_VGGSNN_adam/{args.arch}/{args.dataset}/itearation{_ite}/ckpt.pth.tar")
        best_accuracy = 0
    writer.close()

def TET_loss(outputs, labels, criterion=nn.CrossEntropyLoss(), means=1.0, lamb=1e-3):
    outputs = outputs.permute(1, 0, 2)
    T = outputs.size(1)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[:, t, ...], labels)
    Loss_es = Loss_es / T
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y)
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd

def train(args, epoch, train_data,  model, criterion, optimizer, scheduler=None,iteration=None):
    model.train()
    EPS = 1e-6
    train_loss = 0.0
    train_samples = 0
    for batch_idx, (imgs, labels) in enumerate(train_data):
        optimizer.zero_grad()
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
