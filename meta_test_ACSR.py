import argparse
import copy
import os
import pickle

import numpy as np
import torch.backends.cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import RandomSampler
import torch.nn.functional as F

from tqdm import tqdm
from model.wrn_28_10 import WideResNet
from config import PATH
from datasets.aux_dataset import FSDataset
from datasets.cub import CUB
from datasets.cifar_fs import CIFARFS
from datasets.mini_imagenet import MiniImagenet
from utils import NShotTaskSamplerFromArray, accuracy, LabelSmoothingCrossEntropy

os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='miniImagenet', choices=['miniImagenet', 'CUB', 'CIFAR-FS'])
    parser.add_argument('--arch', default='wrn_28_10', choices=['wrn_28_10', ])

    # params for optimization
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--num_workers_novel', default=2, type=int)
    parser.add_argument('--num_workers_base', default=4, type=int)
    parser.add_argument('--num_workers_task', default=2, type=int)

    parser.add_argument('--tune_mode', default='block3,linear')
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--aux_bs', default=256, type=int)
    parser.add_argument('--epochs_finetune', default=100, type=int)

    parser.add_argument('--n', default=5, type=int)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--q', default=15, type=int)

    parser.add_argument('--tasks', default=500, type=int, help='the number of evaluation tasks')

    args = parser.parse_args()

    #path to load the pre-trained model
    args.load_path = 'checkpoint/{}/{}/S2M2_R.tar'.format(args.dataset, args.arch)

    if args.dataset == 'miniImagenet':
        dataset_class = MiniImagenet
        args.image_size = 84
    elif args.dataset == 'CUB':
        dataset_class = CUB
        args.image_size = 84
    elif args.dataset == 'CIFAR-FS':
        dataset_class = CIFARFS
        args.image_size = 32
    else:
        raise (ValueError, 'Unsupported dataset')

    print(args)

    ###################
    # Create datasets #
    ###################
    taskset = dataset_class(subset='novel', state='center+id', image_size=args.image_size)

    #load pre-sampled tasks/episodes
    tasks_arr = []
    task_dir_name = os.path.join(PATH, 'checkpoint', args.dataset, 'tasks/k_{}'.format(args.k))

    for root, dirs, files in os.walk(task_dir_name):
        files = sorted(files)
        for idx, file in enumerate(files):
            if file.endswith('.pth') and idx < args.tasks:
                with open((os.path.join(root, file)), 'rb') as f:
                    one_task_index = list(pickle.load(f))
                    tasks_arr.append(one_task_index)

    print('len of taskarr', len(tasks_arr))
    sampler = NShotTaskSamplerFromArray(taskset, tasks_arr)

    taskloader = torch.utils.data.DataLoader(
        taskset,
        batch_sampler=sampler,
        pin_memory=True,
        num_workers=args.num_workers_task)

    if args.arch == 'wrn_28_10':
        model = WideResNet(num_classes=args.n)
    else:
        raise NotImplementedError

    print(model)

    store_dict = torch.load(args.load_path, map_location=torch.device('cpu'))

    store_dict_no_fc = {k.replace('module.', ''):v for k,v in store_dict.items() if ('linear' not in k)}

    print('len of valid params', len(store_dict_no_fc))
    now_dict = copy.deepcopy(model.state_dict())
    now_dict.update(store_dict_no_fc)    #now_dict is used to initial model when facing new episode

    #load or generate base set features for computing SR loss
    base_feature_path = os.path.join(PATH, 'checkpoint', args.dataset, args.arch, 'base_features_{}'.format(args.load_path.split('/')[-1]))
    if os.path.exists(base_feature_path):
        with open(base_feature_path, 'rb') as f:
            frozen_baseset_embeddings = pickle.load(f)
            frozen_baseset_embeddings = frozen_baseset_embeddings.cuda(non_blocking=True)
    else:
        model.load_state_dict(now_dict)
        model.cuda()
        model.eval()

        baseset = dataset_class(subset='base', state='center+id', image_size=args.image_size)
        baseset_loader = torch.utils.data.DataLoader(
            baseset,
            batch_size=256,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            drop_last=False)

        with torch.no_grad():
            all_embeddings = torch.zeros(size=[0, ])
            for images, _ in tqdm(baseset_loader):
                images = images.float().cuda(non_blocking=True)
                embeddings, _ = model(images)
                all_embeddings = torch.cat([all_embeddings, embeddings.detach().cpu()])

        feat_dir = os.path.join(PATH, 'checkpoint', args.dataset, args.arch)
        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)

        with open(base_feature_path, 'wb') as f:
            pickle.dump(all_embeddings, f)

        frozen_baseset_embeddings = all_embeddings.cuda(non_blocking=True)


    ############
    # Training #
    ############
    finetune_criterion = LabelSmoothingCrossEntropy(reduction='mean')

    fs_set_novel = FSDataset(num_class=args.n, samples_per_class=args.k, image_size=args.image_size)
    fs_loader_novel = torch.utils.data.DataLoader(
        fs_set_novel,
        batch_size=args.n * args.k,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers_novel)

    baseset = dataset_class(subset='base', state='center+id', image_size=args.image_size)
    baseset_loader = torch.utils.data.DataLoader(
        baseset,
        batch_size=args.aux_bs,
        sampler=RandomSampler(baseset, replacement=True),
        pin_memory=True,
        num_workers=args.num_workers_base,
        drop_last=True)
    aux_iter = iter(baseset_loader)

    accs_lcs = []
    for task_index, (images, indices) in enumerate(taskloader):
        images = images.float().cuda(non_blocking=True)

        test_images = images[args.n * args.k:]

        test_labels = torch.arange(0, args.n, 1 / args.q).long()
        test_labels = test_labels.cuda(non_blocking=True)

        train_indices = indices[:args.n * args.k]


        if task_index == 0:
            for tid in indices:
                print_path = taskset.datasetid_to_filepath[tid.item()]
                print(print_path)

        finetune_paths = []
        for cls_id, row_id in enumerate(train_indices):
                train_path = taskset.datasetid_to_filepath[row_id.item()]
                finetune_paths.append(train_path)

        assert len(finetune_paths) == args.n * args.k
        fs_set_novel.image_path = finetune_paths


        if args.arch == 'wrn_28_10':
            finetune_model = WideResNet(num_classes=args.n)
        else:
            raise NotImplementedError

        finetune_optimizer = torch.optim.SGD(finetune_model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
        finetune_model.load_state_dict(now_dict)
        finetune_model.eval()
        finetune_model.cuda()

        except_layers = args.tune_mode.split(',')

        # adaptation calibration (AC) for block3 and classification layer in wrn_28_10 or (res5+fc) for resnet.
        for name, p in finetune_model.named_parameters():
            if all(except_layer not in name for except_layer in except_layers):
                p.requires_grad = False
        if 'block3' in except_layers:
            finetune_model.bn1.weight.requires_grad = True
            finetune_model.bn1.bias.requires_grad = True
        if task_index == 0:
            for name, p in finetune_model.named_parameters():
                print('requires_grad {:40s} {}'.format(name, '          ' + str(p.requires_grad) if p.requires_grad == False else p.requires_grad))

        for name, module in finetune_model.named_modules():
            if any(except_layer in name for except_layer in except_layers):
                module.train()
        if 'block3' in except_layers:
            finetune_model.bn1.train()
        if task_index == 0:
            for n, module in finetune_model.named_modules():
                print('bn_state {:40s} {}'.format(n, '          ' + str(module.training) if module.training == False else module.training))


        for bid in range(args.epochs_finetune):
            for batch_data in fs_loader_novel:
                finetune_images, finetune_labels = batch_data

                finetune_images = finetune_images.cuda(non_blocking=True)
                finetune_labels = finetune_labels.cuda(non_blocking=True)

                try:
                    aux_data = next(aux_iter)
                except StopIteration:
                    aux_iter = iter(baseset_loader)
                    aux_data = next(aux_iter)

                aux_images, aux_indices = aux_data
                aux_images = aux_images.cuda(non_blocking=True)
                aux_indices = aux_indices.cuda(non_blocking=True)

                with torch.no_grad():
                    aux_embeddings = frozen_baseset_embeddings[aux_indices].data

                finetune_model.zero_grad()

                _, finetune_logits = finetune_model(finetune_images)
                finetune_loss = finetune_criterion(finetune_logits, finetune_labels)   #classification loss

                finetune_base_embeds, _ = finetune_model(aux_images)

                aux_loss = (F.normalize(finetune_base_embeds, dim=-1) * F.normalize(aux_embeddings, dim=-1)).sum(dim=1).mean()   #SR loss

                finetune_optimizer.zero_grad()
                loss = finetune_loss - aux_loss * args.beta
                loss.backward()

                finetune_optimizer.step()

                print('\rfinetune   epoch : {}      finetune_loss : {:.4f}      aux loss : {:.4f}'.format(bid, finetune_loss.item(), aux_loss.item()), end='')

        with torch.no_grad():
            finetune_model.eval()
            _, test_logits = finetune_model(test_images)

            acc_lcs = accuracy(test_logits, test_labels)
            accs_lcs.append(acc_lcs)
            print('\rTask : {}    finetune Loss : {:<.4f}    aux Loss : {:<.4f}     acc_lcs : {:.4f}     accm : {:.4f}    std : {:.4f}    acc1 : {:.4f}    acc10 : {:.4f}    acc100 : {:.4f}'
                  .format(task_index, finetune_loss.item(), aux_loss.item(),  acc_lcs, np.mean(accs_lcs), np.std(accs_lcs), np.min(accs_lcs), np.mean(np.sort(accs_lcs)[:10]), np.mean(np.sort(accs_lcs)[:100])))

if __name__ == '__main__':
    main()