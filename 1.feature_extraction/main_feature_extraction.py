import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from multiview_detector.datasets import *
from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.utils.channel import Channel
from multiview_detector.trainer import PerspectiveTrainer
from multiview_detector.datasets import sequenceDataset_delay
from multiview_detector.datasets import sequenceDataset
    

def main(args):
    global device  # 声明 device 为全局变量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize, ])
    data_path = os.path.expanduser(args.dataset_path)
    base = Wildtrack(data_path)
    # train_set = sequenceDataset(base, tau = args.tau1, train=True, transform=train_trans, grid_reduce=4)
    train_set = sequenceDataset_delay(base, tau = args.tau1, train=True, transform=train_trans, grid_reduce=4, delays=args.delays)
    # test_set是被delay的数据集
    test_set = sequenceDataset_delay(base, tau=args.tau1, train=False, transform=train_trans, grid_reduce=4, delays=args.delays)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    
    # num_workers=args.num_workers, pin_memory=True分别表示使用多少个进程来加载数据和是否将数据保存在内存中 
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    criterion = GaussianMSE().to(device) # 暂时没搞懂这个criterion是干嘛的

    # logging
    logdir = f'logs_feature_extraction/' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(logdir, exist_ok=True) # 创建文件夹
    copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector') 

    # Iterate over all files in the current directory
    for script in os.listdir('.'):
        # Check if the file has a .py extension
        if script.split('.')[-1] == 'py':
            # Create the destination file path by joining the log directory, 'scripts' folder, and the base name of the script
            dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
            # Copy the script file to the destination file path
            shutil.copyfile(script, dst_file)

    # Redirect the standard output to a log file in the log directory
    sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )


    print('Settings: \n', vars(args))

    # train部分
    
    model = PerspTransDetector(train_set, args)
    trainer = PerspectiveTrainer(model, criterion, logdir, denormalize, args.cls_thres, args.beta)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    optimizer = optim.SGD([
        {'params': model.parameters()},
        {'params': trainer.weights_generator.parameters()}
    ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)
    max_MODA = 0


    for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
        print('Training...')
        train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, args.delays, args.log_interval, scheduler)
        print('Testing...')
        test_loss, test_prec, moda = trainer.test(test_loader, os.path.join(logdir, 'test.txt'),
                                                  train_set.gt_fpath, True)

        if moda > max_MODA:
            max_MODA = moda
            torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))
        #print("maximum_MODA is ", max_MODA)
        print("maximum_MODA is {:.2f} %".format(max_MODA))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=1e-5, help='Beta in equation (3)')
    parser.add_argument('-j', '--num_workers', type=int, default=8)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='Training epoch')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')
    parser.add_argument('--target_rate', type=float, default= 250,  help='(KB)')
    parser.add_argument('--tau1', type=int, default=0)
    parser.add_argument('--dataset_path', type=str, default='../Wildtrack_dataset')
    parser.add_argument('--delays', nargs='+', type=int, default=[0, 0, 0, 0, 0, 0, 0], help='Delay values for different cameras')

    
    args = parser.parse_args()
    
    #delays是根据信道情况计算出来的
    # args.delays = Channel.calculate_delays()
    args.tau = args.tau1
    main(args)
