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
from multiview_detector.datasets.sequenceDataset4phase2_delay import sequenceDataset4phase2_delay
from multiview_detector.models.UCB import UCB
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize, ])
    data_path = os.path.expanduser(args.dataset_path)
    base = Wildtrack(data_path)

    tau = max(args.tau_1, args.tau_2)

    # Load dataset with delay
    train_set = sequenceDataset4phase2_delay(base, tau=tau, train=True, transform=train_trans, grid_reduce=4, delays=args.delays)
    test_set = sequenceDataset4phase2_delay(base, tau=tau, train=False, transform=train_trans, grid_reduce=4, delays=args.delays)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
                                                                                  

    # model
    model = PerspTransDetector(train_set, args)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)

    criterion = GaussianMSE().to(device)

    logdir = f'logs/' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

    os.makedirs(logdir, exist_ok=True)
    copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector')
    for script in os.listdir('.'):
        if script.split('.')[-1] == 'py':
            dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
    sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )

    print('Settings: \n', vars(args))

    trainer = PerspectiveTrainer(model, criterion, logdir, denormalize, args.cls_thres)

    max_MODA = 0
    minimum_bits_loss = 2e6

    # Initial training and testing without UCB gating
    for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
        print('Training...')
        train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, args.delays, args.log_interval, scheduler)
        print('Testing...')
        test_loss, test_prec, moda, avg_bit_loss = trainer.test(test_loader, os.path.join(logdir, 'test.txt'),
                                                                train_set.gt_fpath, True, valid_cam_indices=None)

        if minimum_bits_loss > avg_bit_loss:
            minimum_bits_loss = avg_bit_loss

        if moda > max_MODA:
            max_MODA = moda
        print("maximum_MODA is {:.2f} %".format(max_MODA), "minimum_bits_loss {:.2f} KB".format(minimum_bits_loss))

        # Save the model
        model_save_path = os.path.join(logdir, f'model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_save_path)

    # Calculate baseline_loss, provide to UCB
    print('Calculating baseline modp...')
    baseline_cam_indices = list(range(args.num_cams))
    baseline_loss, baseline_precision, baseline_moda, baseline_bits_loss, baseline_modp = trainer.ucb_baseline(
        test_loader, res_fpath=os.path.join(logdir, 'baseline.txt'), gt_fpath=train_set.gt_fpath, visualize=True
    )

    # print("Baseline loss: {:.2f}".format(baseline_loss), "Baseline precision: {:.2f}".format(baseline_precision), "Baseline moda: {:.2f}".format(baseline_moda), "Baseline bits_loss: {:.2f}".format(baseline_bits_loss), "Baseline modp: {:.2f}".format(baseline_modp))

    max_MODA = 0
    minimum_bits_loss = 2e6
    input_threshold = 0.80

    print("Baseline baseline_loss: {:.2f}".format(baseline_loss))

    # Initialize UCB
    ucb_in = UCB(args.num_cams, alpha=1.0, input_threshold=0.80, penalty=-100, delay_threshold=0.5, exploration_rate=0.1)

    # UCB training loop
    for epoch in tqdm.tqdm(range(1, args.ucb_epochs + 1)):
        print('Testing with UCB gating...')
        valid_cam_indices = ucb_in.select_arms(np.array(args.delays))  # Get the list of valid camera indices using UCB
        test_loss, test_prec, moda, avg_bit_loss, modp = trainer.test_ucb(test_loader, os.path.join(logdir, 'test_ucb.txt'),
                                                                    train_set.gt_fpath, True, ucb=ucb_in, input_baseline=baseline_precision*0.01,valid_cam_indices=valid_cam_indices)
        reward = ucb_in.calculate_reward(avg_bit_loss, moda, baseline_moda)
        ucb_in.update(valid_cam_indices, reward)
        if moda >= baseline_moda * 0.8:
            if minimum_bits_loss > avg_bit_loss:
                minimum_bits_loss = avg_bit_loss
                print("When we get minimum_bits_loss = {:.2f} KB; moda = {:.2f} %; valid_cam_indices = {}".format(minimum_bits_loss, moda, valid_cam_indices))

            if moda > max_MODA:
                max_MODA = moda
                print("When we get maximum_MODA = {:.2f} %; avg_bit_loss = {:.2f} KB; valid_cam_indices = {}".format(max_MODA, avg_bit_loss, valid_cam_indices))

    print("Final maximum_MODA is {:.2f} %".format(max_MODA))
    print("Final minimum_bits_loss is {:.2f} KB".format(minimum_bits_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('-j', '--num_workers', type=int, default=80)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='Training epoch (default: 10)')
    parser.add_argument('--ucb_epochs', type=int, default=10, metavar='N', help='UCB training epoch (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')
    parser.add_argument('--tau_1', type=int, default=0) # for fusion model
    parser.add_argument('--tau_2', type=int, default=1) # for temporal entropy module
    parser.add_argument('--dataset_path', type=str, default='/data1/fangzr/Research/PIB/Data/Wildtrack')
    parser.add_argument('--model_path', type=str, default="")
    
    parser.add_argument('--delays', nargs='+', type=int, default=[0, 0, 0, 0, 0, 0, 0], help='Delay values for different cameras')
    parser.add_argument('--num_cams', type=int, default=7, help='Number of cameras')

    args = parser.parse_args()
    print('delay:', parser.parse_args().delays)
    main(args)
    print('Settings:')
    print(vars(args))
