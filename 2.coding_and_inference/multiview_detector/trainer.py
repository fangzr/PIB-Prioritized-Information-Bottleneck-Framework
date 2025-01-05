import time
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from multiview_detector.utils.image_utils import add_heatmap_to_image
from multiview_detector.models.WeightsGenerator import WeightsGenerator
from multiview_detector.models.WeightsGenerator import calculate_loss


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, criterion, logdir, denormalize, cls_thres=0.4):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.beta = 1e-3 #beta
        self.weights_generator = WeightsGenerator()
        self.calculate_loss = calculate_loss

    def train(self, epoch, data_loader, optimizer, delays, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        gt_losses = 0 
        bits_losses = 0

        precision_s, recall_s = AverageMeter(), AverageMeter()
        
        # Generate camera_weights from the input delays and apply them to the calculation of bits_loss
        delays = torch.tensor(delays, dtype=torch.float32).to(device)
        delays = delays.unsqueeze(0)  # reshape delays from (7,)reshape to (1, 7)
        camera_weights = delays * 20 + 0.1
        
        for batch_idx, (data, map_gt, _, _) in enumerate(data_loader):

            optimizer.zero_grad()
            map_res, bits_loss = self.model(data)
            loss = 0
            gt_loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel)
            
            
            # camera_weights = self.weights_generator(delays)
            weighted_bits_loss = bits_loss * camera_weights
            
            gt_loss = gt_loss.mean()
            weighted_bits_loss = weighted_bits_loss.mean()

            loss = gt_loss + weighted_bits_loss * self.beta 

            loss.backward()
            optimizer.step()


            losses += loss.item()
            gt_losses += gt_loss.item()
            bits_losses += bits_loss.item()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                print('Epoch: {}, batch: {}, loss: {:.6f}, gt_losses: {:.6f}, communication cost: {:.2f} KB'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), gt_losses / (batch_idx + 1), bits_losses/(batch_idx + 1)))
                print(f"camera_weights: {camera_weights}") 

        return losses / len(data_loader), precision_s.avg * 100

    def test(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False, valid_cam_indices=None):
        print("res_fpath", res_fpath)
        print("gt_fpath", gt_fpath)
        self.model.eval()
        losses = 0
        bits_losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        output_map_res_statistic = 0
        if res_fpath is not None:
            assert gt_fpath is not None
        for batch_idx, (data, map_gt, _, frame) in enumerate(data_loader):
            with torch.no_grad():
                map_res, bits_loss = self.model(data, use_ucb=False, valid_cam_indices=valid_cam_indices, is_train=False)
            if res_fpath is not None:
                map_grid_res = map_res.detach().cpu().squeeze()
                v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)

                grid_ij = (map_grid_res > self.cls_thres).nonzero()
                if data_loader.dataset.base.indexing == 'xy':
                    grid_xy = grid_ij[:, [1, 0]]
                else:
                    grid_xy = grid_ij
                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                            data_loader.dataset.grid_reduce, v_s], dim=1))

            loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel)

            output_map_res_statistic += torch.sum(map_res)

            losses += loss.item()
            bits_losses += bits_loss.item()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

        moda = 0

        print('test gt losses', losses, 'statistic', output_map_res_statistic)

        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, 20, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                    data_loader.dataset.base.__name__)

            print('moda: {:.2f}%, modp: {:.2f}%, precision: {:.2f}%, recall: {:.2f}%'.
                format(moda, modp, precision, recall))

        print('Communication cost: {:.2f} KB'.format(bits_losses / (len(data_loader))))

        return losses / len(data_loader), precision_s.avg * 100, moda, bits_losses / (len(data_loader))

    def ucb_baseline(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False):
        print("res_fpath", res_fpath)
        print("gt_fpath", gt_fpath)
        self.model.eval()
        losses = 0
        bits_losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        output_map_res_statistic = 0
        if res_fpath is not None:
            assert gt_fpath is not None

        
        for batch_idx, (data, map_gt, _, frame) in enumerate(data_loader):

            with torch.no_grad():
                map_res, bits_loss = self.model(data, use_ucb=False, valid_cam_indices=None)
            
            if res_fpath is not None:
                map_grid_res = map_res.detach().cpu().squeeze()
                v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)

                grid_ij = (map_grid_res > self.cls_thres).nonzero()
                if data_loader.dataset.base.indexing == 'xy':
                    grid_xy = grid_ij[:, [1, 0]]
                else:
                    grid_xy = grid_ij
                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                               data_loader.dataset.grid_reduce, v_s], dim=1))

            loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel)
            output_map_res_statistic += torch.sum(map_res)
            losses += loss.item()
            # print("bits_loss: ", bits_loss)
            bits_losses += bits_loss.item()

            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

        moda = 0

        print('test gt losses', losses, 'statistic', output_map_res_statistic)

        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, 20, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                     data_loader.dataset.base.__name__)

            print('moda: {:.2f}%, modp: {:.2f}%, precision: {:.2f}%, recall: {:.2f}%'.
                  format(moda, modp, precision, recall))

        print('Communication cost: {:.2f} KB'.format(bits_losses / (len(data_loader))))

        return losses / len(data_loader), precision_s.avg * 100, moda, bits_losses / (len(data_loader)), modp

    def test_ucb(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False, ucb=None, input_baseline=None,valid_cam_indices=None):
        print("res_fpath", res_fpath)
        print("gt_fpath", gt_fpath)
        self.model.eval()
        losses = 0
        bits_losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        output_map_res_statistic = 0
        if res_fpath is not None:
            assert gt_fpath is not None

        if input_baseline is None:
            raise ValueError("input_baseline must be provided.")
        
        for batch_idx, (data, map_gt, _, frame) in enumerate(data_loader):
            if ucb is not None:#UCB gating
                # valid_cam_indices = ucb.select_arms()
                valid_cam_indices = valid_cam_indices
            else: #No UCB gating, get threshold about loss
                valid_cam_indices = [0, 1, 2, 4, 5]

            print("valid_cam_indices: ", valid_cam_indices)
            with torch.no_grad():
                map_res, bits_loss = self.model(data, use_ucb=True, valid_cam_indices=valid_cam_indices, is_train=False)
            
            if res_fpath is not None:
                map_grid_res = map_res.detach().cpu().squeeze()
                v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)

                grid_ij = (map_grid_res > self.cls_thres).nonzero()
                if data_loader.dataset.base.indexing == 'xy':
                    grid_xy = grid_ij[:, [1, 0]]
                else:
                    grid_xy = grid_ij
                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                               data_loader.dataset.grid_reduce, v_s], dim=1))

            loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel)
            output_map_res_statistic += torch.sum(map_res)
            losses += loss.item()
            bits_losses += bits_loss.item()

            # print("loss.item(): ", loss.item())
            # print("bits_loss: ", bits_loss)

            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

            # print("precision: ", precision)

            # Update UCB with the current batch's bits_loss

            # if ucb is not None:#UCB gating         
            #     # reward = ucb.calculate_reward(bits_loss.item(), loss.item(), input_baseline)
            #     print("Temp moda: ", moda)
            #     reward = ucb.calculate_reward(bits_loss.item(), moda, 70)
            #     ucb.update(valid_cam_indices, reward)
            # else:
            #     pass

        moda = 0

        print('test gt losses', losses, 'statistic', output_map_res_statistic)

        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, 20, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                     data_loader.dataset.base.__name__)

            print('moda: {:.2f}%, modp: {:.2f}%, precision: {:.2f}%, recall: {:.2f}%'.
                  format(moda, modp, precision, recall))

        print('Communication cost: {:.2f} KB'.format(bits_losses / (len(data_loader))))

        return losses / len(data_loader), precision_s.avg * 100, moda, bits_losses / (len(data_loader)), modp


class BBOXTrainer(BaseTrainer):
    def __init__(self, model, criterion, cls_thres):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres

    