import sys
sys.path.append('./2.coding_and_inference/')
import multiview_detector.utils.projection
import os
import json
from scipy.stats import multivariate_normal
from PIL import Image
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from torchvision.datasets import VisionDataset
import torch
from torchvision.transforms import ToTensor
from multiview_detector.utils.projection import *
from multiview_detector.datasets import *
import cv2


class sequenceDataset4phase2_delay(VisionDataset):
    def __init__(self, base,  tau, train=True, transform=ToTensor(), target_transform=ToTensor(),
                 reID=False, grid_reduce=4, img_reduce=4, train_ratio=0.9, force_download=True, delays=None):
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        map_sigma, map_kernel_size = 20 / grid_reduce, 20
        img_sigma, img_kernel_size = 10 / img_reduce, 10
        self.reID, self.grid_reduce, self.img_reduce = reID, grid_reduce, img_reduce

        self.tau = tau
        
        # delays is a tensor with a shape of (1, num_cam), indicating the delay for each camera.
        self.delays = [delay * 5 for delay in delays] if delays is not None else [0] * base.num_cam

        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))

        if train:
            frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            frame_range = range(0, self.num_frame)
        self.min_frame = min(frame_range)   
        
        self.img_fpaths = self.base.get_image_fpaths(frame_range) # path to the frames
        self.map_gt = {}
        self.imgs_head_foot_gt = {}
        self.download(frame_range)

        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()

        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)

        x, y = np.meshgrid(np.arange(-img_kernel_size, img_kernel_size + 1),
                           np.arange(-img_kernel_size, img_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        img_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * img_sigma)
        img_kernel = img_kernel / img_kernel.max()
        kernel_size = img_kernel.shape[0]
        self.img_kernel = torch.zeros([2, 2, kernel_size, kernel_size], requires_grad=False)
        self.img_kernel[0, 0] = torch.from_numpy(img_kernel)
        self.img_kernel[1, 1] = torch.from_numpy(img_kernel)
        pass

    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                i_s, j_s, v_s = [], [], []
                head_row_cam_s, head_col_cam_s = [[] for _ in range(self.num_cam)], \
                                                 [[] for _ in range(self.num_cam)]
                foot_row_cam_s, foot_col_cam_s, v_cam_s = [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)]
                for single_pedestrian in all_pedestrians:
                    x, y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                    if self.base.indexing == 'xy':
                        i_s.append(int(y / self.grid_reduce))
                        j_s.append(int(x / self.grid_reduce))
                    else:
                        i_s.append(int(x / self.grid_reduce))
                        j_s.append(int(y / self.grid_reduce))
                    v_s.append(single_pedestrian['personID'] + 1 if self.reID else 1)
                    for cam in range(self.num_cam):
                        x = max(min(int((single_pedestrian['views'][cam]['xmin'] +
                                         single_pedestrian['views'][cam]['xmax']) / 2), self.img_shape[1] - 1), 0)
                        y_head = max(single_pedestrian['views'][cam]['ymin'], 0)
                        y_foot = min(single_pedestrian['views'][cam]['ymax'], self.img_shape[0] - 1)
                        if x > 0 and y > 0:
                            head_row_cam_s[cam].append(y_head)
                            head_col_cam_s[cam].append(x)
                            foot_row_cam_s[cam].append(y_foot)
                            foot_col_cam_s[cam].append(x)
                            v_cam_s[cam].append(single_pedestrian['personID'] + 1 if self.reID else 1)
                occupancy_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reducedgrid_shape)
                self.map_gt[frame] = occupancy_map
                self.imgs_head_foot_gt[frame] = {}
                for cam in range(self.num_cam):
                    img_gt_head = coo_matrix((v_cam_s[cam], (head_row_cam_s[cam], head_col_cam_s[cam])),
                                             shape=self.img_shape)
                    img_gt_foot = coo_matrix((v_cam_s[cam], (foot_row_cam_s[cam], foot_col_cam_s[cam])),
                                             shape=self.img_shape)
                    self.imgs_head_foot_gt[frame][cam] = [img_gt_head, img_gt_foot]

    def __getitem__(self, index):

        sequence_imgs = []

        first_frame_name = list(self.map_gt.keys())[index]
        last_frame_name = list(self.map_gt.keys())[index + self.tau]
        
        for i in range(self.tau + 1):
            frame = list(self.map_gt.keys())[index + i]
            #print("i,frame",i,frame)
            imgs, map_gt, _, frame = self.get_single_frame_data(frame)
            sequence_imgs.append(imgs.unsqueeze(0))

        sequence_imgs = torch.cat(sequence_imgs, dim = 0)

        return sequence_imgs, map_gt, first_frame_name, last_frame_name



    def get_single_frame_data(self, frame):
        imgs = []
        for cam in range(self.num_cam):
            min_frame = self.min_frame           
            delayed_frame = max(min_frame, min(frame - self.delays[cam], 2000))
            fpath = self.img_fpaths[cam][delayed_frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        map_gt = self.map_gt[frame].toarray()
        if self.reID:
            map_gt = (map_gt > 0).int()
        if self.target_transform is not None:
            map_gt = self.target_transform(map_gt)

        imgs_gt = []
        for cam in range(self.num_cam):
            img_gt_head = self.imgs_head_foot_gt[frame][cam][0].toarray()
            img_gt_foot = self.imgs_head_foot_gt[frame][cam][1].toarray()
            img_gt = np.stack([img_gt_head, img_gt_foot], axis=2)
            if self.reID:
                img_gt = (img_gt > 0).int()
            if self.target_transform is not None:
                img_gt = self.target_transform(img_gt)
            imgs_gt.append(img_gt.float())

        return imgs, map_gt.float(), imgs_gt, frame

    # Get position of a person in the grid
    def get_person_gt_position(self,json_file_path, person_id, base):
        with open(json_file_path) as json_file:
            all_pedestrians = json.load(json_file)
        for single_pedestrian in all_pedestrians:
            if single_pedestrian['personID'] == person_id:
                grid_x, grid_y = base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                return grid_x, grid_y
        return None, None

    # Plot person positions on the grid
    def plot_person_positions_on_grid(self, json_file_path, dataset, save_path):
        with open(json_file_path) as json_file:
            all_pedestrians = json.load(json_file)

        fig, ax = plt.subplots()
        ax.set_xlim(0, 480)
        ax.set_ylim(0, 1440)
        
        for single_pedestrian in all_pedestrians:
            grid_x, grid_y = dataset.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
            ax.plot(grid_x , grid_y, 'ro')  # Mark the position as a red dot
            print(f'Person ID: {single_pedestrian["personID"]}, X: {grid_x}, Y: {grid_y}')
        
        plt.gca().invert_yaxis()  # Invert the y-axis so that the image is aligned with the image coordinates
        plt.savefig(save_path)
        plt.close(fig)

    def __len__(self):
        return len(self.map_gt.keys()) - self.tau - max(self.delays)

if __name__ == '__main__':
    json_file_path = '/Wildtrack/annotations_positions/00001835.json'
    dataset_path = '/Wildtrack'
    person_id = 177
    base = Wildtrack(dataset_path)  
    dataset = sequenceDataset4phase2_delay(base, tau=0)  # Initialize sequenceDataset4phase2_delay class
    grid_x, grid_y = dataset.get_person_gt_position(json_file_path, person_id, base)
    print(f'Person ID: {person_id}, X: {grid_x}, Y: {grid_y}')

