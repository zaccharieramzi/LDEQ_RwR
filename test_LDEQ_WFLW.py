from pathlib import Path
import time
import argparse
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from tqdm import tqdm

from utils.helpers import *
from utils.loss_function import *
from utils.normalize import Normalize, HeatmapsToKeypoints
from datasets.WFLW_V.helpers import *
from models.ldeq import LDEQ, weights_init

heatmaps_to_keypoints = HeatmapsToKeypoints()


class DEQInference(object):

    def __init__(self, args):
        self.args = args

        ## Model
        ckpt = torch.load(args.landmark_model_weights, map_location='cpu')
        self.train_args = ckpt['args']
        self.train_args.stochastic_max_iters = False #use maximum iters at inference time so perf repeatable
        self.train_args.max_iters = args.n_forward
        self.train_args.rel_diff_target = 1e-7  # making sure we always reach the max number of iterations
        self.gpu_avail = torch.cuda.is_available()
        self.device = 'cuda' if self.gpu_avail else 'cpu'
        self.model = LDEQ(self.train_args)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.apply(weights_init)
        self.model.load_state_dict(ckpt['state_dict'], strict=False)
        self.model.eval()
        print(f'Restored weights for {self.train_args.landmark_model_name} from {self.args.landmark_model_weights}')

        ## Video stuff
        self.normalize = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def get_z0(self, batch_size):
        if self.train_args.z0_mode == 'zeros':
            return torch.zeros(batch_size, self.train_args.z_width, self.train_args.heatmap_size, self.train_args.heatmap_size, device=self.device)
        else:
            raise NotImplementedError

    def test_WFLW(self):
        """test code adapted from https://github.com/starhiking/HeatmapInHeatmap"""
        from datasets.WFLW.dataset import FaceDataset
        from torch.utils.data import DataLoader
        if args.debug_largepose:
            WFLW_splits = ["test_largepose"]
        else:
            WFLW_splits = ["test", "test_largepose", "test_expression", "test_illumination",
                           "test_makeup", "test_occlusion", "test_blur"]
        self.model.eval()
        print(f'Running inference for splits {WFLW_splits}')

        for split in WFLW_splits:
            test_dataset = FaceDataset(root_dir=args.dataset_path, split=split)
            dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

            SME = 0.0
            IONs = None
            fwd_logs = []

            with torch.no_grad():
                for data in tqdm(dataloader_test):
                    x, keypoints = data["image"], data["kpts"]
                    if self.gpu_avail:
                        x = x.cuda()
                        keypoints = keypoints.cuda()
                    output = self.model(x, mode=self.train_args.model_mode, args=self.train_args, z0=self.get_z0(x.shape[0]))
                    pred_keypoints = output['keypoints']
                    fwd_logs.append(output['fwd_logs'])

                    sum_ion, ion_list = calc_nme(pred_keypoints, keypoints)
                    SME += sum_ion
                    IONs = np.concatenate((IONs, ion_list), 0) if IONs is not None else ion_list

            nme, fr, auc = compute_fr_and_auc(IONs, thres=0.10, step=0.0001)
            rel_diff, abs_diff, mean_iters = [
                np.mean([fwd_log[key] for fwd_log in fwd_logs])
                for key in ['final_rel_diff', 'final_abs_diff', 'n_iters']
            ]

            print(f'\n------------ {split} ------------')
            print("NME %: {}".format(nme * 100))
            print("FR_{}% : {}".format(0.10, fr * 100))
            print("AUC_{}: {}".format(0.10, auc))

            if args.output_csv is not None:
                df_results = pd.DataFrame({
                    'split': [split],
                    'n_forward': [args.n_forward],
                    'nme': [nme],
                    'fr': [fr],
                    'auc': [auc],
                    'rel_diff': [rel_diff],
                    'abs_diff': [abs_diff],
                    'mean_iters': [mean_iters],
                })
                df_results.to_csv(args.output_csv, mode='a', header=not Path(args.output_csv).exists(), index=False)


##########################


def main(args):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    t0 = time.time()
    solver = DEQInference(args)
    solver.test_WFLW()

    print(f'Total time: {format_time(time.time()-t0)}')
    if torch.cuda.is_available():
        print(f'Max mem: {torch.cuda.max_memory_allocated(device="cuda") / (1024 ** 3):.1f} GB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEQ Inference')
    parser.add_argument('--landmark_model_weights', default='/home/paul/Documents/RWR_publishing/WFLW/final.pth.tar')
    parser.add_argument('--dataset_path', type=str, default="/home/paul/Datasets/Keypoints/WFLW/HIH/")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--n-forward', type=int, default=1)
    parser.add_argument('--debug-largepose', action='store_true')  # set to true to only run test for largepose split
    parser.add_argument('--output-csv', type=str, default=None)

    args = parser.parse_args()

    print('\nStarting...')

    main(args)
