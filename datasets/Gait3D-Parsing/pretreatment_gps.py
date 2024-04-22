# This source is based on https://github.com/AbnerHqC/GaitSet/blob/master/pretreatment.py
import argparse
import logging
import multiprocessing as mp
import os
import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

ORG_KEYPOINTS = {
    'nose'          :0,
    'neck'      :1,
    'right_shoulder'     :2,
    'right_elbow'      :3,
    'right_wrist'     :4,
    'left_shoulder' :5,
    'left_elbow':6,
    'left_wrist'    :7,
    'MidHip'   :8,
    'right_hip'    :9,
    'right_knee'   :10,
    'right_ankle'      :11,
    'left_hip'     :12,
    'left_knee'     :13,
    'left_ankle'    :14,
    'REye'    :15,
    'LEye'   :16,
    'REar'   :17,
    'LEar'   :18,
    'LBigToe'   :19,
    'LSmallToe'   :20,
    'LHeel'   :21,
    'RBigToe'   :22,
    'RSmallToe'   :23,
    'RHeel'   :24,

}

NEW_KEYPOINTS = {
    0: 'right_shoulder',
    1: 'right_elbow',
    2: 'right_knee',
    3: 'right_hip',
    4: 'left_elbow',
    5: 'left_knee',
    6: 'left_shoulder',
    7: 'right_wrist',
    8: 'right_ankle',
    9: 'left_hip',
    10: 'left_wrist',
    11: 'left_ankle',
}

def get_index_mapping():
    index_mapping = {}
    for _key in NEW_KEYPOINTS.keys():
        map_index = ORG_KEYPOINTS[NEW_KEYPOINTS[_key]]
        index_mapping[_key] = map_index
    return index_mapping
def imgs2pickle(img_groups: Tuple, output_path: Path, img_size: int = 64, verbose: bool = False, parsing: bool = False, dataset='CASIAB') -> None:
    """Reads a group of images and saves the data in pickle format.

    Args:
        img_groups (Tuple): Tuple of (sid, seq, view) and list of image paths.
        output_path (Path): Output path.
        img_size (int, optional): Image resizing size. Defaults to 64.
        verbose (bool, optional): Display debug info. Defaults to False.
    """
    index_mapping = get_index_mapping()
    sinfo = img_groups[0]
    img_paths = img_groups[1]
    to_pickle = []
    merge_seq = []
    for img_file in sorted(img_paths):
        if verbose:
            logging.debug(f'Reading sid {sinfo[0]}, seq {sinfo[1]}, view {sinfo[2]} from {img_file}')

        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        
        if dataset == 'GREW':
            to_pickle.append(img.astype('uint8'))
            continue

        if parsing:
            img_sil = (img>0).astype('uint8') * 255
        else:
            img_sil = img
        if img_sil.sum() <= 10000:
            if verbose:
                logging.debug(f'Image sum: {img_sil.sum()}')
            logging.warning(f'{img_file} has no data.')
            continue
        # Get the upper and lower points
        y_sum = img_sil.sum(axis=1)
        y_top = (y_sum != 0).argmax(axis=0)
        y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)
        img = img[y_top: y_btm + 1, :]
        img_sil = img_sil[y_top: y_btm + 1, :]

        # As the height of a person is larger than the width,
        # use the height to calculate resize ratio.
        ratio = img.shape[1] / img.shape[0]
        ratio_sil = img_sil.shape[1] / img_sil.shape[0]
        assert ratio == ratio_sil
        if parsing:
            img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_NEAREST)
            img_sil = cv2.resize(img_sil, (int(img_size * ratio), img_size), interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)
            img_sil = cv2.resize(img_sil, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)

        # Get the median of the x-axis and take it as the person's x-center.
        x_csum = img_sil.sum(axis=0).cumsum()
        x_center = None
        for idx, csum in enumerate(x_csum):
            if csum > img_sil.sum() / 2:
                x_center = idx
                break

        if not x_center:
            logging.warning(f'{img_file} has no center.')
            continue
        # 这里同样的对smpl数据相关帧进行加载，根据后缀名进行匹配
        # 分割路径，获取目录和文件名（包括后缀）
        file_dir, file_name_with_extension = os.path.split(img_file)

        # 进一步分割文件名，获取文件名和原始后缀
        file_name, original_extension = os.path.splitext(file_name_with_extension)

        # 新的文件扩展名
        new_extension = '.npz'

        # 构建新的文件名和新的文件路径
        new_file_name = file_name + new_extension
        smpl_path = os.path.join(r"E:\BAIDU\3d_smpl\3D_SMPLs", *sinfo, new_file_name)
        try:
            with open(smpl_path, 'rb') as f:
                data = np.load(smpl_path, allow_pickle=True)['results'][()]
                # print("shape",data.shape)
                # 获取3d关键点
                data = data[0]["j3d_op25"]
                # print(data)
                # print(data[0]["j3d_op25"])
                # 这里对数据进行重写映射
                mapped_keypoints = np.zeros((12, 3))
                for i in range(mapped_keypoints.shape[0]):
                    mapped_keypoints[i] = data[index_mapping[i]]
                merge_seq.append(mapped_keypoints)
        except Exception as e:
            print("Error: ", e)
        # Get the left and right points
        half_width = img_size // 2
        left = x_center - half_width
        right = x_center + half_width
        if left <= 0 or right >= img.shape[1]:
            left += half_width
            right += half_width
            _ = np.zeros((img.shape[0], half_width))
            img = np.concatenate([_, img, _], axis=1)

        to_pickle.append(img[:, left: right].astype('uint8'))

    if to_pickle:
        to_pickle = np.asarray(to_pickle)
        dst_path = os.path.join(output_path, *sinfo)
        # print(img_paths[0].as_posix().split('/'),img_paths[0].as_posix().split('/')[-5])
        # dst_path = os.path.join(output_path, img_paths[0].as_posix().split('/')[-5], *sinfo) if dataset == 'GREW' else dst
        os.makedirs(dst_path, exist_ok=True)
        pkl_path = os.path.join(dst_path, f'{sinfo[2]}.pkl')
        if verbose:
            logging.debug(f'Saving {pkl_path}...')
        pickle.dump(to_pickle, open(pkl_path, 'wb'))   
        logging.info(f'Saved {len(to_pickle)} valid frames to {pkl_path}.')
    if merge_seq:
        to_pickle = np.asarray(merge_seq)
        dst_path = os.path.join("E:\BAIDU\Gait3D-Parsing\skeleton_4000", *sinfo)
        # print(img_paths[0].as_posix().split('/'),img_paths[0].as_posix().split('/')[-5])
        # dst_path = os.path.join(output_path, img_paths[0].as_posix().split('/')[-5], *sinfo) if dataset == 'GREW' else dst
        os.makedirs(dst_path, exist_ok=True)
        pkl_path = os.path.join(dst_path, f'{sinfo[2]}.pkl')
        if verbose:
            logging.debug(f'Saving {pkl_path}...')
        pickle.dump(to_pickle, open(pkl_path, 'wb'))
        logging.info(f'Saved {len(to_pickle)} valid frames to {pkl_path}.')


    if len(to_pickle) < 5:
        logging.warning(f'{sinfo} has less than 5 valid data.')



def pretreat(input_path: Path, output_path: Path, img_size: int = 64, workers: int = 4, verbose: bool = False, parsing: bool = False, dataset: str = 'CASIAB') -> None:
    """Reads a dataset and saves the data in pickle format.

    Args:
        input_path (Path): Dataset root path.
        output_path (Path): Output path.
        img_size (int, optional): Image resizing size. Defaults to 64.
        workers (int, optional): Number of thread workers. Defaults to 4.
        verbose (bool, optional): Display debug info. Defaults to False.
    """
    img_groups = defaultdict(list)
    logging.info(f'Listing {input_path}')
    total_files = 0
    for img_path in input_path.rglob('*.png'):
        if 'gei.png' in img_path.as_posix():
            continue
        if verbose:
            logging.debug(f'Adding {img_path}')
        *_, sid, seq, view, _ = img_path.as_posix().split('/')
        img_groups[(sid, seq, view)].append(img_path)
        total_files += 1

    logging.info(f'Total files listed: {total_files}')

    progress = tqdm(total=len(img_groups), desc='Pretreating', unit='folder')

    with mp.Pool(workers) as pool:
        logging.info(f'Start pretreating {input_path}')
        for _ in pool.imap_unordered(partial(imgs2pickle, output_path=output_path, img_size=img_size, verbose=verbose, parsing=parsing, dataset=dataset), img_groups.items()):
            progress.update(1)
    logging.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenGait dataset pretreatment module.')
    parser.add_argument('-i', '--input_path', default='', type=str, help='Root path of raw dataset.')
    parser.add_argument('-o', '--output_path', default='', type=str, help='Output path of pickled dataset.')
    parser.add_argument('-l', '--log_file', default='./pretreatment.log', type=str, help='Log file path. Default: ./pretreatment.log')
    parser.add_argument('-n', '--n_workers', default=4, type=int, help='Number of thread workers. Default: 4')
    parser.add_argument('-r', '--img_size', default=64, type=int, help='Image resizing size. Default 64')
    parser.add_argument('-d', '--dataset', default='CASIAB', type=str, help='Dataset for pretreatment.')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Display debug info.')
    parser.add_argument('-p', '--parsing', default=False, action='store_true', help='Display debug info.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.log_file, filemode='w', format='[%(asctime)s - %(levelname)s]: %(message)s')
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info('Verbose mode is on.')
        for k, v in args.__dict__.items():
            logging.debug(f'{k}: {v}')

    print(f"parsing: {args.parsing}")
    pretreat(input_path=Path(args.input_path), output_path=Path(args.output_path), img_size=args.img_size, workers=args.n_workers, verbose=args.verbose, parsing=args.parsing, dataset=args.dataset)
