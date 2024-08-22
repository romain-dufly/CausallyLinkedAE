
from torch.utils.data import Dataset
import numpy as np
import os
from scipy import interpolate
from einops import rearrange
import json
import csv
import torch
import pickle
from pathlib import Path
import torchvision.transforms as transforms
from collections import defaultdict

import albumentations
from PIL import Image

def lmx_image_transforms(image_array, size=256, random_crop=False):
    rescaler = albumentations.SmallestMaxSize(max_size=size)
    if not random_crop:
        cropper = albumentations.CenterCrop(height=size, width=size)
    else:
        cropper = albumentations.RandomCrop(height=size, width=size)
    preprocessor = albumentations.Compose([rescaler,cropper])

    image = Image.fromarray(image_array)
    if not image.mode=="RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = preprocessor(image=image)['image']
    image = (image/127.5 - 1.0).astype(np.float32)
    return image

def identity(x):
    return x

def create_THINGS_dataset(patch_size=16, fmri_transform=identity, image_transform=identity, restriction=60):
    with open('THINGS_data.pkl', 'rb') as f:
        data = pickle.load(f)
        res_train, im_train, cat_train, res_test, im_test, cat_test = data[0], data[1], data[2], data[3], data[4], data[5]
    res_train = (res_train-0.009)/0.065 # normalize
    res_test = (res_test-0.009)/0.065
    res_train = res_train[:,:patch_size*(res_train.shape[1]//patch_size)]

    ### TO FIX ASYNC ISSUE, temporary test set restriction
    if restriction is None:
        restriction = 1000
    res_test = res_test[:restriction,:patch_size*(res_test.shape[1]//patch_size)]
    im_test = im_test[:restriction]
    cat_test = cat_test[:restriction]

    if isinstance(image_transform, list):
        train_set = Kamitani_dataset(res_train, im_train, [(c, c_name, naive) for c, c_name, naive in zip(np.argmax(cat_train, axis=1), ['' for _ in range(len(cat_train))], [0 for _ in range(len(cat_train))])], fmri_transform, image_transform[0], num_voxels=res_train.shape[-1], num_per_sub=res_train.shape[0])
        test_set = Kamitani_dataset(res_test, im_test, [(c, c_name, naive) for c, c_name, naive in zip(np.argmax(cat_test, axis=1), ['' for _ in range(len(cat_test))], [0 for _ in range(len(cat_test))])], torch.FloatTensor, image_transform[1], num_voxels=res_test.shape[-1], num_per_sub=res_test.shape[0])
    else:
        train_set = Kamitani_dataset(res_train, im_train, [(c, c_name, naive) for c, c_name, naive in zip(np.argmax(cat_train, axis=1), ['' for _ in range(len(cat_train))], [0 for _ in range(len(cat_train))])], fmri_transform, image_transform, num_voxels=res_train.shape[-1], num_per_sub=res_train.shape[0])
        test_set = Kamitani_dataset(res_test, im_test, [(c, c_name, naive) for c, c_name, naive in zip(np.argmax(cat_test, axis=1), ['' for _ in range(len(cat_test))], [0 for _ in range(len(cat_test))])], torch.FloatTensor, image_transform, num_voxels=res_test.shape[-1], num_per_sub=res_test.shape[0])
    return train_set, test_set


def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')

def pad_to_length(x, length):
    assert x.ndim == 3
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x

    return np.pad(x, ((0,0),(0,0), (0, length - x.shape[-1])), 'wrap')

def pad_to_length_dim2(x, length):
    assert x.ndim == 2
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x

    return np.pad(x, ((0,0), (0, length - x.shape[-1])), 'wrap')

def normalize(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def process_voxel_ts(v, p, t=8):
    '''
    v: voxel timeseries of a subject. (1200, num_voxels)
    p: patch size
    t: time step of the averaging window for v. Kamitani used 8 ~ 12s
    return: voxels_reduced. reduced for the alignment of the patch size (num_samples, num_voxels_reduced)

    '''
    # average the time axis first
    num_frames_per_window = t // 0.75 # ~0.75s per frame in HCP
    v_split = np.array_split(v, len(v) // num_frames_per_window, axis=0)
    v_split = np.concatenate([np.mean(f,axis=0).reshape(1,-1) for f in v_split],axis=0)
    # pad the num_voxels
    # v_split = np.concatenate([v_split, np.zeros((v_split.shape[0], p - v_split.shape[1] % p))], axis=-1)
    v_split = pad_to_patch_size(v_split, p)
    v_split = normalize(v_split)
    return v_split

def augmentation(data, aug_times=2, interpolation_ratio=0.5):
    '''
    data: num_samples, num_voxels_padded
    return: data_aug: num_samples*aug_times, num_voxels_padded
    '''
    num_to_generate = int((aug_times-1)*len(data)) 
    if num_to_generate == 0:
        return data
    pairs_idx = np.random.choice(len(data), size=(num_to_generate, 2), replace=True)
    data_aug = []
    for i in pairs_idx:
        z = interpolate_voxels(data[i[0]], data[i[1]], interpolation_ratio)
        data_aug.append(np.expand_dims(z,axis=0))
    data_aug = np.concatenate(data_aug, axis=0)

    return np.concatenate([data, data_aug], axis=0)

def interpolate_voxels(x, y, ratio=0.5):
    ''''
    x, y: one dimension voxels array
    ratio: ratio for interpolation
    return: z same shape as x and y

    '''
    values = np.stack((x,y))
    points = (np.r_[0, 1], np.arange(len(x)))
    xi = np.c_[np.full((len(x)), ratio), np.arange(len(x)).reshape(-1,1)]
    z = interpolate.interpn(points, values, xi)
    return z

def img_norm(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = (img / 255.0) * 2.0 - 1.0 # to -1 ~ 1
    return img

def channel_first(img):
        if img.shape[-1] == 3:
            return rearrange(img, 'h w c -> c h w')
        return img

class hcp_dataset(Dataset):
    def __init__(self, path='../data/HCP/npz', roi='VC', patch_size=16, transform=identity, aug_times=2, 
                num_sub_limit=None, include_kam=False, include_hcp=True):
        super(hcp_dataset, self).__init__()
        data = []
        images = []
        
        if include_hcp:
            for c, sub in enumerate(os.listdir(path)):
                if os.path.isfile(os.path.join(path,sub,'HCP_visual_voxel.npz')) == False:
                    continue 
                if num_sub_limit is not None and c > num_sub_limit:
                    break
                npz = dict(np.load(os.path.join(path,sub,'HCP_visual_voxel.npz')))
                voxels = np.concatenate([npz['V1'],npz['V2'],npz['V3'],npz['V4']], axis=-1) if roi == 'VC' else npz[roi] # 1200, num_voxels
                voxels = process_voxel_ts(voxels, patch_size) # num_samples, num_voxels_padded
                data.append(voxels)
                
            data = augmentation(np.concatenate(data, axis=0), aug_times) # num_samples, num_voxels_padded
            data = np.expand_dims(data, axis=1) # num_samples, 1, num_voxels_padded
            images += [None] * len(data)

        if include_kam:
            kam_path = os.path.join(str(Path(path).parent.parent), 'Kamitani', 'npz')
            k = Kamitani_pretrain_dataset(kam_path, roi, patch_size, transform, aug_times)
            if len(data) != 0:
                padding_len = max([data.shape[-1],  k.data.shape[-1]])
                data = pad_to_length(data, padding_len)
                data_k = pad_to_length(k.data, padding_len)
                data = np.concatenate([data, data_k], axis=0)
            else:
                data = k.data
            images += k.images

        assert len(data) != 0, 'No data found'
        
        self.roi = roi
        self.patch_size = patch_size
        self.num_voxels = data.shape[-1]
        self.data = data
        self.transform = transform
        self.images = images
        self.images_transform = transforms.Compose([
                                            img_norm,
                                            transforms.Resize((112, 112)), 
                                            channel_first
                                        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.images[index]
        images_transform = self.images_transform if img is not None else identity
        img = img if img is not None else torch.zeros(3, 112, 112)

        return {'fmri': self.transform(self.data[index]),
                'image': images_transform(img)}
       
class Kamitani_pretrain_dataset(Dataset):
    def __init__(self, path='../data/Kamitani/npz', roi='VC', patch_size=16, transform=identity, aug_times=2):
        super(Kamitani_pretrain_dataset, self).__init__()
        k1, k2 = create_Kamitani_dataset(path, roi, patch_size, transform, include_nonavg_test=True)
        # data = np.concatenate([k1.fmri, k2.fmri], axis=0)
        # self.images = [img for img in k1.image] + [None] * len(k2.fmri)

        data = k1.fmri
        self.images = [(img*255.0).astype(np.uint8) for img in k1.image]

        # data = augmentation(data, aug_times)
        self.data = np.expand_dims(data, axis=1)
        self.roi = roi
        self.patch_size = patch_size
        self.num_voxels = data.shape[-1]
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.transform(self.data[index])

def get_img_label(class_index:dict, img_filename:list, naive_label_set=None, return_image_name=False):
    img_label = []
    wind = []
    desc = []
    for _, v in class_index.items():
        n_list = []
        for n in v[:-1]:
            n_list.append(int(n[1:]))
        wind.append(n_list)
        desc.append(v[-1])

    if return_image_name:
        image_name_return = []

    naive_label = {} if naive_label_set is None else naive_label_set
    for _, file in enumerate(img_filename):
        name = int(file[0].split('.')[0])
        naive_label[name] = []
        nl = list(naive_label.keys()).index(name)
        for c, (w, d) in enumerate(zip(wind, desc)):
            if name in w:
                img_label.append((c, d, nl))
                if return_image_name: image_name_return.append(name)
                break
    if return_image_name:
        return img_label, naive_label, image_name_return
    else:
        return img_label, naive_label

def create_Kamitani_dataset(path='../data/Kamitani/npz',  roi='VC', patch_size=16, fmri_transform=identity,
            image_transform=identity, subjects = ['sbj_1', 'sbj_2', 'sbj_3', 'sbj_4', 'sbj_5'], 
            test_category=None, include_nonavg_test=False):
    img_npz = dict(np.load(os.path.join(path, 'images_256.npz')))
    with open(os.path.join(path, 'imagenet_class_index.json'), 'r') as f:
        img_class_index = json.load(f)

    with open(os.path.join(path, 'imagenet_training_label.csv'), 'r') as f:
        csvreader = csv.reader(f)
        img_training_filename = [row for row in csvreader]

    with open(os.path.join(path, 'imagenet_testing_label.csv'), 'r') as f:
        csvreader = csv.reader(f)
        img_testing_filename = [row for row in csvreader]

    train_img_label, naive_label_set = get_img_label(img_class_index, img_training_filename)
    test_img_label, _ = get_img_label(img_class_index, img_testing_filename, naive_label_set)

    test_img = [] # img_npz['test_images']
    train_img = [] # img_npz['train_images']
    train_fmri = []
    test_fmri = []
    train_img_label_all = []
    test_img_label_all = []
    for sub in subjects:
        npz = dict(np.load(os.path.join(path, f'{sub}.npz')))
        test_img.append(img_npz['test_images'])
        train_img.append(img_npz['train_images'][npz['arr_3']])
        train_lb = [train_img_label[i] for i in npz['arr_3']]
        test_lb = test_img_label
        train_img_name = [img_training_filename [i] for i in npz['arr_3']]
        
        roi_mask = npz[roi]
        tr = npz['arr_0'][..., roi_mask] # train
        tt = npz['arr_2'][..., roi_mask] 
        if include_nonavg_test:
            tt = np.concatenate([tt, npz['arr_1'][..., roi_mask]], axis=0)

        # train_fmri.append(tr[..., :tr.shape[-1] - tr.shape[-1] % patch_size])
        # test_fmri.append(tt[..., :tt.shape[-1] - tt.shape[-1] % patch_size])
        tr = normalize(pad_to_patch_size(tr, patch_size))
        tt = normalize(pad_to_patch_size(tt, patch_size), np.mean(tr), np.std(tr))
        train_fmri.append(tr)
        test_fmri.append(tt)
        if test_category is not None:
            train_img_, train_fmri_, test_img_, test_fmri_, train_lb, test_lb = reorganize_train_test(train_img[-1], train_fmri[-1], 
                                                            test_img[-1], test_fmri[-1], train_lb, test_lb,
                                                            test_category, npz['arr_3'])
            train_img[-1] = train_img_
            train_fmri[-1] = train_fmri_
            test_img[-1] = test_img_
            test_fmri[-1] = test_fmri_
        
        train_img_label_all += train_lb
        test_img_label_all += test_lb

    len_max = max([i.shape[-1] for i in test_fmri])
    test_fmri = [np.pad(i, ((0, 0),(0, len_max-i.shape[-1])), mode='wrap') for i in test_fmri]
    train_fmri = [np.pad(i, ((0, 0),(0, len_max-i.shape[-1])), mode='wrap') for i in train_fmri]

    # len_min = min([i.shape[-1] for i in test_fmri])
    # test_fmri = [i[:,:len_min] for i in test_fmri]
    # train_fmri = [i[:,:len_min] for i in train_fmri]


    test_fmri = np.concatenate(test_fmri, axis=0)
    train_fmri = np.concatenate(train_fmri, axis=0)
    test_img = np.concatenate(test_img, axis=0)
    train_img = np.concatenate(train_img, axis=0)
    num_voxels = train_fmri.shape[-1]

    # test_img = rearrange(test_img, 'n h w c -> n c h w')
    # train_img = rearrange(train_img, 'n h w c -> n c h w')

    if isinstance(image_transform, list):
        return (Kamitani_dataset(train_fmri, train_img, train_img_label_all, fmri_transform, image_transform[0], num_voxels, len(npz['arr_0'])), 
                Kamitani_dataset(test_fmri, test_img, test_img_label_all, torch.FloatTensor, image_transform[1], num_voxels, len(npz['arr_2'])))
    else:
        return (Kamitani_dataset(train_fmri, train_img, train_img_label_all, fmri_transform, image_transform, num_voxels, len(npz['arr_0'])), 
                Kamitani_dataset(test_fmri, test_img, test_img_label_all, torch.FloatTensor, image_transform, num_voxels, len(npz['arr_2'])))

def create_Kamitani_dataset_distill(path='../data/Kamitani/npz',  roi='VC', patch_size=16, fmri_transform=identity,
            image_transform=identity, subjects = ['sbj_1', 'sbj_2', 'sbj_3', 'sbj_4', 'sbj_5'], 
            test_category=None, include_nonavg_test=False, return_image_name=False):
    img_npz = dict(np.load(os.path.join(path, 'images_256.npz')))
    with open(os.path.join(path, 'imagenet_class_index.json'), 'r') as f:
        img_class_index = json.load(f)

    with open(os.path.join(path, 'imagenet_training_label.csv'), 'r') as f:
        csvreader = csv.reader(f)
        img_training_filename = [row for row in csvreader]

    with open(os.path.join(path, 'imagenet_testing_label.csv'), 'r') as f:
        csvreader = csv.reader(f)
        img_testing_filename = [row for row in csvreader]

    if not return_image_name:
        train_img_label, naive_label_set = get_img_label(img_class_index, img_training_filename)
        test_img_label, _ = get_img_label(img_class_index, img_testing_filename, naive_label_set)
    else:
        train_img_label, naive_label_set, train_img_name = get_img_label(img_class_index, img_training_filename, return_image_name=True)
        test_img_label, _, test_img_name = get_img_label(img_class_index, img_testing_filename, naive_label_set, return_image_name=True)


    test_img = [] # img_npz['test_images']
    train_img = [] # img_npz['train_images']
    train_fmri = []
    test_fmri = []
    train_img_label_all = []
    test_img_label_all = []
    if type(subjects) == str:
        subjects = [subjects]
    for sub in subjects:
        npz = dict(np.load(os.path.join(path, f'{sub}.npz')))
        test_img.append(img_npz['test_images'])
        train_img.append(img_npz['train_images'][npz['arr_3']])
        train_lb = [train_img_label[i] for i in npz['arr_3']]
        test_lb = test_img_label
        # train_img_name = [img_training_filename[i][0].split('.')[0] for i in npz['arr_3']]
        # test_img_name = [ii[0].split('.')[0] for ii in img_testing_filename]
        
        roi_mask = npz[roi]
        tr = npz['arr_0'][..., roi_mask] # train
        tt = npz['arr_2'][..., roi_mask] 
        if include_nonavg_test:
            tt = np.concatenate([tt, npz['arr_1'][..., roi_mask]], axis=0)

        # train_fmri.append(tr[..., :tr.shape[-1] - tr.shape[-1] % patch_size])
        # test_fmri.append(tt[..., :tt.shape[-1] - tt.shape[-1] % patch_size])
        tr = normalize(pad_to_patch_size(tr, patch_size))
        tt = normalize(pad_to_patch_size(tt, patch_size), np.mean(tr), np.std(tr))
        train_fmri.append(tr)
        test_fmri.append(tt)
        if test_category is not None:
            train_img_, train_fmri_, test_img_, test_fmri_, train_lb, test_lb = reorganize_train_test(train_img[-1], train_fmri[-1], 
                                                            test_img[-1], test_fmri[-1], train_lb, test_lb,
                                                            test_category, npz['arr_3'])
            train_img[-1] = train_img_
            train_fmri[-1] = train_fmri_
            test_img[-1] = test_img_
            test_fmri[-1] = test_fmri_
        
        train_img_label_all += train_lb
        test_img_label_all += test_lb

    len_max = max([i.shape[-1] for i in test_fmri])
    test_fmri = [np.pad(i, ((0, 0),(0, len_max-i.shape[-1])), mode='wrap') for i in test_fmri]
    train_fmri = [np.pad(i, ((0, 0),(0, len_max-i.shape[-1])), mode='wrap') for i in train_fmri]

    # len_min = min([i.shape[-1] for i in test_fmri])
    # test_fmri = [i[:,:len_min] for i in test_fmri]
    # train_fmri = [i[:,:len_min] for i in train_fmri]


    test_fmri = np.concatenate(test_fmri, axis=0)
    train_fmri = np.concatenate(train_fmri, axis=0)
    test_img = np.concatenate(test_img, axis=0)
    train_img = np.concatenate(train_img, axis=0)
    num_voxels = train_fmri.shape[-1]

    # test_img = rearrange(test_img, 'n h w c -> n c h w')
    # train_img = rearrange(train_img, 'n h w c -> n c h w')

    if return_image_name:
        if isinstance(image_transform, list):
            return (Kamitani_dataset_distill(train_fmri, train_img, train_img_label_all, fmri_transform, image_transform[0], num_voxels, len(npz['arr_0']), train_img_name), 
                    Kamitani_dataset_distill(test_fmri, test_img, test_img_label_all, torch.FloatTensor, image_transform[1], num_voxels, len(npz['arr_2']), test_img_name))
        else:
            return (Kamitani_dataset_distill(train_fmri, train_img, train_img_label_all, fmri_transform, image_transform, num_voxels, len(npz['arr_0']), train_img_name), 
                    Kamitani_dataset_distill(test_fmri, test_img, test_img_label_all, torch.FloatTensor, image_transform, num_voxels, len(npz['arr_2']), test_img_name))
    else:
        if isinstance(image_transform, list):
            return (Kamitani_dataset(train_fmri, train_img, train_img_label_all, fmri_transform, image_transform[0], num_voxels, len(npz['arr_0'])), 
                    Kamitani_dataset(test_fmri, test_img, test_img_label_all, torch.FloatTensor, image_transform[1], num_voxels, len(npz['arr_2'])))
        else:
            return (Kamitani_dataset(train_fmri, train_img, train_img_label_all, fmri_transform, image_transform, num_voxels, len(npz['arr_0'])), 
                    Kamitani_dataset(test_fmri, test_img, test_img_label_all, torch.FloatTensor, image_transform, num_voxels, len(npz['arr_2'])))


def reorganize_train_test(train_img, train_fmri, test_img, test_fmri, train_lb, test_lb, 
                    test_category, train_index_lookup):
    test_img_ = []
    test_fmri_ = []
    test_lb_ = []
    train_idx_list = []
    num_per_category = 8
    for c in test_category:
        c_idx = c * num_per_category + np.random.choice(num_per_category, 1)[0]
        train_idx = train_index_lookup[c_idx]
        test_img_.append(train_img[train_idx])
        test_fmri_.append(train_fmri[train_idx])
        test_lb_.append(train_lb[train_idx])
        train_idx_list.append(train_idx)
    
    train_img_ = np.stack([img for i, img in enumerate(train_img) if i not in train_idx_list])
    train_fmri_ = np.stack([fmri for i, fmri in enumerate(train_fmri) if i not in train_idx_list])
    train_lb_ = [lb for i, lb in enumerate(train_lb) if i not in train_idx_list] + test_lb

    train_img_ = np.concatenate([train_img_, test_img], axis=0)
    train_fmri_ = np.concatenate([train_fmri_, test_fmri], axis=0)

    test_img_ = np.stack(test_img_)
    test_fmri_ = np.stack(test_fmri_)
    return train_img_, train_fmri_, test_img_, test_fmri_, train_lb_, test_lb_

class Kamitani_dataset(Dataset):
    def __init__(self, fmri, image, img_label, fmri_transform=identity, image_transform=identity, num_voxels=0, num_per_sub=50):
        super(Kamitani_dataset, self).__init__()
        self.fmri = fmri
        self.image = image
        if len(self.image) != len(self.fmri):
            print('!!mismatch happened, len(self.image) != len(self.fmri)')
            print(self.image.shape, self.fmri.shape)
            self.image = np.repeat(self.image, 35, axis=0)
        self.fmri_transform = fmri_transform
        self.image_transform = image_transform
        self.num_voxels = num_voxels
        self.num_per_sub = num_per_sub
        self.img_class = [i[0] for i in img_label]
        self.img_class_name = [i[1] for i in img_label]
        self.naive_label = [i[2] for i in img_label]
        self.return_image_class_info = True

    def __len__(self):
        return len(self.fmri)
    
    def __getitem__(self, index):
        fmri = self.fmri[index]
        if index >= len(self.image):
            img = np.zeros_like(self.image[0])
        else:
            img = self.image[index] / 255.0
        fmri = np.expand_dims(fmri, axis=0) # (1, num_voxels)
        if self.return_image_class_info:
            img_class = self.img_class[index]
            img_class_name = self.img_class_name[index]
            naive_label = torch.tensor(self.naive_label[index])
            return {'fmri': self.fmri_transform(fmri), 'image': self.image_transform(img),
                    'image_class': img_class, 'image_class_name': img_class_name, 'naive_label':naive_label}
        else:
            return {'fmri': self.fmri_transform(fmri), 'image': self.image_transform(img)}

class Kamitani_dataset_distill(Dataset):
    def __init__(self, fmri, image, img_label, fmri_transform=identity, image_transform=identity, num_voxels=0, num_per_sub=50, image_name=None):
        super(Kamitani_dataset_distill, self).__init__()
        self.fmri = fmri
        self.image = image
        if len(self.image) != len(self.fmri):
            print('!!mismatch happened, len(self.image) != len(self.fmri)')
            self.image = np.repeat(self.image, 35, axis=0)
        self.fmri_transform = fmri_transform
        self.image_transform = image_transform
        self.num_voxels = num_voxels
        self.num_per_sub = num_per_sub
        self.img_class = [i[0] for i in img_label]
        self.img_class_name = [i[1] for i in img_label]
        self.naive_label = [i[2] for i in img_label]
        self.return_image_class_info = True
        self.image_name = image_name

    def __len__(self):
        return len(self.fmri)
    
    def __getitem__(self, index):
        fmri = self.fmri[index]
        if index >= len(self.image):
            img = np.zeros_like(self.image[0])
        else:
            img = self.image[index] / 255.0
        fmri = np.expand_dims(fmri, axis=0) # (1, num_voxels)
        if self.return_image_class_info:
            img_class = self.img_class[index]
            img_class_name = self.img_class_name[index]
            naive_label = torch.tensor(self.naive_label[index])
            return {'fmri': self.fmri_transform(fmri), 'image': self.image_transform(img),
                    'image_class': img_class, 'image_class_name': img_class_name, 'naive_label':naive_label,
                    'image_name': self.image_name[index]}
        else:
            return {'fmri': self.fmri_transform(fmri), 'image': self.image_transform(img)}

class base_dataset(Dataset):
    def __init__(self, x, y=None, transform=identity):
        super(base_dataset, self).__init__()
        self.x = x
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        if self.y is None:
            return self.transform(self.x[index])
        else:
            return self.transform(self.x[index]), self.transform(self.y[index])
    
def remove_repeats(fmri, img_lb):
    assert len(fmri) == len(img_lb), 'len error'
    fmri_dict = {}
    for f, lb in zip(fmri, img_lb):
        if lb in fmri_dict.keys():
            fmri_dict[lb].append(f)
        else:
            fmri_dict[lb] = [f]
    lbs = []
    fmris = []
    for k, v in fmri_dict.items():
        lbs.append(k)
        fmris.append(np.mean(np.stack(v), axis=0))
    return np.stack(fmris), lbs

def get_stimuli_list(root, sub):
    sti_name = []
    path = os.path.join(root, 'Stimuli_Presentation_Lists', sub)
    folders = os.listdir(path)
    folders.sort()
    for folder in folders:
        if not os.path.isdir(os.path.join(path, folder)):
            continue
        files = os.listdir(os.path.join(path, folder))
        files.sort()
        for file in files:
            if file.endswith('.txt'):
                sti_name += list(np.loadtxt(os.path.join(path, folder, file), dtype=str))

    sti_name_to_return = []
    for name in sti_name:
        if name.startswith('rep_'):
            name = name.replace('rep_', '', 1)
        sti_name_to_return.append(name)
    return sti_name_to_return

def list_get_all_index(list, value):
    return [i for i, v in enumerate(list) if v == value]
    
def create_BOLD5000_dataset(path='../data/BOLD5000', patch_size=16, fmri_transform=identity,
            image_transform=identity, subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4'], include_nonavg_test=False):
    roi_list = ['EarlyVis', 'LOC', 'OPA', 'PPA', 'RSC']
    fmri_path = os.path.join(path, 'BOLD5000_GLMsingle_ROI_betas/py')
    img_path = os.path.join(path, 'BOLD5000_Stimuli')
    imgs_dict = np.load(os.path.join(img_path, 'Scene_Stimuli/Presented_Stimuli/img_dict.npy'),allow_pickle=True).item()
    repeated_imgs_list = np.loadtxt(os.path.join(img_path, 'Scene_Stimuli', 'repeated_stimuli_113_list.txt'), dtype=str)

    fmri_files = [f for f in os.listdir(fmri_path) if f.endswith('.npy')]
    fmri_files.sort()
    
    fmri_train_major = []
    fmri_test_major = []
    img_train_major = []
    img_test_major = []
    for sub in subjects:
        # load fmri
        fmri_data_sub = []
        for roi in roi_list:
            for npy in fmri_files:
                if npy.endswith('.npy') and sub in npy and roi in npy:
                    fmri_data_sub.append(np.load(os.path.join(fmri_path, npy)))
        fmri_data_sub = np.concatenate(fmri_data_sub, axis=-1) # concatenate all rois
        fmri_data_sub = normalize(pad_to_patch_size(fmri_data_sub, patch_size))
      
        # load image
        img_files = get_stimuli_list(img_path, sub)
        img_data_sub = [imgs_dict[name] for name in img_files]
        
        # split train test
        test_idx = [list_get_all_index(img_files, img) for img in repeated_imgs_list]
        test_idx = [i for i in test_idx if len(i) > 0] # remove empy list for CSI4
        test_fmri = np.stack([fmri_data_sub[idx].mean(axis=0) for idx in test_idx])
        test_img = np.stack([img_data_sub[idx[0]] for idx in test_idx])
        
        test_idx_flatten = []
        for idx in test_idx:
            test_idx_flatten += idx # flatten
        if include_nonavg_test:
            test_fmri = np.concatenate([test_fmri, fmri_data_sub[test_idx_flatten]], axis=0)
            test_img = np.concatenate([test_img, np.stack([img_data_sub[idx] for idx in test_idx_flatten])], axis=0)

        train_idx = [i for i in range(len(img_files)) if i not in test_idx_flatten]
        train_img = np.stack([img_data_sub[idx] for idx in train_idx])
        train_fmri = fmri_data_sub[train_idx]

        fmri_train_major.append(train_fmri)
        fmri_test_major.append(test_fmri)
        img_train_major.append(train_img)
        img_test_major.append(test_img)
    fmri_train_major = np.concatenate(fmri_train_major, axis=0)
    fmri_test_major = np.concatenate(fmri_test_major, axis=0)
    img_train_major = np.concatenate(img_train_major, axis=0)
    img_test_major = np.concatenate(img_test_major, axis=0)

    num_voxels = fmri_train_major.shape[-1]
    if isinstance(image_transform, list):
        return (BOLD5000_dataset(fmri_train_major, img_train_major, fmri_transform, image_transform[0], num_voxels), 
                BOLD5000_dataset(fmri_test_major, img_test_major, torch.FloatTensor, image_transform[1], num_voxels))
    else:
        return (BOLD5000_dataset(fmri_train_major, img_train_major, fmri_transform, image_transform, num_voxels), 
                BOLD5000_dataset(fmri_test_major, img_test_major, torch.FloatTensor, image_transform, num_voxels))

class BOLD5000_dataset(Dataset):
    def __init__(self, fmri, image, fmri_transform=identity, image_transform=identity, num_voxels=0):
        self.fmri = fmri[:10]
        self.image = image
        self.fmri_transform = fmri_transform
        self.image_transform = image_transform
        self.num_voxels = num_voxels
    
    def __len__(self):
        return len(self.fmri)
    
    def __getitem__(self, index):
        fmri = self.fmri[index]
        img = self.image[index] / 255.0
        fmri = np.expand_dims(fmri, axis=0) 
        
        pad_fmri = np.pad(fmri, ((0,0), (0, 4192-fmri.shape[-1])), 'wrap') 
        return {'fmri': self.fmri_transform(fmri),
                'pad_fmri': pad_fmri,
                'image': self.image_transform(img)}
    
    def switch_sub_view(self, sub, subs):
        # Not implemented
        pass

def create_BOLD5000_dataset_crosssub(path='../data/BOLD5000', patch_size=16, fmri_transform=identity,
            image_transform=identity, subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4'], include_nonavg_test=True, do_normalize=False,
            cross_sub=False, train_subs=['CSI1', 'CSI2', 'CSI3'], test_subs=['CSI4'], target_sub_train_proportion=1.0):
    # 通过subjects参数传入目标被试，例如需要CS1的数据，则subjects=['CS1'],需要传入list
    # cross_sub参数为True时，代表进行跨subject解码。此时subjects参数无效，使用train_subs和test_subs参数传入作为训练集和测试集的被试。
    # target_sub_train_proportion参数为目标被试数据的多大比例作为训练数据。
        # 例如在train_subs=['CSI1', 'CSI2', 'CSI3'], test_subs=['CSI4']，target_sub_train_proportion=0.5时，CSI4为目标被试。
        # CSI4全部训练数据的一半加入训练集，和CS1-CSI3的全部数据一起，训练fmri_encoder，而测试集仍为CSI4的原本的测试集。

    if cross_sub:
        print('loading data for cross subject decoding...')
        all_subs = train_subs + test_subs
    else:
        all_subs = subjects

    roi_list = ['EarlyVis', 'LOC', 'OPA', 'PPA', 'RSC']

    #fmri_path对应文件夹中存放有BOLD5000的四个被试CS1-CS4的fmri文件，已经按照脑区抽取为np array存放。
    #每个文件大小为5254*size(ROI), size(ROI)为对应脑区包含的voxel数量，可理解为特征维度。 5254为被试看到的图片数量
    fmri_path = os.path.join(path, 'BOLD5000_GLMsingle_ROI_betas/py')
    # img_path对应文件夹中存放有被试看到的图片
    img_path = os.path.join(path, 'BOLD5000_Stimuli')

    # imgs_dict中存放了被试看到的所有图片的RGB pixel matrix, 尺寸为256*256*3，如需使用需除以255
    imgs_dict = np.load(os.path.join(img_path, 'Scene_Stimuli/Presented_Stimuli/img_dict.npy'),allow_pickle=True).item()

    # repeated_imgs_list中存放了重复采集的所有图片的文件名，重复采集的图像将作为测试集
    repeated_imgs_list = np.loadtxt(os.path.join(img_path, 'Scene_Stimuli', 'repeated_stimuli_113_list.txt'), dtype=str)

    fmri_files = [f for f in os.listdir(fmri_path) if f.endswith('.npy')]
    fmri_files.sort()
    
    fmri_train_major = []
    fmri_test_major = []
    img_train_major = []
    img_test_major = []

    # fmri_name_train_major = []
    img_name_train_major = []
    # fmri_name_test_major = []
    img_name_test_major = []

    for sub in all_subs:
        # load fmri
        fmri_data_sub = []
        # fmri_file_name = []
        for roi in roi_list:
            for npy in fmri_files:
                if npy.endswith('.npy') and sub in npy and roi in npy:
                    fmri_data_sub.append(np.load(os.path.join(fmri_path, npy)))
                    # fmri_file_name.append(os.path.join(fmri_path, npy))
        fmri_data_sub = np.concatenate(fmri_data_sub, axis=-1) # concatenate all rois
        if do_normalize:
            fmri_data_sub = normalize(pad_to_patch_size(fmri_data_sub, patch_size))

        # load image
        # 图像在实验中呈现给被试者的顺序存放在Stimuli_Presentation_Lists文件夹中
        # img_files， 即get_stimuli_list返回的列表中包含按呈现顺序排列的图像文件名
        img_files = get_stimuli_list(img_path, sub)
        # img_data_sub中存放了image的pixel matrix
        img_data_sub = [imgs_dict[name] for name in img_files]
        
        # split train test
        # test split
        # 找出重复采集的图片在img_files中的index
        test_idx = [list_get_all_index(img_files, img) for img in repeated_imgs_list]
        test_idx = [i for i in test_idx if len(i) > 0] # remove empy list for CSI4
        #按index取出fmri，因可能存在重复采集，需要将重复采集的fmri做平均
        test_fmri = np.stack([fmri_data_sub[idx].mean(axis=0) for idx in test_idx])
        #按index取出图片
        test_img = np.stack([img_data_sub[idx[0]] for idx in test_idx])

        # test_fmri_name = [fmri_file_name[idx[0]] for idx in test_idx]
        # 按index取出图片文件名
        test_img_name = np.stack([img_files[idx[0]] for idx in test_idx])
        
        test_idx_flatten = []
        for idx in test_idx:
            test_idx_flatten += idx # flatten
        if include_nonavg_test:
            test_fmri = np.concatenate([test_fmri, fmri_data_sub[test_idx_flatten]], axis=0)
            test_img = np.concatenate([test_img, np.stack([img_data_sub[idx] for idx in test_idx_flatten])], axis=0)
            test_img_name = np.concatenate([test_img_name, np.stack([img_files[idx] for idx in test_idx_flatten])], axis=0)

        # train split
        # 找出测试数据以外其他数据的index
        train_idx = [i for i in range(len(img_files)) if i not in test_idx_flatten]
        train_img = np.stack([img_data_sub[idx] for idx in train_idx])
        train_fmri = fmri_data_sub[train_idx]

        # train_fmri_name = fmri_file_name[train_idx]
        train_img_name = [img_files[idx] for idx in train_idx]

        fmri_train_major.append(train_fmri)
        fmri_test_major.append(test_fmri)
        img_train_major.append(train_img)
        img_test_major.append(test_img)

        # fmri_name_test_major.append(test_fmri_name)
        img_name_test_major.append(test_img_name)
        # fmri_name_train_major.append(train_fmri_name)
        img_name_train_major.append(train_img_name)


    if len(all_subs) > 1:
        max_fmri_dim = np.max([fmri.shape[-1] for fmri in fmri_train_major])
        # print('max_fmri_dim', max_fmri_dim, fmri_train_major[0].shape)
        fmri_train_major = [pad_to_length_dim2(ii, max_fmri_dim) for ii in fmri_train_major]
        fmri_test_major = [pad_to_length_dim2(ii, max_fmri_dim) for ii in fmri_test_major]

    if cross_sub:
        fmri_train_major_cross = []
        fmri_test_major_cross = []
        img_train_major_cross = []
        img_test_major_cross = []
        img_name_train_major_cross = []
        img_name_test_major_cross = []

        # 对于train_subs中的每个subject，将其训练集和测试集合并，作为cross subject decoding的训练集
        for sub_index in range(len(train_subs)):
            fmri_train_major_cross.append(np.concatenate([fmri_train_major[sub_index], fmri_test_major[sub_index]], axis=0))
            img_train_major_cross.append(np.concatenate([img_train_major[sub_index], img_test_major[sub_index]], axis=0))
            img_name_train_major_cross.append(np.concatenate([img_name_train_major[sub_index], img_name_test_major[sub_index]], axis=0))
        # 对于test_subs中的每个subject，将其训练集中前target_sub_train_proportion*fmri_train_major[sub_index_test].shape[0]条合并入cross subject decoding的训练集
        # 对于test_subs中的每个subject，其测试集作为cross subject decoding的测试集
        for sub_index in range(len(test_subs)):
            sub_index_test = len(train_subs) + sub_index
            if target_sub_train_proportion > 0:
                tsb = int(target_sub_train_proportion*fmri_train_major[sub_index_test].shape[0])
                fmri_train_major_cross.append(fmri_train_major[sub_index_test][:tsb])
                img_train_major_cross.append(img_train_major[sub_index_test][:tsb])
                img_name_train_major_cross.append(img_name_train_major[sub_index_test][:tsb])

            fmri_test_major_cross.append(fmri_test_major[sub_index_test])
            img_test_major_cross.append(img_test_major[sub_index_test])
            img_name_test_major_cross.append(img_name_test_major[sub_index_test])
        
        fmri_train_major = np.concatenate(fmri_train_major_cross, axis=0)
        fmri_test_major = np.concatenate(fmri_test_major_cross, axis=0)
        img_train_major = np.concatenate(img_train_major_cross, axis=0)
        img_test_major = np.concatenate(img_test_major_cross, axis=0)
        img_name_test_major = np.concatenate(img_name_test_major_cross, axis=0)
        img_name_train_major = np.concatenate(img_name_train_major_cross, axis=0)
        print('fmri_train_major.shape', fmri_train_major.shape)
        print('fmri_test_major.shape', fmri_test_major.shape)
        print('img_train_major.shape', img_train_major.shape)
        print('img_test_major.shape', img_test_major.shape)
        print('img_name_test_major.shape', img_name_test_major.shape)
        print('img_name_train_major.shape', img_name_train_major.shape)
    
    else:
        fmri_train_major = np.concatenate(fmri_train_major, axis=0)
        fmri_test_major = np.concatenate(fmri_test_major, axis=0)
        img_train_major = np.concatenate(img_train_major, axis=0)
        img_test_major = np.concatenate(img_test_major, axis=0)
        img_name_test_major = np.concatenate(img_name_test_major, axis=0)
        img_name_train_major = np.concatenate(img_name_train_major, axis=0)

    num_voxels = fmri_train_major.shape[-1]
    if isinstance(image_transform, list):
        return (BOLD5000_dataset_classify(fmri_train_major, img_train_major, img_name_train_major, fmri_transform, image_transform[0], num_voxels),
                BOLD5000_dataset_classify(fmri_test_major, img_test_major, img_name_test_major, torch.FloatTensor, image_transform[1], num_voxels))
    else:
        return (BOLD5000_dataset_classify(fmri_train_major, img_train_major, img_name_train_major, fmri_transform, image_transform, num_voxels),
                BOLD5000_dataset_classify(fmri_test_major, img_test_major, img_name_test_major, torch.FloatTensor, image_transform, num_voxels))

def create_BOLD5000_dataset_classify(path='../data/BOLD5000', patch_size=16, fmri_transform=identity,
            image_transform=identity, subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4'], include_nonavg_test=True, do_normalize=False,
            target_sub_train_proportion=1):
    #通过subjects参数传入目标被试，例如需要CS1的数据，则subjects=['CS1'],需要传入list
    roi_list = ['EarlyVis', 'LOC', 'OPA', 'PPA', 'RSC']

    #fmri_path对应文件夹中存放有BOLD5000的四个被试CS1-CS4的fmri文件，已经按照脑区抽取为np array存放。
    #每个文件大小为5254*size(ROI), size(ROI)为对应脑区包含的voxel数量，可理解为特征维度。 5254为被试看到的图片数量
    fmri_path = os.path.join(path, 'BOLD5000_GLMsingle_ROI_betas/py')
    # img_path对应文件夹中存放有被试看到的图片
    img_path = os.path.join(path, 'BOLD5000_Stimuli')

    # imgs_dict中存放了被试看到的所有图片的RGB pixel matrix, 尺寸为256*256*3，如需使用需除以255
    imgs_dict = np.load(os.path.join(img_path, 'Scene_Stimuli/Presented_Stimuli/img_dict.npy'),allow_pickle=True).item()

    # repeated_imgs_list中存放了重复采集的所有图片的文件名，重复采集的图像将作为测试集
    repeated_imgs_list = np.loadtxt(os.path.join(img_path, 'Scene_Stimuli', 'repeated_stimuli_113_list.txt'), dtype=str)

    fmri_files = [f for f in os.listdir(fmri_path) if f.endswith('.npy')]
    fmri_files.sort()
    
    fmri_train_major = []
    fmri_test_major = []
    img_train_major = []
    img_test_major = []

    # fmri_name_train_major = []
    img_name_train_major = []
    # fmri_name_test_major = []
    img_name_test_major = []
    if type(subjects) == str:
        subjects = [subjects]

    for sub in subjects:
        # load fmri
        fmri_data_sub = []
        # fmri_file_name = []
        for roi in roi_list:
            for npy in fmri_files:
                if npy.endswith('.npy') and sub in npy and roi in npy:
                    fmri_data_sub.append(np.load(os.path.join(fmri_path, npy)))
                    # fmri_file_name.append(os.path.join(fmri_path, npy))
        fmri_data_sub = np.concatenate(fmri_data_sub, axis=-1) # concatenate all rois
        if do_normalize:
            fmri_data_sub = normalize(pad_to_patch_size(fmri_data_sub, patch_size))

        # load image
        # 图像在实验中呈现给被试者的顺序存放在Stimuli_Presentation_Lists文件夹中
        # img_files， 即get_stimuli_list返回的列表中包含按呈现顺序排列的图像文件名
        img_files = get_stimuli_list(img_path, sub)
        # img_data_sub中存放了image的pixel matrix
        img_data_sub = [imgs_dict[name] for name in img_files]
        
        # split train test
        # test split
        # 找出重复采集的图片在img_files中的index
        test_idx = [list_get_all_index(img_files, img) for img in repeated_imgs_list]
        test_idx = [i for i in test_idx if len(i) > 0] # remove empy list for CSI4
        #按index取出fmri，因可能存在重复采集，需要将重复采集的fmri做平均
        test_fmri = np.stack([fmri_data_sub[idx].mean(axis=0) for idx in test_idx])
        #按index取出图片
        test_img = np.stack([img_data_sub[idx[0]] for idx in test_idx])

        # test_fmri_name = [fmri_file_name[idx[0]] for idx in test_idx]
        # 按index取出图片文件名
        test_img_name = np.stack([img_files[idx[0]] for idx in test_idx])
        
        test_idx_flatten = []
        for idx in test_idx:
            test_idx_flatten += idx # flatten
        if include_nonavg_test:
            test_fmri = np.concatenate([test_fmri, fmri_data_sub[test_idx_flatten]], axis=0)
            test_img = np.concatenate([test_img, np.stack([img_data_sub[idx] for idx in test_idx_flatten])], axis=0)
            test_img_name = np.concatenate([test_img_name, np.stack([img_files[idx] for idx in test_idx_flatten])], axis=0)

        # train split
        # 找出测试数据以外其他数据的index
        train_idx = [i for i in range(len(img_files)) if i not in test_idx_flatten]
        train_img = np.stack([img_data_sub[idx] for idx in train_idx])
        train_fmri = fmri_data_sub[train_idx]

        # train_fmri_name = fmri_file_name[train_idx]
        train_img_name = [img_files[idx] for idx in train_idx]

        fmri_train_major.append(train_fmri)
        fmri_test_major.append(test_fmri)
        img_train_major.append(train_img)
        img_test_major.append(test_img)

        # fmri_name_test_major.append(test_fmri_name)
        img_name_test_major.append(test_img_name)
        # fmri_name_train_major.append(train_fmri_name)
        img_name_train_major.append(train_img_name)


    fmri_train_major = np.concatenate(fmri_train_major, axis=0)
    fmri_test_major = np.concatenate(fmri_test_major, axis=0)
    img_train_major = np.concatenate(img_train_major, axis=0)
    img_test_major = np.concatenate(img_test_major, axis=0)

    if target_sub_train_proportion < 1:
        fmri_train_major = fmri_train_major[:int(fmri_train_major.shape[0]*target_sub_train_proportion)]
        img_train_major = img_train_major[:int(img_train_major.shape[0]*target_sub_train_proportion)]

    # fmri_name_test_major = np.concatenate(fmri_name_test_major, axis=0)
    img_name_test_major = np.concatenate(img_name_test_major, axis=0)
    # fmri_name_train_major = np.concatenate(fmri_name_train_major, axis=0)
    img_name_train_major = np.concatenate(img_name_train_major, axis=0)

    num_voxels = fmri_train_major.shape[-1]
    if isinstance(image_transform, list):
        return (BOLD5000_dataset_classify(fmri_train_major, img_train_major, img_name_train_major, fmri_transform, image_transform[0], num_voxels),
                BOLD5000_dataset_classify(fmri_test_major, img_test_major, img_name_test_major, torch.FloatTensor, image_transform[1], num_voxels))
    else:
        return (BOLD5000_dataset_classify(fmri_train_major, img_train_major, img_name_train_major, fmri_transform, image_transform, num_voxels),
                BOLD5000_dataset_classify(fmri_test_major, img_test_major, img_name_test_major, torch.FloatTensor, image_transform, num_voxels))


class BOLD5000_dataset_classify(Dataset):
    def __init__(self, fmri, image, image_name, fmri_transform=identity, image_transform=identity, num_voxels=0):
        self.fmri = fmri
        self.image = image
        # self.fmri_name = fmri_name
        self.image_name = image_name

        self.fmri_transform = fmri_transform
        self.image_transform = image_transform
        self.num_voxels = num_voxels
        self.imagename2index = defaultdict(list)
        for i, iname in enumerate(self.image_name):
            self.imagename2index[iname].append(i)
            
    def __len__(self):
        return len(self.fmri)
    
    def __getitem__(self, index):
        fmri = self.fmri[index]
        img = self.image[index] 
        # fmri_name = self.fmri_name[index]
        img_name = self.image_name[index]
        fmri = np.expand_dims(fmri, axis=0) 
        if self.image_transform == identity:
            return {'fmri': self.fmri_transform(fmri), 
                    'image': self.image_transform(img/ 255.0),
                    'image_name': img_name,
                    'data_index': index}
        else:
            return {'fmri': self.fmri_transform(fmri), 
                    'image': self.image_transform(img),
                    'image_name': img_name,
                    'data_index': index}
    
    def switch_sub_view(self, sub, subs):
        # Not implemented
        pass

def create_BOLD5000_dataset_newtest(path='../data/BOLD5000', patch_size=16, fmri_transform=identity,
            image_transform=identity, subjects = ['CSI4'], include_nonavg_test=False, do_normalize=False):
    #通过subjects参数传入目标被试，例如需要CS1的数据，则subjects=['CS1'],需要传入list
    roi_list = ['EarlyVis', 'LOC', 'OPA', 'PPA', 'RSC']

    #fmri_path对应文件夹中存放有BOLD5000的四个被试CS1-CS4的fmri文件，已经按照脑区抽取为np array存放。
    #每个文件大小为5254*size(ROI), size(ROI)为对应脑区包含的voxel数量，可理解为特征维度。 5254为被试看到的图片数量
    fmri_path = os.path.join(path, 'BOLD5000_GLMsingle_ROI_betas/py')
    # img_path对应文件夹中存放有被试看到的图片
    img_path = os.path.join(path, 'BOLD5000_Stimuli')

    # imgs_dict中存放了被试看到的所有图片的RGB pixel matrix, 尺寸为256*256*3，如需使用需除以255
    imgs_dict = np.load(os.path.join(img_path, 'Scene_Stimuli/Presented_Stimuli/img_dict.npy'),allow_pickle=True).item()

    # repeated_imgs_list中存放了重复采集的所有图片的文件名，重复采集的图像将作为测试集
    repeated_imgs_list = np.loadtxt(os.path.join(img_path, 'Scene_Stimuli', 'repeated_stimuli_113_list.txt'), dtype=str)

    fmri_files = [f for f in os.listdir(fmri_path) if f.endswith('.npy')]
    fmri_files.sort()
    
    fmri_train_major = []
    fmri_test_major = []
    img_train_major = []
    img_test_major = []

    # fmri_name_train_major = []
    img_name_train_major = []
    # fmri_name_test_major = []
    img_name_test_major = []

    for sub in subjects:
        # load fmri
        fmri_data_sub = []
        # fmri_file_name = []
        for roi in roi_list:
            for npy in fmri_files:
                if npy.endswith('.npy') and sub in npy and roi in npy:
                    fmri_data_sub.append(np.load(os.path.join(fmri_path, npy)))
                    # fmri_file_name.append(os.path.join(fmri_path, npy))
        fmri_data_sub = np.concatenate(fmri_data_sub, axis=-1) # concatenate all rois
        fmri_data_sub = normalize(pad_to_patch_size(fmri_data_sub, patch_size))
      
        # load image
        # 图像在实验中呈现给被试者的顺序存放在Stimuli_Presentation_Lists文件夹中
        # img_files， 即get_stimuli_list返回的列表中包含按呈现顺序排列的图像文件名
        img_files = get_stimuli_list(img_path, sub)
        # img_data_sub中存放了image的pixel matrix
        img_data_sub = [imgs_dict[name] for name in img_files]
        
        # split train test
        # test split
        # 找出重复采集的图片在img_files中的index
        test_idx = [list_get_all_index(img_files, img) for img in repeated_imgs_list]
        test_idx = [i for i in test_idx if len(i) > 0] # remove empy list for CSI4
        #按index取出fmri，因可能存在重复采集，需要将重复采集的fmri做平均
        test_fmri = np.stack([fmri_data_sub[idx[-1]] for idx in test_idx])
        # test_fmri = np.stack([fmri_data_sub[idx].mean(axis=0) for idx in test_idx])
        #按index取出图片
        test_img = np.stack([img_data_sub[idx[-1]] for idx in test_idx])
        # test_img = np.stack([img_data_sub[idx[0]] for idx in test_idx])

        # test_fmri_name = [fmri_file_name[idx[0]] for idx in test_idx]
        # 按index取出图片文件名
        test_img = np.stack([img_data_sub[idx[-1]] for idx in test_idx])
        # test_img_name = np.stack([img_files[idx[0]] for idx in test_idx])
        
        test_idx_flatten = []
        for idx in test_idx:
            test_idx_flatten += idx # flatten
        if include_nonavg_test:
            test_fmri = np.concatenate([test_fmri, fmri_data_sub[test_idx_flatten]], axis=0)
            test_img = np.concatenate([test_img, np.stack([img_data_sub[idx] for idx in test_idx_flatten])], axis=0)
            test_img_name = np.concatenate([test_img_name, np.stack([img_files[idx] for idx in test_idx_flatten])], axis=0)

        # train split
        # 找出测试数据以外其他数据的index
        train_idx = [i for i in range(len(img_files)) if i not in test_idx_flatten]
        train_img = np.stack([img_data_sub[idx] for idx in train_idx])
        train_fmri = fmri_data_sub[train_idx]

        # train_fmri_name = fmri_file_name[train_idx]
        train_img_name = [img_files[idx] for idx in train_idx]

        fmri_train_major.append(train_fmri)
        fmri_test_major.append(test_fmri)
        img_train_major.append(train_img)
        img_test_major.append(test_img)

        # fmri_name_test_major.append(test_fmri_name)
        img_name_test_major.append(test_img_name)
        # fmri_name_train_major.append(train_fmri_name)
        img_name_train_major.append(train_img_name)


    fmri_train_major = np.concatenate(fmri_train_major, axis=0)
    fmri_test_major = np.concatenate(fmri_test_major, axis=0)
    img_train_major = np.concatenate(img_train_major, axis=0)
    img_test_major = np.concatenate(img_test_major, axis=0)

    # fmri_name_test_major = np.concatenate(fmri_name_test_major, axis=0)
    img_name_test_major = np.concatenate(img_name_test_major, axis=0)
    # fmri_name_train_major = np.concatenate(fmri_name_train_major, axis=0)
    img_name_train_major = np.concatenate(img_name_train_major, axis=0)

    num_voxels = fmri_train_major.shape[-1]
    if isinstance(image_transform, list):
        return (BOLD5000_dataset_classify(fmri_train_major, img_train_major, img_name_train_major, fmri_transform, image_transform[0], num_voxels),
                BOLD5000_dataset_classify(fmri_test_major, img_test_major, img_name_test_major, torch.FloatTensor, image_transform[1], num_voxels))
    else:
        return (BOLD5000_dataset_classify(fmri_train_major, img_train_major, img_name_train_major, fmri_transform, image_transform, num_voxels),
                BOLD5000_dataset_classify(fmri_test_major, img_test_major, img_name_test_major, torch.FloatTensor, image_transform, num_voxels))
    
    
