import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
import time

sys.path.append('../')
sys.path.append('code/')

from phase2_finetune_baseline import load_model_image
from dataset import create_Kamitani_dataset_distill, create_BOLD5000_dataset_classify

from sc_mbm.mae_for_fmri import MAEforFMRICross
from config import Config_MBM_finetune_cross
from phase2_finetune_baseline import eval_one_epoch


def god_bold_test(model_path, config_path):

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(model_path, map_location='cpu')
    config_ckpt = torch.load(config_path, map_location='cpu')
    config_pretrain = config_ckpt['config']
    config = Config_MBM_finetune_cross()
    config.do_cross_attention = False

    model_image, model_image_config, image_feature_extractor = load_model_image(config)
    model_image.to(device)

    num_voxels = (ckpt['model_fmri']['pos_embed'].shape[1] - 1)* config_pretrain.patch_size

    model = MAEforFMRICross(num_voxels=num_voxels, patch_size=config_pretrain.patch_size, embed_dim=config_pretrain.embed_dim,
                            decoder_embed_dim=config_pretrain.decoder_embed_dim, depth=config_pretrain.depth, 
                            num_heads=config_pretrain.num_heads, decoder_num_heads=config_pretrain.decoder_num_heads, 
                            mlp_ratio=config_pretrain.mlp_ratio, focus_range=None, use_nature_img_loss=False, 
                            do_cross_attention=False, cross_encoder_config=model_image_config,
                            decoder_depth=config.fmri_decoder_layers)

    model.load_state_dict(ckpt['model_fmri'])
    model.to(device)
    model.eval()

    model_image.load_state_dict(ckpt['model_image'])
    model_image.eval()
    
    ## load all subjects for god
    train = []
    test = []

    for sbj in ['sbj_1', 'sbj_2', 'sbj_3', 'sbj_4', 'sbj_5']:
        train_set, test_set = load_dataset('GOD', sbj, config_pretrain, config, num_voxels)
        train.append(train_set)
        test.append(test_set)

    start_time = time.time()

    saved_metrics = []

    for i in range(5):
        print(f'Subject {i+1}')
        print('Train')
        eval_out_train = eval_one_epoch(model, model_image, train[i], device, 0, log_writer=None, config=config, 
                            start_time=start_time, model_without_ddp=model, img_feature_extractor=image_feature_extractor)
        print('Test')
        eval_out_test = eval_one_epoch(model, model_image, test[i], device, 0, log_writer=None, config=config, 
                            start_time=start_time, model_without_ddp=model, img_feature_extractor=image_feature_extractor)
        saved_metrics.append((eval_out_train, eval_out_test))

    ## load all subjects for bold5000
    btrain = []
    btest = []

    for sbj in ['CSI1', 'CSI2', 'CSI3', 'CSI4']:
        train_set, test_set = load_dataset('BOLD5000', sbj, config_pretrain, config, num_voxels)
        btrain.append(train_set)
        btest.append(test_set)

    start_time = time.time()

    bsaved_metrics = []

    for i in range(4):
        print(f'Subject {i+1}')
        print('Train')
        eval_out_train = eval_one_epoch(model, model_image, btrain[i], device, 0, log_writer=None, config=config, 
                            start_time=start_time, model_without_ddp=model, img_feature_extractor=image_feature_extractor)
        print('Test')
        eval_out_test = eval_one_epoch(model, model_image, btest[i], device, 0, log_writer=None, config=config, 
                            start_time=start_time, model_without_ddp=model, img_feature_extractor=image_feature_extractor)
        bsaved_metrics.append((eval_out_train, eval_out_test))

    return saved_metrics, bsaved_metrics


def load_dataset(dataset, sbj, config_pretrain, config, num_voxels):
    if dataset == 'GOD':
        print('Dataset: GOD')
        train_set, test_set = create_Kamitani_dataset_distill(path=config.kam_path, patch_size=config_pretrain.patch_size, 
                                subjects=sbj, fmri_transform=torch.FloatTensor, include_nonavg_test=config.include_nonavg_test,
                                return_image_name=True)
    elif dataset == 'BOLD5000':
        print('Dataset: BOLD5000')
        train_set, test_set = create_BOLD5000_dataset_classify(path=config.bold5000_path, patch_size=config_pretrain.patch_size, 
                fmri_transform=torch.FloatTensor, subjects=sbj, include_nonavg_test=config.include_nonavg_test)
    else:
        raise NotImplementedError

    if train_set.fmri.shape[-1] < num_voxels:
        train_set.fmri = np.pad(train_set.fmri, ((0,0), (0, num_voxels - train_set.fmri.shape[-1])), 'wrap')
    else:
        train_set.fmri = train_set.fmri[:, :num_voxels]

    if test_set.fmri.shape[-1] < num_voxels:
        test_set.fmri = np.pad(test_set.fmri, ((0,0), (0, num_voxels - test_set.fmri.shape[-1])), 'wrap')
    else:
        test_set.fmri = test_set.fmri[:, :num_voxels]

    # scarmble test_set with random data
    #test_set.image = np.array([np.random.permutation(test_set.image[i]) for i in range(len(test_set.image))])

    sampler = torch.utils.data.DistributedSampler(train_set) if (torch.cuda.device_count() > 1 and config.distr) else torch.utils.data.RandomSampler(train_set) 
    dataloader_hcp = DataLoader(train_set, batch_size=config.batch_size, sampler=sampler)
    test_sampler = torch.utils.data.DistributedSampler(test_set) if (torch.cuda.device_count() > 1 and config.distr) else torch.utils.data.RandomSampler(test_set) 
    dataloader_hcp_test = DataLoader(test_set, batch_size=config.batch_size, sampler=test_sampler)

    return dataloader_hcp, dataloader_hcp_test