
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import os
import sys
import datetime
import time
import shutil
import math

sys.path.append('../')
sys.path.append('code/')

from sc_mbm.trainer import NativeScalerWithGradNormCount as NativeScaler

from config import Config_MBM_finetune_cross, merge_needed_cross_config
from dataset import create_Kamitani_dataset_distill, create_BOLD5000_dataset_classify, create_THINGS_dataset
from sc_mbm.mae_for_fmri import MAEforFMRICross
from sc_mbm.mae_for_image import ViTMAEForPreTraining, ViTMAEConfig
from sc_mbm.trainer import NativeScalerWithGradNormCount as NativeScaler
from sc_mbm.utils import save_model_merge_conf
import sc_mbm.utils as ut

from transformers import AutoFeatureExtractor

def finetune_baseline(config_update):
    config = Config_MBM_finetune_cross()
    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')

    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print('No GPU available, using the CPU')
    else:
        try:
            os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
            memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
            print(memory_available)
            device = 'cuda:%d' % np.argmax(memory_available)
            print(device)
        except Exception as e:
            print(e)
            print('This is exception')
            device = torch.device('cuda')

    if config_update.pretrain_path is not None:
        config.pretrain_mbm_path = config_update.pretrain_path
    if config_update.fmri_recon_weight != -1 :
        config.fmri_recon_weight = config_update.fmri_recon_weight
    if config_update.img_recon_weight != -1 :
        config.img_recon_weight = config_update.img_recon_weight
    if config_update.img_mask_ratio != -1 :
        config.img_mask_ratio = config_update.img_mask_ratio
    if config_update.mask_ratio != -1 :
        config.mask_ratio = config_update.mask_ratio
    if config_update.batch_size != -1 :
        config.batch_size = config_update.batch_size
    if config_update.dataset != None :
        config.dataset = config_update.dataset
    if config_update.num_epoch != -1:  
        config.num_epoch = config_update.num_epoch

    if config_update.do_cross_attention != None:
        config.do_cross_attention = config_update.do_cross_attention

    print("Batch size:", config.batch_size)
    print("Number of epochs:", config.num_epoch)

    sd = torch.load(config.pretrain_mbm_path, map_location='cpu')
    config_pretrain = sd['config']

    if config_update.seed != -1:
        config_pretrain.seed = config_update.seed
    torch.manual_seed(config_pretrain.seed)
    np.random.seed(config_pretrain.seed)

    output_sub = config.bold5000_subs if config.dataset == 'BOLD5000' else config.kam_subs
    output_path = os.path.join(config.output_path, 'results', 'fmri_finetune_{}_{}'.format(config.dataset, output_sub),  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))

    config.output_path = output_path
    os.makedirs(output_path, exist_ok=True)

    ## Load the pretrained model

    model_image, model_image_config, image_feature_extractor = load_model_image(config)
    model_image.to(device)

    # num_voxels = 4480

    # pretrained model
    num_voxels = (sd['model']['pos_embed'].shape[1] - 1)* config_pretrain.patch_size
    if config.dataset == 'THINGS' and (num_voxels != 7664) and False: # deactivated for now
        size_mismatch = True
        num_voxels = 7664
    else:
        size_mismatch = False

    print("Cross attention:", config.do_cross_attention)
    print("Number of voxels:", num_voxels)
    print("Cross residual", model_image_config.do_cross_residual)

    model = MAEforFMRICross(num_voxels=num_voxels, patch_size=config_pretrain.patch_size, embed_dim=config_pretrain.embed_dim,
                        decoder_embed_dim=config_pretrain.decoder_embed_dim, depth=config_pretrain.depth, 
                        num_heads=config_pretrain.num_heads, decoder_num_heads=config_pretrain.decoder_num_heads, 
                        mlp_ratio=config_pretrain.mlp_ratio, focus_range=None, use_nature_img_loss=False, 
                        do_cross_attention=config.do_cross_attention, cross_encoder_config=model_image_config,
                        decoder_depth=config.fmri_decoder_layers)

    ### Load pretrained model

    if not(size_mismatch):
        model.load_state_dict(sd['model'], strict=False)
        print('Loaded model from', config.pretrain_mbm_path, ", number of voxels", num_voxels)
    else:
        print('Did not load pretrained model due to size mismatch with THINGS dataset')

    #for param in model.blocks.parameters():
    #    param.requires_grad = False

    #for param in model.decoder_blocks.parameters():
    #    param.requires_grad = False

    model.to(device)
    model_without_ddp = model

    ## Load the dataset

    if config.dataset == 'GOD':
        print('Dataset: GOD')
        train_set, test_set = create_Kamitani_dataset_distill(path=config.kam_path, patch_size=config_pretrain.patch_size, 
                                subjects=config.kam_subs, fmri_transform=torch.FloatTensor, include_nonavg_test=config.include_nonavg_test,
                                return_image_name=True)
    elif config.dataset == 'BOLD5000':
        print('Dataset: BOLD5000')
        train_set, test_set = create_BOLD5000_dataset_classify(path=config.bold5000_path, patch_size=config_pretrain.patch_size, 
                fmri_transform=torch.FloatTensor, subjects=config.bold5000_subs, include_nonavg_test=config.include_nonavg_test)
    elif config.dataset == 'THINGS':
        print('Dataset: THINGS')
        train_set, test_set = create_THINGS_dataset(patch_size=config_pretrain.patch_size)
    else:
        raise NotImplementedError

    ## Adjust padding

    print('Original train_set.fmri.shape:', train_set.fmri.shape)
    print('Original test_set.fmri.shape:', test_set.fmri.shape)

    if train_set.fmri.shape[-1] < num_voxels:
        train_set.fmri = np.pad(train_set.fmri, ((0,0), (0, num_voxels - train_set.fmri.shape[-1])), 'wrap')
    else:
        train_set.fmri = train_set.fmri[:, :num_voxels]

    if test_set.fmri.shape[-1] < num_voxels:
        test_set.fmri = np.pad(test_set.fmri, ((0,0), (0, num_voxels - test_set.fmri.shape[-1])), 'wrap')
    else:
        test_set.fmri = test_set.fmri[:, :num_voxels]

    print('New train_set.fmri.shape:', train_set.fmri.shape)
    print('New test_set.fmri.shape:', test_set.fmri.shape)

    print(f'Dataset size: {len(train_set)}, {len(test_set)}')

    sampler = torch.utils.data.DistributedSampler(train_set) if (torch.cuda.device_count() > 1 and config.distr) else torch.utils.data.RandomSampler(train_set) 
    dataloader_hcp = DataLoader(train_set, batch_size=config.batch_size, sampler=sampler)
    test_sampler = torch.utils.data.DistributedSampler(test_set) if (torch.cuda.device_count() > 1 and config.distr) else torch.utils.data.RandomSampler(test_set) 
    dataloader_hcp_test = DataLoader(test_set, batch_size=config.batch_size, sampler=test_sampler)

    ## Set optimizer

    param_groups = add_weight_decay([model, model_image], config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epoch)

    ## Start training

    print('Finetuning MAE on train fMRI ... ...')
    num_ep = config.num_epoch

    addition_config = {'num_voxels':num_voxels}
    merged_config = merge_needed_cross_config(config_pretrain, config, model_image_config, addition_config)

    # define two arrays to hold training metrics and evaluation metrics
    train_metrics = np.zeros((num_ep, 5))
    eval_metrics = np.zeros((num_ep, 4))
    
    eval_cor_init = 0.5
    best_eval_corr_epoch = 0
    saved_epoch_list = []
    start_time = time.time()

    addition_config = {'num_voxels':num_voxels}
    merged_config = merge_needed_cross_config(config_pretrain, config, model_image_config, addition_config)

    num_ep = config.num_epoch

    for ep in range(num_ep):

        #print(model.cross_map_in.weight)
        #print(model.cross_map_in.bias)

        ckpt_file_name = f'checkpoint_singlesub_{config.wandb_name}_epo{ep}_mergconf.pth'

        train_out = train_one_epoch(model, model_image, dataloader_hcp, optimizer, device, ep, loss_scaler, None, config, start_time, model_without_ddp,
                                    img_feature_extractor=image_feature_extractor, fmri_recon_weight=config.fmri_recon_weight, 
                                    img_recon_weight=config.img_recon_weight)
        eval_out = eval_one_epoch(model, model_image, dataloader_hcp_test, device, ep, None, config, start_time, model_without_ddp,  
                                        img_feature_extractor=image_feature_extractor) 
        for i,e in enumerate(eval_out):
            eval_metrics[ep][i] = e
        for i,t in enumerate(train_out):
            train_metrics[ep][i] = t
        cor = train_out[0]
        ecor, ecor_img = eval_out[0], eval_out[1]

        if (ep != 0 and (2*ecor + ecor_img) > eval_cor_init) or ep==num_ep - 1:
            save_path = os.path.join(output_path, f'checkpoints_{ep}')
            print('Saving models in file: %s' % output_path)

            save_model_merge_conf(config_pretrain, ep, model_without_ddp, optimizer, loss_scaler, save_path, merged_config, ckpt_file_name)

            to_save = {'model_fmri': model.state_dict(),
                'model_image': model_image.state_dict(),}
            torch.save(to_save, os.path.join(save_path, 'checkpoint_models.pth'))

            eval_cor_init = (2*ecor + ecor_img)
            best_eval_corr_epoch = ep
            saved_epoch_list.append(ep)
            if len(saved_epoch_list) > 3:
                for del_ep in saved_epoch_list[:-3]:
                    print('Deleting model at ep {}'.format(del_ep))
                    shutil.rmtree(os.path.join(output_path, f'checkpoints_{del_ep}'))
                saved_epoch_list = saved_epoch_list[-3:]
            print('Saving model at ep {} eval_image_corr_train {} eval_image_corr_test {}, best ep {},  best eval_loss {}'.format(
                ep, cor, ecor, best_eval_corr_epoch, eval_cor_init))
        
    print('Training done, best joint correlation:', eval_cor_init/2, 'at epoch:', best_eval_corr_epoch)
    return train_metrics, eval_metrics, (output_path, ckpt_file_name, best_eval_corr_epoch)

def load_model_image(config):
    image_feature_extractor = AutoFeatureExtractor.from_pretrained(config.vit_mae_model)
    model_image_config =  ViTMAEConfig.from_pretrained(config.vit_mae_model)
    model_image = ViTMAEForPreTraining.from_pretrained(config.vit_mae_model)

    model_image_config.num_cross_encoder_layers = config.num_cross_encoder_layers
    # model_image_config.do_cross_attention = config.do_cross_attention  ##################################
    model_image_config.do_cross_residual = config.do_cross_residual
    model_image_config.decoder_num_hidden_layers = config.img_decoder_layers
    model_image_new = ViTMAEForPreTraining(model_image_config, do_cross_attention=config.do_cross_attention)

    pretrained_state_dict = model_image.state_dict()
    new_model_state_dict = model_image_new.state_dict()
    for key in pretrained_state_dict.keys():
        if key in new_model_state_dict:
            new_model_state_dict[key] = pretrained_state_dict[key]

    model_image_new.load_state_dict(new_model_state_dict)

    # model_image_new.vit.eval()

    for param in model_image_new.vit.parameters():
        param.requires_grad = False

    model_image_new.decoder.train()
    
    for param in model_image_new.decoder.parameters():
        param.requires_grad = True

    return model_image_new, model_image_config, image_feature_extractor

def add_weight_decay(models, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for model in models:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def train_one_epoch(model, model_image, data_loader, optimizer, device, epoch, 
                        loss_scaler, log_writer=None, config=None, start_time=None, model_without_ddp=None, 
                        img_feature_extractor=None, preprocess=None, optimizer_img=None, loss_scaler_img=None,
                        fmri_recon_weight=1.0, img_recon_weight=1.0):
    model.train(True)
    model_image.train(True)
    optimizer.zero_grad()
    total_loss = []
    total_loss_image = []
    total_cor = []
    total_cor_image = []
    total_full_loss = []
    accum_iter = config.accum_iter

    for data_iter_step, (data_dcit) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            ut.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config) # per iteration (instead of per epoch) lr scheduler

        samples = data_dcit['fmri']
        images = data_dcit['image']
        valid_idx = torch.nonzero(images.sum(dim=(1,2,3)) != 0).squeeze(1)

        img_prep = img_feature_extractor(images=images, return_tensors="pt")
        img_prep["pixel_values"] = img_prep["pixel_values"].to(device)
        samples = samples.to(device)

        optimizer.zero_grad()
        #optimizer_crl.zero_grad()

        with torch.cuda.amp.autocast(enabled=True):
            # reconstruct fmri
            img_support = model_image(pixel_values=img_prep["pixel_values"], given_mask_ratio=0, encoder_only=True)
            loss_fmri_recon, pred, _ = model(samples, valid_idx=valid_idx, mask_ratio=config.mask_ratio, image_support=img_support.last_hidden_state)
            # reconstruct image
            fmri_support = model(samples, mask_ratio=0, encoder_only=True)
            fmri_support = None
            img_recons_output = model_image(pixel_values=img_prep["pixel_values"], given_mask_ratio=config.img_mask_ratio, fmri_support=fmri_support)

        loss = fmri_recon_weight*loss_fmri_recon + img_recon_weight*img_recons_output.loss

        loss_value = loss.item()
        #lr = scheduler.get_last_lr()[0]

        #if optimizer_img is None or loss_scaler_img is None:
        #    loss_scaler(loss, optimizer, parameters=[model.parameters(), model_image.parameters()], clip_grad=config.clip_grad)
        #else:
        #    loss_scaler(loss_fmri_recon, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)
        #    loss_scaler_img(img_recons_output.loss, optimizer_img, parameters=model_image.parameters(), clip_grad=config.clip_grad)

        loss.backward(retain_graph=True)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optimizer.step()
        #scheduler.step()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        pred = pred.to('cpu').detach()
        samples = samples.to('cpu').detach()
        pred = model_without_ddp.unpatchify(pred)

        cor = torch.mean(torch.tensor([torch.corrcoef(torch.cat([p, s],axis=0))[0,1] for p, s in zip(pred, samples)])).item()

        cor_image = img_recons_output.corr
        optimizer.zero_grad()

        total_loss.append(loss_fmri_recon.item())
        total_loss_image.append(img_recons_output.loss.item())
        total_cor.append(cor)
        total_cor_image.append(cor_image)
        total_full_loss.append(loss_value)
   
    print(f'[Epoch {epoch}] Train loss/corr fMRI: {sci(np.mean(total_loss))} | {sci(np.mean(total_cor))}')
    print(f'           Train loss/corr image: {sci(np.mean(total_loss_image))} | {sci(np.mean(total_cor_image))}')
    #print(f'           LR: {sci(lr)}')
    print(f'           Train loss TOTAL: {sci(np.mean(total_full_loss))}')

    return np.mean(total_cor), np.mean(total_cor_image), np.mean(total_loss), np.mean(total_loss_image), np.mean(total_full_loss)

def eval_one_epoch(model, model_image, data_loader, device, epoch, log_writer=None, config=None, 
                        start_time=None, model_without_ddp=None, img_feature_extractor=None):
    model.eval()
    model_image.eval()
    total_loss = []
    total_loss_image = []
    total_cor = []
    total_cor_image = []

    for data_iter_step, (data_dcit) in enumerate(data_loader):
        
        samples = data_dcit['fmri']
        images = data_dcit['image']
        valid_idx = torch.nonzero(images.sum(dim=(1,2,3)) != 0).squeeze(1)

        img_prep = img_feature_extractor(images=images, return_tensors="pt")
        img_prep["pixel_values"] = img_prep["pixel_values"].to(device)
        samples = samples.to(device)

        with torch.no_grad():
            # reconstruct fmri
            img_support = model_image(pixel_values=img_prep["pixel_values"], given_mask_ratio=0, encoder_only=True)
            #img_support.last_hidden_state = None
            loss_fmri_recon, pred, _ = model(samples, valid_idx=valid_idx, mask_ratio=config.mask_ratio, image_support=img_support.last_hidden_state)
            # reconstruct image
            fmri_support = model(samples, mask_ratio=0, encoder_only=True)
            fmri_support = None
            # print('fmri_support ', fmri_support.shape)
            img_recons_output = model_image(pixel_values=img_prep["pixel_values"], given_mask_ratio=config.img_mask_ratio, fmri_support=fmri_support)

        loss = loss_fmri_recon + img_recons_output.loss
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        # loss_scaler(img_recons_output.loss, optimizer, parameters=model_image.parameters(), clip_grad=config.clip_grad)

        pred = pred.to('cpu').detach()
        samples = samples.to('cpu').detach()
        pred = model_without_ddp.unpatchify(pred)
        cor = torch.mean(torch.tensor([torch.corrcoef(torch.cat([p, s],axis=0))[0,1] for p, s in zip(pred, samples)])).item()
        cor_image = img_recons_output.corr

        total_loss.append(loss_fmri_recon.item())
        total_loss_image.append(img_recons_output.loss.item())
        total_cor.append(cor)
        total_cor_image.append(cor_image)

    if config.local_rank == 0:        
        # convert all printed numbers to scientific format with 3 decimal points
        print(f'[Test {epoch}] Test loss/corr fMRI: {sci(np.mean(total_loss))} | {sci(np.mean(total_cor))}')
        print(f'           Test loss/corr image: {sci(np.mean(total_loss_image))} | {sci(np.mean(total_cor_image))}')
        print(f'Epoch end, total time: {int((time.time() - start_time) // 60)}:{"0" if int((time.time() - start_time) % 60)<10 else ""}{int((time.time() - start_time) % 60)}')

    return np.mean(total_cor), np.mean(total_cor_image), np.mean(total_loss), np.mean(total_loss_image)

def sci(num):
    if num < 0.01:
        return "{:.3e}".format(num)
    else:
        return "{:.3f}".format(num)