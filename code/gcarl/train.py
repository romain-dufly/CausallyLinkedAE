"""GCaRL training"""


from datetime import datetime
import os.path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from gcarl import gcarl
from subfunc.showdata import *


# =============================================================
# =============================================================
def train(data,
          num_h_nodes,
          num_hz_nodes,
          num_hp_nodes,
          initial_learning_rate,
          momentum,
          max_steps,
          decay_steps,
          decay_factor,
          batch_size,
          train_dir,
          noise_factor=0,
          test_set=None,
          site_information=None,
          test_site_information=None,
          directions=None,
          weight_decay=0,
          weight_decay_l1=None,
          phi_type='maxout',
          phi_share=False,
          moving_average_decay=0.999,
          summary_steps=500,
          checkpoint_steps=10000,
          save_file='model.pt',
          load_file=None,
          device=None,
          random_seed=None):
    """Build and train a model
    Args:
        data: data [data, group, dim]
        num_h_nodes: number of nodes for each layer. 1D array [num_layer]
        num_hz_nodes:
        num_hp_nodes:
        initial_learning_rate: initial learning rate
        momentum: momentum parameter
        max_steps: number of iterations (mini-batches)
        decay_steps: decay steps
        decay_factor: decay factor
        batch_size: mini-batch size
        train_dir: save directory
        weight_decay: weight decay
        weight_decay_L1: (option) weight decay
        phi_type: model type of phi (needs to be consistent with the source model)
        phi_share: share parameters of phi across group-pairs or not
        moving_average_decay: (option) moving average decay of variables to be saved
        summary_steps: (option) interval to save summary
        checkpoint_steps: (option) interval to save checkpoint
        save_file: (option) name of model file to save
        load_file: (option) name of model file to load
        device: device to be used
        random_seed: (option) random seed
    Returns:
    """

    # set random_seed
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # num_data, num_group, num_dim = data.shape
    num_data = data[0].shape[0]
    num_group = len(data)
    num_dim = [xi.shape[-1] for xi in data]

    # decide device to be used
    if device is None:
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
    else:
        device = 'cuda:%d' % device if type(device) == int else device

    # define network
    model = gcarl.Net(num_group=num_group,
                      directions=directions,
                      num_xdim=num_dim,
                      h_sizes=num_h_nodes if num_h_nodes is not None else [num_dim],
                      hz_sizes=num_hz_nodes if num_hz_nodes is not None else [num_dim],
                      hp_sizes=num_hp_nodes,
                      phi_type=phi_type,
                      phi_share=phi_share)
    model = model.to(device)
    model.train()

    # define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    if weight_decay_l1 is None:
        optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=momentum)
    # optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=weight_decay)

    if type(decay_steps) == list:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_steps, gamma=decay_factor)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_factor)
    writer = SummaryWriter(log_dir=train_dir)

    state_dict_ema = model.state_dict()

    trained_step = 0
    if load_file is not None:
        print('Load trainable parameters from %s...' % load_file)
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trained_step = checkpoint['step']

    # training iteration
    for step in range(trained_step, max_steps):
        start_time = time.time()

        shuffle_by_site = not(site_information is None)

        # make shuffled batch
        if shuffle_by_site:
            # choose a random site
            site = np.random.choice(np.unique(site_information))
            idx = np.where(site_information == site)[0]
            random_idx = np.random.choice(idx, batch_size)
            x0 = [x[random_idx, :] for x in data]
            xast = [x.copy() for x in x0]
            fix_group = np.random.randint(num_group)  # single group has the same value to x0
            for mi, m in enumerate(np.setdiff1d(np.arange(num_group), fix_group)):
                idx_ast = np.random.choice(idx, batch_size)
                xast[m] = data[m][idx_ast, :]

        else:
            idx = np.random.choice(num_data, batch_size)
            x0 = [x[idx, :] for x in data]
            xast = [x.copy() for x in x0]
            fix_group = np.random.randint(num_group)  # single group has the same value to x0
            for mi, m in enumerate(np.setdiff1d(np.arange(num_group), fix_group)):
                idx_ast = np.random.choice(num_data, batch_size)
                xast[m] = data[m][idx_ast, :]
        
        # add some random noise to the batches
        if noise_factor > 0:
            x0 = [x0[i] + noise_factor * np.random.randn(*x0[i].shape) for i in range(num_group)]
            xast = [xast[i] + noise_factor * np.random.randn(*xast[i].shape) for i in range(num_group)]

        # convert to torch
        x_batch = [np.concatenate([x0[i], xast[i]], axis=0) for i in range(num_group)]
        x_torch = [torch.from_numpy(x_batch[i].astype(np.float32)).to(device) for i in range(num_group)]
        y_torch = torch.cat([torch.ones([batch_size]), torch.zeros([batch_size])]).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logits, h, phi = model(x_torch)
        loss = criterion(logits, y_torch)

        # add regularization
        if weight_decay_l1 is not None:
            l1list = []
            l2list = []
            # l1names = ['w', 'h.0.0.weight', 'h.1.0.weight']
            # l1names = ['w', 'h.0.0.weight']
            # l1names = ['w', 'h.1.0.weight']
            # l1names = ['h.0.0.weight', 'h.1.0.weight']
            # l1names = ['h.0.0.weight']
            l1names = ['h.1.0.weight']
            # l1names = ['w']
            for name, param in model.named_parameters():
                if param.requires_grad and not any([n1 == name for n1 in l1names]):
                    l2list.append(name)
                    loss = loss + torch.norm(param, p=2) * weight_decay
            for name, param in model.named_parameters():
                if param.requires_grad and any([n1 == name for n1 in l1names]):
                    l1list.append(name)
                    loss = loss + torch.norm(param, p=1) * weight_decay_l1

        loss.backward()
        optimizer.step()
        scheduler.step()

        if phi_type.startswith('lap'):
            model.w.data = model.w.data.clamp(max=0)

        # moving average of parameters
        state_dict_n = model.state_dict()
        for key in state_dict_ema:
            state_dict_ema[key] = moving_average_decay * state_dict_ema[key] \
                                  + (1.0 - moving_average_decay) * state_dict_n[key]

        # accuracy
        predicted = (logits > 0.0).float()
        accu_val = (predicted == y_torch).sum().item()/(batch_size*2)
        loss_val = loss.item()
        lr = scheduler.get_last_lr()[0]

        duration = time.time() - start_time

        assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

        # display stats
        if step % 100 == 0:
            num_examples_per_step = batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            format_str = '%s: step %d, lr = %f, loss = %.2f, accuracy = %3.2f (%.1f examples/sec; %.3f sec/batch)'
            print(format_str % (datetime.now(), step, lr, loss_val, accu_val * 100,
                                examples_per_sec, sec_per_batch))
            
            if test_set is not None:
                test_shuffle_by_site = not(test_site_information is None)
                if test_shuffle_by_site:
                    # choose a random site
                    site = np.random.choice(np.unique(test_site_information))
                    idx = np.where(test_site_information == site)[0]
                    random_idx = np.random.choice(idx, batch_size)
                    x0 = [x[random_idx, :] for x in test_set]
                    xast = [x.copy() for x in x0]
                    fix_group = np.random.randint(num_group)
                    for mi, m in enumerate(np.setdiff1d(np.arange(num_group), fix_group)):
                        idx_ast = np.random.choice(idx, batch_size)
                        xast[m] = test_set[m][idx_ast, :]
                    
                else:
                    if shuffle_by_site:
                        print('Warning: test data is not shuffled by site, but training data is')
                    idx = np.random.choice(test_set[0].shape[0], batch_size)
                    x0 = [x[idx, :] for x in test_set]
                    xast = [x.copy() for x in x0]
                    fix_group = np.random.randint(num_group)
                    for mi, m in enumerate(np.setdiff1d(np.arange(num_group), fix_group)):
                        idx_ast = np.random.choice(test_set[m].shape[0], batch_size)
                        xast[m] = test_set[m][idx_ast, :]
                
                x_batch = [np.concatenate([x0[i], xast[i]], axis=0) for i in range(num_group)]
                x_torch = [torch.from_numpy(x_batch[i].astype(np.float32)).to(device) for i in range(num_group)]
                y_torch = torch.cat([torch.ones([batch_size]), torch.zeros([batch_size])]).to(device)
                logits, h, phi = model(x_torch)
                loss = criterion(logits, y_torch)
                predicted = (logits > 0.0).float()
                accu_test = (predicted == y_torch).sum().item()/(batch_size*2)
                loss_test = loss.item()
                writer.add_scalar('scalar/test_loss', loss_test, step)
                writer.add_scalar('scalar/test_accu', accu_test, step)

                format_str = '%s: step %d, test_loss = %.2f, test_accuracy = %3.2f'
                print(format_str % (datetime.now(), step, loss_test, accu_test * 100))
                print('-----')

        # save summary
        if step % summary_steps == 0:
            writer.add_scalar('scalar/lr', lr, step)
            writer.add_scalar('scalar/loss', loss_val, step)
            writer.add_scalar('scalar/accu', accu_val, step)
            h_val = h.cpu().detach().numpy()
            h_comp = np.split(h_val, indices_or_sections=h.shape[1], axis=1)
            for (i, cm) in enumerate(h_comp):
                writer.add_histogram('h/h%d' % i, cm)
            for k, v in state_dict_n.items():
                writer.add_histogram('w/%s' % k, v)


        # save the model checkpoint periodically.
        if step % checkpoint_steps == 0:
            checkpoint_path = os.path.join(train_dir, save_file)
            torch.save({'step': step,
                        'model_state_dict': model.state_dict(),
                        'ema_state_dict': state_dict_ema,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()}, checkpoint_path)

    # save trained model
    save_path = os.path.join(train_dir, save_file)
    print('Save model in file: %s' % save_path)
    torch.save({'step': max_steps,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': state_dict_ema,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()}, save_path)
