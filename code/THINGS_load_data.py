### script to load all the necessary tables for the THINGS dataset
### also filters the ROIs, selects stable voxels, extracts image ResNet features...

from itertools import combinations
from os.path import join as pjoin
from scipy.io import loadmat

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

import sys
sys.path.append('../')

from thingsmri.dataset import ThingsmriLoader

def load_all(sub='01', selectedROI='glasser', ratio=1, test_size=0.2, reload_resnet=False):
    '''
    sub : the subject to use, from ['01','02','03']
    selectedROI : brain parcels to use. Either a list of str, or a str to use all parcels whose name contains selectedROI
    reload_resnet : recalculate the precomputed resnet features of the images. Set to True if the images changed.
    '''
    dl = ThingsmriLoader(
        thingsmri_dir='data/THINGS/'
    )
    responses, stimdata, voxdata = dl.load_responses(sub)

    # Selecting specific ROIs from voxdata
    if type(selectedROI) == str:
        selectedROI = voxdata.columns[voxdata.columns.str.contains(selectedROI)]

    roimask = voxdata[selectedROI].sum(axis=1).values.astype(bool)
    v_voxdata = voxdata[roimask].copy()

    # drop all other columns
    to_drop = [c for c in voxdata.columns if c not in list(selectedROI) + ['voxel_id']]
    v_voxdata.drop(columns=to_drop, inplace=True)

    # choose only trial_type = train
    v_stimdata = stimdata[stimdata['trial_type'] == 'train'].copy()
    v_stimdata = v_stimdata.drop(columns=['trial_type','session','run','subject_id'])
    v_responses = responses[roimask].copy()
    v_responses = v_responses[[int(e) for e in v_stimdata.index]]

    print("ROIs selected. Number of ROIs :", len(selectedROI))

    # stability selection
    t_stimdata = stimdata[stimdata['trial_type'] == 'test'].copy()
    t_responses = responses[roimask].copy()
    t_responses = t_responses[[int(e) for e in t_stimdata.index]]
    
    number_selected = int(ratio*t_responses.shape[0])

    data = t_responses.values.reshape(12, 100, -1)

    def stability_selection(data, n=None):
        """Return the indices of the n voxels with best stability"""
        n_repetitions, n_items, n_voxels = data.shape

        if n is None:
            n = n_voxels
        elif n > n_voxels:
            raise ValueError('n must be a number between 0 and ' + n_voxels)

        # Drop all voxels don't contain NaN's for any items
        non_nan_mask = ~np.any(np.any(np.isnan(data), axis=1), axis=0)
        non_nan_indices = np.flatnonzero(non_nan_mask)
        data_trimmed = data[:, :, non_nan_mask]

        data_means = data_trimmed.mean(axis=1)
        data_stds = data_trimmed.std(axis=1)

        # Loop over all pairwise combinations and compute correlations
        stability_scores = []
        for x, y in combinations(range(n_repetitions), 2):
            x1 = (data_trimmed[x] - data_means[x]) / data_stds[x]
            y1 = (data_trimmed[y] - data_means[y]) / data_stds[y]
            stability_scores.append(np.sum(x1 * y1, axis=0) / n_items)

        # Compute the N best voxels
        best_voxels = np.mean(stability_scores, axis=0).argsort()[-n:]
        scores = np.mean(stability_scores, axis=0)

        # Return the (original) indices of the best voxels in decreasing order
        return non_nan_indices[best_voxels][::-1], scores

    if ratio < 1:
        top_indices, scores = stability_selection(data, n=number_selected)

        voxels_per_roi = np.zeros(180)
        score_per_roi = np.zeros(180)
        errors = 0

        for voxel_index in top_indices:
            voxel = t_responses.index[voxel_index]
            if sum(voxdata.iloc[voxel, -180:]) != 1:
                errors += 1
            else:
                roi = np.argmax(voxdata.iloc[voxel, -180:].values)
                voxels_per_roi[roi] += 1
                score_per_roi[roi] += scores[voxel_index]
        print("ROI not found for", errors, "voxels")

        s_responses = v_responses.iloc[top_indices]
        print("Selected", number_selected, "voxels from the original", t_responses.shape[0])
    else:
        s_responses = v_responses
        print("Skipped stability selection")
    s_stimdata = v_stimdata

    # Load images
    images_path = '/home/aip/dufly/Documents/THINGS-CRL/images'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if reload_resnet:
        print(f'Using {device} for feature extraction')

        resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)

        resnet50.eval().to(device)
        resnet50.fc = nn.Identity()

        def extract_features_from_images(images_path, model, stim):
            features = []
            images = []
            for i, row in stim.iterrows():
                img = Image.open(pjoin(images_path, row['concept'], row['stimulus']))
                img = img.convert('RGB')
                img = img.resize((224, 224))
                img_torch = transforms.ToTensor()(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(img_torch)
                features.append(output.cpu().numpy().flatten())
                images.append(img)
            return np.array(features), np.array(images)
        
        s_image_features, images = extract_features_from_images(images_path, resnet50, s_stimdata)
        print("Shape of extracted features table:", s_image_features.shape)
        print("Shape of images table:", images.shape)
        np.save('s_image_features.npy', s_image_features)
    else:
        s_image_features = np.load('s_image_features.npy')

    # Load semantic embeddings and categories

    semantic_emb = pd.read_csv('semantic_emb.csv', header=None)
    semantic_emb = semantic_emb.fillna(0)
    categories = pd.read_csv('/home/aip/dufly/Documents/THINGS-CRL/MRI/data/Categories_final_20200131.tsv', sep='\t')
    categories = categories.fillna(0)

    # create a new table. for each row in stimdata, retrieve the semantic embedding by matching the location of the concept in the categories table

    full_semantic_emb = np.zeros((s_stimdata.shape[0], semantic_emb.shape[1]))
    for i, row in enumerate(s_stimdata.iterrows()):
        row = row[1]
        concept = row['concept']
        concept = concept.replace('_', ' ')
        idx = categories[categories['Word'] == concept].index[0]
        full_semantic_emb[i] = semantic_emb.iloc[idx].values

    with open('/home/aip/dufly/Documents/THINGS-CRL/spose_embedding_49d_sorted.txt', 'r') as f:
        lines = f.readlines()
        simi_emb = np.zeros((len(lines), 49))
        for i, line in enumerate(lines):
            simi_emb[i] = np.array([float(e) for e in line.split()])

    labels_short = loadmat('/home/aip/dufly/Documents/THINGS-CRL/labels_short49.mat')
    labels_short = labels_short['labels_short']
    labels = [l[0] for l in labels_short[0]]

    full_similarity_emb = np.zeros((s_stimdata.shape[0], simi_emb.shape[1]))
    for i, row in enumerate(s_stimdata.iterrows()):
        row = row[1]
        concept = row['concept']
        concept = concept.replace('_', ' ')
        idx = categories[categories['Word'] == concept].index[0]
        full_similarity_emb[i] = simi_emb[idx]

    full_categories = np.zeros((s_stimdata.shape[0], categories.shape[1]-2))
    for i, row in enumerate(s_stimdata.iterrows()):
        row = row[1]
        concept = row['concept']
        concept = concept.replace('_', ' ')
        idx = categories[categories['Word'] == concept].index[0]
        full_categories[i] = categories.iloc[idx][2:].values

    # set dummy images variables if reload_resnet is False

    if not reload_resnet:
        images = np.zeros((s_stimdata.shape[0], 224, 224, 3))

    # split into train and test

    i_train, i_test = train_test_split(range(s_image_features.shape[0]), test_size=test_size)
    vox_train, vox_test = s_responses.T.iloc[i_train].to_numpy(), s_responses.T.iloc[i_test].to_numpy()
    imf_train, imf_test = s_image_features[i_train], s_image_features[i_test]
    #pca_train, pca_test = pca_responses[i_train], pca_responses[i_test]
    sem_train, sem_test = full_semantic_emb[i_train], full_semantic_emb[i_test]
    simi_train, simi_test = full_similarity_emb[i_train], full_similarity_emb[i_test]
    cat_train, cat_test = full_categories[i_train], full_categories[i_test]
    im_train, im_test = images[i_train], images[i_test]

    full_sets = (s_responses, v_voxdata, s_stimdata, images, s_image_features, full_semantic_emb, full_similarity_emb, full_categories)
    train_sets = (vox_train, im_train, imf_train, sem_train, simi_train, cat_train)
    test_sets = (vox_test, im_test, imf_test, sem_test, simi_test, cat_test)

    return full_sets, train_sets, test_sets, labels