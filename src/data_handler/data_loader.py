"""
Python package for loading the data and creating batches.
Code adapted from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""
from torch.utils import data
from PIL import Image
from torchvision import transforms

import os
import h5py
import numpy as np
import csv

import helper


# to be cleaned
class Dataset(data.Dataset):
    """
    This characterizes a custom PyTorch dataset.
    """
    def __init__(self, img_ids, labels, labels_hot, data_folder):
        """
        Initialization of the custom Dataset.
        :param img_ids: list of the id of the images in the dataset_path
        :param labels: list of all the corresponding labels
        :param labels_hot: labels in the one-hot format (for our case multiple hots)
        :param data_folder: the folder containing the data.
        :param scale: determines the type of the input images. If scale is 'gray', it will be converted to RGB in the
        __getitem__ function.
        """
        self.img_ids = img_ids
        self.labels = labels
        self.labels_hot = labels_hot
        self.data_folder = data_folder

        # resize the image to 256x256, then normalize it (required for using ImageNet pre-trained models)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        """
        :param index:
        :return: a dictionary accessible with sample['image'] and sample['label'] to obtain the image and the label
        respectively.
        """
        img_id = self.img_ids[index]
        img = Image.open(f'{self.data_folder}/{img_id}')  # load the image
        img = to_rgb(img)  # convert to gray scale to RGB
        img_tensor = self.preprocess(img)  # pre-processing the image

        labels = np.array(self.labels_hot[img_id])  # convert 1d list to np array (more easily converted to tensor)

        sample = {'image': img_tensor, 'label': labels}
        return sample


def init_data_loaders(params, loader_params):
    """
    This function creates the PyTorch data loaders.
    :param params:
    :param loader_params:
    :return:
    """
    batch_size = loader_params['batch_size']
    shuffle = loader_params['shuffle']
    num_workers = loader_params['num_workers']
    data_folder = params['data_folder']

    partition, labels, labels_hot = read_already_partitioned(params)

    # creating the train data loader
    train_set = Dataset(partition['train'], labels, labels_hot, data_folder)
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers)
    # creating the validation data loader
    val_set = Dataset(partition['validation'], labels, labels_hot, data_folder)
    val_loader = data.DataLoader(dataset=val_set, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)

    # creating the test data loader
    test_set = Dataset(partition['test'], labels, labels_hot, data_folder)
    test_loader = data.DataLoader(dataset=test_set, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def to_rgb(gray_image):
    """
    Converts the gray-scale image to RGB.
    :param gray_image:
    :return:
    """
    rgb = Image.new('RGB', gray_image.size)
    rgb.paste(gray_image)
    return rgb


def read_formatted_data(params):
    formatted_data_file = params['formatted_data_path']

    # create the formatted data file if not exists
    if not os.path.exists(formatted_data_file):
        print(f'In [read_formatted_data]: Reading the h5 file and saving the formatted data to: \n"{formatted_data_file}"')
        img_ids, labels, labels_hot = read_ids_and_labels(params['h5_file'])  # read img ids and labels from the h5 file
        save_formatted_data(params, img_ids, labels, labels_hot)  # save the formatted data for future use

    formatted_data = np.load(formatted_data_file, allow_pickle=True).item()
    img_ids = formatted_data['image_ids']  # not actually used since the image ids are already in the .txt files
    labels = formatted_data['labels']
    labels_hot = formatted_data['labels_hot']
    return img_ids, labels, labels_hot


def read_already_partitioned(params):
    """
    This function reads all the 112,120 images of the dataset, and partitions them according to the 'train_val_list.txt'
    and 'test_list.txt' files obtained from the dataset website. It also extracts the labels from the .h5 file already
    created to simplify this label extraction process. It then stores the extracted labels as a Python dictionary
    for future references (called 'formatted_data.npy').

    :param params:
    :return: the dictionaries partition, labels, labels_hot.
    partition is accessed through 'train', 'validation', or 'test' and returns the list of image ids corresponding to
    that set. labels is accesses through labels['img_id'] and returns the list of the labels corresponding to that
    image. The same holds for labels_hot except that it returns the (multiple) hot encoded version of the labels
    corresponding to the image.

    Notes:
        - Fro info about saving Python dictionary to file, refer to:
          https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file.
    """
    # reading labels and labels hot from the formatted data
    _, labels, labels_hot = read_formatted_data(params)
    # print(f'In [read_already_partitioned]: reading the formatted data: done.')

    # read the .txt file containing the image ids used for training, validation, and test
    with open(params['train_val_path'], 'r') as f:
        train_val_list = f.read().splitlines()

    with open(params['test_path'], 'r') as f:
        test_list = f.read().splitlines()

    # extract the train_val list to train and validation lists, based on the percentages in the paper
    train_size = int(.875 * len(train_val_list))  # 70% out of 80% for training, 10% out of 80% for validation
    train_list = train_val_list[:train_size]
    val_list = train_val_list[train_size:]

    # the value for each key is the list of ids for the corresponding set
    partition = {
        'train': train_list,
        'validation': val_list,
        'test': test_list
    }

    print(f'In [read_already_partitioned]: returning with size \n'
          f'train: {len(partition["train"]):,} \n'
          f'validation: {len(partition["validation"]):,} \n'
          f'test: {len(partition["test"]):,}')

    return partition, labels, labels_hot


# to be cleaned
def read_ids_and_labels(h5_file):
    """
    This function reads the ids and labels of the images in an h5 file.
    :param params:
    :return: img_ids, labels, labels_hot: img_ids is the list containing all the image names, labels and labels hot
    are two dictionaries. labels is accesses through labels['img_id'] and returns the list of the labels corresponding
    to that image. The same holds for labels_hot except that it returns the (multiple) hot encoded version of the labels
    corresponding to the image.
    """
    pathologies = {'Atelectasis': 0,
                   'Cardiomegaly': 1,
                   'Consolidation': 2,
                   'Edema': 3,
                   'Effusion': 4,
                   'Emphysema': 5,
                   'Fibrosis': 6,
                   'Hernia': 7,
                   'Infiltration': 8,
                   'Mass': 9,
                   'Nodule': 10,
                   'Pleural_Thickening': 11,
                   'Pneumonia': 12,
                   'Pneumothorax': 13}

    # read the h5 file containing information about the images of the dataset
    with h5py.File(h5_file, 'r') as h5_data:
        image_ids = h5_data['Image Index']
        finding_labels = h5_data['Finding Labels']

        print(f'In [read_ids_and_labels]: found {len(image_ids):,} images in the h5 file. Extracting the labels...')

        img_ids, labels, labels_hot = [], {}, {}
        # create disease encoding for each image in the data
        for i in range(len(image_ids)):
            img_id = image_ids[i].decode('utf-8')  # file name is stored as byte-formatted in the h5 file
            diseases = finding_labels[i].decode("utf-8") .split('|')  # diseases split by '|'
            # vector containing multiple hot elements based on what pathologies are present
            diseases_hot_enc = [0] * len(pathologies.keys())

            # if verbose:
            #    print('In [read_indices_and_labels]: diseases found:', diseases)

            # check if the patient has any diseases
            if diseases != ['']:
                pathology_ids = [pathologies[disease] for disease in diseases]  # get disease index
                # if verbose:
                #    print('In [read_indices_and_labels]: pathology indexes:', pathology_ids)

                for pathology_id in pathology_ids:
                    diseases_hot_enc[pathology_id] = 1  # change values from 0 to 1

            # if verbose:
            #    print('In [read_indices_and_labels]: pathology indexes: one_hot: ', diseases_hot_enc)

            img_ids.append(img_id)  # append to the lists
            labels.update({img_id: pathology_ids})  # adding to the dictionaries
            labels_hot.update({img_id: diseases_hot_enc})

    print('In [read_ids_and_labels]: reading imag ids and extracting the labels done.')
    return img_ids, labels, labels_hot


def save_formatted_data(params, image_ids, labels, labels_hot):
    """
    This function saves the image ids, labels, and labels_hot into an .npy file for further reference. This is useful
    because reading the .h5 file and extracting the labels and labels_hot every time is time-consuming for all the
    112120 images.
    :param params:
    :param image_ids:
    :param labels:
    :param labels_hot:
    :return:
    """
    # packing all the data into a dictionary
    formatted_data = {'image_ids': image_ids,
                      'labels': labels,
                      'labels_hot': labels_hot}

    # helper.make_dir_if_not_exists(params['formatted_data_path'].split('/')[:-1])
    np.save(params['formatted_data_path'], formatted_data)
    print('In [save_formatted_data]: save the formatted data.')


# to be refactored
def read_bbox(bbox_file):
    """
    Note: If one needs to get the class of the disease, he/she could look at the 'read_and_partition_data' and use a
    similar dictionary to convert the disease name to the disease class.
    :param bbox_file: the name of the bbox file. NOTE: this file should exist in the 'data' folder, and only the
    name of the file should be given to this function, like 'BBox_List_2017.csv', not the full path. See the test module
    for usage.
    :return: list containing the rows of the file.
    """
    # bbox_path = 'data/' + bbox_file_name
    with open(bbox_file, 'rt') as f:
        reader = csv.reader(f)
        rows = list(reader)[1:]  # ignoring the first row because it is the titles

    for row in rows:
        for idx in [2, 3, 4, 5]:
            row[idx] = float(row[idx])  # convert the box coordinates from str to float

    print(f'In [read_bbox]: read bounding boxes for {len(rows)} images')
    return rows
