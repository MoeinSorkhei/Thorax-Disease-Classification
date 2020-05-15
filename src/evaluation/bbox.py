from queue import Queue
from PIL import Image
from torchvision import transforms

import csv
import cv2
import numpy as np
import os.path

import data_handler
import helper
import networks
from . import plotting

class BBox():
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def scale(self, scaleX, scaleY=None):
        """
        :param scaleX: x and width scale.
        :param scaleY: y and height scale. If not given it will scale uniformly.
        """
        if scaleY is None:
            scaleY = scaleX

        self.x *= scaleX
        self.y *= scaleY
        self.width  *= scaleX
        self.height *= scaleY

    def IoU(self, other):
        """
        Calculates and returns the Intersection over Union between this BBox and the other BBox.
        IoU is defined as a simple ratio: <intersected area> / <union area>
        
        :param other: the other BBox to intersect with.
        :return: the Intersection over Union, between 0 and 1.
        """
        # Calculate top-left and bottom-right points of intersection.
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)

        intersectArea = max(0, x2 - x1) * max(0, y2 - y1)
        unionArea = self.area() + other.area() - intersectArea

        return intersectArea / unionArea

    def IoBB(self, ground_truth):
        """
        Very similar to IoU, but instead of dividing by union area we divide by
        the area of the predicted BBox. Thus, if the predicted BBox is completely
        inside the ground truth BBox but is much smaller, IoBB will still be 1.

        Also known as Intersection over Region (IoR).
        
        :param other: the ground truth BBox. Does NOT work the other way around.
        :return: the Intersection over detected Bounding Box, between 0 and 1.
        """
        # Calculate top-left and bottom-right points of intersection.
        x1 = max(self.x, ground_truth.x)
        y1 = max(self.y, ground_truth.y)
        x2 = min(self.x + self.width, ground_truth.x + ground_truth.width)
        y2 = min(self.y + self.height, ground_truth.y + ground_truth.height)

        intersectArea = max(0, x2 - x1) * max(0, y2 - y1)

        return intersectArea / self.area()

    def __str__(self):
        return f'x:{self.x} y:{self.y}, width:{self.width}, height:{self.height}'


def bbox_from_point(bin_img, x, y):
    """
    Generates a BBox by starting at (x,y) and traversing the binary img, finding
    all connected 1's.

    :param bin_img: a binary 2D array.
    :param x: starting x.
    :param y: starting y.
    :return: a BBox fully covering the area of 1's connected to (x,y) in bin_img.
    """
    visited = np.zeros_like(bin_img, dtype=bool)
    q = Queue()
    q.put((y, x))

    # Height and width. -1 for simpler conditionals in while loop.
    h = bin_img.shape[0] - 1
    w = bin_img.shape[1] - 1

    # First populate the visited matrix by using a queue.
    while not q.empty():
        yx = q.get()

        # Don't process this position if already visited or corresponds to 0 in img.
        if visited[yx] or bin_img[yx] == 0:
            continue
        visited[yx] = True

        # For every direction, check that new position is valid and add to queue.
        if yx[0] > 0:
            q.put((yx[0] - 1, yx[1]))
        
        if yx[0] < h:
            q.put((yx[0] + 1, yx[1]))
        
        if yx[1] > 0:
            q.put((yx[0], yx[1] - 1))

        if yx[1] < w:
            q.put((yx[0], yx[1] + 1))

    # Columns and rows in visited that contain True.
    cols = np.nonzero(visited.any(axis=0))[0]
    rows = np.nonzero(visited.any(axis=1))[0]
    
    return BBox(cols[0], rows[0], cols.size, rows.size)


def evaluate_iobb(args, params, img_name=None, img_disease=None):
    """
    Goes through every image in the BBox csv file, and calculates IoBB.
    Results are stored in data/iobb.npy together with image name and disease type.

    :param img_name: If given, only load one image and plot bbox and heatmap.
    :param img_disease: Which disease to plot bbox for, used together with img_name.
    """
    pathologies = {'Atelectasis': 0,
                   'Cardiomegaly': 1,
                   'Effusion': 4,
                   'Infiltrate': 8,
                   'Mass': 9,
                   'Nodule': 10,
                   'Pneumonia': 12,
                   'Pneumothorax': 13}

    ############################################
    ### Load model checkpoint and get heatmap.
    ############################################
    path_to_load = helper.compute_paths(args, params)['save_path']
    net = networks.init_unified_net(args.model, params)
    net, _ = helper.load_model(path_to_load, args.epoch, net, optimizer=None, resume_train=False)

    # Get bbox_data from csv file.
    f = open('../data/BBox_List_2017.csv', 'rt')
    reader = csv.reader(f)
    rows = list(reader)[1:]  # ignoring the first row because it is the titles

    # A list of tuples (img_name, disease_index, iobb)
    results = []

    for i, img_data in enumerate(rows):
        # Make sure image exists.
        file_path = f'../data/extracted/images/{img_data[0]}'
        if not os.path.isfile(file_path):
            continue

        # If only loading one image, check if this row contains the img and correct disease, otherwise continue.
        if img_name is not None:
            if img_data[0] != img_name:
                continue
            if img_disease is not None and pathologies[img_data[1]] != img_disease:
                continue

        ############################################
        ### Load image and turn into tensor.
        ############################################
        xray_img = Image.open(file_path)
        rgb = Image.new('RGB', xray_img.size)
        rgb.paste(xray_img)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img_tensor = preprocess(rgb)
        img_tensor.unsqueeze_(0)

        # Get heatmap for correct disease and turn into numpy array.
        heatmap = net.forward(img_tensor, return_cam=True)
        disease = pathologies[img_data[1]]
        heatmap = heatmap[0, disease].numpy()

        ground_truth = BBox(float(img_data[2]), float(img_data[3]), float(img_data[4]), float(img_data[5]))

        # Save results if evaluating all images, not just plotting one.
        if img_name is None:
            iobb = evaluate_single_bbox(ground_truth, heatmap, iobb=True)
            results.append((img_data[0], disease, iobb))
            print(f'{i:} {iobb}')
        else:
            iobb = evaluate_single_bbox(ground_truth, heatmap, iobb=True, xray_img=xray_img)
            print(f'iobb: {iobb}')
            break


    if img_name is None:
        # Order results by IoBB value.
        results = sorted(results, key=lambda x: x[2], reverse=True)

        results = np.array(results)

        # Save as numpy array.
        np.save('../data/iobb.npy', results)

        # Save as txt
        with open('../data/iobb.txt', 'w') as txt:
            for i in range(results.shape[0]):
                txt.write(f'{float(results[i][2])}, {int(results[i][1])}, {results[i][0]}\n')

    f.close() # Close csv file.


def evaluate_single_bbox(ground_truth, heatmap, iobb=True, xray_img=None):
    """ 
    :param ground_truth: the ground truth BBox.
    :param heatmap: the 8x8 heatmap numpy array.
    :param iobb: if True use IoBB, otherwise use IoU.
    :param xray_img: the x-ray image. If given, it will plot the bboxes and the heatmap.
    :return: the IoBB or IoU metric.
    """
    do_plot = xray_img is not None

    # Normalize heatmap to [0, 255].    
    heatmap -= heatmap.min() # Move to make smallest value 0.
    heatmap *= 255 / heatmap.max() # Scale to make largest value 255.

    # Resize heatmap to 256x256.
    heatmap = cv2.resize(heatmap, (256, 256), interpolation=cv2.INTER_CUBIC)

    # Find maximum peak coordinate in heatmap.
    peak = np.unravel_index(heatmap.argmax(), heatmap.shape)
    
    # Only calculate 'red' bbox when plotting, not when just getting IoBB.
    if do_plot:
        thresholds = [180, 60]
    else:
        thresholds = [180]

    bboxes = [None, None]

    # For every threshold, generate a single bbox.
    for i in range(len(thresholds)):
        # Threshold heatmap and make it binary.
        bin_heatmap = np.where(heatmap >= thresholds[i], 1, 0)
        
        # Generate bbox covering peak in heatmap.
        bboxes[i] = bbox_from_point(bin_heatmap, peak[1], peak[0])
        
        # Scale bbox from 256x256 to 1024x1024 to compare with ground truth.
        bboxes[i].scale(4)

    # Plot the bboxes and heatmap if the x-ray image is given.
    if do_plot:
        plotting.plot_bbox(xray_img, bboxes[0], bboxes[1], ground_truth, heatmap=heatmap)

    # Return the IoBB or IoU.
    if iobb:
        return bboxes[0].IoBB(ground_truth)
    else:
        return bboxes[0].IoU(ground_truth)
