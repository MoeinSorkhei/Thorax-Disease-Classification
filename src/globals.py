import torch

# this device is accessible in all the modules who import this module
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

pathology_names = ['Atelectasis',
               'Cardiomegaly',
               'Consolidation',
               'Edema',
               'Effusion',
               'Emphysema',
               'Fibrosis',
               'Hernia',
               'Infiltration',
               'Mass',
               'Nodule',
               'Pleural_Thickening',
               'Pneumonia',
               'Pneumothorax']

