import os, random, glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import h5py, torch, cv2


class DepthDataset(Dataset):
    def __init__(self, data_path = "/home3/fsml62/LLM_and_SGG_for_MDE/dataset/nyu_depth_v2/official_splits", transform=None, ext="jpg", mode='train'):
        
        self.data_path = os.path.join(data_path, mode)
        
        self.filenames = glob.glob(os.path.join(self.data_path, '**', '*.{}'.format(ext)), recursive=True)
        self.pt_path = "/home3/fsml62/LLM_and_SGG_for_MDE/GNN_for_MDE/results/nyu_depth_v2"
        self.depth_map_path = "/home3/fsml62/LLM_and_SGG_for_MDE/GNN_for_MDE/results/depth_map/nyu_depth_v2"
        self.sg_path = "/home3/fsml62/LLM_and_SGG_for_MDE/GNN_for_MDE/results/SGG/nyu_depth_v2"

        self.mode = mode



    def __getitem__(self, idx):

        # image path
        img_path = self.filenames[idx]
        # get the image
        img = Image.open(img_path)

        # get the relative path
        relative_path = os.path.relpath(img_path, self.data_path)
        # get depth embedding path
        depth_emb_path = os.path.join(self.pt_path, '{}.pt'.format(relative_path.split('.')[0]))
        # get depth map path
        depth_path = os.path.join(self.depth_map_path, '{}.pt'.format(relative_path.split('.')[0]))

        #get the scene graph path
        scenegraph_path = os.path.join(self.sg_path, '{}.h5'.format(relative_path.split('.')[0]))



        ## Depth Embedding
        depth_emb = torch.load(depth_emb_path)

        ## depth map
        depth_map = torch.load(depth_map_path)

        ## get the actual depth
        actual_depth_path = img_path.replace("rgb", "sync_depth").replace('.jpg', '.png')
        actual_depth = Image.open(actual_depth_path)



        ## Scene Graph
        threshold = 0.1

        with h5py.File(scenegraph_path, 'r') as h5_file:
            # Load each tensor into a dictionary
            # outputs = {key: torch.from_numpy(h5_file[key][:]) for key in h5_file.keys()}
            loaded_output_dict = {key: torch.tensor(h5_file[key]) for key in h5_file.keys()}

        probas = loaded_output_dict['rel_logits'].softmax(-1)[0, :, :-1]
        probas_sub = loaded_output_dict['sub_logits'].softmax(-1)[0, :, :-1]
        probas_obj = loaded_output_dict['obj_logits'].softmax(-1)[0, :, :-1]
        keep = torch.logical_and(probas.max(-1).values > threshold, 
                                torch.logical_and(probas_sub.max(-1).values > threshold, probas_obj.max(-1).values > threshold))
        
        sub_bboxes_scaled = rescale_bboxes(loaded_output_dict['sub_boxes'][0, keep], img.size)
        obj_bboxes_scaled = rescale_bboxes(loaded_output_dict['obj_boxes'][0, keep], img.size)
        relations = outputs['rel_logits'][0, keep]
        
        probas_dic = {
            'probas': probas[keep],
            'probas_sub': probas_sub[keep],
            'probas_obj': probas_obj[keep]
        }

        obj_relationship = {
            'relation': relations,
            'bbox_sub': sub_bboxes_scaled,
            'bbox_obj': obj_bboxes_scaled
        }


        ## return data
        data = {
            'image': img,
            'depth_emb': depth_emb,
            'depth_map': depth_map,
            'depth': actual_depth,
            'scene_graph': obj_relationship

        }

        return data

    def __len__(self):
        return len(self.filenames)


    def box_cxcywh_to_xyxy(x):

        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    
        return torch.stack(b, dim=1)

    def rescale_bboxes(out_bbox, size):

        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

        return b