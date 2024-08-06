import os, random, glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import h5py, torch, cv2

from torch import nn, optim



class DepthDataset(Dataset):
    
    def __init__(self, data_path = "/home3/fsml62/LLM_and_SGG_for_MDE/dataset/nyu_depth_v2/official_splits", transform=None, ext="jpg", mode='train'):
        
        self.data_path = data_path
        
        self.filenames = glob.glob(os.path.join(self.data_path, mode, '**', '*.{}'.format(ext)), recursive=True)
        self.pt_path = "/home3/fsml62/LLM_and_SGG_for_MDE/GNN_for_MDE/results/depth_embedding/nyu_depth_v2/official_splits"
        self.depth_map_path = "/home3/fsml62/LLM_and_SGG_for_MDE/GNN_for_MDE/results/depth_map/nyu_depth_v2/official_splits"
        self.sg_path = "/home3/fsml62/LLM_and_SGG_for_MDE/GNN_for_MDE/results/SGG/nyu_depth_v2/official_splits"

        self.transform = transform if transform else transforms.ToTensor()
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
        depth_path = os.path.join(self.mode, self.depth_map_path, '{}.pt'.format(relative_path.split('.')[0]))

        #get the scene graph path
        scenegraph_path = os.path.join(self.sg_path, '{}.h5'.format(relative_path.split('.')[0]))



        ## Depth Embedding
        depth_emb = torch.load(depth_emb_path)

        ## depth map
        depth_map = torch.load(depth_path)

        ## get the actual depth
        actual_depth_path = img_path.replace("rgb", "sync_depth").replace('.jpg', '.png')
        actual_depth = Image.open(actual_depth_path)



        ## Scene Graph
        threshold = 0.5

        with h5py.File(scenegraph_path, 'r') as h5_file:
            # Load each tensor into a dictionary
            # outputs = {key: torch.from_numpy(h5_file[key][:]) for key in h5_file.keys()}
            loaded_output_dict = {key: torch.tensor(h5_file[key]) for key in h5_file.keys()}

        probas = loaded_output_dict['rel_logits'].softmax(-1)[0, :, :-1]
        probas_sub = loaded_output_dict['sub_logits'].softmax(-1)[0, :, :-1]
        probas_obj = loaded_output_dict['obj_logits'].softmax(-1)[0, :, :-1]
        
        
        keep = torch.logical_and(probas.max(-1).values > threshold, 
                                torch.logical_and(probas_sub.max(-1).values > threshold, probas_obj.max(-1).values > threshold))
        
#         mini_threshold = 0.1
#         # Calculate the keep mask with an additional check for valid bounding boxes
#         valid_bboxes = (loaded_output_dict['sub_boxes'][0, :, 2] > mini_threshold) & (loaded_output_dict['sub_boxes'][0, :, 3] > mini_threshold) & \
#                        (loaded_output_dict['obj_boxes'][0, :, 2] > mini_threshold) & (loaded_output_dict['obj_boxes'][0, :, 3] > mini_threshold)

#         keep = torch.logical_and(
#             probas.max(-1).values > threshold, 
#             torch.logical_and(
#                 probas_sub.max(-1).values > threshold, 
#                 torch.logical_and(probas_obj.max(-1).values > threshold, valid_bboxes)
#             )
#         )
        
        
        
        
        sub_bboxes_scaled = self.rescale_bboxes(loaded_output_dict['sub_boxes'][0, keep], img.size)
        obj_bboxes_scaled = self.rescale_bboxes(loaded_output_dict['obj_boxes'][0, keep], img.size)
        relations = loaded_output_dict['rel_logits'][0, keep]

        valid_sub_bboxes = self.validate_bounding_boxes(sub_bboxes, img.size)
        valid_obj_bboxes = self.validate_bounding_boxes(obj_bboxes, img.size)

        # Combine validity of subject and object bounding boxes
        valid_pairs = torch.tensor([vs and vo for vs, vo in zip(valid_sub_bboxes, valid_obj_bboxes)])

        # Apply the updated keep mask
        sub_bboxes_scaled = self.rescale_bboxes(loaded_output_dict['sub_boxes'][0, valid_pairs], img.size)
        obj_bboxes_scaled = self.rescale_bboxes(loaded_output_dict['obj_boxes'][0, valid_pairs], img.size)
        relations = loaded_output_dict['rel_logits'][0, valid_pairs]


        
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
        
        
        # Apply transform to image and depth
        img = self.transform(img)
        actual_depth = self.transform(actual_depth).float()
        
        
        
        target_size = (25, 25)
        
        # Apply the pooling function to the current batch element
        pooled_sub_images, pooled_obj_images = self.pool_visual_content_and_depth(
            sub_bboxes_list=sub_bboxes_scaled,
            obj_bboxes_list=obj_bboxes_scaled,
            image=img,
            target_size=target_size)
        
        pooled_sub_depths, pooled_obj_depths = self.pool_visual_content_and_depth(
            sub_bboxes_list=sub_bboxes_scaled,
            obj_bboxes_list=obj_bboxes_scaled,
            image=depth_emb[0],
            target_size=target_size)
        
        pooled_sub_act_depths, pooled_obj_act_depths = self.pool_visual_content_and_depth(
            sub_bboxes_list=sub_bboxes_scaled,
            obj_bboxes_list=obj_bboxes_scaled,
            image=actual_depth,
            target_size=target_size)
         
        
        pooled_visuals = {
            'sub_imgs': pooled_sub_images, 
            'obj_imgs': pooled_obj_images, 
            'sub_depth_emb': pooled_sub_depths, 
            'obj_depth_emb': pooled_obj_depths,
            'sub_act_depths': pooled_sub_act_depths,
            'obj_act_depths': pooled_obj_act_depths
        }
#         pooled_visuals = None
        


        ## return data
        data = {
            'image': img,
            'depth_emb': depth_emb,
            'depth_map': depth_map,
            'depth': actual_depth,
            'scene_graphs': obj_relationship,
            'pooled_visuals': pooled_visuals

        }
        
        
        return data

    def __len__(self):
        return len(self.filenames)
    
    def box_cxcywh_to_xyxy(self, x):

        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    
        return torch.stack(b, dim=1)
    
    def rescale_bboxes(self, out_bbox, size):

        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        
        b = torch.round(b).int()

        return b

    def validate_bounding_boxes(self, bboxes, img_size):
        """Return a list of booleans indicating whether each bounding box is valid."""
        valid_bboxes = []
        img_w, img_h = img_size

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > img_w: x2 = img_w
            if y2 > img_h: y2 = img_h
            if x2 > x1 and y2 > y1:
                valid_bboxes.append(True)
            else:
                valid_bboxes.append(False)

        return valid_bboxes


    
    def pool_visual_content_and_depth(self, sub_bboxes_list, obj_bboxes_list, image, target_size=(224, 224), actual=False):
        # Define the pooling operation
        pool = nn.AdaptiveAvgPool2d(target_size)

        # Pooling subject images
        pooled_sub_images = []

        for bbox in sub_bboxes_list:
            # Crop the image based on the bounding box
            bbox = self.validate_bounding_box(bbox, image.size()[1:])  # Adjust bbox

            cropped_img = image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            if cropped_img.dim() == 2:
                cropped_img = cropped_img.unsqueeze(0)
            
            
            # Apply pooling
            pooled_img = pool(cropped_img.unsqueeze(0)).squeeze(0)  # Add batch dim, then remove after pooling
            # Flatten the pooled image
            flattened_img = pooled_img.view(-1)
            pooled_sub_images.append(flattened_img)


        # Pooling object images
        pooled_obj_images = []
        
        for bbox in obj_bboxes_list:
            # Crop the image based on the bounding box
            cropped_img = image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            
            if cropped_img.dim() == 2:
                cropped_img = cropped_img.unsqueeze(0)
            
            
            # Apply pooling
            pooled_img = pool(cropped_img.unsqueeze(0)).squeeze(0)
            # Flatten the pooled image
            flattened_img = pooled_img.view(-1)
            pooled_obj_images.append(flattened_img)


        # Convert lists to tensors
        pooled_sub_images = torch.stack(pooled_sub_images) if pooled_sub_images else torch.empty(0)
        pooled_obj_images = torch.stack(pooled_obj_images) if pooled_obj_images else torch.empty(0)

        return pooled_sub_images, pooled_obj_images
