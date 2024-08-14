import os, random, glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch_geometric.data import Data
import h5py, torch, cv2
import numpy as np
from statistics import mode

from torch import nn, optim



class DepthDataset(Dataset):
    
    def __init__(self, data_path = "/home3/fsml62/LLM_and_SGG_for_MDE/dataset/nyu_depth_v2/official_splits", transform=None, ext="jpg", mode='train', threshold=0.06):
        
        self.data_path = data_path
        
        self.filenames = glob.glob(os.path.join(self.data_path, mode, '**', '*.{}'.format(ext)), recursive=True)
        self.pt_path = "/home3/fsml62/LLM_and_SGG_for_MDE/GNN_for_MDE/results/depth_embedding/nyu_depth_v2/official_splits"
        self.depth_map_path = "/home3/fsml62/LLM_and_SGG_for_MDE/GNN_for_MDE/results/depth_map/nyu_depth_v2/official_splits"
        self.sg_path = "/home3/fsml62/LLM_and_SGG_for_MDE/GNN_for_MDE/results/SGG/nyu_depth_v2/official_splits"

        self.transform = transform if transform else transforms.ToTensor()
        self.mode = mode
        self.threshold = threshold
        
        self.cache = {}
        
        



    def __getitem__(self, idx):
        
        
        if idx in self.cache:
            return self.cache[idx]
        

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
        depth_emb = self.normalize(torch.load(depth_emb_path))

        ## depth map
        depth_map = self.normalize(torch.load(depth_path))

        ## get the actual depth
        actual_depth_path = img_path.replace("rgb", "sync_depth").replace('.jpg', '.png')
        actual_depth = Image.open(actual_depth_path)

        
        # the resize size
        target_size = (25, 25)


        with h5py.File(scenegraph_path, 'r') as h5_file:
            loaded_output_dict = {key: torch.tensor(np.array(h5_file[key])) for key in h5_file.keys()}


        probas = loaded_output_dict['rel_logits'].softmax(-1)[0, :, :-1]
        probas_sub = loaded_output_dict['sub_logits'].softmax(-1)[0, :, :-1]
        probas_obj = loaded_output_dict['obj_logits'].softmax(-1)[0, :, :-1]
        
        
        keep = torch.logical_and(probas.max(-1).values > self.threshold, 
                                torch.logical_and(probas_sub.max(-1).values > self.threshold, probas_obj.max(-1).values > self.threshold))
        
        
        
      
        
        sub_bboxes_scaled = self.rescale_bboxes(loaded_output_dict['sub_boxes'][0, keep], img.size)
        obj_bboxes_scaled = self.rescale_bboxes(loaded_output_dict['obj_boxes'][0, keep], img.size)
        relations = loaded_output_dict['rel_logits'][0, keep]

        valid_sub_bboxes = self.validate_bounding_boxes(sub_bboxes_scaled, img.size, target_size)
        valid_obj_bboxes = self.validate_bounding_boxes(obj_bboxes_scaled, img.size, target_size)

        # Combine validity of subject and object bounding boxes
        valid_pairs = torch.tensor([vs and vo for vs, vo in zip(valid_sub_bboxes, valid_obj_bboxes)], dtype=torch.bool)

        # Apply the updated keep mask
        sub_bboxes_scaled = sub_bboxes_scaled[valid_pairs]
        obj_bboxes_scaled = obj_bboxes_scaled[valid_pairs]
        relations = relations[valid_pairs]
        
        
        
        sub_idxs, nodes1 = self.assign_index(sub_bboxes_scaled, [], threshold=0.7)
        obj_idxs, nodes2 = self.assign_index(obj_bboxes_scaled, nodes1, threshold=0.7)
        
        all_idxs = sub_idxs + obj_idxs
        bbox_lists = torch.concat((sub_bboxes_scaled, obj_bboxes_scaled), dim=0)
        
        unique_idxs = set()
        filtered_idxs = []
        filtered_bboxes = []

        for idx, bbox in zip(all_idxs, bbox_lists):   
            if idx not in unique_idxs:
                unique_idxs.add(idx)
                filtered_idxs.append(idx)
                filtered_bboxes.append(bbox.tolist())  

        # Sort the filtered indices along with their corresponding bounding boxes
        sorted_wrapped = sorted(zip(filtered_idxs, filtered_bboxes), key=lambda x: x[0])
        sorted_idxs, sorted_bboxes = zip(*sorted_wrapped)

        # Convert them back to torch tensors
        sorted_idxs = torch.tensor(sorted_idxs, dtype=torch.long)
        sorted_bboxes = torch.tensor(sorted_bboxes, dtype=torch.int32)        
        

        
        
        # Apply transform to image and depth
        img = self.transform(img)
        img = self.normalize(img)
        actual_depth = self.transform(actual_depth).float()
        actual_depth = self.normalize(actual_depth)
        
        device = 'cpu'
        
        pooled_images = self.pool_visual_content_and_depth(
            sorted_bboxes=sorted_bboxes,
            embedding=img,
            target_size=target_size).to(device)
        
        pooled_depths = self.pool_visual_content_and_depth(
            sorted_bboxes=sorted_bboxes,
            embedding=depth_emb[0],
            target_size=target_size).to(device)
        
        pooled_act_depths = self.resize_depth_map(
            sorted_bboxes=sorted_bboxes,
            embedding=actual_depth.squeeze(0),
            target_size=target_size).to(device)
        
        
        
        # node embedding
        node_embeddings = torch.cat([pooled_images, pooled_depths], dim=1).to(device)
        
        edge_index = torch.tensor([sub_idxs, obj_idxs]).to(device)
        
#         edge_embeddings = torch.tensor(relations, dtype=torch.float).to(device)
        edge_embeddings = relations.clone().detach().float().to(device)
        
        
        
        gnndata = Data(x=node_embeddings, edge_index=edge_index, edge_attr=edge_embeddings)


        ## return data
        data = {
            'image': img,
            'depth_emb': depth_emb,
            'depth_map': depth_map,
            'depth': actual_depth,
            'pooled_act_depths': pooled_act_depths,
            'bboxs': filtered_bboxes,
            'gnndata': gnndata
        }
        
        self.cache[idx] = data
        
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

        b[:, 0] = torch.clamp(b[:, 0], min=0, max=img_w)
        b[:, 1] = torch.clamp(b[:, 1], min=0, max=img_h)
        b[:, 2] = torch.clamp(b[:, 2], min=0, max=img_w)
        b[:, 3] = torch.clamp(b[:, 3], min=0, max=img_h)

        return b

    def validate_bounding_boxes(self, bboxes, img_size, target_size):
        """Return a list of booleans indicating whether each bounding box is valid."""
        valid_bboxes = []
        img_w, img_h = img_size
        
        t_w, t_h = target_size

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > img_w: x2 = img_w
            if y2 > img_h: y2 = img_h
            
            if (x2 - x1) >= t_w and (y2 - y1) >= t_h:
                valid_bboxes.append(True)
            else:
                valid_bboxes.append(False)

        return valid_bboxes


    
    def pool_visual_content_and_depth(self, sorted_bboxes, embedding, target_size=(50, 50)):

        pool = nn.AdaptiveAvgPool2d(target_size)

        pooled_embs = []

        for bbox in sorted_bboxes:
            cropped_emb = embedding[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

            if cropped_emb.dim() == 2:
                cropped_emb = cropped_emb.unsqueeze(0)  

            pooled_emb = pool(cropped_emb.unsqueeze(0)).squeeze(0)

            flattened_emb = pooled_emb.view(-1)
            
            pooled_embs.append(flattened_emb)


        pooled_embs = torch.stack(pooled_embs) if pooled_embs else torch.empty(0)

        return pooled_embs
    
    
    def calculate_iou(self, box1, box2):

        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        # Calculate intersection coordinates
        x_inter1 = max(x1, x3)
        y_inter1 = max(y1, y3)
        x_inter2 = min(x2, x4)
        y_inter2 = min(y2, y4)

        # Calculate intersection dimensions
        width_inter = max(0, x_inter2 - x_inter1)
        height_inter = max(0, y_inter2 - y_inter1)

        # Calculate intersection area
        area_inter = width_inter * height_inter

        # Calculate areas of the input boxes
        width_box1 = abs(x2 - x1)
        height_box1 = abs(y2 - y1)
        area_box1 = width_box1 * height_box1

        width_box2 = abs(x4 - x3)
        height_box2 = abs(y4 - y3)
        area_box2 = width_box2 * height_box2

        # Calculate union area
        area_union = area_box1 + area_box2 - area_inter

        # Calculate IoU
        if area_union == 0:
            return 0  # avoid division by zero
        iou = area_inter / area_union

        return iou
    
    
    def assign_index(self, bounding_boxes, nodes, threshold=0.5):
        indices = []
        existing_boxes = nodes

        for box in bounding_boxes:
            found_match = False
            for idx, existing_box in enumerate(existing_boxes):
                if self.calculate_iou(box, existing_box) > threshold:
                    indices.append(idx)
                    found_match = True
                    break

            if not found_match:
                existing_boxes.append(box)
                indices.append(len(existing_boxes) - 1)

        return indices, existing_boxes
    
    
    def normalize(self, tensor):
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        normalized_tensor = (tensor - tensor_min) / ((tensor_max - tensor_min) + 1e-8)
        return normalized_tensor
    
    
    
    def resize_depth_map(self, sorted_bboxes, embedding, target_size=(25, 25)):

        pooled_embs = []

        for bbox in sorted_bboxes:
            cropped_emb = embedding[bbox[1]:bbox[3], bbox[0]:bbox[2]]


            pooled_emb = self.downsize_depth_map_mode(cropped_emb, target_size)

            flattened_emb = pooled_emb.view(-1)
            
            pooled_embs.append(flattened_emb)


        pooled_embs = torch.stack(pooled_embs) if pooled_embs else torch.empty(0)

        return pooled_embs
    
    
    def downsize_depth_map_mode(self, depth_map, new_size):
        
        original_height, original_width = depth_map.shape[-2], depth_map.shape[-1]
        new_height, new_width = new_size

        window_height = original_height // new_height
        window_width = original_width // new_width

        downsized_map = torch.zeros((new_height, new_width), dtype=depth_map.dtype, device=depth_map.device)

        for i in range(new_height):
            for j in range(new_width):
                # Define the window boundaries
                start_row = i * window_height
                end_row = start_row + window_height
                start_col = j * window_width
                end_col = start_col + window_width

                # Extract the window
                window = depth_map[start_row:end_row, start_col:end_col]

                # Flatten the window
                flat_window = window.flatten().cpu().numpy()  # Convert to numpy for mode calculation

                # Calculate the mode and handle exceptions
                try:
                    downsized_map[i, j] = mode(flat_window)
                except:
                    downsized_map[i, j] = torch.tensor(np.median(flat_window), dtype=depth_map.dtype)

        return downsized_map
    
    