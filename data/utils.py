import torch


def custom_collate(batch):
    
    
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None
    
    # Handle images, depth maps, and other tensors separately
    images = torch.stack([item['image'] for item in batch])
    depth_embs = torch.stack([item['depth_emb'] for item in batch])
    depth_maps = torch.stack([item['depth_map'] for item in batch])
    depths = torch.stack([item['depth'] for item in batch])

    # scene_graphs = [item['scene_graphs'] for item in batch]
    relations = torch.stack([item['relation'] for item in batch])
    bbox_subs = torch.stack([item['bbox_sub'] for item in batch])
    bbox_objs = torch.stack([item['bbox_obj'] for item in batch])
    # pooled_visuals = [item['pooled_visuals'] for item in batch]
        # Stack the visual pooling results
    sub_imgs = torch.stack([item['sub_imgs'] for item in batch])
    obj_imgs = torch.stack([item['obj_imgs'] for item in batch])
    sub_depth_emb = torch.stack([item['sub_depth_emb'] for item in batch])
    obj_depth_emb = torch.stack([item['obj_depth_emb'] for item in batch])
    sub_act_depths = torch.stack([item['sub_act_depths'] for item in batch])
    obj_act_depths = torch.stack([item['obj_act_depths'] for item in batch])

    return {
        'image': images,
        'depth_emb': depth_embs,
        'depth_map': depth_maps,
        'depth': depths,
        # 'scene_graphs': scene_graphs,
        'relation': relations,
        'bbox_sub': bbox_subs,
        'bbox_obj': bbox_objs,
        #'pooled_visuals': pooled_visuals
        'sub_imgs': sub_imgs,
        'obj_imgs': obj_imgs,
        'sub_depth_emb': sub_depth_embs,
        'obj_depth_emb': obj_depth_embs,
        'sub_act_depths': sub_act_depths,
        'obj_act_depths': obj_act_depths

    }