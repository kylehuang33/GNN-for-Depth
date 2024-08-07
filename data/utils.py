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

    scene_graphs = [item['scene_graphs'] for item in batch]
    pooled_visuals = [item['pooled_visuals'] for item in batch]

    return {
        'image': images,
        'depth_emb': depth_embs,
        'depth_map': depth_maps,
        'depth': depths,
        'scene_graphs': scene_graphs,
        'pooled_visuals': pooled_visuals
    }