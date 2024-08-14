import torch
from torch_geometric.data import Batch, Data


def custom_collate(batch):
    
    
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None
    
    # Handle images, depth maps, and other tensors separately
    images = torch.stack([item['image'] for item in batch])
    depth_embs = torch.stack([item['depth_emb'] for item in batch])
    depth_maps = torch.stack([item['depth_map'] for item in batch])
    depths = torch.stack([item['depth'] for item in batch])
    
#     pooled_act_depths = [torch.tensor(item['pooled_act_depths'], dtype=torch.int) for item in batch]
    pooled_act_depths = [item['pooled_act_depths'].clone().detach().float() for item in batch]




#     bboxs = torch.stack([item['bboxs'] for item in batch])
    bboxs = [torch.tensor(item['bboxs'], dtype=torch.int) for item in batch]

    gnndata_list = [item['gnndata'] for item in batch]

    # Batch the graph data using PyTorch Geometric's Batch
    graph_batch = Batch.from_data_list(gnndata_list)

    return {
        'image': images,
        'depth_emb': depth_embs,
        'depth_map': depth_maps,
        'depth': depths,
        'pooled_act_depths': pooled_act_depths,
        'bboxs': bboxs,
        'gnndata': graph_batch,  # Batched graph data
    }