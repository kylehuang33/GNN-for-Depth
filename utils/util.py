

def calculate_iou(box1, box2):
    """ 
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    We assume that the box follows the format:
    box1 = [x1,y1,x2,y2], and box2 = [x3,y3,x4,y4],
    where (x1,y1) and (x3,y3) represent the top left coordinate,
    and (x2,y2) and (x4,y4) represent the bottom right coordinate.
    """
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


def assign_index(bounding_boxes, nodes, threshold=0.5):
    """
    Assign indices to bounding boxes based on IoU.
    
    Parameters:
    - bounding_boxes: List of bounding boxes in the format [x1, y1, x2, y2]
    - threshold: IoU threshold to decide if a box is the same as a previously seen box
    
    Returns:
    - indices: List of indices assigned to each bounding box
    """
    indices = []
    existing_boxes = nodes

    for box in bounding_boxes:
        found_match = False
        for idx, existing_box in enumerate(existing_boxes):
            if calculate_iou(box, existing_box) > threshold:
                indices.append(idx)
                found_match = True
                break
        
        if not found_match:
            existing_boxes.append(box)
            indices.append(len(existing_boxes) - 1)

    return indices, existing_boxes
