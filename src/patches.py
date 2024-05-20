import numpy as np
import torch

def sample_patch(origin, mask:np.array, patch_numwh:list):
    ''' 
    randomly samples patch from masked region (chest).
    width and height of patch is selected by user.
    
    # parameters 
    origin: input array of original chest tensor image.
    mask: segmented tensor image. 
    patch_numwh: list of number, width and heigh of patch. 
    # return
    list of sampled patch Tensor images.
    '''
    patch_num, patch_w, patch_h = 0, 0, 0
    if len(patch_numwh) != 3:
        ValueError("should contain all three values (patch num, width, heigh)")
    else:
        patch_num, patch_w, patch_h = patch_numwh
        if patch_num < 0:
            ValueError("patch number should be possitive")
        if (patch_w < 0) or (patch_h < 0):
            ValueError("patch width or height should be possitive")
    
    np_mask = np.array(mask)
    lung_axes = np.argwhere(np_mask == 1) 

    patch_indexes = np.random.choice(range(len(lung_axes)), patch_num)
    patch_axes = lung_axes[patch_indexes]
    
    list_patches = []

    for patch_x, patch_y in patch_axes:

        origin_np = np.array(origin).squeeze()
        l_dx, r_dx, d_dy, u_dy = 0, 0, 0, 0

        dhx = (patch_w - 1) // 2
        dhy = (patch_h - 1) // 2

        w_is_even = (patch_w - 1) % 2
        h_is_even = (patch_h - 1) % 2

        if w_is_even and h_is_even:
            l_dx = dhx + 1
            r_dx = dhx
            d_dy = dhy + 1
            u_dy = dhy

        elif h_is_even:
            l_dx = dhx
            r_dx = dhx
            d_dy = dhy + 1
            u_dy = dhy

        elif w_is_even:
            l_dx = dhx + 1
            r_dx = dhx
            d_dy = dhy
            u_dy = dhy

        else:
            l_dx = dhx
            r_dx = dhx
            d_dy = dhy
            u_dy = dhy

        x_0, x_1, y_0, y_1 = 0, 0, 0, 0 # initial values

        if (patch_x-l_dx < 0): x_0 = 0
        else: x_0 = patch_x-l_dx
        if (patch_x+r_dx+1 > origin_np.shape[0]): x_1 = origin_np.shape[0]
        else: x_1 = patch_x+r_dx+1
        if (patch_y-d_dy < 0): y_0 = 0
        else: y_0 = patch_y-d_dy
        if (patch_y+u_dy+1 > origin_np.shape[1]): y_1 = origin_np.shape[1]
        else: y_1 = patch_y+u_dy+1

        patch_np = origin_np[x_0:x_1, y_0:y_1]

        # patch_np = origin_np[patch_x-l_dx:patch_x+r_dx+1, 
        #                     patch_y-d_dy:patch_y+u_dy+1]

        patch = torch.from_numpy(patch_np)

        list_patches.append(patch)

    return list_patches