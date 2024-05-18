import numpy as np

def sample_patch(origin, mask:np.array, patch_numwh:list, pil_or_ten:bool=1):
    ''' 
    randomly samples patch from masked region (chest).
    width and height of patch is selected by user.
    
    # parameters 
    origin: input array of original chest tensor image.
    mask: segmented tensor image. 
    patch_numwh: list of number, width and heigh of patch. 
    pil_or_ten: output patches as pil(1; default) or tensor img(0).
    # return
    list of sampled patch pil images
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

    patch_idexes = np.random.choice(range(len(lung_axes)), patch_num)
    patch_axes = lung_axes[patch_idexes]
    
    list_patches_np = []

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

        patch_np = origin_np[patch_x-l_dx:patch_x+r_dx+1, 
                            patch_y-d_dy:patch_y+u_dy+1]

        list_patches_np.append(patch_np)

    return list_patches_np