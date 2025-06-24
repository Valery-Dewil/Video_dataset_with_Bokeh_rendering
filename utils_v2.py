import argparse
from gblurs import G
import iio
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d, griddata, interpn
import torch
from torch import nn
from torch.nn import functional as F
import os
from torch.autograd import Variable

#np.random.seed(2025)




def translate(img, translation_y, translation_x):
    H,W,_ = img.shape
    result = img*0
    flow = np.zeros((H,W,2), dtype=np.float32)
    flow[:,:,0] = -translation_x   #si on veut un flow qui met image i+1 sur image i, il faut + translation_x
    flow[:,:,1] = -translation_y   #si on veut un flow qui met image i+1 sur image i, il faut + translation_y


    if translation_y > 0:
        if translation_x > 0:
            result[translation_y:, translation_x:] = img[:-translation_y, :-translation_x]
        elif translation_x < 0:
            result[translation_y:, 0:translation_x] = img[:-translation_y, -translation_x:]
        else:
            result[translation_y:, :] = img[:-translation_y, :]
    elif translation_y <0:
        if translation_x > 0:
            result[0:translation_y, translation_x:] = img[-translation_y:, :-translation_x]
        elif translation_x < 0:
            result[0:translation_y, 0:translation_x] = img[-translation_y:, -translation_x:]
        else:
            result[0:translation_y, :] = img[-translation_y:, :]
    else:
        if translation_x > 0:
            result[:, translation_x:] = img[:, :-translation_x]
        elif translation_x < 0:
            result[:, 0:translation_x] = img[:, -translation_x:]
        else:
            result = img #case ty=tx=0 !
    
    return result, flow







def update_crop_index(i,j,k,l,ty,tx):
    if ty>=0:
        i = i+ty
    if ty<0:
        k=k-ty

    if tx>=0:
        j = j+tx
    if tx<0:
        l=l-tx

    return i,j,k,l




def downscale(img, scale):
    result = img*0
    for c in range(img.shape[-1]):
        result[:,:,c] = torch.nn.functional.interpolate(torch.Tensor(img).permute(2,0,1).unsqueeze(0), scale_factor=scale, mode='bicubic', align_corners=True).numpy().squeeze().transpose(1,2,0)
    return result




def resize(img, mask, depth_pro_mask, h,w,max_ratio=(1/2,1/2), min_ratio=(1/3,1/3), antialiasing=True):
    """
    Take a foreground image img and the corresponding mask.
    1. Crop the img in a rectangle image where there is no complete line or column equal to 0.
    2. Resize the image and the mask so that it is not greater than __ratio__ (for instance 1/3) of the h and w. The h and w come from the background image
    """
    # Ensure the mask has 3 channels (convention, because some datasets have 3 channels mask)
    if mask.shape[-1] == 1:
        mask  = np.repeat(mask, 3, axis=-1)
    if mask.shape[-1] == 2:
        mask  = np.repeat(mask[:,:,0:1], 3, axis=-1)
    if depth_pro_mask.shape[-1] == 1:
        depth_pro_mask  = np.repeat(depth_pro_mask, 3, axis=-1)

    # isolate the foreground object
    object_img = img*mask

    # determine the zeros lines and columns
    H,W,_ = object_img.shape
    i,j,k,l=0,0,H,W
    # while the lines (or columns) are zero (or almost zero), we can crop them
    while np.max(object_img[i,:]) < 1.1 and i<H:
        i = i+1
    while np.max(object_img[:,j]) < 1.1 and j<W:
        j = j+1
    while np.max(object_img[k-1,:]) < 1.1 and k>0:
        k = k-1
    while np.max(object_img[:,l-1]) < 1.1 and l>0:
        l = l-1

    # crop zeros lines and columns
    print("before: ", object_img.shape, mask.shape)
    object_img   = object_img[i:k, j:l]
    mask_cropped = mask[      i:k, j:l]
    depth_pro_mask_cropped = depth_pro_mask[      i:k, j:l]
    print("after: ", object_img.shape, mask_cropped.shape)

    # if dimensions already fit the wanted ratio, we can return
    ratio=(np.random.rand()*(max_ratio[0]-min_ratio[0])+min_ratio[0], np.random.rand()*(max_ratio[1]-min_ratio[1])+min_ratio[1])
    max_size_y, max_size_x = int(h*ratio[0]), int(w*ratio[1])


    if (k-i) <= max_size_y and (l-j) <= max_size_x:
        return object_img, mask_cropped
    # else, we need to resize. First, let's determine in which axis it should be down
    # In x-axis
    ratio = (l-j) / max_size_x  #np.ceil((l-j) / max_size_x)
    if (l-j) > max_size_x and (k-i)/ratio < max_size_y:  #resize in the x-axis dominates
        # apply antialiasing
        if antialiasing:
            object_img = G('borelli', object_img, 0.5*np.sqrt(ratio**2-1))
        # scale the image by ratio and return them
        object_img   = torch.nn.functional.interpolate(torch.Tensor(object_img  ).permute(2,0,1).unsqueeze(0), scale_factor=1/ratio, mode='bicubic', align_corners=True).numpy().squeeze().transpose(1,2,0)
        mask_cropped = torch.nn.functional.interpolate(torch.Tensor(mask_cropped).permute(2,0,1).unsqueeze(0), scale_factor=1/ratio, mode='bicubic', align_corners=True).numpy().squeeze().transpose(1,2,0)
        depth_pro_mask_cropped = torch.nn.functional.interpolate(torch.Tensor(depth_pro_mask_cropped).permute(2,0,1).unsqueeze(0), scale_factor=1/ratio, mode='bicubic', align_corners=True).numpy().squeeze().transpose(1,2,0)
    else:                                                #resize in the y-axis dominates
        #the ratio should be taken into account in the y-axis instead:
        ratio = (k-i) / max_size_y   # np.ceil((k-i) / max_size_y)
        # apply antialiasing
        if antialiasing:
            object_img = G('borelli', object_img, 0.5*np.sqrt(ratio**2-1))
        # scale the image by ratio and return them
        object_img   = torch.nn.functional.interpolate(torch.Tensor(object_img  ).permute(2,0,1).unsqueeze(0), scale_factor=1/ratio, mode='bicubic', align_corners=True).numpy().squeeze().transpose(1,2,0)
        mask_cropped = torch.nn.functional.interpolate(torch.Tensor(mask_cropped).permute(2,0,1).unsqueeze(0), scale_factor=1/ratio, mode='bicubic', align_corners=True).numpy().squeeze().transpose(1,2,0)
        depth_pro_mask_cropped = torch.nn.functional.interpolate(torch.Tensor(depth_pro_mask_cropped).permute(2,0,1).unsqueeze(0), scale_factor=1/ratio, mode='bicubic', align_corners=True).numpy().squeeze().transpose(1,2,0)

    return object_img, mask_cropped, depth_pro_mask_cropped
    



def compose_image(background, background_depth_pro_mask, foreground_list, mask_list, depth_pro_mask_list, delta_list, sigma):
    H,W,_ = background.shape

    # Blur the background
    blurred_bg = background*0
    if len(sigma)==2:
        length = sigma[1] - sigma[0]
        sigma = np.random.rand()*length+sigma[0] # random between sigma[0] and sigma[1]
    else:
        sigma=sigma[0]
    for c in range(3):
        blurred_bg[:,:,c] = gaussian_filter(background[:,:,c], sigma)

    all_in_focus = background.copy()
    with_Bokeh = blurred_bg.copy()

    for foreground, mask, depth_pro_mask, delta in zip(foreground_list, mask_list, depth_pro_mask_list, delta_list):
        # all in focus
        all_in_focus = all_in_focus*(1-mask) + foreground*mask

        # with Bokeh
        with_Bokeh = with_Bokeh*(1-mask) + foreground*mask

    # composed mask  
    #Mask = np.sum(np.array(mask_list), axis=0).clip(0,1)
    Mask = compose_mask(background_depth_pro_mask, mask_list, depth_pro_mask_list, delta_list)

    return all_in_focus, with_Bokeh, Mask, sigma

def compose_mask(background_depth_pro_mask, mask_list, depth_pro_mask_list, delta_list, foreground_depth=0.1):
    mask_list = np.array(mask_list)
    depth_pro_mask_list = np.array(depth_pro_mask_list)
    #####################"delta_list = np.array(delta_list)
    N,H,W,C = mask_list.shape
    Mask = background_depth_pro_mask.copy()
    max_delta=0
    print("N = ", N, ", for delta, it is ", delta_list.shape)
    for n in range(N):
        max_delta = np.max([max_delta, delta_list[n]+foreground_depth/2])
        max_delta = np.min([max_delta, 1])
        ##################Mask = Mask + (mask_list[n]*delta_list[n] + mask_list[n+1]*delta_list[n+1]).clip(0, max_delta)
        #print('delta ', delta_list[n])
        #print('min ', (mask_list[n]*scale_DepthPro_mask(depth_pro_mask_list[n], delta_list[n]-foreground_depth/2, delta_list[n]+foreground_depth/2)).min())
        #print('max ', (mask_list[n]*scale_DepthPro_mask(depth_pro_mask_list[n], delta_list[n]-foreground_depth/2, delta_list[n]+foreground_depth/2)).max())
        #Mask = (Mask + mask_list[n]*scale_DepthPro_mask(depth_pro_mask_list[n], delta_list[n]-foreground_depth/2, delta_list[n]+foreground_depth/2))#.clip(0,max_delta)
       

        scaled_DepthPro_mask = scale_DepthPro_mask(depth_pro_mask_list[n], delta_list[n]-foreground_depth/2, delta_list[n]+foreground_depth/2)
        print(Mask.shape, mask_list[n].shape, scaled_DepthPro_mask.shape)
        for i in range(H):
            for j in range(W):
                if Mask[i,j] < mask_list[n,i,j,0]*scaled_DepthPro_mask[i,j,0]:
                    Mask[i,j] = (mask_list[n,i,j,0]*scaled_DepthPro_mask[i,j,0]).clip(0,max_delta)

    return Mask



def update_lists(fg_list, mask_list, depth_pro_mask_list, foreground, mask, depth_pro_mask, start_y, start_x, H, W, h, w):

        Foreground, Mask, DepthProMask = np.zeros((H,W,3), dtype=np.float32), np.zeros((H,W,3), dtype=np.float32), np.zeros((H,W,3), dtype=np.float32)
        if start_y>H or start_y<0 or start_x>W or start_x<0:
            #we are out of the image, so we don't add this as an object anymore
            return fg_list, mask_list, depth_pro_mask_list, False
       
        #else, we copy paste the object at the good location and add it in the lists
        if start_y+h<=H:
            if start_x+w<=W:
                Foreground[start_y:start_y+h, start_x:start_x+w] = foreground
                Mask[      start_y:start_y+h, start_x:start_x+w] = mask
                DepthProMask[      start_y:start_y+h, start_x:start_x+w] = depth_pro_mask
            else:
                Foreground[start_y:start_y+h, start_x:] = foreground[:,:W-start_x]
                Mask[      start_y:start_y+h, start_x:] = mask[     :, :W-start_x]
                DepthProMask[      start_y:start_y+h, start_x:] = depth_pro_mask[     :, :W-start_x]
        else:
            if start_x+w<=W:
                Foreground[start_y:, start_x:start_x+w] = foreground[:H-start_y,:]
                Mask[      start_y:, start_x:start_x+w] = mask[      :H-start_y,:]      
                DepthProMask[      start_y:, start_x:start_x+w] = depth_pro_mask[      :H-start_y,:]      
            else:
                Foreground[start_y:, start_x:] = foreground[:H-start_y,:W-start_x]
                Mask[      start_y:, start_x:] = mask[      :H-start_y,:W-start_x]      
                DepthProMask[      start_y:, start_x:] = depth_pro_mask[      :H-start_y,:W-start_x]      

        fg_list.append(Foreground)
        mask_list.append(Mask)
        depth_pro_mask_list.append(DepthProMask)

        return fg_list, mask_list, depth_pro_mask_list, True


def update_foreground_lists(fg_list, mask_list, depth_pro_mask_list, foreground, mask, depth_pro_mask, start_y, start_x, H, W, h, w):

        Foreground, Mask, DepthProMask = np.zeros((H,W,3), dtype=np.float32), np.zeros((H,W,3), dtype=np.float32), np.zeros((H,W,3), dtype=np.float32)

        Foreground[start_y:start_y+h, start_x:start_x+w] = foreground
        Mask[      start_y:start_y+h, start_x:start_x+w] = mask
        DepthProMask[      start_y:start_y+h, start_x:start_x+w] = depth_pro_mask

        fg_list.append(Foreground)
        mask_list.append(Mask)
        depth_pro_mask_list.append(DepthProMask)

        return fg_list, mask_list, depth_pro_mask_list



def read_mask(path, threshold=2):
    mask = iio.read(path)
    # See if the mask is [0,1] or [0,255]. The threshold is used for that.
    if np.max(mask)>2:
        return  mask/255
    return mask




def zig_zag_borders_spiky(mask, threshold=100, length=10):
    H,W,_ = mask.shape
    if np.any(mask[0,:]):
        r = np.random.randint(length)
        # if the non masked part intercept this line, look where the non masked part is.
        for i in range(W):
            if np.min(mask[0,i])>threshold: # in pixel i, there is non masked part.
                j=0 #look which length in the vertical direction. Take the minimum between this length and a fix length given in parameter.
                while np.min(mask[j,i])>threshold and j<=length:
                    j = j + 1
                # in this vertical direction, erode the mask randomly
                r = np.array([r + np.random.randint(2)-1]).clip(0,length).item()
                if r != 0:
                    mask[:r, i] = 0

    if np.any(mask[-1,:]):
        r = np.random.randint(length)
        for i in range(W):
            if np.min(mask[-1,i])>threshold: 
                j=-1
                while np.min(mask[j,i])>threshold and -j<=length:
                    j = j - 1
                r = np.array([r + np.random.randint(2)-1]).clip(0,length).item()
                if r != 0:
                    mask[-r:, i] = 0

    if np.any(mask[:,0]):
        r = np.random.randint(length)
        for i in range(H):
            if np.min(mask[i,0])>threshold: 
                j=1
                while np.min(mask[i,j])>threshold and j<=length:
                    j = j + 1
                r = np.array([r + np.random.randint(2)-1]).clip(0,length).item()
                if r != 0:
                    mask[i, :r] = 0

    if np.any(mask[:,-1]):
        r = np.random.randint(length)
        for i in range(H):
            if np.min(mask[i,-1])>threshold: 
                j=-1
                while np.min(mask[i,j])>threshold and -j<=length:
                    j = j - 1
                #r = np.random.randint(-j)
                r = np.array([r + np.random.randint(2)-1]).clip(0,length).item()
                if r != 0:
                    mask[i, -r:] = 0

    return mask

#def zig_zag_borders(mask, threshold=100, length=10, smoothness=5):
#def zig_zag_borders(mask, threshold=0.1, length=10, smoothness=20):
def zig_zag_borders(mask, threshold=0.1, length=20, smoothness=15):
    try:
        H,W,_ = mask.shape
        if np.any(mask[0,:]>threshold):  #the straight line is on the top
            # find the coordinate where the line starts and ends
            start, end = 0,W-1
            while np.min(mask[0, start]<threshold) and start < W-smoothness-1:
                start = start+1
            while np.min(mask[0, end]<threshold) and end > start:
                end = end-1

            coords = np.linspace(start, end, end - start + 1)
            control_x = np.linspace(start, end, smoothness)
            control_y = np.random.randint(1, length + 1, size=smoothness)

            interp_curve = interp1d(control_x, control_y, kind='cubic', fill_value='extrapolate')
            offsets = np.clip(interp_curve(coords).astype(int), 1, length)
            for i, offset in enumerate(offsets):
                pos = start + i
                if offset < H:
                    mask[0:offset, pos] = 0

        if np.any(mask[-1,:]>threshold): # the straight line is on the bottom
            # find the coordinate where the line starts and ends
            start, end = 0,W-1
            while np.min(mask[-1, start]<threshold) and start < W-smoothness-1:
                start = start+1
            while np.min(mask[-1, end]<threshold) and end > start:
                end = end-1

            coords = np.linspace(start, end, end - start + 1)
            control_x = np.linspace(start, end, smoothness)
            control_y = np.random.randint(1, length + 1, size=smoothness)

            interp_curve = interp1d(control_x, control_y, kind='cubic', fill_value='extrapolate')
            offsets = np.clip(interp_curve(coords).astype(int), 1, length)
            for i, offset in enumerate(offsets):
                pos = start + i
                if offset < H:
                    mask[-offset:, pos] = 0

        if np.any(mask[:,0]>threshold):
            # find the coordinate where the line starts and ends
            start, end = 0,H-1
            while np.min(mask[start,0]<threshold) and start < H-smoothness-1:
                start = start+1
            while np.min(mask[end,0]<threshold) and end > start:
                end = end-1

            coords = np.linspace(start, end, end - start + 1)
            control_y = np.linspace(start, end, smoothness)
            control_x = np.random.randint(1, length + 1, size=smoothness)

            interp_curve = interp1d(control_y, control_x, kind='cubic', fill_value='extrapolate')
            offsets = np.clip(interp_curve(coords).astype(int), 1, length)
            for i, offset in enumerate(offsets):
                pos = start + i
                if offset < W:
                    mask[pos, :offset] = 0

        if np.any(mask[:,-1]>threshold):
            start, end = 0,H-1
            while np.min(mask[start,-1]<threshold) and start < H-smoothness-1:
                start = start+1
            while np.min(mask[end,-1]<threshold) and end > start:
                end = end-1

            coords = np.linspace(start, end, end - start + 1)
            control_y = np.linspace(start, end, smoothness)
            control_x = np.random.randint(1, length + 1, size=smoothness)

            interp_curve = interp1d(control_y, control_x, kind='cubic', fill_value='extrapolate')
            offsets = np.clip(interp_curve(coords).astype(int), 1, length)
            for i, offset in enumerate(offsets):
                pos = start + i
                if offset < W:
                    mask[pos, -offset:] = 0


        return mask
    except:
        return mask
        #iio.write("/mnt/adisk/dewil/dummy/PROBLEMATIC_MASK.tif", mask)
        #print(np.any(mask[0,:]),np.any( mask[-1,:]),np.any(mask[:,0]), np.any(mask[:,-1]))
        #print(coords)
        #print(control_y)
        #print(control_x)
        #exit()





def create_flow_old(H, W, theta, tx, ty):
    """
    H,W = shape of the image
    theta = rotation angle (in degree)
    tx, ty = translation parameters
    """

    # Create mesh grid of coordinates
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Convert angle to radians
    theta = theta * np.pi / 180

    # Compute center of the image
    cx, cy = W / 2, H / 2

    # Shift coordinates to the center
    x_shifted = x - cx
    y_shifted = y - cy

    # Apply rotation
    x_rot = np.cos(theta) * x_shifted - np.sin(theta) * y_shifted
    y_rot = np.sin(theta) * x_shifted + np.cos(theta) * y_shifted

    # Shift back from center and apply translation
    x_new = x_rot + cx + tx
    y_new = y_rot + cy + ty

    # Flow is the difference between current coordinates and new coordinates
    flow = np.zeros((H, W, 2), dtype=np.float32)
    #flow[..., 0] = x - x_new  # vx
    #flow[..., 1] = y - y_new  # vy
    flow[..., 0] = x_new - x  # vx
    flow[..., 1] = y_new - y  # vy

    return flow


def create_flow(H, W, theta, tx, ty):
    """
    H,W = shape of the image
    theta = rotation angle (in degree)
    tx, ty = translation parameters
    """

    # Convert angle to radians
    theta = theta * np.pi / 180

    # Compute size of new canvas (diagonal)
    diag = int(np.ceil(np.sqrt(H**2 + W**2)))
    H, W = diag, diag

    # Centers
    cx, cy = W/2, H/2

    # Create grid for new canvas
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Shift coordinates to center of new image
    x_shifted = x - cx 
    y_shifted = y - cy 

    # Apply rotation
    x_rot = np.cos(theta) * x_shifted - np.sin(theta) * y_shifted
    y_rot = np.sin(theta) * x_shifted + np.cos(theta) * y_shifted

    # Shift back from center and apply translation
    x_new = x_rot + cx + tx
    y_new = y_rot + cy + ty

    # Flow is the difference between current coordinates and new coordinates
    flow = np.zeros((H, W, 2), dtype=np.float32)
    #flow[..., 0] = x - x_new  # vx
    #flow[..., 1] = y - y_new  # vy
    flow[..., 0] = x_new - x  # vx
    flow[..., 1] = y_new - y  # vy

    return flow





def apply_rotation_translation_with_flow(u, theta, tx=0, ty=0):
    H, W = u.shape[:2]
    flow = create_flow(H, W, theta, tx, ty)

    # You must place the original image in the center of a padded canvas
    diag = flow.shape[0]  # flow is (diag, diag, 2)
    pad_y = (diag - H) // 2
    pad_x = (diag - W) // 2
    
    # Pad original image into center of new canvas
    u_padded = np.zeros((diag, diag, 3), dtype=u.dtype)
    u_padded[pad_y:pad_y+H, pad_x:pad_x+W] = u

    # Warp using your custom function
    warped = warp(u_padded, flow)

    return u_padded, warped, flow








def pad_image_for_rotation(img):
    H, W = img.shape[:2]
    diag = int(np.ceil(np.sqrt(H**2 + W**2)))

    pad_y = (diag - H) // 2
    pad_x = (diag - W) // 2

    padded = np.pad(
        img,
        ((pad_y, diag - H - pad_y), (pad_x, diag - W - pad_x), (0, 0)),
        mode='constant',
        constant_values=0
    )
    return padded, pad_x, pad_y



def apply_rotation_translation(u, theta, tx, ty, mask=None, return_also_transformed_mask=False):
    H,W = u.shape[:2]
    flow = create_flow_old(H, W, theta, tx, ty)
    if return_also_transformed_mask:
        return warp(u, flow), flow, warp(mask, flow)
    return warp(u, flow), flow






def scale_DepthPro_mask(mask, new_min, new_max):
    m,M = mask.min(), mask.max()
    ########delta = np.random.rand()*0.5
    #result = (1-delta)*(mask-M)/(m-M)
    result = ( new_max*(mask-m) + new_min*(M-mask) ) / (M-m)
    return result








### FUNCTIONS FOR INTERPOLATION AND WARPING

def cubic_interpolation(A, B, C, D, x):
    a,b,c,d = A.size()
    x = x.view(a,1,c,d).repeat(1,b,1,1)
    return B + 0.5*x*(C - A + x*(2.*A - 5.*B + 4.*C - D + x*(3.*(B - C) + D - A)))

def bicubic_interpolation_slow(x, vgrid):
    B, C, H, W = x.size()
    if B>0:
        output = torch.cat( [bicubic_interpolation(x[i:(i+1),:,:,:], vgrid[i:(i+1),:,:,:]) for i in range(B)], 0)
    else:
        output = bicubic_interpolation(x, vgrid)
    return output

def bicubic_interpolation(im, grid):
    B, C, H, W = im.size()
    assert B == 1, "For the moment, this interpolation only works for B=1."

    x0 = torch.floor(grid[0, 0, :, :] - 1).long()
    y0 = torch.floor(grid[0, 1, :, :] - 1).long()
    x1 = x0 + 1
    y1 = y0 + 1
    x2 = x0 + 2
    y2 = y0 + 2
    x3 = x0 + 3
    y3 = y0 + 3

    x0 = x0.clamp(0, W-1)
    y0 = y0.clamp(0, H-1)
    x1 = x1.clamp(0, W-1)
    y1 = y1.clamp(0, H-1)
    x2 = x2.clamp(0, W-1)
    y2 = y2.clamp(0, H-1)
    x3 = x3.clamp(0, W-1)
    y3 = y3.clamp(0, H-1)

    A = cubic_interpolation(im[:, :, y0, x0], im[:, :, y1, x0], im[:, :, y2, x0],
                                 im[:, :, y3, x0], grid[:, 1, :, :] - torch.floor(grid[:, 1, :, :]))
    B = cubic_interpolation(im[:, :, y0, x1], im[:, :, y1, x1], im[:, :, y2, x1],
                                 im[:, :, y3, x1], grid[:, 1, :, :] - torch.floor(grid[:, 1, :, :]))
    C = cubic_interpolation(im[:, :, y0, x2], im[:, :, y1, x2], im[:, :, y2, x2],
                                 im[:, :, y3, x2], grid[:, 1, :, :] - torch.floor(grid[:, 1, :, :]))
    D = cubic_interpolation(im[:, :, y0, x3], im[:, :, y1, x3], im[:, :, y2, x3],
                                 im[:, :, y3, x3], grid[:, 1, :, :] - torch.floor(grid[:, 1, :, :]))
    return cubic_interpolation(A, B, C, D, grid[:, 0, :, :] - torch.floor(grid[:, 0, :, :]))


def warp(x, flow, interp='bicubic'):
    """
    Differentiably warp a tensor according to the given optical flow.

    Args:
        x    : numpy array of dimension [H, W, 3], image to be warped.
        flow : numpy array of dimension [H, W, 2], optical flow
        inter: str, can be 'nearest', 'bilinear' or 'bicubic'
    
    Returns:
        y   : numpy array of dimension [H, W, 3], image warped according to flow
    """
    
    x    = torch.Tensor(x   ).permute(2,0,1).unsqueeze(0).cuda()
    flow = torch.Tensor(flow).permute(2,0,1).unsqueeze(0).cuda()
    _,C,H,W = x.shape

    yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                            torch.arange(W, device=x.device))

    xx, yy = map(lambda x: x.view(1,1,H,W), [xx,yy])

    grid = torch.cat((xx, yy), 1).float()
    vgrid = Variable(grid) + flow.to(x.device)

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0*vgrid[:, 0, :, :]/(W-1) - 1.0
    vgrid[:, 1, :, :] = 2.0*vgrid[:, 1, :, :]/(H-1) - 1.0
    mask = (vgrid[:, 0, :, :] >= -1) * (vgrid[:, 0, :, :] <= 1) *\
           (vgrid[:, 1, :, :] >= -1) * (vgrid[:, 1, :, :] <= 1)
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, padding_mode="zeros",
                                       mode=interp, align_corners=True)

    return output[0].detach().cpu().numpy().transpose(1,2,0)




def random_float(m,M):  
    """
    Returns a random float number in [m,M]
    """
    return np.random.rand()*(M-m) + m




#def crop_biggest_rectangle(all_in_focus, bokeh, mask, flows, size_should_be_multiple_of=1):
#    """
#    The research of the black border will be done on the images all_on_focus, but the same crops will be applied to all the images (the mask, the bokeh, the flows)
#    """
#    N,H,W,C = all_in_focus.shape
#    # Crop parameters
#    y1, y2 = 0,H-1
#    x1, x2 = 0,W-1
#    
#    for p in range(N):
#        # Create mask of non-black pixels
#        #black_border_mask = np.any(all_in_focus[p] != [0, 0, 0], axis=2)
#        black_border_mask = np.any(all_in_focus[p] < [0.4, 0.4, 0.4], axis=2)
#
#        # Find rows and columns where mask is True
#        rows = np.any(black_border_mask, axis=1)
#        cols = np.any(black_border_mask, axis=0)
#
#        y_min, y_max = np.where(rows)[0][[0, -1]]
#        x_min, x_max = np.where(cols)[0][[0, -1]]
#        if y_min > y1:
#            y1 = y_min
#        if y_max < y2:
#            y2 = y_max
#        if x_min > x1:
#            x1 = x_min
#        if x_max < x2:
#            x2 = x_max
#
#        # Ensure that the crop size is a multiple of the desired value
#        if (y2-y1) % size_should_be_multiple_of != 0:
#            y1 = y1 + (y2-y1)%size_should_be_multiple_of
#        if (x2-x1) % size_should_be_multiple_of != 0:
#            x1 = x1 + (x2-x1)%size_should_be_multiple_of
#
#    # Add +1 to x_max and y_max and return all the dataset, after a crop
#    return all_in_focus[:, y1:y2+1, x1:x2+1], bokeh[:, y1:y2+1, x1:x2+1], mask[:, y1:y2+1, x1:x2+1], flows[:, y1:y2+1, x1:x2+1]
#
#


def update_delta_list(delta_list, delta, mask_list):
    """ 
    Update the liste of depth (delta) of foreground object. The goal is that the foregournd objects can move (in the z-axis) from frame to frame.
    However, if they overlap, the one which is in front of the other remains in front of the other!
    """
    nb_objects = len(mask_list)
    for index in range(nb_objects):
        delta_list[index] = np.clip(delta_list[index] + random_float(-delta/3, delta/3), 1-delta, 1)

    return delta_list
    

    #distance   = np.argsort(np.array(delta_list)) #distance of each foreground object

    #
    #for index in range(nb_objects):
    #    if mask_list[index] * mask_list
    #    delta_list[index] = np.clip(delta_list[index] + random_float(-delta/3, delta/3), 1-delta, 1)


    #
    ##1st case: nothing overlap. Each object is free to move forward or backward in the z-axis
    #if np.max(np.prod(np.array(mask_list), axis=0)) == 0:
    #    for index in range(nb_objects):
    #        delta_list[index] = np.clip(delta_list[index] + random_float(-delta/3, delta/3), 1-delta, 1)
    ##2nd case: objects overlap
    #else:
    #    if mask_list[0] * mask_list[1] == 0  and mask_list[0] * mask_list[2] == 0: #the object 0 is not overlapping any other object
    #        delta_list[0] = np.clip(delta_list[0] + random_float(-delta/3, delta/3), 1-delta, 1)
    #    if mask_list[1] * mask_list[0] == 0  and mask_list[1] * mask_list[2] == 0: #the object 1 is not overlapping any other object
    #        delta_list[1] = np.clip(delta_list[1] + random_float(-delta/3, delta/3), 1-delta, 1)
    #    if mask_list[2] * mask_list[0] == 0  and mask_list[2] * mask_list[1] == 0: #the object 2 is not overlapping any other object
    #        delta_list[2] = np.clip(delta_list[2] + random_float(-delta/3, delta/3), 1-delta, 1)
    #    if np.max(mask_list[0]*mask_list[1]*mask_list[2]) > 0: #all objects overlap
    #        distance = np.argsort(np.array(delta_list))




    #return delta_list




#def largest_rotated_rect(w, h, angle):
#    """
#    Given a rectangle of size wxh that has been rotated by 'angle' (in
#    radians), computes the width and height of the largest possible
#    axis-aligned rectangle within the rotated rectangle.
#
#    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
#
#    Converted to Python by Aaron Snoswell
#    """
#
#    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
#    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
#    alpha = (sign_alpha % math.pi + math.pi) % math.pi
#
#    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
#    bb_h = w * math.sin(alpha) + h * math.cos(alpha)
#
#    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)
#
#    delta = math.pi - alpha - gamma
#
#    length = h if (w < h) else w
#
#    d = length * math.cos(alpha)
#    a = d * math.sin(alpha) / math.sin(delta)
#
#    y = a * math.cos(gamma)
#    x = y * math.tan(gamma)
#
#    return (
#        bb_w - 2 * x,
#        bb_h - 2 * y
#    )
#

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]





def look_if_complete_black_triangle(im, i, j, border):
    H,W,_ = im.shape
    #coordinates of the center
    cy,cx = H//2, W//2
    
    k = 0
    result = True
    #print("we start at : ", cy-i, cx-j)
    if border == 'top':       
        while cy-i-k>0 and result: 
            #print(-(np.any(im[cy-i-k, cx-j:cx+j], axis=1)-1).sum())
            #print(im[cy-i-k, cx-j:cx+j])
            if -(np.any(im[cy-i-k, cx-j:cx+j], axis=1)-1).sum()<=k: #I compte the number of pixel equal to [0,0,0]
                result=False
            k = k+1
            #print("next iter")
            #print("")
            #print("")

    if border == 'bottom':       
        while cy+i+k<H and result: 
            if -(np.any(im[cy+i+k, cx-j:cx+j], axis=1)-1).sum()<=k: #I compte the number of pixel equal to [0,0,0]
                result=False
            k = k+1

    if border == 'left':       
        while cx-j-k>0 and result: 
            if -(np.any(im[cy-i:cy+i, cx-j-k], axis=1)-1).sum()<=k: #I compte the number of pixel equal to [0,0,0]
                result=False
            k = k+1

    if border == 'right':       
        while cx+j+k<W and result: 
            if -(np.any(im[cy-i:cy+i, cx+j+k], axis=1)-1).sum()<=k: #I compte the number of pixel equal to [0,0,0]
                result=False
            k = k+1


    return result





def crop_square_around_center(im):
    H,W,_ = im.shape
    #coordinates of the center
    cy,cx = H//2, W//2

    # offset from the center
    i,j = 1,1
  
    Continue = True #should we continue or stop
    while Continue and i<H//2 and j<W//2:
        top_border    = im[cy-i, cx-j:cx+j]
        bottom_border = im[cy+i, cx-j:cx+j]
        left_border   = im[cy-i:cy+i, cx-j]
        right_border  = im[cy-i:cy+i, cx+j]
            
        # if there is no zero in all borders, we continue to increase the square
        if np.all(np.any(top_border, axis=1)) and np.all(np.any(bottom_border, axis=1)) and np.all(np.any(left_border, axis=1)) and np.all(np.any(right_border, axis=1)):
            #print(np.any(top_border))
            i,j = i+1, j+1
        # if not, we look if the border a an isole zeros (a black pixel) of if it is a full black triangle
        else:
            #print("we enter this else")
            if not np.all(np.any(top_border, axis=1)):
                #print("top")
                Continue = (Continue and not look_if_complete_black_triangle(im, i, j, 'top'))
            if Continue and not np.all(np.any(bottom_border, axis=1)):
                #print("bottom")
                Continue = (Continue and not look_if_complete_black_triangle(im, i, j, 'bottom'))
            if Continue and not np.all(np.any(left_border, axis=1)):
                #print("left")
                Continue = (Continue and not look_if_complete_black_triangle(im, i, j, 'left'))
            if Continue and not np.all(np.any(right_border, axis=1)):
                #print("right")
                Continue = (Continue and not look_if_complete_black_triangle(im, i, j, 'right'))

            i,j = i+1, j+1

    return  i-1, j-1 ###################,look_if_complete_black_triangle(im, 129,129, 'top')



def crop_biggest_rectangle(all_in_focus, bokeh, mask, flows, size_should_be_multiple_of=1):
    """
    The research of the black border will be done on the images all_on_focus, but the same crops will be applied to all the images (the mask, the bokeh, the flows)
    """
    N,H,W,C = all_in_focus.shape
    # Crop parameters
    y1, y2 = 0,H-1
    x1, x2 = 0,W-1
    
    for p in range(N):
        i, j = crop_square_around_center(all_in_focus[p])
        k, l = crop_square_around_center(mask        [p])
        i, j = np.min([i,k]), np.min([j,l]) #maybe we should crop the frames, maybe the mask? We take the smallest rectangle so that there is no black border for sure in any of these two options


        if H//2-i > y1:
            y1 = H//2-i
        if H//2+i < y2:
            y2 = H//2+i
        if W//2-j > x1:
            x1 = W//2-j
        if W//2+j < x2:
            x2 = W//2+j

    # Ensure that the crop size is a multiple of the desired value
    if (y2-y1) % size_should_be_multiple_of != 0:
        y1 = y1 + (y2-y1)%size_should_be_multiple_of
    if (x2-x1) % size_should_be_multiple_of != 0:
        x1 = x1 + (x2-x1)%size_should_be_multiple_of

    return all_in_focus[:, y1:y2, x1:x2], bokeh[:, y1:y2, x1:x2], mask[:, y1:y2, x1:x2], flows[:, y1:y2, x1:x2]



