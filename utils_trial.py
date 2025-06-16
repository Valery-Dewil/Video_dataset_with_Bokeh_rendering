import argparse
from gblurs import G
import iio
import numpy as np
from scipy.ndimage import gaussian_filter
import torch




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




def resize(img,mask,h,w,max_ratio=(1/2,1/2), min_ratio=(1/3,1/3), antialiasing=True):
    """
    Take a foreground image img and the corresponding mask.
    1. Crop the img in a rectangle image where there is no complete line or column equal to 0.
    2. Resize the image and the mask so that it is not greater than __ratio__ (for instance 1/3) of the h and w. The h and w come from the background image
    """
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
    object_img   = object_img[i:k, j:l]
    mask_cropped = mask[      i:k, j:l]

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
        object_img   = torch.nn.functional.interpolate(torch.Tensor(object_img  ).permute(0,1,2).unsqueeze(0), scale_factor=1/ratio, mode='bicubic', align_corners=True).numpy().squeeze().transpose(1,2,0)
        mask_cropped = torch.nn.functional.interpolate(torch.Tensor(mask_cropped).permute(0,1,2).unsqueeze(0), scale_factor=1/ratio, mode='bicubic', align_corners=True).numpy().squeeze().transpose(1,2,0)
    else:                                                #resize in the y-axis dominates
        #the ratio should be taken into account in the y-axis instead:
        ratio = (k-i) / max_size_y   # np.ceil((k-i) / max_size_y)
        # apply antialiasing
        if antialiasing:
            object_img = G('borelli', object_img, 0.5*np.sqrt(ratio**2-1))
        # scale the image by ratio and return them
        object_img   = torch.nn.functional.interpolate(torch.Tensor(object_img  ).permute(2,0,1).unsqueeze(0), scale_factor=1/ratio, mode='bicubic', align_corners=True).numpy().squeeze().transpose(1,2,0)
        mask_cropped = torch.nn.functional.interpolate(torch.Tensor(mask_cropped).permute(2,0,1).unsqueeze(0), scale_factor=1/ratio, mode='bicubic', align_corners=True).numpy().squeeze().transpose(1,2,0)

    return object_img, mask_cropped
    



def compose_image(background, initial_mask, foreground_list, mask_list, delta_list, sigma):
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

    for foreground, mask, delta in zip(foreground_list, mask_list, delta_list):
        # all in focus
        all_in_focus = all_in_focus*(1-mask) + foreground*mask

        # with Bokeh
        with_Bokeh = with_Bokeh*(1-mask) + foreground*mask

    # composed mask  
    #Mask = np.sum(np.array(mask_list), axis=0).clip(0,1)
    Mask = compose_mask(initial_mask, mask_list, delta_list)

    return all_in_focus, with_Bokeh, Mask, sigma

def compose_mask(initial_mask, mask_list, delta_list):
    mask_list = np.array(mask_list)
    delta_list = np.array(delta_list)
    N,H,W,C = mask_list.shape
    print(H,W,C)
    Mask = initial_mask.copy()
    max_delta=0
    for i in range(N):
        max_delta = np.max([max_delta, delta_list[i]])
        #Mask = Mask + (mask_list[i]*delta_list[i] + mask_list[i+1]*delta_list[i+1]).clip(0, max_delta)
        Mask = (Mask + mask_list[i]*delta_list[i]).clip(0,max_delta)

    return Mask



def update_lists(fg_list, mask_list, foreground, mask,start_y, start_x, H, W, h, w):

        Foreground, Mask = np.zeros((H,W,3), dtype=np.float32), np.zeros((H,W,3), dtype=np.float32)
        if start_y>H or start_y<0 or start_x>W or start_x<0:
            #we are out of the image, so we don't add this as an object anymore
            return fg_list, mask_list, False
       
        #else, we copy paste the object at the good location and add it in the lists
        if start_y+h<=H:
            if start_x+w<=W:
                Foreground[start_y:start_y+h, start_x:start_x+w] = foreground
                Mask[      start_y:start_y+h, start_x:start_x+w] = mask
            else:
                Foreground[start_y:start_y+h, start_x:] = foreground[:,:W-start_x]
                Mask[      start_y:start_y+h, start_x:] = mask[     :, :W-start_x]
        else:
            if start_x+w<=W:
                Foreground[start_y:, start_x:start_x+w] = foreground[:H-start_y,:]
                Mask[      start_y:, start_x:start_x+w] = mask[      :H-start_y,:]      
            else:
                Foreground[start_y:, start_x:] = foreground[:H-start_y,:W-start_x]
                Mask[      start_y:, start_x:] = mask[      :H-start_y,:W-start_x]      

        fg_list.append(Foreground)
        mask_list.append(Mask)

        return fg_list, mask_list, True



def read_mask(path, threshold=2):
    mask = iio.read(path)
    # See if the mask is [0,1] or [0,255]. The threshold is used for that.
    if np.max(mask)>2:
        return  mask/255
    return mask





def scale_initial_mask(mask):
    m,M = mask.min(), mask.max()
    delta = np.random.rand()/3
    result = (1-delta)*(mask-M)/(m-M)
    return result, delta
