import argparse
import glob
import iio
import numpy as np
from os import makedirs
from os.path import join, isdir
from scipy.ndimage import gaussian_filter
from utils_v2 import translate, resize, compose_image, update_foreground_lists, update_crop_index, read_mask, scale_initial_mask, zig_zag_borders, apply_rotation_translation, random_float, crop_biggest_rectangle, update_delta_list
import torch


np.random.seed(2025)




if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Create a dataset with Bokeh")
    parser.add_argument("--background", type=str, default="", help='path to background frames (C type)')
    parser.add_argument("--foreground", type=str, default="", help='path to foreground frames (C type)')
    parser.add_argument("--alpha_matte_mask", type=str, default="", help='path to mask (alpha-matting) of foreground frames (C type)')
    parser.add_argument("--DepthPro_mask", type=str, default="", help='path to mask (DepthPro) of foreground frames (C type)')
    parser.add_argument("--foregroundDepthPro_mask", type=str, default="", help='path to mask (DepthPro) of foreground frames (C type)')
    parser.add_argument("--sigma", type=float, nargs='+', default=(4.5,8.0), help='value(s) of sigma for the Gaussian blur. If single value, this value. If 2 values, a random value is sampled between these two values')
    parser.add_argument("--output_all_in_focus", type=str, default="", help='path to the output of all-in-focus frames')
    parser.add_argument("--output_bokeh_masks", type=str, default="", help='path to the output of Bokeh mask')
    parser.add_argument("--output_bokeh_frames", type=str, default="", help='path to the output of Bokeh frames')
    parser.add_argument("--output_flows", type=str, default="", help='path to the output of the flows')
    parser.add_argument("--nb_videos", type=int, default=1, help='number of videos we want to generate')
    parser.add_argument("--nb_frames", type=int, default=3, help='number of frames to generate')
    parser.add_argument("--max_speed_translation", type=int, default=15, help='Max number of pixels for a translation')
    parser.add_argument("--max_angle_rotation", type=float, default=5, help='Max angle for the rotation')
    parser.add_argument("--max_inter_frames_acceleration", type=int, default=3, help='Max difference of number of pixels for a translation between two contiguous frames (to have a more or less continuous motion without drastic and unrealistic acceleration and deceleration)')
    parser.add_argument("--max_rotation_acceleration", type=int, default=2, help='Max difference of angle for a rotation between two contiguous frames (to have a more or less continuous motion without drastic and unrealistic acceleration and deceleration)')

    args = parser.parse_args()

    # List of images
    list_of_background_images = sorted(glob.glob(join(args.background      , '*')))
    list_of_foreground_images = sorted(glob.glob(join(args.foreground      , '*')))
    list_of_masks             = sorted(glob.glob(join(args.alpha_matte_mask, '*')))
    list_of_DepthProMasks     = sorted(glob.glob(join(args.foregroundDepthPro_mask, '*')))
    list_of_depthMaps         = sorted(glob.glob(join(args.DepthPro_mask   , '*')))
    nb_background_images = len(list_of_background_images)
    nb_foreground_images = len(list_of_foreground_images)
    assert len(list_of_masks    )==nb_foreground_images
    print("%1d background images found, %1d foreground images (and masks) found"%(nb_background_images, nb_foreground_images))
    
    
    # Main loop that generates the videos
    for video in range(args.nb_videos):
        print('\r' + "Generating video %04d"%video, end='')
        # Load a random background image
        i = np.random.randint(nb_background_images) #choose randomly a background image
        background = iio.read(list_of_background_images[i])
        DepthPro_mask_BG = iio.read(list_of_depthMaps[i])
        DepthPro_mask_BG, delta = scale_initial_mask(DepthPro_mask_BG)
        print("initial delta is ", delta)
        H,W,_ = background.shape
        # Sample the translation and rotation parameters (for the background)
        ty_bg    = np.zeros(args.nb_frames-1, dtype=np.float32) #translation in y-axis
        tx_bg    = np.zeros(args.nb_frames-1, dtype=np.float32) #translation in x-axis
        theta_bg = np.zeros(args.nb_frames-1, dtype=np.float32) #rotation angle (in degree)
        ty_bg[0]    = random_float(-args.max_speed_translation, args.max_speed_translation) 
        tx_bg[0]    = random_float(-args.max_speed_translation, args.max_speed_translation) 
        theta_bg[0] = random_float(-args.max_angle_rotation   , args.max_angle_rotation   )
        for t in range(1, args.nb_frames-1):
            ty_bg[t] = np.clip(ty_bg[t-1]+random_float(-args.max_inter_frames_acceleration, args.max_inter_frames_acceleration), -args.max_speed_translation, args.max_speed_translation)
            tx_bg[t] = np.clip(tx_bg[t-1]+random_float(-args.max_inter_frames_acceleration, args.max_inter_frames_acceleration), -args.max_speed_translation, args.max_speed_translation)
            theta_bg[t-1] = np.clip(theta_bg[t-1]+random_float(-args.max_rotation_acceleration, args.max_rotation_acceleration), -args.max_angle_rotation, args.max_angle_rotation)

        ########### Set starting crop rectangle
        ##############i,j,k,l = 0,0,0,0 # crop to do at each border to avoid black triangles
        #"##########try:    
        
        # Sample number of foreground objects (either 1, 2 or 3)
        random = np.random.rand() #use to determine if we have 1, 2 or 3 objects
        random = 0.9
        nb_foreground_objects = (random > 0.5) + (random > 0.2) + 1
        
        # Create a list of foreground object and mask (1,2, or 3 objects) and rotation/translation parameters
        foreground_list, mask_list, depth_pro_mask_list, delta_list = [], [], [], []
        # Sample the translation and rotation parameters (for the foreground)
        ty_fg    = np.zeros((args.nb_frames-1, nb_foreground_objects), dtype=np.float32) #translation in the y-axis for the foreground objects 
        tx_fg    = np.zeros((args.nb_frames-1, nb_foreground_objects), dtype=np.float32) #translation in the x-axis for the foreground objects  
        theta_fg = np.zeros((args.nb_frames-1, nb_foreground_objects), dtype=np.float32) #rotation angle (in degree) for the foreground objects 
        delta_fg = np.zeros((args.nb_frames  , nb_foreground_objects), dtype=np.float32) #``depth'' of the foreground object (alway above the background). Between 0 and 1.  0 in ``infinity'', 1 is against the sensor
        for foreground_object in range(nb_foreground_objects):
            ty_fg[   0, foreground_object] = random_float(-args.max_speed_translation, args.max_speed_translation)
            tx_fg[   0, foreground_object] = random_float(-args.max_speed_translation, args.max_speed_translation)
            theta_fg[0, foreground_object] = random_float(-args.max_angle_rotation   , args.max_angle_rotation   )
            delta_fg[0, foreground_object] = random_float(1-delta                    , delta                     )
            for t in range(1, args.nb_frames-1):
                ty_fg[   t, foreground_object]    = np.clip(ty_fg[t-1, foreground_object]+random_float(-args.max_inter_frames_acceleration, args.max_inter_frames_acceleration), -args.max_speed_translation, args.max_speed_translation)
                tx_fg[   t, foreground_object]    = np.clip(tx_fg[t-1, foreground_object]+random_float(-args.max_inter_frames_acceleration, args.max_inter_frames_acceleration), -args.max_speed_translation, args.max_speed_translation)
                theta_fg[t, foreground_object] = np.clip(theta_fg[t-1, foreground_object]+random_float(-args.max_rotation_acceleration    , args.max_rotation_acceleration    ), -args.max_angle_rotation   , args.max_angle_rotation   )
                delta_fg[t, foreground_object] = np.clip(delta_fg[t-1, foreground_object]+random_float(-delta/3                           , delta/3                           ), 1 - delta                  ,                          1)
            delta_fg[-1, foreground_object] = np.clip(delta_fg[-2, foreground_object]+random_float(-delta/3                           , delta/3                           ), 1 - delta                  ,                          1)



        
        
        # First object
        p = np.random.randint(nb_foreground_images) # choose randomly a foreground object
        foreground = iio.read(list_of_foreground_images[p])
        mask       = read_mask(list_of_masks[p])
        mask       = zig_zag_borders(mask) #to avoid straight line at borders
        DepthPro_mask = iio.read(list_of_DepthProMasks[p])
        foreground, mask, DepthPro_mask = resize(foreground, mask, DepthPro_mask, H,W)
        h,w,_ = foreground.shape
        start_y, start_x = np.random.randint(H-h), np.random.randint(W-w) #initial position of the top-left (array[0,0] in python) corner of the foreground object
        print(DepthPro_mask.shape)
        foreground_list, mask_list, depth_pro_mask_list = update_foreground_lists(foreground_list, mask_list, depth_pro_mask_list, foreground, mask, DepthPro_mask, start_y, start_x, H, W, h, w)
        
        # Second object
        #if random > 0.2:
        if True:
            p = np.random.randint(nb_foreground_images)
            foreground = iio.read(list_of_foreground_images[p])
            mask       = read_mask(list_of_masks[p])
            mask       = zig_zag_borders(mask) #to avoid straight line at borders
            DepthPro_mask = iio.read(list_of_DepthProMasks[p])
            foreground, mask, DepthPro_mask = resize(foreground, mask, DepthPro_mask, H,W)
            h,w,_ = foreground.shape
            start_y, start_x = np.random.randint(H-h), np.random.randint(W-w) #initial position of the top-left (array[0,0] in python) corner of the foreground object
            foreground_list, mask_list, depth_pro_mask_list = update_foreground_lists(foreground_list, mask_list, depth_pro_mask_list, foreground, mask, DepthPro_mask, start_y, start_x, H, W, h, w)
        
        # Third object
            #if random > 0.5:
            if True:
                p = np.random.randint(nb_foreground_images)
                foreground = iio.read(list_of_foreground_images[p])
                mask       = read_mask(list_of_masks[p])
                mask       = zig_zag_borders(mask) #to avoid straight line at borders
                DepthPro_mask = iio.read(list_of_DepthProMasks[p])
                foreground, mask, DepthPro_mask = resize(foreground, mask, DepthPro_mask, H,W)
                h,w,_ = foreground.shape
                start_y, start_x = np.random.randint(H-h), np.random.randint(W-w) #initial position of the top-left (array[0,0] in python) corner of the foreground object
                foreground_list, mask_list, depth_pro_mask_list = update_foreground_lists(foreground_list, mask_list, depth_pro_mask_list, foreground, mask, DepthPro_mask, start_y, start_x, H, W, h, w)
        

        ALL_IN_FOCUS, BOKEH, MASK, FLOWS = np.zeros((args.nb_frames, H, W, 3), dtype=np.float32), np.zeros((args.nb_frames, H, W, 3), dtype=np.float32), np.zeros((args.nb_frames, H, W, 3), dtype=np.float32), np.zeros((args.nb_frames-1, H, W, 2), dtype=np.float32)


        print(ALL_IN_FOCUS.shape)
        print(BOKEH.shape)
        print(MASK.shape)
        print(FLOWS.shape)
        print(background.shape)
        print(np.array(foreground_list).shape)
        print(np.array(mask_list).shape)
        print(np.array(depth_pro_mask_list).shape)

        # First frames
        ALL_IN_FOCUS[0], BOKEH[0], MASK[0], sigma = compose_image(background, DepthPro_mask_BG, foreground_list, mask_list, depth_pro_mask_list, delta_fg[0], args.sigma)


        for p in range(1,args.nb_frames):
            ############ Update the crop rectangle
            #################i,j,k,l = update_crop_index(i,j,k,l,ty_bg,tx_bg)

            # Translate the background
            print(tx_bg)
            print(tx_bg[:p])

            rotated_background, flow, DepthPro_mask_BG = apply_rotation_translation(background, theta_bg[:p].sum(), tx_bg[:p].sum(), ty_bg[:p].sum(), mask=DepthPro_mask_BG, return_also_transformed_mask=True)
            
            # Translate the foreground objects
            new_foreground_list, new_mask_list, new_depth_pro_mask_list = [], [], []
            for (fg, Mask, DepthProMask, ty, tx, theta) in zip(foreground_list, mask_list, depth_pro_mask_list, ty_fg[:p].sum(axis=0), tx_fg[:p].sum(axis=0), theta_fg[:p].sum(axis=0)):
                transformed, _ = apply_rotation_translation(fg, theta, tx, ty)
                new_foreground_list.append(transformed)
                transformed, flow_fg = apply_rotation_translation(Mask, theta, tx, ty)
                flow[transformed[:,:,:2]>2/255] = flow_fg[transformed[:,:,:2]>2/255]
                new_mask_list.append(transformed)
                transformed, _ = apply_rotation_translation(DepthProMask, theta, tx, ty)
                new_depth_pro_mask_list.append(transformed)

            # Update the list of foreground objects and masks
            #foreground_list = new_foreground_list
            #mask_list = new_mask_list
            #depth_pro_mask_list = new_depth_pro_mask_list


            # Compose the image
            #print(delta_fg)
            #print(delta_fg[:p+1])
            ALL_IN_FOCUS[p], BOKEH[p], MASK[p], sigma = compose_image(rotated_background, DepthPro_mask_BG, new_foreground_list, new_mask_list, new_depth_pro_mask_list, delta_fg[:p+1].sum(axis=0), args.sigma)
            FLOWS[p-1] = flow

        
                    


            ############## Update the list of delta (the foreground object can move slighly from frame to frame)
            #############print("avant update ", delta_list) 
            #############delta_list = update_delta_list(delta_list, delta, mask_list)
            #############print("après update ", delta_list) 
        


        
        ######################""""# Ensure that the crop size will be a multiple of 8
        ######################""""if (H-k-i) % 8 != 0:
        ######################""""    k = k+(H-k-i)%8-8
        ######################""""if (W-l-j) % 8 != 0:
        ######################""""    l = l+(W-l-j)%8-8
        ######################""""# Convert the list to numpy array and crop the frames to remove black triangles
        ######################""""#ALL_IN_FOCUS = np.array(ALL_IN_FOCUS)[:  , i:H-k, j:W-l]
        ######################""""#BOKEH        = np.array(BOKEH       )[:  , i:H-k, j:W-l]
        ######################""""#MASK         = np.array(MASK        )[:  , i:H-k, j:W-l]*255
        ######################""""#FLOWS        = np.array(FLOWS       )[:-1, i:H-k, j:W-l]

        ALL_IN_FOCUS, BOKEH, MASK, FLOWS = crop_biggest_rectangle(ALL_IN_FOCUS, BOKEH, MASK*255, FLOWS)



        # Save the frames
        if not isdir(join(args.output_all_in_focus, '%04d'%video)):
            makedirs(join(args.output_all_in_focus, '%04d'%video))
        if not isdir(join(args.output_bokeh_frames, '%04d'%video)):
            makedirs(join(args.output_bokeh_frames, '%04d'%video))
        if not isdir(join(args.output_bokeh_masks , '%04d'%video)):
            makedirs(join(args.output_bokeh_masks , '%04d'%video))
        if not isdir(join(args.output_flows       , '%04d'%video)):
            makedirs(join(args.output_flows       , '%04d'%video))
        
        for p in range(args.nb_frames-1):
            iio.write(join(args.output_all_in_focus, '%04d/%03d.png'%(video,p)), ALL_IN_FOCUS[p])
            iio.write(join(args.output_bokeh_frames, '%04d/%03d.png'%(video,p)), BOKEH[       p])
            iio.write(join(args.output_bokeh_masks , '%04d/%03d.png'%(video,p)), MASK[        p])
            iio.write(join(args.output_flows       , '%04d/%03d.flo'%(video,p)), FLOWS[       p])
        iio.write(join(args.output_all_in_focus, '%04d/%03d.png'%(video,args.nb_frames-1)), ALL_IN_FOCUS[args.nb_frames-1])
        iio.write(join(args.output_bokeh_frames, '%04d/%03d.png'%(video,args.nb_frames-1)), BOKEH[       args.nb_frames-1])
        iio.write(join(args.output_bokeh_masks , '%04d/%03d.png'%(video,args.nb_frames-1)), MASK[        args.nb_frames-1])
        
