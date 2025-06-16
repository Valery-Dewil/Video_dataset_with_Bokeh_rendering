import argparse
import glob
import iio
import numpy as np
from os import makedirs
from os.path import join, isdir
from scipy.ndimage import gaussian_filter
from utils import translate, resize, compose_image, update_lists, update_crop_index, read_mask, scale_initial_mask, apply_rotation_translation, random_float, crop_biggest_rectangle, update_delta_list
import torch


np.random.seed(2025)




if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Create a dataset with Bokeh")
    parser.add_argument("--background", type=str, default="", help='path to background frames (C type)')
    parser.add_argument("--foreground", type=str, default="", help='path to foreground frames (C type)')
    parser.add_argument("--alpha_matte_mask", type=str, default="", help='path to mask (alpha-matting) of foreground frames (C type)')
    parser.add_argument("--DepthPro_mask", type=str, default="", help='path to mask (DepthPro) of foreground frames (C type)')
    parser.add_argument("--sigma", type=float, nargs='+', default=(4.5,8.0), help='value(s) of sigma for the Gaussian blur. If single value, this value. If 2 values, a random value is sampled between these two values')
    parser.add_argument("--output_all_in_focus", type=str, default="", help='path to the output of all-in-focus frames')
    parser.add_argument("--output_bokeh_masks", type=str, default="", help='path to the output of Bokeh mask')
    parser.add_argument("--output_bokeh_frames", type=str, default="", help='path to the output of Bokeh frames')
    parser.add_argument("--output_flows", type=str, default="", help='path to the output of the flows')
    parser.add_argument("--nb_videos", type=int, default=1, help='number of videos we want to generate')
    parser.add_argument("--nb_frames", type=int, default=3, help='number of frames to generate')
    parser.add_argument("--max_speed_translation", type=int, default=15, help='Max number of pixels for a translation')
    parser.add_argument("--max_angle_rotation", type=float, default=45, help='Max angle for the rotation')
    parser.add_argument("--max_inter_frames_acceleration", type=int, default=3, help='Max difference of number of pixels for a translation between two contiguous frames (to have a more or less continuous motion without drastic and unrealistic acceleration and deceleration)')
    parser.add_argument("--max_rotation_acceleration", type=int, default=10, help='Max difference of angle for a rotation between two contiguous frames (to have a more or less continuous motion without drastic and unrealistic acceleration and deceleration)')

    args = parser.parse_args()

    # List of images
    list_of_background_images =        glob.glob(join(args.background      , '*'))
    list_of_foreground_images = sorted(glob.glob(join(args.foreground      , '*')))
    list_of_masks             = sorted(glob.glob(join(args.alpha_matte_mask, '*')))
    list_of_depthMaps         = sorted(glob.glob(join(args.DepthPro_mask   , '*')))
    nb_background_images = len(list_of_background_images)
    nb_foreground_images = len(list_of_foreground_images)
    assert len(list_of_masks    )==nb_foreground_images
    print("%1d background images found, %1d foreground images (and masks) found"%(nb_background_images, nb_foreground_images))
    
    
    # Main loop that generates the videos
    video = 0
    while video < args.nb_videos:
        print('\r' + "Generating video %04d"%video, end='')
        # Load a random background image
        i = np.random.randint(nb_background_images)
        background = iio.read(list_of_background_images[i])
        initial_mask = iio.read(list_of_depthMaps[i])
        initial_mask, delta = scale_initial_mask(initial_mask)
        H,W,_ = background.shape
        # Sample a first translation
        ty_bg = random_float(-args.max_speed_translation, args.max_speed_translation) #translation in y axis
        tx_bg = random_float(-args.max_speed_translation, args.max_speed_translation) #translation in x axis
        # Sample a first rotation
        theta_bg = random_float(-args.max_angle_rotation, args.max_angle_rotation)    #rotation angle
        # Set starting crop rectangle
        i,j,k,l = 0,0,0,0 # crop to do at each border to avoid black triangles
        #try:    
        print("c0")

        # Create a list of foreground object and mask (1,2, or 3 objects)
        foreground_list, mask_list, delta_list = [], [], []
        ty_foregrounds, tx_foregrounds = [], [] #the translation for each foreground object
        theta_foregrounds              = []     #the rotation    for each foreground object
        
        random = np.random.rand() #use to determine if we have 1, 2 or 3 objects
        
        # First object
        p = np.random.randint(nb_foreground_images)
        foreground = iio.read(list_of_foreground_images[p])
        mask       = read_mask(list_of_masks[p])
        print(foreground.shape, mask.shape)
        delta_list.append(np.random.rand()*delta+1-delta)
        foreground, mask = resize(foreground, mask, H,W)
        h,w,_ = foreground.shape
        start_y, start_x = np.random.randint(H-h), np.random.randint(W-w)
        foreground_list, mask_list, isupdated = update_lists(foreground_list, mask_list, foreground, mask, start_y, start_x, H, W, h, w)
        ty_foregrounds.append(random_float(-args.max_speed_translation, args.max_speed_translation)) #translation in y axis
        tx_foregrounds.append(random_float(-args.max_speed_translation, args.max_speed_translation)) #translation in x axis
        theta_foregrounds.append(random_float(-args.max_angle_rotation, args.max_angle_rotation   )) #rotation
        print("c11")
        
        # Second object
        #if random > 0.2:
        if True:
            p = np.random.randint(nb_foreground_images)
            foreground = iio.read(list_of_foreground_images[p])
            mask       = read_mask(list_of_masks[p])
            delta_list.append(np.random.rand()*delta+1-delta)
            foreground, mask = resize(foreground, mask, H,W)
            h,w,_ = foreground.shape
            start_y, start_x = np.random.randint(H-h), np.random.randint(W-w)
            foreground_list, mask_list, isupdated = update_lists(foreground_list, mask_list, foreground, mask, start_y, start_x, H, W, h, w)
            ty_foregrounds.append(random_float(-args.max_speed_translation, args.max_speed_translation)) #translation in y axis
            tx_foregrounds.append(random_float(-args.max_speed_translation, args.max_speed_translation)) #translation in x axis
            theta_foregrounds.append(random_float(-args.max_angle_rotation, args.max_angle_rotation   )) #rotation
        
        # Third object
            #if random > 0.5:
            if True:
                p = np.random.randint(nb_foreground_images)
                foreground = iio.read(list_of_foreground_images[p])
                mask       = read_mask(list_of_masks[p])
                delta_list.append(np.random.rand()*delta+1-delta)
                foreground, mask = resize(foreground, mask, H,W)
                h,w,_ = foreground.shape
                start_y, start_x = np.random.randint(H-h), np.random.randint(W-w)
                foreground_list, mask_list, isupdated = update_lists(foreground_list, mask_list, foreground, mask, start_y, start_x, H, W, h, w)
                ty_foregrounds.append(random_float(-args.max_speed_translation, args.max_speed_translation)) #translation in y axis
                tx_foregrounds.append(random_float(-args.max_speed_translation, args.max_speed_translation)) #translation in x axis
                theta_foregrounds.append(random_float(-args.max_angle_rotation, args.max_angle_rotation   )) #rotation
        
        print("c1")

        ALL_IN_FOCUS, BOKEH, MASK, FLOWS = [], [], [], [] 
        for p in range(args.nb_frames):
            # Update the crop rectangle
            i,j,k,l = update_crop_index(i,j,k,l,ty_bg,tx_bg)
            # Compose the image
            all_in_focus, bokeh, mask, sigma = compose_image(background, initial_mask, foreground_list, mask_list, delta_list, args.sigma)
            ALL_IN_FOCUS.append(all_in_focus)
            BOKEH.append(bokeh)
            MASK.append(mask)
            print("c2")
            
            # Translate the background
            background, flow = apply_rotation_translation(background, theta_bg, tx_bg, ty_bg)
            # Update the translation and rotation for next step
            ty_bg = np.clip(ty_bg+random_float(-args.max_inter_frames_acceleration, args.max_inter_frames_acceleration), -args.max_speed_translation, args.max_speed_translation)
            tx_bg = np.clip(tx_bg+random_float(-args.max_inter_frames_acceleration, args.max_inter_frames_acceleration), -args.max_speed_translation, args.max_speed_translation)
            theta_bg = np.clip(theta_bg+random_float(-args.max_rotation_acceleration, args.max_rotation_acceleration), -args.max_angle_rotation, args.max_angle_rotation)

            # Translate the foreground object
            new_foreground_list, new_mask_list, new_start_y_list, new_start_x_list, new_ty_foregrounds, new_tx_foregrounds, new_theta_foregrounds = [], [], [], [], [], [], []
            for (fg, Mask, ty, tx, theta) in zip(foreground_list, mask_list, ty_foregrounds, tx_foregrounds, theta_foregrounds):
                transformed, _ = apply_rotation_translation(fg, theta, tx, ty)
                new_foreground_list.append(transformed)
                transformed, flow_fg = apply_rotation_translation(Mask, theta, tx, ty)
                flow[transformed[:,:,:2]>2/255] = flow_fg[transformed[:,:,:2]>2/255]
                new_mask_list.append(transformed)
                new_ty_foregrounds.append(np.clip(ty+random_float(-args.max_inter_frames_acceleration, args.max_inter_frames_acceleration), -args.max_speed_translation, args.max_speed_translation))
                new_tx_foregrounds.append(np.clip(tx+random_float(-args.max_inter_frames_acceleration, args.max_inter_frames_acceleration), -args.max_speed_translation, args.max_speed_translation))
                new_theta_foregrounds.append(np.clip(theta+random_float(-args.max_rotation_acceleration, args.max_rotation_acceleration), -args.max_angle_rotation, args.max_angle_rotation))

            FLOWS.append(flow)

            # Update the list of delta (the foreground object can move slighly from frame to frame)
            print("avant update ", delta_list) 
            delta_list = update_delta_list(delta_list, delta, mask_list)
            print("après update ", delta_list) 
        
            # Update the list of foreground objects and masks
            foreground_list = new_foreground_list
            mask_list = new_mask_list
            ty_foregrounds    = new_ty_foregrounds
            tx_foregrounds    = new_tx_foregrounds
            theta_foregrounds = new_theta_foregrounds


        print("c3")
        
        # Ensure that the crop size will be a multiple of 8
        if (H-k-i) % 8 != 0:
            k = k+(H-k-i)%8-8
        if (W-l-j) % 8 != 0:
            l = l+(W-l-j)%8-8
        # Convert the list to numpy array and crop the frames to remove black triangles
        #ALL_IN_FOCUS = np.array(ALL_IN_FOCUS)[:  , i:H-k, j:W-l]
        #BOKEH        = np.array(BOKEH       )[:  , i:H-k, j:W-l]
        #MASK         = np.array(MASK        )[:  , i:H-k, j:W-l]*255
        #FLOWS        = np.array(FLOWS       )[:-1, i:H-k, j:W-l]

        ALL_IN_FOCUS, BOKEH, MASK, FLOWS = crop_biggest_rectangle(np.array(ALL_IN_FOCUS), np.array(BOKEH), np.array(MASK)*255, np.array(FLOWS))



        # Save the frames
        if not isdir(join(args.output_all_in_focus, '%04d'%video)):
            makedirs(join(args.output_all_in_focus, '%04d'%video))
        if not isdir(join(args.output_bokeh_frames, '%04d'%video)):
            makedirs(join(args.output_bokeh_frames, '%04d'%video))
        if not isdir(join(args.output_bokeh_masks , '%04d'%video)):
            makedirs(join(args.output_bokeh_masks , '%04d'%video))
        if not isdir(join(args.output_flows       , '%04d'%video)):
            makedirs(join(args.output_flows       , '%04d'%video))
        print("c4")
        
        for p in range(args.nb_frames-1):
            iio.write(join(args.output_all_in_focus, '%04d/%03d.png'%(video,p)), ALL_IN_FOCUS[p])
            iio.write(join(args.output_bokeh_frames, '%04d/%03d.png'%(video,p)), BOKEH[       p])
            iio.write(join(args.output_bokeh_masks , '%04d/%03d.png'%(video,p)), MASK[        p])
            iio.write(join(args.output_flows       , '%04d/%03d.flo'%(video,p)), FLOWS[       p])
        iio.write(join(args.output_all_in_focus, '%04d/%03d.png'%(video,args.nb_frames-1)), ALL_IN_FOCUS[args.nb_frames-1])
        iio.write(join(args.output_bokeh_frames, '%04d/%03d.png'%(video,args.nb_frames-1)), BOKEH[       args.nb_frames-1])
        iio.write(join(args.output_bokeh_masks , '%04d/%03d.png'%(video,args.nb_frames-1)), MASK[        args.nb_frames-1])
        
        video = video + 1
        #except:
        #    pass 
