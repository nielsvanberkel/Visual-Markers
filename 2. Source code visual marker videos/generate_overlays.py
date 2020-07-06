import cv2
import numpy as np
import glob
import os
import argparse
from PIL import Image

#
# Instructions:
#
# 1. Install python3 from https://www.python.org/downloads/.
# 2. Install the packages listed above using pip install. For example, 'pip install opencv-python', 'pip install numpy'.
# 3. Change working directory to the folder with the generate_overlays.py file. This folder should contain both the video file and a subfolder with the image mask files.
# 4. Execute command 'python3 generate_overlays.py --input_video_path "apparent.mov" --input_mask_path "apparent-mask" --condition 1' to generate your first video marker. 
# 5. The program will run and generate a video file with the specified video condition using the specified video file and annotation files.
#    See parameter overview below for additional arguments (e.g., video condition, coloured stills).
# 
# Additional example commands:
# 
# Generate a Segmentation Outline marker on the challenging video file:
# python3 generate_overlays.py --input_video_path "challenging.mov" --input_mask_path "challenging-mask" --condition 5
#
# Generate screenshots showing different colour options:
# python3 generate_overlays.py --input_video_path "apparent.mov" --input_mask_path "apparent-mask" --mode 1
#
# Bonus. Use ffmpeg to convert video files to expected size and video format.
# ffmpeg -i apparent_1_detection_output.mp4 -vf scale=640:480 apparent_1_480.mp4
#

global_colour = (255, 0, 0) # Blue
global_thickness = 4 # Thickness for 1080p videos. Adjust accordingly.

parser = argparse.ArgumentParser(description='Parse input and streams.')
parser.add_argument('--input_video_path', type=str,
                    help='Path to input video file')

parser.add_argument('--input_mask_path', type=str,
                    help='Path to input mask folder to overlay')

parser.add_argument('--condition', type=int, default=1,
                    help='Marker overlay. Default is 1. 1 = Wide bounding circle, 2 = Tight bounding box, 3 = Spotlight, 4 = Segmentation, 5 = Segmentation outline, 6 = Detection confidence, 7 = Detection signal, 10 = Nothing.')

parser.add_argument('--smoothness', type=int, default=2,
                    help='How smooth the output should be. Default is 2. Original = 0, showing unaltered detection.')

parser.add_argument('--expand', type=int, default=0,
                    help='How much to expand the bounding box. Default is 0.')

parser.add_argument('--flip_hori', action='store_true',
                    help='Flip image horizontally.')

parser.add_argument('--crop_xl', type=int, default=0,
                    help='How much to crop from left. Default is 0.')

parser.add_argument('--crop_xr', type=int, default=0,
                    help='How much to crop from right. Default is 0.')

parser.add_argument('--mode', type=int, default=0,
                    help='Video (0), coloured options (1), screenshots (2). Default is 0.')

args = parser.parse_args()

def blend_normal_mask(background_img, foreground_img, foreground_color=(1, 1, 1),
                      blend_ratio=0.3):

    # make float on range 0-1
    b = background_img.astype(float)/255
    f = foreground_img.astype(float)/255

    bf = np.zeros_like(b)

    mask_fg = f > 0

    for i in range(len(foreground_color)):
        f[:, :, i] *= foreground_color[i]

    bf[mask_fg] = (1.0 - blend_ratio) * b[mask_fg] + blend_ratio * f[mask_fg]
    bf[~mask_fg] = b[~mask_fg]

    return (255 * bf).astype(np.uint8)


def blend_normal_mask_border(background_img, foreground_img, frame = 1):
    # make float on range 0-1
    b = background_img.astype(float)/255
    f = foreground_img.astype(float)/255

    im_bw = cv2.cvtColor((f).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def smooth_signal(signal_in, loops=1):
    if loops == 0:
        return signal_in

    else:
        signal_out = signal_in.copy()

        for i in range(len(signal_in) - 2):
            if not np.isnan(signal_out[i+1]):
                signal_out[i+1] = np.nanmean(signal_in[i:i+3])

        return smooth_signal(signal_out, loops - 1)

# function to overlay a transparent image on background.
def transparentOverlay(src , overlay , pos=(0,0),scale = 1):
    """
    :param src: Input Color Background Image
    :param overlay: transparent Image (BGRA)
    :param pos:  position where the image to be blit.
    :param scale : scale factor of transparent image.
    :return: Resultant Image
    """
    overlay = cv2.resize(overlay,(0,0),fx=scale,fy=scale)
    h,w,_ = overlay.shape  # Size of foreground
    rows,cols,_ = src.shape  # Size of background Image
    y,x = pos[0],pos[1]    # Position of foreground/overlay image
    
    #loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x+i >= rows or y+j >= cols:
                continue
            alpha = float(overlay[i][j][3]/255.0) # read the alpha channel 
            src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
    return src

path_to_video = args.input_video_path

scaling_factor = 1

cap = cv2.VideoCapture(path_to_video)

print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(cap.get(cv2.CAP_PROP_FPS))

frames = []
ret, frame = cap.read()

i = 0

while ret:
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor)
    frames.append(frame)
    ret, frame = cap.read()
    i = i +1
    print(i)

frames = np.array(frames)
print(len(frames))

frame_shape = frames[0].shape

# initialize video writer
# format = 'XVID'
format = 'MP4V'

fourcc = cv2.VideoWriter_fourcc(*format)
fps = cap.get(cv2.CAP_PROP_FPS)

main_folder = os.path.split(path_to_video)[0]
video_in_filename = os.path.split(path_to_video)[1]
condition = args.condition

if format == 'MP4V':
    # os.path.join(main_folder, r'detection_output.avi')
    video_out_filename = os.path.splitext(
        path_to_video)[0] + '_' + str(condition) + '_detection_output.mp4'
else:
    # os.path.join(main_folder, r'detection_output.avi')
    video_out_filename = os.path.splitext(
        path_to_video)[0] + '_' + str(condition) + '_detection_output.avi'

if args.mode == 2:
    video_out_filename = os.path.splitext(path_to_video)[0] +'temp.mp4'

width = frame_shape[1] - scaling_factor * (args.crop_xl + args.crop_xr)
height = frame_shape[0]
out = cv2.VideoWriter(video_out_filename, fourcc, fps, (width, height))

# read in seg masks
path_mask_folder = args.input_mask_path

mask_files = glob.glob(os.path.join(path_mask_folder, r'*'))
mask_files = sorted(mask_files)
mask_files = sorted(mask_files, key=len)
mask_frames = []
frames_with_boxes = []
rects = []

print(len(mask_files))
print(len(frames))

for f in mask_files:
    if len(mask_frames) <= len(frames):
        #print(f)
        mask_frame = cv2.imread(f)

        # convert non-black to white (PixelAnnotationTool does not produce 100% white annotations)
        mask_frame[np.all(mask_frame != [0, 0, 0], axis=2)] = [255, 255, 255]

        mask_frame_gray = np.average(mask_frame, 2)
        mask_frame_gray = cv2.resize(
            mask_frame_gray, None, fx=scaling_factor, fy=scaling_factor)

        try:
            ytl = np.min(np.argwhere(mask_frame_gray)[:, 0])
            ybr = np.max(np.argwhere(mask_frame_gray)[:, 0])

            xtl = np.min(np.argwhere(mask_frame_gray)[:, 1])
            xbr = np.max(np.argwhere(mask_frame_gray)[:, 1])

        except:
            ytl = np.nan
            ybr = np.nan

            xtl = np.nan
            xbr = np.nan

        rect = np.array([xtl, ytl, xbr, ybr])
        rects.append(rect)

        mask_frames.append(mask_frame)

rects = np.array(rects)

# smooth bounding boxes
rects_smooth = rects.copy()

smooth_loops = args.smoothness

for i in range(len(rects[0, :])):
    rects_smooth[:, i] = smooth_signal(rects[:, i], smooth_loops)

# expand bounding boxes, draw rectangles and crop output frames
expand_by = scaling_factor * args.expand

crop_xl = args.crop_xl
crop_xr = args.crop_xr

crop_xl = scaling_factor * crop_xl
crop_xr = scaling_factor * crop_xr

alpha = 0.1

condition = args.condition
if args.mode == 1:
    condition = 1 # When taking screenshots of the different colours, we only want to use condition 1

for i in range(len(frames)):
#for i in range(248):
#max i 254 for easy video
#max i 248 for difficult video
    show_overlay = True

    frame_with_box = frames[i].copy()
    frame_overlay = frames[i].copy()

    if show_overlay:
        # difficult condition has a strange grey/black box -- make it completely black
        cv2.rectangle(frame_with_box, (0,0), (668, 668), (0,0,0), -1)

        # for consistency, cover the icon in bottom right corner with a black triangle
        br_1 = (1720, 1080)
        br_2 = (1913, 1080)
        br_3 = (1913, 864)
        triangle_cnt = np.array( [br_1, br_2, br_3] )
        cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,0), -1, cv2.LINE_AA)

        try:
            xtl = int(rects_smooth[i, 0]) - expand_by
            ytl = int(rects_smooth[i, 1]) - expand_by
            xbr = int(rects_smooth[i, 2]) + expand_by
            ybr = int(rects_smooth[i, 3]) + expand_by

            if condition == 1:
                # Wide bounding circle
                center = (int((xtl + xbr) / 2.), int((ytl + ybr) / 2.))
                radius = abs(int(xbr - xtl))

                if (radius > 550) :
                    radius = 550

                cv2.circle(frame_with_box, center, radius=radius, color=(global_colour), thickness=global_thickness, lineType=cv2.LINE_AA)

                # if circle extends beyond edge, draw black triangles + rectangle
                tl_1 = (670, 0)
                tl_2 = (853, 0)
                tl_3 = (670, 215)
                triangle_cnt = np.array( [tl_1, tl_2, tl_3] )
                cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,0), -1, cv2.LINE_AA)

                tr_1 = (1716, 0)
                tr_2 = (1912, 0)
                tr_3 = (1912, 214)
                triangle_cnt = np.array( [tr_1, tr_2, tr_3] )
                cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,0), -1, cv2.LINE_AA)

                bl_1 = (670, 1080)
                bl_2 = (854, 1080)
                bl_3 = (670, 864)
                triangle_cnt = np.array( [bl_1, bl_2, bl_3] )
                cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,0), -1, cv2.LINE_AA)

                br_1 = (1720, 1080)
                br_2 = (1913, 1080)
                br_3 = (1913, 864)
                triangle_cnt = np.array( [br_1, br_2, br_3] )
                cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,0), -1, cv2.LINE_AA)

                total_height, total_width, channels = frame_with_box.shape
                cv2.rectangle(frame_with_box, (1913, 0), (total_width, total_height), (0, 0, 0), thickness=-1)

                if args.mode == 1 and i == 202:
                    blue = global_colour
                    frame_color = frame_with_box.copy()
                    cv2.circle(frame_color, center, radius=radius, color=(blue), thickness=global_thickness, lineType=cv2.LINE_AA)
                    frame_color = cv2.resize(frame_color, (854, 480))
                    write_name = 'screen_blue'+str(i)+'.png'
                    cv2.imwrite(write_name, frame_color)

                    red = (11, 11, 255) # red
                    frame_color = frame_with_box.copy()
                    cv2.circle(frame_color, center, radius=radius, color=(red), thickness=global_thickness, lineType=cv2.LINE_AA)
                    frame_color = cv2.resize(frame_color, (854, 480))
                    write_name = 'screen_red'+str(i)+'.png'
                    cv2.imwrite(write_name, frame_color)

                    yellow = (0, 242, 252) # yellow
                    frame_color = frame_with_box.copy()
                    cv2.circle(frame_color, center, radius=radius, color=(yellow), thickness=global_thickness, lineType=cv2.LINE_AA)
                    frame_color = cv2.resize(frame_color, (854, 480))
                    write_name = 'screen_yellow'+str(i)+'.png'
                    cv2.imwrite(write_name, frame_color)

                    green = (0, 224, 0) # green
                    frame_color = frame_with_box.copy()
                    cv2.circle(frame_color, center, radius=radius, color=(green), thickness=global_thickness, lineType=cv2.LINE_AA)
                    frame_color = cv2.resize(frame_color, (854, 480))
                    write_name = 'screen_green'+str(i)+'.png'
                    cv2.imwrite(write_name, frame_color)

                    orange = (0, 145, 255) # orange
                    frame_color = frame_with_box.copy()
                    cv2.circle(frame_color, center, radius=radius, color=(orange), thickness=global_thickness, lineType=cv2.LINE_AA)
                    frame_color = cv2.resize(frame_color, (854, 480))
                    write_name = 'screen_orange'+str(i)+'.png'
                    cv2.imwrite(write_name, frame_color)

                    purple = (198, 67, 198) # purple
                    frame_color = frame_with_box.copy()
                    cv2.circle(frame_color, center, radius=radius, color=(purple), thickness=global_thickness, lineType=cv2.LINE_AA)
                    frame_color = cv2.resize(frame_color, (854, 480))
                    write_name = 'screen_purple'+str(i)+'.png'
                    cv2.imwrite(write_name, frame_color)

                    black = (0, 0, 0) # black
                    frame_color = frame_with_box.copy()
                    cv2.circle(frame_color, center, radius=radius, color=(black), thickness=global_thickness, lineType=cv2.LINE_AA)
                    frame_color = cv2.resize(frame_color, (854, 480))
                    write_name = 'screen_black'+str(i)+'.png'
                    cv2.imwrite(write_name, frame_color)

            elif condition == 2:
                # Tight bounding box
                cv2.rectangle(frame_with_box, (xtl, ytl), (xbr, ybr), global_colour, thickness=global_thickness, lineType=cv2.LINE_AA)

                # if rectangle extends beyond edge, draw black triangles + rectangle
                tl_1 = (670, 0)
                tl_2 = (853, 0)
                tl_3 = (670, 215)
                triangle_cnt = np.array( [tl_1, tl_2, tl_3] )
                cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,0), -1, cv2.LINE_AA)

                tr_1 = (1716, 0)
                tr_2 = (1912, 0)
                tr_3 = (1912, 214)
                triangle_cnt = np.array( [tr_1, tr_2, tr_3] )
                cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,0), -1, cv2.LINE_AA)

                bl_1 = (670, 1080)
                bl_2 = (854, 1080)
                bl_3 = (670, 864)
                triangle_cnt = np.array( [bl_1, bl_2, bl_3] )
                cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,0), -1, cv2.LINE_AA)
                
                br_1 = (1720, 1080)
                br_2 = (1913, 1080)
                br_3 = (1913, 864)
                triangle_cnt = np.array( [br_1, br_2, br_3] )
                cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,0), -1, cv2.LINE_AA)

                total_height, total_width, channels = frame_with_box.shape
                cv2.rectangle(frame_with_box, (1913, 0), (total_width, total_height), (0, 0, 0), thickness=-1)

            elif condition == 3:
                # Spotlight
                overlay = frame_with_box.copy()
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

                total_height, total_width, channels = frame_with_box.shape
                rect = cv2.rectangle(overlay, (540, 0), (total_width, total_height), (0, 0, 0, 130), thickness=-1)

                center = (int((xtl + xbr) / 2.), int((ytl + ybr) / 2.))
                radius = abs(int(xbr - xtl))

                if (radius > 550) :
                    radius = 550

                overlay = cv2.circle(overlay, center, radius=radius, color=(255, 255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

                # replace white pixels with completely transparent
                overlay[np.all(overlay == [255, 255, 255, 255], axis=2)] = [255, 255, 255, 0]

                # overlay the two images
                frame_with_box = transparentOverlay(frame_with_box, overlay, (0,0), 1)         

                # draw highlight circle
                cv2.circle(frame_with_box, center, radius=radius, color=global_colour, thickness=global_thickness, lineType=cv2.LINE_AA)

                 # if circle extends beyond edge, draw black triangles + rectangle
                tl_1 = (670, 0)
                tl_2 = (853, 0)
                tl_3 = (670, 215)
                triangle_cnt = np.array( [tl_1, tl_2, tl_3] )
                cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,0), -1, cv2.LINE_AA)

                tr_1 = (1716, 0)
                tr_2 = (1912, 0)
                tr_3 = (1912, 214)
                triangle_cnt = np.array( [tr_1, tr_2, tr_3] )
                cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,0), -1, cv2.LINE_AA)

                bl_1 = (670, 1080)
                bl_2 = (854, 1080)
                bl_3 = (670, 864)
                triangle_cnt = np.array( [bl_1, bl_2, bl_3] )
                cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,0), -1, cv2.LINE_AA)

                br_1 = (1720, 1080)
                br_2 = (1913, 1080)
                br_3 = (1913, 864)
                triangle_cnt = np.array( [br_1, br_2, br_3] )
                cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,0), -1, cv2.LINE_AA)

                total_height, total_width, channels = frame_with_box.shape
                cv2.rectangle(frame_with_box, (1913, 0), (total_width, total_height), (0, 0, 0), thickness=-1)

            elif condition == 4:
                # Segmentation
                frame_with_box = blend_normal_mask(frame_with_box, mask_frames[i], global_colour, 0.3)

            elif condition == 5:
                # Segmentation outline
                contours = blend_normal_mask_border(frame_with_box, mask_frames[i])
                cv2.drawContours(frame_with_box, contours, 0, global_colour, global_thickness, cv2.LINE_AA)

            elif condition == 6:
                # Detection confidence
                total_height, total_width, channels = frame_with_box.shape
                polyp_width = xbr - xtl

                confidence = round(polyp_width / (total_width / 2) * 100)
                confidence = round(polyp_width / (total_width / 6) * 100) # for easy video
                if confidence > 99:
                    confidence = 96

                #fontface = cv2.FONT_HERSHEY_SIMPLEX
                fontface = cv2.FONT_HERSHEY_DUPLEX
                fontscale = 2.25
                fontcolor = (255, 255, 255)
                #cv2.putText(frame_with_box, str(confidence) + "%", (246, 38), fontface, fontscale, fontcolor, 1, cv2.LINE_AA)
                #cv2.putText(frame_with_box, str(confidence) + "%", (660, 96), fontface, fontscale, fontcolor, 1, cv2.LINE_AA)
                cv2.putText(frame_with_box, str(confidence) + "%", (660, 60), fontface, fontscale, fontcolor, 2, cv2.LINE_AA)

            elif condition == 7:
                # Detection signal
                total_height, total_width, channels = frame_with_box.shape

                tl_1 = (670, 0)
                tl_2 = (853, 0)
                tl_3 = (670, 215)
                triangle_cnt = np.array( [tl_1, tl_2, tl_3] )
                cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,255), -1, cv2.LINE_AA)

                tr_1 = (1716, 0)
                tr_2 = (1912, 0)
                tr_3 = (1912, 214)
                triangle_cnt = np.array( [tr_1, tr_2, tr_3] )
                cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,255), -1, cv2.LINE_AA)

                bl_1 = (670, 1080)
                bl_2 = (854, 1080)
                bl_3 = (670, 864)
                triangle_cnt = np.array( [bl_1, bl_2, bl_3] )
                cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,255), -1, cv2.LINE_AA)
                
                br_1 = (1720, 1080)
                br_2 = (1913, 1080)
                br_3 = (1913, 864)
                triangle_cnt = np.array( [br_1, br_2, br_3] )
                cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,255), -1, cv2.LINE_AA)

            elif condition == 10:
                # Nothing
                frame_with_box = frame_with_box

            #if args.mode == 2 and i == 202:
            if args.mode == 2 and i == 40:
                #frame_cond = cv2.resize(frame_with_box, (854, 480))
                frame_cond = frame_with_box.copy()
                write_name = 'screen_'+str(condition)+'_'+str(i)+'.png'
                cv2.imwrite(write_name, frame_cond)
                sys.exit()

        except ValueError:
            tl_1 = (670, 0)
            tl_2 = (853, 0)
            tl_3 = (670, 215)
            triangle_cnt = np.array( [tl_1, tl_2, tl_3] )
            cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,0), -1, cv2.LINE_AA)

            tr_1 = (1716, 0)
            tr_2 = (1912, 0)
            tr_3 = (1912, 214)
            triangle_cnt = np.array( [tr_1, tr_2, tr_3] )
            cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,0), -1, cv2.LINE_AA)

            bl_1 = (670, 1080)
            bl_2 = (854, 1080)
            bl_3 = (670, 864)
            triangle_cnt = np.array( [bl_1, bl_2, bl_3] )
            cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,0), -1, cv2.LINE_AA)
            
            br_1 = (1720, 1080)
            br_2 = (1913, 1080)
            br_3 = (1913, 864)
            triangle_cnt = np.array( [br_1, br_2, br_3] )
            cv2.drawContours(frame_with_box, [triangle_cnt], 0, (0,0,0), -1, cv2.LINE_AA)
            
            if condition == 6:
                fontface = cv2.FONT_HERSHEY_DUPLEX
                fontscale = 2.25
                fontcolor = (255, 255, 255)
                cv2.putText(frame_with_box, "0%", (660, 60), fontface, fontscale, fontcolor, 2, cv2.LINE_AA)

            print("No annotation for this frame.")

    if args.flip_hori:
        out.write(np.fliplr(frame_with_box[:, crop_xl:width-crop_xr, :]))
    else:
        out.write(frame_with_box[:, crop_xl:width-crop_xr, :])
    print()


out.release()