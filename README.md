# Designing Artificial Intelligence Support Systems for Continuous Adaptation: A Colonoscopy Case Study
Auxilary files for the ACM Health submission 'Designing Artificial Intelligence Support Systems for Continuous Adaptation: A Colonoscopy Case Study'. The following files are included;

* Visual marker videos
* Source code visual marker videos
* Survey data and analysis

## Visual marker videos

The videos as shown to participants, contains all fourteen videos; seven unique visual markers across two patient colon videos (one with an apparent polyp, one with a challenging polyp).

## Source code visual marker videos

Original patient video (without visual markers), annotated image files (frame-by-frame), and included source code required to generate visual marker videos. Source code in Python, instructions to run;

```
1. Install python3 from https://www.python.org/downloads/.
2. Install the packages listed in generate_overlays.py using pip install. For example, 'pip install opencv-python', 'pip install numpy'.
3. Change working directory to the folder containing the generate_overlays.py file. This folder should contain both the video file and a subfolder with the image mask files.
4. Execute command 'python3 generate_overlays.py --input_video_path "apparent.mov" --input_mask_path "apparent-mask" --condition 1' to generate your first video marker. 
5. The program will run and generate a video file with the specified video condition using the specified video file and annotation files.
   See examples below for additional arguments (e.g., video condition, coloured stills).

Additional example commands:

Generate a Segmentation Outline marker on the challenging video file:
python3 generate_overlays.py --input_video_path "challenging.mov" --input_mask_path "challenging-mask" --condition 5

Generate screenshots showing different colour options:
python3 generate_overlays.py --input_video_path "apparent.mov" --input_mask_path "apparent-mask" --mode 1

Bonus. Use ffmpeg to convert video files to expected size and video format.
ffmpeg -i apparent_1_detection_output.mp4 -vf scale=640:480 apparent_1_480.mp4
```

## Survey data and analysis

Anonymised survey responses and analysis files (R).