# camera
CAMERA_DEVICE = "/dev/video0"
CAMERA_RES_W = 1280
CAMERA_RES_H = 720
CAMERA_FPS = 30
CAMERA_TARGET_BRIGHTNESS_DEF = 120.0 # mean value target brightness of exposure
CAMREA_FIXED_EXPOSURE_DEF = 150 #  exposure time in ms
CAMERA_WB_TEMP_DEF = 4500 # white balance in kelvin
CAMERA_IMG_SEQU_AMT_DEF = 3 # amount of images in sequence
CAMERA_IMG_SEQU_STEP_DEF = 1 # skipped images in sequence at 30 FPS

# analyzer
# numerical analysis
HISTOGRAM_OVERALL_BINS_DEF = 16 # amount of histogram bins in overall analysis
HISTOGRAM_FIELDS_BINS_DEF = 8 # amount of histogram bins in spatial analysos
FIELDS_ROWS_DEF = 4 # rows of spatial analysis
FIELDS_COLS_DEF = 4 # cols of spatial analysis
HISTORGAM_ROUND_DECIMALS = 4 # rounding floats in histograms after decimal point
# motion analysis
MOVEMENT_MASK_THRESH_DEF = 25 # delta thresh between pixels in frame differencing
MOTION_MASK_MIN_AREA_DEF = 500 # the minimum area of motion area that will be seen as movement
MOTION_MASK_MERGE_DISTANCE_DEF = 20 # dilate; gaps in px that will be fused with other areas of mask
MIN_PX_MOTION_THRESH_DEF = 20 # minimum of counted pixels in movement area for o-flow calc