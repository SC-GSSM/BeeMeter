from default_boxes.default_boxes_other import SSDInputEncoder
from default_boxes.default_boxes import DefaultBoxHandler
import numpy as np

feature_map_sizes = [(3, 2)]  # [(35, 30), (25, 20), (10, 10)]
img_height = 400
img_width = 300
min_scale = 0.1
max_scale = 0.9
aspect_ratios_global = [0.5, 1.0, 2.0]
use_bonus_square_box = False
variances = [1.0, 1.0, 1.0, 1.0]

handler = SSDInputEncoder(img_height=img_height,
                          img_width=img_width,
                          n_classes=2,
                          predictor_sizes=feature_map_sizes,
                          min_scale=min_scale,
                          max_scale=max_scale,
                          aspect_ratios_global=aspect_ratios_global,
                          two_boxes_for_ar1=use_bonus_square_box,
                          clip_boxes=True,
                          variances=variances,
                          coords='corners',
                          normalize_coords=True)

my_handler = DefaultBoxHandler(image_size=(img_width, img_height), feature_map_sizes=feature_map_sizes,
                               scale_min=min_scale,
                               scale_max=max_scale,
                               aspect_ratios_global=aspect_ratios_global, use_bonus_square_box=use_bonus_square_box,
                               standardizing=None)

np.set_printoptions(suppress=True, threshold=np.inf, linewidth=300, precision=3)
gt = [np.array([[0, 0, 70, 80], [50, 60, 145, 130]])]
gt_1 = [np.array([[1, 0, 0, 70, 80], [1, 50, 60, 145, 130]])]
my_encoded = my_handler.encode_default_boxes(gt)
encoded = handler(gt_1)
print(my_encoded)
print(encoded)

