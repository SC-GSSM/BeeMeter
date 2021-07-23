from time import time
from tracking import VideoProcessor, VideoGenerator

if __name__ == '__main__':
    video_path = "resources/demo/demo_vid.h264"
    detection_weight_path = "resources/demo/keras_model.h5"
    detection_config_path = "resources/demo/model_config.conf"

    log_file_path = "resources/demo/log_file.csv"
    aoi = [0, 440, 1280, 500]
    crop = [40, 200, 1230, 680]

    vid = VideoGenerator(path=video_path, debug=False, start=0, end=None, batch=4)
    start_time = time()

    processor = VideoProcessor(configuration=detection_config_path,
                               weights=detection_weight_path,
                               video=vid,
                               show=True,
                               aoi=aoi,
                               fps=60,
                               crop=crop,
                               handler_kwargs=dict(max_dist=100,
                                                   state_variance=100,
                                                   measurement_variance=0.01,
                                                   filter_type='acceleration',
                                                   max_frames=2,
                                                   path_max_len=100,
                                                   count_cool_down=10),
                               logger_kwargs=dict(frequency=10,
                                                  path=log_file_path),
                               viz_kwargs=dict(scores=False,
                                               true_path=True,
                                               ids=False,
                                               aoi=True,
                                               bee='box',
                                               count=True,
                                               frame_count=True,
                                               pred_path=False))
    processor.start()

    print(f"Count results: IN={processor.handler.in_count}, OUT={processor.handler.out_count}")
    print("Time: {:.3f}s".format(time() - start_time))
