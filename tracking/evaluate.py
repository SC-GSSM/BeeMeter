import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
from collections import defaultdict
import time
from tracking.video_processor import VideoGenerator, VideoProcessor
from sklearn.model_selection import ParameterGrid


class ParameterSearch:
    def __init__(self):
        pass

    @staticmethod
    def f1(pred_in, pred_out, true_in, true_out):
        error_in = abs(1 - pred_in / true_in)
        error_out = abs(1 - pred_out / true_out)
        if (error_in + error_out) == 0:
            _f1 = 0
        else:
            _f1 = 2 * (error_in * error_out) / (error_in + error_out)
        return _f1, error_in, error_out

    def shadowVideo(self):
        file = "/media/t/Bachelor/large_vid_18-09-15-19-47.h264"
        vid = VideoGenerator(path=file, debug=False, start=100, end=1900, batch=8)
        true_in, true_out = 56, 52
        return vid, true_in, true_out

    def test_video(self, nr=1):
        if nr == 1:
            file = "/media/t/Bachelor/videos/25-03/test_video_1.h264"
            vid = VideoGenerator(path=file, debug=False, start=120, end=None, batch=8)
            true_in, true_out = 123, 92
            aoi = [0, 440, 1280, 500]
            crop = [40, 200, 1230, 680]
        return vid, aoi, crop, true_in, true_out

    def parameter_grid(self, max_dist=200, state_variance=10, measurement_variance=0.001, filter_type='acceleration',
                       max_frames=3, count_cool_down=0):
        params = locals()
        del params['self']

        params = {i: [j] if type(j) != list else j for i, j in params.items()}
        grid = ParameterGrid(params)

        print('Number of iterations:', len(grid))

        video, aoi, crop, true_in, true_out = self.test_video(nr=1)
        config = "/media/t/Bachelor_final_training/MobileNetV2_B_10_expand/model_config.conf"
        weights = "/media/t/Bachelor_final_training/MobileNetV2_B_10_expand/checkpoints/MobileNetV2_B_10_expand-58_loss-2.0722_val_loss-2.4338.h5"

        result = []
        times = []
        current_best = 0
        current_best_params = None
        for i in range(len(grid)):
            t1 = time.time()
            processor = VideoProcessor(configuration=config, weights=weights, video=video, show=False, aoi=aoi,
                                       crop=crop, handler_kwargs=grid[i], logger_kwargs=False)
            processor.start()

            f1, error_in, error_out = self.f1(pred_in=processor.handler.in_count, pred_out=processor.handler.out_count,
                                              true_in=true_in, true_out=true_out)
            times.append(time.time() - t1)
            if f1 < current_best:
                current_best = f1
                current_best_params = grid[i]
            d = grid[i]
            d['Total_In'] = processor.handler.in_count
            d['Total_Out'] = processor.handler.out_count
            d['Error_In'] = round(error_in, 3)
            d['Error_Out'] = round(error_out, 3)
            d['Error_f1'] = round(f1, 3)
            result.append(d)
            print("Run {0:<4} with parameters: {1}".format(i, grid[i]))
            print("{0:<4} | {1:<8} | {2:<8} | {3:<8}".format("", "True", "Estimate", "Error"))
            print("{0:<4} | {1:<8} | {2:<8} | {3:<8.2f}".format("IN", true_in, processor.handler.in_count, error_in))
            print(
                "{0:<4} | {1:<8} | {2:<8} | {3:<8.2f}".format("OUT", true_out, processor.handler.out_count, error_out))
            print("F1: {0:.3f} \n".format(f1))
            video.restart()
            print("{0}/{1} estimated time: {2:.2f} (One: {3:.2f})".format(i + 1, len(grid) + 1,
                                                                  ((len(grid) + 1) - (i + 1)) * np.mean(times),
                                                                  np.mean(times)))
        df = pd.DataFrame(result)
        df.to_csv('results_parameter_search.csv')
        print("Finished with best score: {0:.3f}".format(current_best))
        print("Params: {0}".format(current_best_params))


if __name__ == '__main__':
    p = ParameterSearch()
    p.parameter_grid(state_variance=[10, 1000],
                     measurement_variance=[0.001, 1],
                     filter_type=['acceleration', 'velocity', 'none'],
                     max_dist=[60, 100, 120],
                     max_frames=[2, 4],
                     count_cool_down=[0, 10])
