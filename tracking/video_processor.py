import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from detection.model.registry import load_model
from tracking.counter import PathHandler, Logger
from detection.utils.box_tools import draw_box
import time


class VideoGenerator:
    """
    Class to iterate over the images of a video.
    """
    def __init__(self, path, start=0, end=None, fps=24, batch=1, debug=False):
        """
        :param path: Path to the video file.
        :param start: Frame to start.
        :param end:  Frame to end.
        :param fps: The fps of the video.
        :param batch: Images can be returned as a batch of given size.
        :param debug: if True, images are display.
        """
        self.path = path
        self.fps = fps
        self.debug = debug
        self.batch = batch

        self.frame_count = 0
        self.start_frame = start
        self.end_frame = end

        self.cap = None

        self.init_video()

    def init_video(self):
        self.frame_count = 0

        self.cap = cv2.VideoCapture(self.path)

        if not self.cap.isOpened:
            raise ValueError('Unable to open video file: {0}'.format(self.path))

        # iterate to start frame
        while self.frame_count != self.start_frame:
            self._next_frame()

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def __iter__(self):
        return self

    def _next_frame(self):
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.frame_count += 1
        else:
            print("Warning damaged frame or end of video detected. ret: {0}, frame: {1}".format(ret, frame))
            self.frame_count += 1

        return ret, frame

    def __next__(self):
        result = []
        end = False
        for _ in range(self.batch):
            ret, frame = self._next_frame()

            if self.end_frame is not None:
                if self.frame_count > self.end_frame:
                    end = True
            if frame is None or not ret or end:
                if result:
                    return result
                raise StopIteration()

            result.append(frame)

            if self.frame_count % 1000 == 0:
                print("Proceed {0} frames.".format(self.frame_count))

            if self.debug:
                cv2.imshow('Image', frame)
                key = cv2.waitKey(0)
                if key == 27:  # Escape Key
                    cv2.destroyAllWindows()
                    raise StopIteration()

        return result

    def restart(self):
        self.cap.release()
        self.init_video()


class VideoProcessor:
    def __init__(self, configuration, weights, video, aoi, crop,
                 fps=24, show=False, logger_kwargs=None, handler_kwargs=None, viz_kwargs=None):
        """
        Class for analysing video of bees flying in and out. It is mainly used for visualisation.
        :param configuration: Path to the detection models configuration file.
        :param weights: Path to the detection models weights file.
        :param video: Path to the video file.
        :param aoi: Area of interest.
        :param crop: Crop area of the image.
        :param fps: Frames per Second.
        :param show: Show the video frame by frame or just count.
        :param logger_kwargs: Arguments of the logger: See Logger class
        :param handler_kwargs: Arguments of the Path handler: See PathHandler class
        :param viz_kwargs: Arguments for the visualization: See function visualize of this class.
        """
        self.fps = fps
        self.show = show
        self.crop = crop
        self.logger_kwargs = dict(path='log.csv', frequency=1, absolute=False, start_time=None, delta_t=1)
        self.viz_kwargs = dict(scores=False, true_path=True, ids=False, aoi=True, bee='center', count=True,
                               frame_count=True)

        if not logger_kwargs:
            logger = None
        else:
            if logger_kwargs is not None:
                self.logger_kwargs.update(logger_kwargs)
            logger = Logger(**self.logger_kwargs)

        self.handler_kwargs = dict(aoi=aoi, max_frames=3, max_dist=200, measurement_variance=0.05 ** 2,
                                   state_variance=1000, filter_type='acceleration', path_max_len=10)
        if handler_kwargs is not None:
            self.handler_kwargs.update(handler_kwargs)
        if viz_kwargs is not None:
            self.viz_kwargs.update(viz_kwargs)

        self.handler = PathHandler(logger=logger, dt=1.0 / self.fps, **self.handler_kwargs)

        self.model = load_model(config_path=configuration, weights_path=weights)

        self.video = video
        self.font = ImageFont.truetype("/home/t9s9/PycharmProjects/BeeMeter/tracking/RobotoMono-Medium.ttf", 30)
        self.font_small = ImageFont.truetype("/home/t9s9/PycharmProjects/BeeMeter/tracking/RobotoMono-Medium.ttf", 10)

        self.prev_frame = None

    def visualize(self, frame, prediction, bee='Box', scores=False, aoi=True, count=True, true_path=True,
                  pred_path=False, frame_count=True, frame_nr=None, velocity=False, ids=False):
        """
        Visualize different parts of the flight traffic analysis.
        :param frame: The frame to visualize.
        :param prediction: The predictions of the objects.
        :param bee: How to visualize the objects: one of ['box', 'center']
        :param scores: Display the prediction confidence score to each object.
        :param aoi: Display the area of interest = count zone
        :param count: display the counter for in and out
        :param true_path: display the true flight path
        :param pred_path: display the predicted flight path (the next step of the Kalman filter)
        :param frame_count: Display the frame count.
        :param frame_nr: The frame number is only important when the video does not start at frame 0
        :param velocity: Display the velocity of the objects
        :param ids: Display the ID of the objects
        :return: Frame with visualizations.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if bee.lower() in ['box']:
            score = prediction[..., 0] if scores else None
            frame = draw_box(frame, prediction[..., 1:], scores=score, color=(0, 0, 255), thickness=1, cvt_color=False)
        elif bee.lower() in ['center', 'centroid']:
            for path in self.handler.path_list:
                if path.skipped_frames == 0:
                    frame = cv2.circle(frame, tuple(path.path[-1]), radius=3, color=(0, 0, 255),
                                       thickness=cv2.FILLED)

        if aoi:
            aoi = cv2.rectangle(np.zeros(frame.shape, np.uint8), tuple(self.handler.aoi[:2]),
                                tuple(self.handler.aoi[2:]),
                                color=(255, 255, 0), thickness=cv2.FILLED)
            frame = cv2.addWeighted(frame, 1.0, aoi, 0.2, 1)

        # Pillow
        if count or true_path or frame_count or velocity or ids or pred_path:
            pil_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_img)
            if count or frame_count:
                if count:
                    text = " IN: {0:<3}\nOUT: {1:<3}".format(self.handler.in_count, self.handler.out_count)
                    draw.text((0, 0), text, (255, 255, 255), font=self.font)
                if frame_count:
                    text = "{0}".format(self.video.frame_count - (self.video.batch - frame_nr))
                    draw.text((frame.shape[1] - 60, 0), text, (255, 255, 255), font=self.font)
            if true_path or velocity or ids or pred_path:
                for path in self.handler.path_list:
                    if path.skipped_frames == 0:
                        if true_path:
                            draw.line(xy=np.array(path.path).flatten().tolist(), width=3, fill=(24, 176, 0))
                        if pred_path:
                            draw.line(xy=np.array(path.predicted_path).flatten().tolist(), width=3, fill=(42, 31, 216))
                        if velocity:
                            draw.text(xy=path.path[-1], text=str('{0:.2f}\n{1:.2f}'.format(*path.filter.x[2:4])),
                                      color=(255, 255, 255), font=self.font_small)
                        if ids:
                            draw.text(xy=path.path[-1], text=str(path.path_id),
                                      color=(255, 255, 255), font=self.font_small)
            frame = np.array(pil_img)

        if self.prev_frame is None:
            self.prev_frame = np.zeros_like(frame)

        # cv2.imwrite("../resources/gif_images/{0}.png".format(self.video.frame_count - (self.video.batch - frame_nr)), frame)
        # cframe = np.vstack([self.prev_frame, frame])
        cv2.imshow('Images', frame)
        state = True
        while True:
            key = cv2.waitKey(0)
            if key == 32:  # space
                break
            elif key == 83 or key == ord("d"):  # right arrow
                if state:
                    break
                else:
                    cv2.imshow('Images', frame)
                    state = True
            elif key == 81 or key == ord("a"):  # left arrow
                cv2.imshow('Images', self.prev_frame)
                state = False
            elif key == 27:  # esc
                quit()
        self.prev_frame = frame

    def start_timing(self):
        batch_count = 0
        times = []
        start = time.time()
        for batch_frame in self.video:
            t1 = time.time()
            for i in range(len(batch_frame)):
                x1, y1, x2, y2 = self.crop
                batch_frame[i] = batch_frame[i][y1:y2, x1:x2]
                batch_frame[i] = cv2.cvtColor(batch_frame[i], cv2.COLOR_BGR2RGB)
            t2 = time.time()
            batch_prediction = self.model.predict(np.array(batch_frame), batch_size=len(batch_frame),
                                                  conf_threshold=0.5,
                                                  nms_threshold=0.3,
                                                  sigma=0.5, nms_methode='normal', forward_methode=1,
                                                  faster_decode=True)
            t3 = time.time()
            for prediction in batch_prediction:
                if len(prediction) > 0:
                    self.handler.update(np.array(prediction)[:, 1:])
            t4 = time.time()
            times.append([t2 - t1, t3 - t2, t4 - t3, t4 - t1])
            batch_count += 1

        times = np.array(times)
        print("Preprocessing: {0:.4f}s Prediction: {1:.4f}s Tracking: {2:.4f}s Total: {3:.4f}s".format(
            *np.mean(times, axis=0)))
        print("Number batches:", batch_count)
        print("Total time: {0:.4f}s".format(time.time() - start))
        print("IN: {0} OUT: {1}".format(self.handler.in_count, self.handler.out_count))

    def start(self):
        for batch_frame in self.video:
            for i in range(len(batch_frame)):
                x1, y1, x2, y2 = self.crop
                batch_frame[i] = batch_frame[i][y1:y2, x1:x2]
                batch_frame[i] = cv2.cvtColor(batch_frame[i], cv2.COLOR_BGR2RGB)
            batch_prediction = self.model.predict(np.array(batch_frame), batch_size=len(batch_frame),
                                                  conf_threshold=0.9, nms_threshold=0.25,
                                                  sigma=0.5, nms_methode='linear', forward_methode=0,
                                                  faster_decode=True)

            # for each frame in the predicted batch update the path handler and eventually visualize or store the frame
            # or update the logger. For the case that there is no prediction on the frame, update the path handler
            # anyways
            for i, prediction in enumerate(batch_prediction):
                # update paths, first slice out the scores
                self.handler.update(np.array(prediction)[:, 1:])
                if self.show:
                    self.visualize(batch_frame[i], prediction, frame_nr=i, **self.viz_kwargs)


if __name__ == '__main__':
    base = "/media/t/Bachelor/"
    # file = "/media/t/Bachelor/large_vid_18-09-15-19-47.h264"  # [0, 420, 1280, 470] # fps 24  # batch_frame[i] = batch_frame[i][250:720, 0:1280]
    # # file = base + "videos/vid_26-06-07-50-39-tt.h264"
    # aoi = [0, 420, 1280, 470]
    # crop = [0, 250, 1280, 720]
    # 53 43
    # file = base + 'videos/23-02/vid-incan-sport-long.h264'  # [0, 455, 1280, 515] # fps 60  # # batch_frame[i] = batch_frame[i][135:645, 0:1260]
    # file = base + "videos/25-02/25-2-incan-60.h264"

    # file = base + 'videos/24-03/vid_1250-1450.h264'  # aoi=[0, 470, 1280, 530]  # batch_frame[i] = batch_frame[i][125:655, :]

    test_video_1 = base + "videos/25-03/test_video_1.h264"
    aoi = [0, 440, 1280, 500]  # , batch_frame[i][200:680, 40:1230]
    crop = [40, 200, 1230, 680]

    # file = base + "videos/25-03/vid-1053.h264"
    # file2 = base + "videos/25-03/vid-1529.h264"

    # file = base + "videos/29-03/vid818.h264"
    # file2 = base + "videos/29-03/vid1321.h264"
    # file3 = base + "videos/29-03/vid1923.h264"
    aoi = [0, 410, 1280, 480]  # + [0, 430, 1280, 480]
    crop = [45, 220, 1235, 700]

    test_video_3 = "/media/t/Bachelor/videos/test_videos/test_video_3.avi"

    config = "/media/t/Bachelor_final_training/MobileNetV2_B_10_expand/model_config.conf"
    weights = "/media/t/Bachelor_final_training/MobileNetV2_B_10_expand/checkpoints/MobileNetV2_B_10_expand-58_loss-2.0722_val_loss-2.4338.h5"

    # t1 = time.time()
    # vid = VideoGenerator(path=file, debug=False, start=100, end=2200, batch=8)
    #
    # processor = VideoProcessor(configuration=config, weights=weights, video=vid, show=0, aoi=aoi,
    #                            fps=24, crop=crop,
    #                            handler_kwargs=dict(max_dist=100, state_variance=1000, measurement_variance=0.01,
    #                                                filter_type='acceleration', max_frames=2, path_max_len=100,
    #                                                count_cool_down=0),
    #                            logger_kwargs=dict(frequency=20, path='vid818_new_aoi.csv'))
    # processor.start()
    # print(processor.handler.in_count, processor.handler.out_count)
    # print("Time: {:.3f}s".format(time.time() - t1))

    t1 = time.time()
    # vid30 = base + "videos/30-03/vid3003-0820.h264"
    # aoi = [0, 420, 1280, 480]
    # crop = [60, 220, 1220, 700]

    vid = VideoGenerator(path=test_video_3, debug=False, start=100, end=1900, batch=8)

    processor = VideoProcessor(configuration=config, weights=weights, video=vid, show=False, aoi=aoi,
                               fps=60, crop=crop,
                               handler_kwargs=dict(max_dist=100, state_variance=100, measurement_variance=0.01,
                                                   filter_type='acceleration', max_frames=2, path_max_len=100,
                                                   count_cool_down=10),
                               logger_kwargs=dict(frequency=1, path='log_vid3003_09.csv'),
                               viz_kwargs=dict(scores=False, true_path=True, ids=False, aoi=True, bee='center',
                                               count=True, frame_count=True, pred_path=False))
    processor.start()
    print(processor.handler.in_count, processor.handler.out_count)
    print("Time: {:.3f}s".format(time.time() - t1))

    # t1 = time.time()
    # vid = VideoGenerator(path=file, debug=False, start=300, end=None, batch=8)
    #
    # processor = VideoProcessor(configuration=config, weights=weights, video=vid, show=False, aoi=aoi,
    #                            fps=60, crop=crop,
    #                            handler_kwargs=dict(max_dist=120, state_variance=1000, measurement_variance=0.001,
    #                                                filter_type='acceleration', max_frames=3, path_max_len=120,
    #                                                count_cool_down=20),
    #                            logger_kwargs=dict(frequency=1, path='vid818_experimantal.csv'))
    # processor.start()
    # print(processor.handler.in_count, processor.handler.out_count)
    # print("Time: {:.3f}s".format(time.time() - t1))
