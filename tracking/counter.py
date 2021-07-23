import numpy as np
from scipy.optimize import linear_sum_assignment
from tracking.kalman_filter import VelocityKalmanFilter, AccelerationKalmanFilter, DummyFilter
import csv
from datetime import datetime, timedelta
from detection.utils.box_tools import box2center
from collections import deque


class _PathDeque(deque):
    def __init__(self, start=None, maxlen=10):
        """
        Data structure to handle path objects. A simple queue is implemented with a max length function.
        :param start: init the queue with a start object that will be append
        :param maxlen: maximum length of the queue
        """
        super().__init__(maxlen=maxlen)
        if start is not None:
            self.append(start)

    def save_append(self, x):
        """
        Append an object to the queue. If the max length is reached, the first elements is removed.
        :param x: object to append.
        """
        if len(self) == self.maxlen:
            self.popleft()
        self.append(x)

    def get_last(self, s):
        """
        Get the last s objects of the queue.
        :param s: number of last elements.
        :return: list if contents.
        """
        return list(self)[slice(-s, None)]


class Path:
    def __init__(self, path_id, start, filter_type='acceleration', dt=0.04, measurement_variance=0.1 ** 2,
                 state_variance=1000, max_len=100):
        """
        Single path of an object to be tracked.
        :param path_id: Unique ID of a path object.
        :param start:
        :param filter_type: Type of the Kalman Filter. One of ['acceleration', 'velocity', 'none]
        :param dt: Time between frames.
        :param measurement_variance: Variance to measure the noise of the Measurement and the motion.
        :param state_variance: To scale the covariance matrix.
        :param max_len: Maximum length of a path. If a new point is added and the max length is reaches, the earliest
         point is removed => Deque
        """
        self.path_id = path_id
        self.prediction = start
        self.path = _PathDeque(start, maxlen=max_len)
        self.predicted_path = _PathDeque(start, maxlen=max_len)
        self.skipped_frames = 0
        self.cool_down = 0

        if filter_type.lower() == 'acceleration':
            self.filter = AccelerationKalmanFilter(x=start, dt=dt, measurement_variance=measurement_variance,
                                                   state_variance=state_variance)
        elif filter_type.lower() == 'velocity':
            self.filter = VelocityKalmanFilter(x=start, dt=dt, measurement_variance=measurement_variance,
                                               state_variance=state_variance)
        elif filter_type.lower() == 'none':
            self.filter = DummyFilter(x=start)
        else:
            raise ValueError("Filter must be either 'acceleration' or 'velocity'.")

        self.filter.predict()

    def predict(self, measurement):
        self.filter.update(np.array(measurement))
        self.filter.predict()
        self.prediction = self.filter.x[:2]

        self.predicted_path.save_append(self.prediction)

        # if its a new measurement add it to the path, else it is an skipped frame
        if self.skipped_frames == 0:
            self.path.save_append(measurement)

        if self.cool_down > 0:
            self.cool_down -= 1

        # print('Measured: {0:.0f}, {1:.0f}    Prediction: {2:.0f}, {3:.0f}'.format(*measurement, *self.prediction))


class PathHandler:
    def __init__(self, aoi, max_dist=100, max_frames=3, dt=0.04, measurement_variance=0.1 ** 2,
                 filter_type='acceleration', state_variance=1000, logger=None, path_max_len=10, count_cool_down=0):
        """
        This class takes over the task of assigning objects to paths. Paths are dynamically created, deleted, updated or
        paused, as required. A Kalman filter is used to predict the next position.
        :param aoi: The area of interest, where an object is counted when leaving or entering. [xmin, ymin, xmax, ymax]
        :param max_dist: The maximum distance between two objects to be included in the path
        :param max_frames: The maximum number of frames in which no object has been added to a path before it is deleted
        :param path_max_len: maximum length of a path before it gets deleted
        :param filter_type: either 'acceleration' or 'velocity'. Defines type if the kalman filter
        :param count_cool_down: if a path is count wait for this len until you count again
        """
        self.path_list = []
        self.id_count = 0
        self.in_count = 0
        self.out_count = 0

        self.aoi = aoi
        self.max_dist = max_dist
        self.max_frames = max_frames
        self.path_max_len = path_max_len
        self.count_cool_down = count_cool_down

        self.logger = logger
        self.frame = 0
        self.filter_type = filter_type
        self.dt = dt
        self.state_var = state_variance
        self.measure_var = measurement_variance

        self.flight_arrival_coords = []
        self.flight_departure_coords = []

    @staticmethod
    def euclidean_distance(array1, array2):
        """
        Returns the Euclidean distance in a 2d space between two points
        :param array1:
        :param array2:
        :return: Euclidean distance
        """
        uno = np.tile(np.expand_dims(array1, axis=1), reps=(1, array2.shape[0], 1))
        dos = np.tile(np.expand_dims(array2, axis=0), reps=(array1.shape[0], 1, 1))
        return np.sqrt(np.sum((uno - dos) ** 2, axis=2))

    def update_counter_nocd(self):
        """
        Monitors whether a path enters or leaves the area of interest and updates the counters
        """
        points = []
        for path in self.path_list:
            # if the path tracked two object in the exact last two frames
            if len(path.path) >= 2 and path.skipped_frames == 0:
                points.append(np.concatenate(path.path.get_last(2)))
        points = np.array(points)

        # if there are any points
        if points.size > 0:
            in_aoi = np.logical_and.reduce((self.aoi[0] < points[..., [0, 2]], points[..., [0, 2]] < self.aoi[2],
                                            self.aoi[1] < points[..., [1, 3]], points[..., [1, 3]] < self.aoi[3]))
            self.out_count += np.sum(np.logical_and(in_aoi[..., 0], np.logical_not(in_aoi[..., 1])))
            self.in_count += np.sum(np.logical_and(np.logical_not(in_aoi[..., 0]), in_aoi[..., 1]))

    def update_counter(self):
        """
        Monitors whether a path enters or leaves the area of interest and updates the counters
        """
        for path in self.path_list:
            # if the path tracked two object in the exact last two frames
            if len(path.path) >= 2 and path.skipped_frames == 0 and path.cool_down == 0:
                p1, p2 = path.path.get_last(2)
                first_point_in_aoi = self.aoi[0] < p1[0] < self.aoi[2] and self.aoi[1] < p1[1] < self.aoi[3]
                second_point_in_aoi = self.aoi[0] < p2[0] < self.aoi[2] and self.aoi[1] < p2[1] < self.aoi[3]
                if first_point_in_aoi and not second_point_in_aoi:
                    self.out_count += 1
                    path.cool_down = self.count_cool_down
                elif second_point_in_aoi and not first_point_in_aoi:
                    self.in_count += 1
                    path.cool_down = self.count_cool_down

    def update(self, objects):
        """
        This function assigns a suitable previous object to each object. For this purpose, the next position of the
        previous object is predicted with the aid of a Kalman filter and then assigned to the nearest object. For this
        assignment the maximum matching with minimum cost is searched.
        :param objects: list of bounding boxes
        :type objects: array of boxes: [(int, int, int, int)]
        """
        self.frame += 1

        # Convert bounding boxes into center points
        centers = box2center(objects)
        len_centers = centers.shape[0]

        # If there are no tracks create one for every prediction
        if not self.path_list:
            for i, center in enumerate(centers):
                self.path_list.append(
                    Path(i, center, dt=self.dt, state_variance=self.state_var, measurement_variance=self.measure_var,
                         filter_type=self.filter_type, max_len=self.path_max_len))
            self.id_count = len_centers
        else:
            # calculate the distance between every path and every new prediction
            predictions = np.array([instance.prediction for instance in self.path_list])
            cost = self.euclidean_distance(predictions, centers)
            # solve the sum assignment problem to match every path to every new prediction by shortest distance
            row, col = linear_sum_assignment(cost_matrix=cost, maximize=False)
            # print(cost[row, col])
            alloc = [-1 for _ in range(len(self.path_list))]

            for i in range(len(row)):
                alloc[row[i]] = col[i]

            for path_index, center_index in enumerate(alloc):
                if center_index == -1:
                    # no new position is assigned to the path at position path_index in path_list,
                    # so increase the skipped frames counter
                    self.path_list[path_index].skipped_frames += 1

                    # if there is no new detection, predict the next step based on the last measurement
                    # TODO: optimize: no prediction if path will be deleted
                    self.path_list[path_index].predict(self.path_list[path_index].predicted_path[-1])
                else:
                    # a new position is found. If the distance is less than a threshold add it to the path else
                    # increase the frame counter
                    if cost[path_index][center_index] > self.max_dist:
                        self.path_list[path_index].skipped_frames += 1
                        alloc[path_index] = -1  # to create a new path later
                        self.path_list[path_index].predict(self.path_list[path_index].predicted_path[-1])
                    else:
                        self.path_list[path_index].skipped_frames = 0
                        self.path_list[path_index].predict(centers[center_index])

            # delete all paths that exceed the max_frames threshold
            to_delete = []
            for i, path in enumerate(self.path_list):
                if path.skipped_frames > self.max_frames:
                    to_delete.append(i)
            to_delete = to_delete[::-1]  # delete largest index first, so that there is no IndexError by shifting
            for i in to_delete:
                del self.path_list[i]

            # check for each path whether the area of interest was entered or left
            self.update_counter()

            # add a new path for all detections which could not be allocated
            for i in range(len_centers):
                if i not in alloc:
                    self.path_list.append(Path(self.id_count, centers[i], dt=self.dt, state_variance=self.state_var,
                                               measurement_variance=self.measure_var, filter_type=self.filter_type,
                                               max_len=self.path_max_len))
                    self.id_count += 1

        if self.logger:
            self.logger.log(self.frame, self.in_count, self.out_count)


class Logger:
    def __init__(self, path, frequency, start_time=None, delta_t=1, absolute=True):
        """
        Class to log the count of the paths
        :param path: path to a the csv-file or to a sqlite database
        :param frequency: the number of images after which a data set should be created
        :param start_time: [year, month, day, hour, minute=0, second=0]
        :param delta_t: time between two frames in seconds. Its 1/FPS
        :param: absolute: If true, the absolute number of counts per time is stored, otherwise only the change in the
         number is stored
        """
        self.path = path

        if self.path.endswith('.csv'):
            with open(self.path, 'w'):  # clear csv-file
                pass
            self.write = self.write_csv
        elif self.path.endswith('.db'):
            self.write = self.write_db
        else:
            raise ValueError('Unknown file type. Please use .csv or .db file.')

        self.frequency = frequency
        self.absolute = absolute
        self.start_time = datetime(*start_time) if start_time else None
        self.dt = timedelta(seconds=delta_t)

        self.in_buffer, self.out_buffer = 0, 0
        self.last_in, self.last_out = 0, 0
        self.frame_count = 0

    def write_csv(self, data):
        with open(self.path, 'a') as file:
            csv.writer(file).writerow(data)

    def write_db(self, data):
        pass

    def log(self, frame_nr, in_count, out_count):
        if self.start_time:
            frame_nr = self.start_time + frame_nr * self.dt
        if self.absolute:
            self.in_buffer = in_count
            self.out_buffer = out_count
        else:
            self.in_buffer += (in_count - self.last_in)
            self.out_buffer += (out_count - self.last_out)
            self.last_in = in_count
            self.last_out = out_count
        self.frame_count += 1
        if self.frame_count == self.frequency:
            self.write([frame_nr, self.in_buffer, self.out_buffer])
            self.frame_count = 0
            self.in_buffer, self.out_buffer = 0, 0
