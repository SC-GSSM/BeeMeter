import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, F, H, x, Q=None, P=None, R=None):
        """
        Implementation of a Kalman Filter
        :param F: state transition matrix
        :param H: measurement function
        :param x: state
        :param Q: motion/system noise
        :param P: covariance matrix
        :param R: measurement noise
        """
        self.Q = np.eye(F.shape[0]) if Q is None else Q
        self.P = np.eye(F.shape[0]) if P is None else P
        self.R = np.eye(H.shape[0]) if R is None else R
        self.F = F
        self.H = H
        self.x = x
        self.I = np.eye(self.F.shape[0])

    def predict(self):
        """ Predict next state """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """ Add new measurement to the filter """
        PH = self.P @ self.H.T  # Helper
        K = PH @ np.linalg.inv(self.H @ PH + self.R)  # Kalman-Verstärkung / Kalman-Gain-Matrix

        error = measurement - self.H @ self.x  # error between prediction and measurement

        self.x = self.x + K @ error  # prädizierte Zustandsvektor

        self.P = (self.I - K @ self.H) @ self.P  # Kovarianzmatrix des Schätzfehlers


class DummyFilter:
    def __init__(self, x, **kwargs):
        self.x = x

    def predict(self):
        return self.x

    def update(self, measurement):
        self.x = measurement


class VelocityKalmanFilter(KalmanFilter):
    def __init__(self, x=np.array([0, 0, 0, 0]), dt=.1, measurement_variance=0.01 ** 2, state_variance=500):
        """
        Kalman Filter that simulates a movement with no constant velocity
        :param x: initial state, State representation: [x, y, x', y']
        :param dt:  time step in seconds
        :param measurement_variance:
        :param state_variance:
        """
        # State transition matrix: next_x = F dot x
        # next_pos_x = prev_pos_x + x' * dt
        # next_pos_y = prev_pos_y + y' * dt
        # x' = x'
        # y' = y'
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        # Measurement function: position = H dot x
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        # Measurement noise
        R = np.identity(H.shape[0]) * measurement_variance
        # Covariance matrix
        P = np.identity(F.shape[0]) * state_variance

        # Motion noise
        Q = np.identity(F.shape[0]) * measurement_variance

        if x.size == 2:
            x = np.concatenate([x, [0, 0]])

        super(VelocityKalmanFilter, self).__init__(F, H, x, Q=Q, P=P, R=R)


class AccelerationKalmanFilter(KalmanFilter):
    def __init__(self, x=np.array([0, 0, 0, 0, 0, 0]), dt=.1, measurement_variance=1 ** 2, state_variance=500):
        """
        Kalman filter that simulates a movement with constant acceleration
        :param x: initial state, State representation: [x, y, x', y', x'', y'']
        :param dt:  time step in seconds
        :param measurement_variance:
        :param state_variance:
        """
        # State transition matrix: next_x = F @ x
        F = np.array([[1, 0, dt, 0, dt ** 2 / 2, 0],
                      [0, 1, 0, dt, 0, dt ** 2 / 2],
                      [0, 0, 1, 0, dt, 0],
                      [0, 0, 0, 1, 0, dt],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])
        # Measurement function: position = H @ x
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0]])
        # Measurement noise
        R = np.identity(H.shape[0]) * measurement_variance
        # Covariance matrix
        P = np.identity(F.shape[0]) * state_variance
        # Motion noise
        Q = np.identity(F.shape[0]) * measurement_variance

        if x.size == 2:
            x = np.concatenate([x, [0, 0, 0, 0]])
        super(AccelerationKalmanFilter, self).__init__(F, H, x, Q=Q, P=P, R=R)


class ConstantVelocityObject:
    def __init__(self, x0=0, y0=0, scale=0.05, dt=0.1):
        self.scale = scale
        self.dt = dt
        self.v_x = 1
        self.v_y = 1
        self.x = np.array([x0, y0, self.v_x, self.v_y])

    def update(self):
        self.v_x += np.random.randn() * self.scale
        self.v_y += np.random.randn() * self.scale
        F = np.array([[1, 0, self.dt, 0],
                      [0, 1, 0, self.dt],
                      [0, 0, self.v_x, 0],
                      [0, 0, 0, self.v_y]])
        self.x = F @ self.x
        return self.x[0], self.x[1]


if __name__ == '__main__':
    np.random.seed(45)
    dt = 1
    fil = VelocityKalmanFilter(dt=dt, measurement_variance=0.08 ** 2, state_variance=1000)
    obj = ConstantVelocityObject(dt=dt, scale=0.08)

    x_dir, y_dir = [], []
    x_pred, y_pred = [], []
    for i in range(1, 10):
        x, y = obj.update()
        x_dir.append(x)
        y_dir.append(y)
        fil.update(np.array([x, y]))
        fil.predict()
        pred = fil.x[:2]
        x_pred.append(pred[0])
        y_pred.append(pred[1])
    plt.plot(x_pred, y_pred)
    plt.scatter(x_dir, y_dir)
    plt.show()
