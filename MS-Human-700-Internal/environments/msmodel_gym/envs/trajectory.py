import pandas as pd
import scipy
import scipy.signal
from scipy import interpolate
import numpy as np

class Trajectory:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.load_data()

    def load_data(self):
        df_data = pd.read_csv(self.file_path, sep=",", header=0)
        data_time = df_data["time"]
        mj_joint_data = df_data.drop(columns=["time"]).values
        self.mj_joint_data_tck, self.data_time = self.get_scipy_tck_symmetry(mj_joint_data)
        self.num_joints = mj_joint_data.shape[1]

    def get_scipy_tck_symmetry(self, rawdata):
        time_frame = rawdata.shape[0]
        nq = rawdata.shape[1]

        num_periods = 2  # hardcoded
        mean_period = 114

        # cut the single periods
        startloc = int(np.floor(time_frame / 2 - (num_periods / 2 * mean_period)))
        single_periods = np.zeros((nq, int(mean_period), num_periods))

        for loop1 in range(nq):
            startloc_loop = startloc
            for loop2 in range(num_periods):
                single_periods[loop1, :, loop2] = rawdata[startloc_loop : startloc_loop + mean_period, loop1]
                startloc_loop = startloc_loop + mean_period

        # calculate mean
        mean_gait_angle = np.mean(single_periods, axis=2)
        mean_time = np.array([0.00])
        mean_time = np.append(mean_time, np.linspace(start=0.01, stop=(mean_period - 1) * 0.01, num=mean_period - 1))
        mean_gait_angle_expand = np.concatenate([mean_gait_angle, mean_gait_angle, mean_gait_angle], axis=1)

        b, a = scipy.signal.butter(8, 0.2, 'lowpass')

        for i in range(nq):
            if i == 2:
                # tx
                mean_gait_angle[i, :] = rawdata[startloc : startloc + mean_period, i]
                continue
            mean_gait_angle_expand[i, :] = scipy.signal.filtfilt(b, a, mean_gait_angle_expand[i, :])
            mean_gait_angle[i, :] = mean_gait_angle_expand[i, mean_period : 2 * mean_period]

        temp_p = np.expand_dims(mean_gait_angle[:, 0], axis=0)

        mean_time = np.array([0.00])
        mean_time = np.append(mean_time, np.linspace(start=0.01, stop=(mean_period) * 0.01, num=mean_period))
        mean_gait_angle = np.concatenate([mean_gait_angle, temp_p.T], axis=1)

        for i in range(3):
            mean_gait_angle[i, :] = mean_gait_angle[i, :] - mean_gait_angle[i, 0]

        tck_list = []
        for i in range(nq):
            if i == 2:  # tx
                mean_gait_angle[i, -1] = mean_gait_angle[i, -2] + mean_gait_angle[i, 1] - mean_gait_angle[i, 0]
                tck = interpolate.splrep(mean_time, mean_gait_angle[i, :], s=0)
            else:
                tck = interpolate.splrep(mean_time, mean_gait_angle[i, :], s=0, per=1)
            tck_list.append(tck)

        data_time_length = mean_period * 0.01

        return tck_list, data_time_length

    def query(self, time):
        period_now = time // self.data_time
        t_interp = time % self.data_time
        p_interp = np.zeros(self.num_joints)
        v_interp = np.zeros(self.num_joints)

        for i in range(self.num_joints):
            tck_now = self.mj_joint_data_tck[i]
            p_interp[i] = interpolate.splev(t_interp, tck_now, der=0)
            v_interp[i] = interpolate.splev(t_interp, tck_now, der=1)

        # tx
        p_interp[2] += interpolate.splev(self.data_time, self.mj_joint_data_tck[2], der=0) * period_now

        return p_interp, v_interp



