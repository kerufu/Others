import numpy as np
import time
from os.path import dirname, join
from scipy.io import wavfile
import java

SOUND_SPEED = 343.2

MIC_DISTANCE = np.array([0.07081, 0.06737, 0.07081])
MAX_TDOA = MIC_DISTANCE / float(SOUND_SPEED)

SAMPLE_RATE = 16000
CHANNELS = 4
CHUNK_SIZE = 6400

MOVE_CHECK_RATE = 30
MOVE_CHECK_THRESH = 1024 / MOVE_CHECK_RATE
MOVE_DELAY_THRESH = .1
DOA_THRESH = 3
TALKING_THRESH = 4
TALK_BASE_AMPLITUDE = 250
YELL_BASE_AMPLITUDE = 600

HP_ANGLE_MIN = -18
HP_ANGLE_MAX = 37
BY_ANGLE_RANGE = 360

class SoundDirectionDetector:

    ambient_noise = 0

    time_start = None
    last_time = None
    time_length = 0
    last_doa_timer = 0

    talk_threshold = TALK_BASE_AMPLITUDE
    yell_threshold = YELL_BASE_AMPLITUDE

    last_moving = time.time()

    def adjust_threshold(self, amplitude):
        self.talk_threshold = TALK_BASE_AMPLITUDE + .5 * amplitude
        self.yell_threshold = YELL_BASE_AMPLITUDE + .8 * amplitude

    def check_if_moving(self):
        pass
        # for _ in RateLimiter(MOVE_CHECK_RATE):
        #     hy_speed, hp_speed, by_speed = self.mcu_cmd.get_servo_velocities()
        #     total_move = abs(hp_speed) + abs(by_speed) + abs(hy_speed)
        #     # print("Total move:", total_move)
        #     if total_move > MOVE_CHECK_THRESH:
        #         self.last_moving = time.time()

    def process(self, audio):
        # PATH_TO_WAV = join(dirname(__file__), "soundRecording/speak_front.wav")
        # a = wavfile.read(PATH_TO_WAV)
        # audio = np.array(a[1],dtype=float)[:CHUNK_SIZE]
        audio = np.array(audio)
        direction_vector = self.get_direction(audio)
        amplitude = self.cal_amplitude(audio)
        timestamp = time.time()
        self.ambient_noise = 0.95 *  self.ambient_noise + 0.05 * amplitude

        # print("amplitude", amplitude)
        # print("sound direction 1", self.talk_threshold, self.yell_threshold, self.ambient_noise, amplitude)

        self.check_if_moving()
        if amplitude > self.yell_threshold:  # volume threshold for yelling
            if time.time() - self.last_moving > MOVE_DELAY_THRESH:
                self.last_doa_timer = time.time()
                HPBY = self.direction_vector_to_HPBY(direction_vector)
                return [java.jint(2),
                        java.jfloat(float(HPBY[0])),
                        java.jfloat(float(HPBY[1])),
                        java.jfloat(timestamp)
                        ] # loudness, direction_vector, timestamp
        elif amplitude > self.talk_threshold:  # volume threshold for keeping talking
            if not self.time_start:
                self.time_start = timestamp
                self.last_time = self.time_start
            else:
                stop_period = timestamp - self.last_time
                if stop_period > 3:  # 2 seconds
                    self.time_start = None
                    self.last_time = None
                    self.time_length = 0
                else:
                    self.last_time = timestamp
                    self.time_length = self.last_time - self.time_start
                    # keep talking for 5 seconds
                    # print(self.time_length)
                    if self.time_length > TALKING_THRESH and \
                            time.time() - self.last_moving > MOVE_DELAY_THRESH:
                        # time.time() - self.last_doa_timer > DOA_THRESH:
                        self.last_doa_timer = time.time()
                        self.time_start = None
                        self.last_time = None
                        self.time_length = 0
                        HPBY = self.direction_vector_to_HPBY(direction_vector)
                        return [java.jint(1),
                                java.jfloat(float(HPBY[0])),
                                java.jfloat(float(HPBY[1])),
                                java.jfloat(timestamp)
                                ] # loudness, direction_vector, timestamp

        # print("sound direction 2", self.talk_threshold, self.yell_threshold, self.ambient_noise, amplitude)
        self.adjust_threshold(self.ambient_noise)
        # print("sound direction 3", self.talk_threshold, self.yell_threshold, self.ambient_noise, amplitude)
        return [java.jint(0),
                java.jfloat(-100),
                java.jfloat(-100),
                java.jfloat(-100)
                ] # loudness, direction_vector, timestamp

    def direction_vector_to_HPBY(self, direction):
        target_HP, target_BY = -100, -100
        try:
            if True in np.isnan(direction):
                print("******There is nan value******")
                return
            # Calculate Head Pitch degree
            xy = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
            HP_direction = np.arctan2(direction[2], xy) * 180 / np.pi

            target_HP = min(max(HP_direction, HP_ANGLE_MIN), HP_ANGLE_MAX)

            # Calculate Body Yaw degree
            BY_direction = np.arctan2(direction[1], direction[0]) * 180 / np.pi
            # change x-axis to point to the right front
            BY_direction = BY_direction % BY_ANGLE_RANGE - 90

            # _, _, body_yaw = self.base_outputs.mcu_cmd.get_servo_angles()
            # self.target_BY = (BY_direction + body_yaw) % BY_ANGLE_RANGE
            target_BY = BY_direction % BY_ANGLE_RANGE

        finally:
            return target_HP, target_BY

    def get_direction(self, buf):
        MIC_GROUP_N = 3
        MIC_GROUP = [[0, 2], [1, 2], [3, 2]]
        tau = [0] * MIC_GROUP_N
        theta = [0] * MIC_GROUP_N
        for i, v in enumerate(MIC_GROUP):
            tau[i], _ = self.gcc_phat(buf[v[0]::CHANNELS], buf[v[1]::CHANNELS],
                                      fs=SAMPLE_RATE, max_tau=MAX_TDOA[i], interp=16)

        U = np.array([[0 - (-0.5683), 0 - 0.4225, 0 - 0],
                      [0 - 0, 0 - (-0.1771), 0 - (-0.65)],
                      [0 - 0.5683, 0 - 0.4225, 0 - 0]])
        Delta = np.array([[tau[0] * SOUND_SPEED],
                          [tau[1] * SOUND_SPEED],
                          [tau[2] * SOUND_SPEED]])
        vector = np.linalg.solve(U.T.dot(U), U.T.dot(Delta))

        return vector / (np.linalg.norm(vector) or 1)

    def gcc_phat(self, sig, refsig, fs=1, max_tau=None, interp=16):
        '''
        This function computes the offset between the signal sig and
        the reference signal refsig using the Generalized Cross Correlation -
        Phase Transform (GCC-PHAT)method.
        '''
        # make sure the length for the FFT is larger or equal than
        # len(sig) + len(refsig)
        n = sig.shape[0] + refsig.shape[0]

        # Generalized Cross Correlation Phase Transform
        SIG = np.fft.rfft(sig, n=n)
        REFSIG = np.fft.rfft(refsig, n=n)
        R = SIG * np.conj(REFSIG)

        cc = np.fft.irfft(np.divide(R, np.abs(R)), n=(interp * n))

        max_shift = int(interp * n / 2)
        if max_tau:
            max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

        cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

        # find max cross correlation index
        shift = np.argmax(np.abs(cc)) - max_shift

        tau = shift / float(interp * fs)

        return tau, cc

    def cal_amplitude(self, chunk):
        return np.mean(np.abs(chunk))

def getDetector():
    return SoundDirectionDetector()