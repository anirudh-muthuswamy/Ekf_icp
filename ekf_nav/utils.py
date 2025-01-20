import numpy as np

def generate_velocity_sequence(time, vel_seq):
    if time >= 0 and time <= 10:
        return np.array(vel_seq[0])
    elif time > 10 and time <= 20:
        return np.array(vel_seq[1])
    elif time > 20 and time <= 30:
        return np.array(vel_seq[2])
    else:
        return np.array(vel_seq[3])

def add_noise(data, variance):
    return data + np.random.normal(0, np.sqrt(variance), size=data.shape)