import numpy as np

def awgn_channel(bits, ebn0_db, rate):
    '''
    Simulate AWGN channel and return noisy samples
    '''

    # BPSK Mapping
    x = 1 - 2 * bits

    # Eb/N0(dB) to sigma2 conversion
    ebn0_lin = 10 ** (ebn0_db / 10)
    esn0_lin = ebn0_lin * rate
    sigma2 = 1.0 / (2.0 * esn0_lin)
    sigma = np.sqrt(sigma2)

    # AWGN Noise
    noise = np.random.normal(0.0, sigma, size = x.shape)
    y = x  + noise

    return y

''' Testing to ensure different outputs'''
for trial in range(3):
    y = awgn_channel(np.zeros(128, dtype = int), 2, 0.5)
    print(f"Trial {trial + 1}, first 5 samples of y: {y[:5]}")
