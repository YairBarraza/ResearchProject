import numpy as np

def awgn_channel(bits, ebn0_db, rate):
    '''
    Simulate AWGN channel and return llrs
    '''

    # BPSK Mapping
    x = 1 - 2 * bits

    # Eb/N0(dB) to sigma2 conversion
    ebn0_lin = 10 ** (ebn0_db / 10)
    esn0_lin = ebn0_lin * rate
    sigma2 = 1.0 / (2.0 * esn0_lin)
    sigma = np.sqrt(sigma2)

    # AWGN Noise
    noise = np.random.normal(0.0, sigma, size=x.shape)
    y = x + noise

    # LLRs
    llr = 4 * esn0_lin * y

    return llr


def vector_similarity_test(v1, v2, tol):
    for i in range(len(v1)):
        if abs(v1[i] - v2[i]) > tol:
            return False
    return True

"""
''' Testing to ensure different outputs '''
num_trials = 1000
bits = np.zeros(128, dtype=int)
ebn0_db = 2
rate = 0.5
vec_list = []
similar_count = 0
tolerance = 0.1

for trial in range(num_trials):
    llr = awgn_channel(bits, ebn0_db, rate)

    # Adjusting the size of the comparison vectors
    vec = np.round(llr[:5], 1)

    for prev in vec_list:
        if vector_similarity_test(vec, prev, tolerance):
            similar_count += 1
            break
    vec_list.append(vec)

print(f"Total trials: {num_trials}")
print(f"Number of similar vector LLR patterns: {similar_count}")
print(f"Number of unique patterns: {num_trials - similar_count}")
"""
