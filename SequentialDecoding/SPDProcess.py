import numpy as np
import chime
from Projects.AWGNChannel import awgn_channel
from Projects.SSPDecoder import ssp_decoder

def spd_process(chosen_stnr: float) -> np.ndarray:
    i_max = 20000
    total_trials = 0
    total_frame_errors = 0
    total_bit_errors = 0
    codeword_length = 128

    for iteration in range(i_max):
        # Create a codeword, 128 zeros
        codeword = np.zeros(codeword_length, dtype = int)

        # Corrupt the codeword
        ebn0_db = chosen_stnr # This changes
        rate = 0.5
        corrupted_codeword = awgn_channel(codeword, ebn0_db, rate)

        # Decode the codeword
        decoded_codeword = ssp_decoder(corrupted_codeword)

        # Keep track Sum Product mistakes
        sp_frame_errors = 0
        sp_bit_errors = 0

        for current_bit in range(np.size(decoded_codeword)):
            if codeword[current_bit] != decoded_codeword[current_bit]:
                sp_bit_errors += 1

        if sp_bit_errors > 0:
            print("\033[31mv ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\033[0m")
            sp_frame_errors += 1

        total_trials += 1
        total_frame_errors += sp_frame_errors
        total_bit_errors += sp_bit_errors

        print("Trial " + str(total_trials))
        print("          Keyword: " + str(codeword))
        print("Corrupted Keyword: " + str(corrupted_codeword))
        print("  Decoded Keyword: " + str(decoded_codeword))
        print("     Frame Errors: " + str(sp_frame_errors))
        print("       Bit Errors: " + str(sp_bit_errors) + "\n")

        if total_frame_errors == 100:
            break

    print("Total Trials: " + str(total_trials))
    print("Total Frame Errors: " + str(total_frame_errors))
    print("Total Bit Errors: " + str(total_bit_errors))
    print("Frame Error Rate (FER): " + str(total_frame_errors / total_trials))
    print("Bit Error Rate (BER): " + str(total_bit_errors / (total_trials * codeword_length)))

    fer_ber = np.array ([
        total_frame_errors / total_trials, total_bit_errors / (total_trials * codeword_length)
    ], dtype = float)

    return fer_ber

# CHANGE THISSSSSS
the_num = 5

empty_num = spd_process(the_num)
chime.theme('mario')
chime.success(sync = True)

print(f"Framer Error Rate (FER) for signal to noise ({the_num}): {empty_num[0]}")
print(f"   Bit Error Rate (BER) for signal to noise ({the_num}): {empty_num[1]}")
