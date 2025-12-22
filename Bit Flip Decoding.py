import numpy as np

"""
Here are the constants that we need to start creating the Bit Flip Decoding process.
They will consist of: The corrupted code word (which is inside the function), the H matrix, max iterations, and
an E matrix. The corrupted code is what we're trying to figure out, the H matrix is what does the solving, 
the max is for max attempts, and the E matrix is the matrix that will, for each keyword bit, have the opinion
of each check node. This means E must equal H in terms of size.
"""

"""
Description: Takes in a corrupted keyword (1 by 7) and decodes it using bitflip decoding
Depends on: numpy
Parameters: input_list (list data type, 1 by 7)
Returns: m_sub_i (array data type, 1 by 7),
         "Decoding Failed: ... Iterations" (string data type, ... is your Max Iterations number)
"""

"Now we make the function"

def bit_flip_decode(input_list):

    "H Matrix"
    h_matrix = np.array([
        [1, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 1]
    ])

    "Max iterations"
    max_iterations = 1000

    "The E Matrix"
    e_matrix = np.full(
        h_matrix.shape, 2
    )

    "Corrupted Codeword"
    m_sub_i = np.array(input_list)

    "This will repeat the process"
    for iterations in range(max_iterations):

        "This does two things: give a list of where each check node connects to a bit node and holds the empty array"
        for j_row in range(h_matrix.shape[0]):
            particular_row = h_matrix[j_row, :]
            particular_row_tuple = np.where(particular_row == 1)
            ones_places_for_row = particular_row_tuple[0]
            array_of_keyword_bits = np.empty(0)

            "This adds up the bits that the check node points to and takes a mod 2 of their sum"
            for i_column in ones_places_for_row:
                particular_bit = m_sub_i[i_column]
                array_of_keyword_bits = np.append(array_of_keyword_bits, particular_bit)

            "This is the message the check node gives: 1 means the bits are wrong, 0 means the bits are correct"
            bit_message_from_check_node = int(np.sum(array_of_keyword_bits) % 2)

            "Now we need to collect the opinions of these check nodes in an E Matrix"
            for i_column_repeat in ones_places_for_row:
                e_matrix[j_row, i_column_repeat] = bit_message_from_check_node

        "Now that we have collected the opinions of each check node, we analyze what's unique and what's been repeated"
        for i_column_e_matrix in range(h_matrix.shape[1]):
            particular_row_e_matrix = e_matrix[:, i_column_e_matrix]
            uniqueness_of_opinions_tuple = np.unique(particular_row_e_matrix, False, False, True)
            uniqueness_of_opinions = uniqueness_of_opinions_tuple[0]
            repetition_of_opinions = uniqueness_of_opinions_tuple[1]
            # print(str(uniqueness_of_opinions) + " & " + str(repetition_of_opinions))

            "Find the 2's and delete both their uniqueness and repetitions"
            if 2 in uniqueness_of_opinions:
                current_position_of_2_tuple = np.where(uniqueness_of_opinions == 2)
                current_position_of_2 = current_position_of_2_tuple[0]
                no_2_uniqueness_of_opinions = np.delete(uniqueness_of_opinions, current_position_of_2)
                no_2_repetition_of_opinions = np.delete(repetition_of_opinions, current_position_of_2)
                # print("^")
                # print(str(no_2_uniqueness_of_opinions) + " & " + str(no_2_repetition_of_opinions))
                # print(" ")

            "If there's 1 and 0s, check if 1's are greater than 0's to change the bit"
            if np.array_equal(no_2_uniqueness_of_opinions, [0, 1]):
                number_of_0s = no_2_repetition_of_opinions[0]
                number_of_1s = no_2_repetition_of_opinions[1]

                if number_of_0s < number_of_1s:
                    m_sub_i[i_column_e_matrix] = int(((m_sub_i[i_column_e_matrix]) + 1) % 2)

            "If there's only 1's, change the bit"
            if np.array_equal(no_2_uniqueness_of_opinions, [1]):
                m_sub_i[i_column_e_matrix] = int(((m_sub_i[i_column_e_matrix]) + 1) % 2)

        "Now we do this until there's no 1's"
        if 1 not in e_matrix:
            #print("Here's the Code Word: " + str(m_sub_i))
            #print("It took " + str(iterations + 1) + " iterations")
            return m_sub_i
            break

        "If there is still unknowns in the keyword and the max iterations have been reached, we say it reached the max"
        if (iterations + 1) == max_iterations:
            #print("Max iterations have been reached: " + str(max_iterations))
            #print("Decoding failed")
            return(("Decoding failed (" + str(max_iterations) + " Iterations)"))

"This is an improved version of bit flip decode, but it needs comments explaining what's happening"

def decode_bit_flip(input_list):
    h_matrix = np.array([
        [1, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 1]
    ])

    max_iterations = 100

    e_matrix = np.full(
        h_matrix.shape, 2
    )

    m_sub_i = np.array(input_list)


    for iterations in range(max_iterations):
        print(m_sub_i)
        print(e_matrix)
        check_node_decisions(h_matrix, e_matrix, m_sub_i)
        change_m_code(h_matrix, e_matrix, m_sub_i)


        if 1 not in e_matrix:
            return m_sub_i

        if iterations + 1 == max_iterations:
            print(f"Reached Max Iterations ({max_iterations})")
            return m_sub_i



def check_node_decisions(h_matrix, e_matrix, m_sub_i):
    row_num = h_matrix.shape[0]

    for j_row in range(row_num):
        jth_row = h_matrix[j_row, :]
        jth_row_tuple = np.asarray(jth_row == 1).nonzero()
        one_positions = jth_row_tuple[0]

        calculate_decisions(e_matrix, m_sub_i, one_positions, j_row)


def calculate_decisions(e_matrix, m_sub_i, one_positions, jth_row):
    opinions = np.empty(0)

    for bit_place in one_positions:
        bit = m_sub_i[bit_place]
        opinions = np.append(opinions, bit)

    decision = int(np.sum(opinions) % 2)

    for ith_column in one_positions:
        e_matrix[jth_row, ith_column] = decision


def change_m_code(h_matrix, e_matrix, m_sub_i):
    column_num = h_matrix.shape[1]

    for i_column in range(column_num):
        ith_column_opinions = e_matrix[:, i_column]
        current_key_bit = m_sub_i[i_column]

        decision_bool = decide_change(ith_column_opinions)

        if not decision_bool:
            m_sub_i[i_column] = int((current_key_bit + 1) % 2)


def decide_change(opinions):
    change = 0
    stay = 0

    for opinion in opinions:

        if opinion == 1:
            change += 1

        if opinion == 0:
            stay += 1

    if change > stay:
        return False
    return True


