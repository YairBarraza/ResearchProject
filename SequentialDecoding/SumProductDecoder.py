import numpy as np

# This code is for Sum Product decoding but will later be made to sum product sequential coding

"""
Initializes the M matrix with the combination of the current priori codeword (r) and
the j nots of the current i column.

Parameter:
    priori_i (float): The current i for the priori codeword (r).
    j_nots (list): The j nots of the current i column of an E matrix.
    
Returns:
    float: The calculated probability using intrinsic and extrinsic probability for the M matrix.
"""
def calculate_m_matrix(priori_i: float, j_nots: list) -> float:
    pass

"""
Initializes the E matrix with the i nots and some very interesting math.

Parameters:
    i_nots (list): The i nots in the M matrix.
    
Returns:
    float: The extrinsic probability of the current i based on the other i nots for the E matrix.
"""
def calculate_e_matrix(i_nots: list) -> float:
    pass

"""
Creates the l codeword which is the combined chances (LLRs) of the priori codework (r) and the column of the E matrix.

Parameters:
    priori_i (float): The current i for the priori codeword (r).
    all_i (list): All numbers in the column of the E matrix.
    
Returns:
    float: The probability that the codeword bit is a 0 (positive) or a 1 (negative).
"""
def calculate_l_codeword(priori_i: float, all_i: list) -> float:
    pass

"""
Creates the z codeword by taking in the l codeword and actually turning it into a list of 0's and 1's

Parameters:
    l_codeword_i (float): The value inside the l codework saying the probability of the z bit to be 0 or 1.
    
Returns:
    int: The 0 or 1 given by the l_codeword being negative or positive, make sure to put in a special case for
         l_codeword having an LLR of just 0.
"""
def calculate_z_codeword(l_codework_i: float) -> int:
    pass

"""
Checks the syndrome of the z codeword (if the array is 0, yippie, solved).

Parameters:
    z_codeword (np.ndarray): The attempted codeword.
    h_matrix (np.ndarray): The H matrix.
    
Returns:
    bool: If z is a codeword it will return true; returns false if z is not a codeword.
"""
def calculate_syndrome(z_codeword: np.ndarray, h_matrix: np.ndarray) -> bool:
    pass

"""
The algorithm that initiates the M matrix (Uses calculate_m_matrix).

Parameters:
    priori_matrix (np.ndarray): The priori matrix used in the calculations.
    h_matrix (np.ndarray): The H matrix used to find what i nodes and j nodes you need in order to initialize
                           the M matrix, as well as what nodes you need to have to in the e_matrix in order to feed
                           that in the calculations.
    e_matrix (np.ndarray): The E matrix used to initialize the M matrix with its extrinsic values.
                           Do note that the M matrix, on its first iteration, does not use the E matrix
                           because it does not have any information in the first iteration.
    m_matrix (np.ndarray): The M matrix we want to modify.

Returns:
    None: Since were putting in the M matrix, it will modify it and return nothing.
"""
def initiate_m_matrix(priori_matrix: np.ndarray, h_matrix: np.ndarray, e_matrix: np.ndarray, m_matrix: np.ndarray) -> None:
    pass

"""
The algorithm that initiates the H matrix (Uses calculate_e_matrix).

Parameters:
    h_matrix (np.ndarray): The H matrix used to find where you need to get the i nots in in order to find them in the
                           M matrix, which is then used to calculate the extrinsic values of the E matrix.
    m_matrix (np.ndarray): The M matrix, when used with the h_matrix telling you where the i nots are, gives the
                           values needed for the math of setting up the E matrix (You use that Eji equation).
    e_matrix (np.ndarray): The E matrix we want to modify.
    
Returns:
    None: Since were putting in the E matrix, it will modify it and return nothing.
"""
def initiate_e_matrix(h_matrix: np.ndarray, e_matrix: np.ndarray, m_matrix: np.ndarray) -> None:
    pass

"""
The algorithm that initiates the l codeword (Uses calculate_l_codeword).

Parameters:
    priori_matrix (np.ndarray): The priori matrix used in the calculations (intrinsic values).
    e_matrix (np.ndarray): The E matrix; grab a column of this matrix and add all the values (extrinsic values).
    l_codeword (np.ndarray): The combination of intrinsic and extrinsic values to determine a solved codeword.

Returns:
    None: Since were putting in the l codeword, it will modify it and return nothing.
"""
def initiate_l_codeword(priori_matrix: np.ndarray, e_matrix: np.ndarray, l_codeword: np.ndarray) -> None:
    pass

"""
The algorithm that initiates the z codeword (Uses calculate_z_codeword).

Parameters:
    l_codeword (np.ndarray): The matrix of combinations of LLRs to now give a 0 or 1 to the z codeword.
    z_codeword (np.ndarray): The matrix of a corrected and attempted, codeword.
    
Returns:
    None: Since were putting in the z codeword, it will modify it and return nothing.
"""
def initiate_z_codeword(l_codeword: np.ndarray, z_codeword: np.ndarray) -> None:
    pass

"""
Decodes a corrupted codeword using log likelihood ratios (LLRs).

Parameters:
    corrupted_codeword (np.ndarray): The corrupted codeword.

Returns:
    np.ndarray: This returns the allegedly solved codeword.
"""
def sum_product_decoding(corrupted_codeword: np.ndarray) -> np.ndarray:
    # We need some things to be initiated before with start.

    r_codeword = None # For now, we can just make r_codeword = corrupted_codeword, like as a placeholder.
    l_codeword = None # This variable is to hold all the LLR codewords we try out, it's np.ndarray type.
    z_codeword = None # This variable is to hold all the codewords we actually test, it's np.ndarray type.
    syndrome = False # We assume syndrome is false until the codeword is solved.
    h_matrix = None # An np.ndarray.
    m_matrix = None # An empty, same size as h_matrix, np.ndarray.
    e_matrix = None # An empty, same size as h_matrix, np.ndarray.
    i_iterations = 0 # A counter that counts the total iterations.
    i_max_iterations = 10 # The maximum times we want this program to go for.

    # None we start the decoding process.

    # Step 0: Make a for loop and make the iterations end when the max iterations occur.
    while i_iterations >= i_max_iterations:
        # Step 1: We want to initialize the M matrix; takes in the priori, H matrix, and E matrix.
        initiate_m_matrix(r_codeword, h_matrix, e_matrix, m_matrix)

        # Step 2: Start putting numbers in the E matrix.
        initiate_e_matrix(h_matrix, e_matrix, m_matrix)

        # Step 3: Start calculating the l codeword
        initiate_l_codeword(r_codeword, e_matrix, l_codeword)

        # Step 4: Initiate the z codeword (You can make an initiate_z_codeword if you'd like, it's a simple for loop)
        calculate_z_codeword(l_codeword, z_codeword)

        # Step 5: Test the z codeword out.
        syndrome = calculate_syndrome(z_codeword, h_matrix)

        # Step 6: If the syndrome is true, return the solved codeword and print an iteration amount
        if (syndrome):
            print("Codeword solved, it took this many iterations: " + str(i_iterations))
            return z_codeword

        # Step 7: If you are here, that means the decoder didn't solve it in one iteration, so it will try until it
        #         the codeword is solved or the iterations is equal to the max iterations (the while loop up top).
        i_iterations += 1

        # Note: The first iteration of initiate_m_matrix, it needs to realize that the e_matrix is empty and just use
        #       the priori and the H matrix. After the first iteration, the E matrix will no longer be empty and the
        #       code in initiate_m_matrix needs to recognize this and start using the E matrix for making the M matrix.

    # Step 8: Since the while loop broke, that means the code was unsolved. We will return what is left of the
    #         codeword and print the max iterations
    print("Codeword unsolved, max iterations met: " + str(i_max_iterations))
    return z_codeword

# Step 9: Test the codeword.
corrupt_c = None
solved_c = sum_product_decoding(corrupt_c)
print("Here is our attempted codeword solving: " + str(solved_c))

# Step 10: Verify if the codeword that's solved is the original, uncorrupted codeword.
