import numpy as np

def calculate_m_matrix(priori_i: float, j_nots: list) -> float:
    return priori_i + float(np.sum(j_nots))

# Implementing the 2arctanh formula for the E-matrix
def calculate_e_matrix(i_nots: list) -> float:
    i_nots = np.array(i_nots, dtype = float)
    if len(i_nots) == 0:
        return 0.0

    x = np.tanh(i_nots / 2.0)
    prod = np.prod(x)

    # setting the bounds for prod so no infinite value is returned
    if prod == 1:
        prod = 0.999999
    elif prod == -1:
        prod = -0.999999

    return 2.0 * np.arctanh(prod)

def initiate_m_matrix(priori_matrix: np.array,
                      h_matrix: np.array,
                      e_matrix: np.array,
                      m_matrix: np.array) -> None:
    """
    Fill the M matrix with variable check messages.
    M[j, i] = r_i + sum of all E[k, i] for k != j.
    """

    dimensions = h_matrix.shape
    num_checks = dimensions[0]
    num_bits = dimensions[1]

    for j in range(num_checks):
        for i in range(num_bits):

            if h_matrix[j, i] == 1:

                # find other checks connected to bit i
                connected_checks = []
                column_to_check = h_matrix[:, i]
                for index in range(len(column_to_check)):
                    if column_to_check[index] == 1:
                        connected_checks.append(index)


                # Remove the current check j
                other_checks = []
                for check_idx in connected_checks:
                    if check_idx != j:
                        other_checks.append(check_idx)

                # Gather E[k, i] for all k != j
                j_nots = e_matrix[other_checks, i]

                # Compute M[j, i]
                m_matrix[j, i] = calculate_m_matrix(priori_matrix[i], j_nots)

            else:
                # No edge then no message
                m_matrix[j, i] = 0.0

def initiate_e_matrix(h_matrix: np.ndarray,
                      e_matrix: np.ndarray,
                      m_matrix: np.ndarray) -> None:
    dimensions = h_matrix.shape
    num_checks = dimensions[0]
    num_bits = dimensions[1]

    for j in range (num_checks):        # loop through the check nodes
        for i in range(num_bits):       # loop over variable nodes
            if h_matrix[j, i] == 1:     # only compute messages on check nodes

                connected_bits = []
                row_to_check = h_matrix[j, :]
                # find bits connected to j
                for index in range(len(row_to_check)):
                    if row_to_check[index] == 1:
                        connected_bits.append(index)

                # remove the current bit i
                other_bits = []
                for bit_idx in connected_bits:
                    if bit_idx != i:
                        other_bits.append(bit_idx)

                # gather M[j, k] for all k != i
                i_nots = []
                for bit_index in other_bits:
                    value = m_matrix[j][bit_index]
                    i_nots.append(value)

                # Compute E[j, i]
                e_matrix[j, i] = calculate_e_matrix(i_nots)
            else:
                e_matrix[j, i] = 0.0

def initiate_l_codeword(r_vector, e_matrix, l_codeword):
    dimensions = h_matrix.shape
    num_checks = dimensions[0]
    num_bits = dimensions[1]

    for i in range(num_bits):
        #find all checks connected to bit i
        connected_checks = []
        for j in range(num_checks):
            if h_matrix[j, i] == 1:
                connected_checks.append(j)

        # Sum all E[j, i] from those checks
        extrinsic_sum = 0.0
        for j in connected_checks:
            extrinsic_sum += e_matrix[j, i]

        # Final LLR
        l_codeword[i] = r_vector[i] + extrinsic_sum

def initiate_z_codeword(l_codeword, z_codeword):
    for i in range(len(l_codeword)):
        #Check the sign of the LLR's
        if l_codeword[i] <= 0:
            # Negative or zero LLR means bit is likely 1
            z_codeword[i] = 1
        else:
            # Positive LLR means bit is likely 0
            z_codeword[i] = 0

def check_syndrome(h_matrix, z_codeword):
    h_transpose = h_matrix.transpose()
    syndrome = np.dot(z_codeword, h_transpose) % 2

    return syndrome


### -----------------------------------------
# H Matrix
### -----------------------------------------

h_matrix = np.array ([
    [1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1]
], dtype = int)

#-------------------------
# Test 1: Testing codeword corruption and priori vector creation
#-------------------------

# Utilizing received codework from example 2.6 in Sarah Johnson doc
received_codeword = np.array([1, 0, 1, 0, 1, 1])


pr_vec = np.array([-0.5, 2.5, -4.0, 5.0, -3.5, 2.5])

print(f"The r vector:\n{pr_vec}")

e_matrix = np.zeros_like(h_matrix, dtype = float)
m_matrix = np.zeros_like(h_matrix, dtype = float)

invalid_codeword = 1
iteration_max = 10
iteration_count = 0

while invalid_codeword == 1:
    iteration_count += 1
    initiate_m_matrix(pr_vec, h_matrix, e_matrix, m_matrix)
    print(f"The M Matrix:\n{m_matrix}")

    #-------------------------
    # Test 2: Testing E Matrix creation with set received codeword
    #-------------------------

    initiate_e_matrix(h_matrix, e_matrix, m_matrix)
    print(f"The E Matrix:\n{e_matrix}")


    #-------------------------
    # Test 3: Testing L-codeword creation
    #-------------------------
    l_codeword = np.zeros_like(pr_vec)
    initiate_l_codeword(pr_vec, e_matrix, l_codeword)
    print(f"The L Codeword:\n{l_codeword}")

    #-------------------------
    # Test 4: Testing z codeword creation
    #-------------------------
    z_codeword = np.zeros_like(pr_vec)
    initiate_z_codeword(l_codeword, z_codeword)
    print(f"The Z Codeword:\n{z_codeword}")

    #-------------------------
    # Test 5: Testing syndrome check
    #-------------------------
    syndrome = check_syndrome(h_matrix, z_codeword)
    print(f"The syndrome:\n{syndrome}")

    print(f"Iteration count: {iteration_count}")

    if np.all(syndrome == 0):
        print("The codeword is valid.")
        invalid_codeword = 0
    elif iteration_count == iteration_max:
        print("Iteration max reached.")
        invalid_codeword = 0
    else:
        print("The codeword is not valid.")

# For Sequenctial we update E then go back and update the M, then go back to E....