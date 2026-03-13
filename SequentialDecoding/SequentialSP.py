import numpy as np

def get_row(gen_matrix: np.ndarray, index: int) -> np.ndarray:
    row_slice = gen_matrix[index, :]
    return row_slice


def get_column(gen_matrix: np.ndarray, index: int) -> np.ndarray:
    column_slice = gen_matrix[:, index]
    return column_slice


def get_one_pos(gen_matrix_slice: np.ndarray) -> np.ndarray:
    ones_positions_tuple = np.asarray(gen_matrix_slice == 1).nonzero()
    ones_positions = ones_positions_tuple[0]
    return ones_positions


def initiate_m_matrix(p_matrix: np.ndarray, h_matrix: np.ndarray, m_matrix: np.ndarray) -> None:
    j_rows = h_matrix.shape[0]
    i_columns = h_matrix.shape[1]

    for j_row in range(j_rows):
        row_slice = get_row(h_matrix, j_row)
        row_one_pos = get_one_pos(row_slice)

        for i_column in row_one_pos:
            m_matrix[j_row, i_column] = p_matrix[i_column]


def calculate_e_matrix(gen_matrix: np.ndarray) -> float:
    matrix_tan_h = np.zeros(np.shape(gen_matrix), dtype = float)
    column_size = gen_matrix.shape[0]

    for i in range(column_size):
        value = gen_matrix[i]
        matrix_tan_h[i] = np.tanh(value / 2)

    product_notation_calculated = 1.0

    for i in range(column_size):
        product_notation_calculated *= matrix_tan_h[i]

    calculated_num = np.log((1 + product_notation_calculated) / (1 - product_notation_calculated))

    return calculated_num


def initialize_e_matrix(h_matrix: np.ndarray, m_matrix: np.ndarray, e_matrix: np.ndarray) -> None:
    rows = h_matrix.shape[0]

    for j_row in range(rows):
        row_slice = get_row(h_matrix, j_row)
        row_ones = get_one_pos(row_slice)

        for i_column in row_ones:
            current_ones = row_ones
            i_nots_ones = current_ones[current_ones != i_column]
            i_nots_m_vals = np.zeros(np.shape(i_nots_ones), dtype = float)
            count = 0

            for i_column_m in i_nots_ones:
                i_nots_m_vals[count] = m_matrix[j_row, i_column_m]
                count += 1

            e_matrix[j_row, i_column] = calculate_e_matrix(i_nots_m_vals)


def modify_l_matrix(p_matrix: np.ndarray, l_matrix: np.ndarray, h_matrix: np.ndarray, e_matrix: np.ndarray) -> None:
    column_size = h_matrix.shape[1]

    for i_column in range(column_size):
        h_column_slice = get_column(h_matrix, i_column)
        h_column_one_pos = get_one_pos(h_column_slice)
        sum_e = 0.0

        for j_row_e in h_column_one_pos:
            sum_e += e_matrix[j_row_e, i_column]

        l_matrix[i_column] = p_matrix[i_column] + sum_e


def modify_z_matrix(l_matrix: np.ndarray, z_matrix: np.ndarray) -> None:
    column_size = l_matrix.shape[0]

    for i_column in range(column_size):
        l_value = l_matrix[i_column]
        if l_value < 0:
            z_matrix[i_column] = 1
        elif l_value > 0:
            z_matrix[i_column] = 0
        else:
            z_matrix[i_column] = np.random.randint(0, 2)


def syndrome_check(z_matrix: np.ndarray, h_matrix: np.ndarray) -> bool:
    row_size = h_matrix.shape[0]
    column_size = h_matrix.shape[1]

    for row in range(row_size):
        int_sum = 0

        for column in range(column_size):
            int_sum += h_matrix[row, column] * z_matrix[column]

        if (int_sum % 2) == 1:
            return False

    return True


def sequential_m_matrix(r_matrix: np.ndarray, h_matrix: np.ndarray, m_matrix: np.ndarray,
                        e_matrix: np.ndarray, chosen_check: int) -> None:
    check_node_row = get_row(h_matrix, chosen_check)
    check_row_one = get_one_pos(check_node_row)

    for i_column in check_row_one:
        check_node_column = get_column(h_matrix, i_column)
        check_column_one = get_one_pos(check_node_column)
        # Take away the J node because they don't update anything.
        check_column_one = check_column_one[check_column_one != chosen_check]
        sum_non_js = 0.0

        for j_row in check_column_one:
            sum_non_js += e_matrix[j_row, i_column]

        m_matrix[chosen_check, i_column] = sum_non_js +  r_matrix[i_column]


def sequential_e_matrix(h_matrix: np.ndarray, m_matrix: np.ndarray, e_matrix: np.ndarray, chosen_check: int) -> None:
    row_slice = get_row(h_matrix, chosen_check)
    row_ones = get_one_pos(row_slice)

    for i_column in row_ones:
        current_ones = row_ones
        i_nots_ones = current_ones[current_ones != i_column]
        i_nots_m_vals = np.zeros(np.shape(i_nots_ones), dtype = float)
        count = 0

        for i_column_m in i_nots_ones:
            i_nots_m_vals[count] = m_matrix[chosen_check, i_column_m]
            count += 1

        e_matrix[chosen_check, i_column] = calculate_e_matrix(i_nots_m_vals)


def user_input_sequence() -> np.ndarray:
    print("Sequence must follow this exact structure: #, #, #, #")
    print("If not inputted properly, errors will occur...\n")
    print("Example sequence: 0, 2, 3, 1")
    user_in_string = input("Input sequence: ")
    user_in_string_list = user_in_string.split(", ")
    user_in_matrix = np.zeros([len(user_in_string_list)], dtype = int)

    for index_num in range(len(user_in_string_list)):
        user_in_matrix[index_num] = int(user_in_string_list[index_num])

    print()

    return user_in_matrix


def sequential_sum_product_decoder(priori_matrix: np.ndarray) -> None:
    l_matrix = np.zeros(np.shape(priori_matrix), dtype = float)
    z_matrix = np.zeros(np.shape(priori_matrix), dtype = int)
    h_matrix = np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1]
    ], dtype = int)
    m_matrix = np.zeros(np.shape(h_matrix), dtype = float)
    e_matrix = np.zeros(np.shape(h_matrix), dtype = float)
    syndrome = False
    iteration_count = 0
    iteration_max = 5

    print("Printing initial M matrix:")
    initiate_m_matrix(priori_matrix, h_matrix, m_matrix)
    print(m_matrix)
    print("\n")

    print("Printing initial E matrix:")
    print(e_matrix)
    print("\n")

    while not syndrome and iteration_count < iteration_max:

        print(f"\033[91mCurrent Iteration: {iteration_count}\033[0m\n\n")

        input_sequence = user_input_sequence()

        for chosen_check in input_sequence:
            print(f"\033[92mCurrent J_node: {chosen_check}\033[0m\n\n")

            print("Printing sequentially modified M matrix:")
            sequential_m_matrix(priori_matrix, h_matrix, m_matrix, e_matrix, chosen_check)
            print(m_matrix)
            print("\n")

            print("Printing sequentially modified E matrix:")
            sequential_e_matrix(h_matrix, m_matrix, e_matrix, chosen_check)
            print(e_matrix)
            print("\n")

        print("Printing L matrix:")
        modify_l_matrix(priori_matrix, l_matrix, h_matrix, e_matrix)
        print(l_matrix)
        print("\n")

        print("Printing Z matrix:")
        modify_z_matrix(l_matrix, z_matrix)
        print(z_matrix)
        print("\n")

        print("Printing if Z is a codeword:")
        syndrome = syndrome_check(z_matrix, h_matrix)
        print(syndrome)
        print("\n")

        iteration_count += 1

    if syndrome:
        print("Decoding Success:")
        print(f"- Completed Codeword: {z_matrix}")
        print(f"- Global Iterations: {iteration_count}")
    else:
        print("Decoding Failure:")
        print(f"- Attempted Codeword: {z_matrix}")
        print(f"- Max Iterations: {iteration_count}")

r_codeword = np.array([-0.5, 2.5, -4.0, 5.0, -3.5, 2.5], dtype = float)
sequential_sum_product_decoder(r_codeword)
