import numpy as np
import time
np.set_printoptions(precision=2, suppress=True)

def awgn_channel(bits: np.ndarray, ebn0_db: float, rate: float) -> np.ndarray:
    x = 1 - 2 * bits

    ebn0_lin = 10 ** (ebn0_db / 10)
    esn0_lin = ebn0_lin * rate
    sigma2 = 1.0 / (2.0 * esn0_lin)
    sigma = np.sqrt(sigma2)

    noise = np.random.normal(0.0, sigma, size=x.shape)
    y = x + noise

    llr = 4 * esn0_lin * y
    
    return llr

def training_set(codeword_bits: np.ndarray, snr_set: np.ndarray, size: int, rate: float) -> np.ndarray:
    training_llrs = np.zeros(size, dtype = np.ndarray)
    snr_count = 0
    snr_change = size / np.shape(snr_set)[0]

    for element in range(size):
        if (element % snr_change == 0 and element != 0):
            snr_count += 1
        
        corrupted_llr = awgn_channel(codeword_bits, snr_set[snr_count], rate)
        training_llrs[element] = corrupted_llr

    return training_llrs

def return_h_matrix(file_name: str) -> None:
    empty_string = "np.array(["

    with open(file_name, "r") as file:
        list_of_lines = file.readlines()

        for i in range(len(list_of_lines)):

            item = list_of_lines[i]
            new_item = item.strip()

            empty_string += "["

            for j in range(len(new_item)):

                empty_string += new_item[j]

                if (j != len(new_item) - 1):
                    empty_string += ", "

            if (i != len(list_of_lines) - 1):
                empty_string += "],"
                empty_string += "\n"

            else:
                empty_string += "]"
                empty_string += "], dtype = object)"

    return eval(empty_string)

def highest_check_degree(h_matrix: np.ndarray) -> int:
    max_check_degree = 0
    rows, columns = np.shape(h_matrix)
    temp_count = 0
    
    for j_row in range(rows):
        for i_column in range(columns):
            temp_count += h_matrix[j_row][i_column]

        if temp_count > max_check_degree:
            max_check_degree = temp_count
        temp_count = 0

    return max_check_degree

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

    for j_row in range(j_rows):
        row_slice = get_row(h_matrix, j_row)
        row_one_pos = get_one_pos(row_slice)

        for i_column in row_one_pos:
            m_matrix[j_row, i_column] = p_matrix[i_column]

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

def local_state_indexes(h_matrix: np.ndarray, global_state: np.ndarray, q_table: np.ndarray) -> np.ndarray:
    action_size = np.shape(h_matrix)[0]
    all_local_state_indexes = np.zeros(action_size, dtype = np.int32)
    temp_local_state = np.zeros(round(np.log(np.shape(q_table)[1])), dtype = np.int32)

    for j_check_node in range(action_size):
        check_node_connections = get_one_pos(h_matrix[j_check_node])
        counter = np.shape(temp_local_state)[0] - np.shape(check_node_connections)[0]
        
        for index in check_node_connections:
            temp_local_state[counter] = global_state[index]
            counter += 1
        
        all_local_state_indexes[j_check_node] = int("".join(temp_local_state.astype(str)), 2)

        for local_index in range(np.shape(temp_local_state)[0]):
            temp_local_state[local_index] = 0
        
        counter = 0

    return all_local_state_indexes

def action_choice(h_matrix: np.ndarray, global_state: np.ndarray, q_table: np.ndarray, epsilon: float) -> int:
    rng = np.random.default_rng()

    if rng.random() < epsilon:
        action_size = np.shape(q_table)[0]
        action_index = rng.integers(low = 0, high = action_size, size = 1)[0]
        return action_index
    
    else:
        action_size = np.shape(q_table)[0]
        all_local_state_indexes = local_state_indexes(h_matrix, global_state, q_table)
        max_action_index = 0

        for j_row in range(1, action_size):
            if q_table[j_row][all_local_state_indexes[j_row]] > q_table[max_action_index][all_local_state_indexes[j_row]]:
                max_action_index = j_row

        return max_action_index

def calculate_e_matrix(gen_matrix: np.ndarray) -> float:
    matrix_tan_h = np.zeros(np.shape(gen_matrix), dtype = float)
    column_size = gen_matrix.shape[0]

    for i in range(column_size):
        value = gen_matrix[i]
        matrix_tan_h[i] = np.tanh(value / 2)

    product_notation_calculated = 1.0

    for i in range(column_size):
        product_notation_calculated *= matrix_tan_h[i]
        product_notation_calculated = max(min(product_notation_calculated, 1 - 1e-15), -1 + 1e-15)

    calculated_num = np.log((1 + product_notation_calculated) / (1 - product_notation_calculated))

    return calculated_num

def sequential_m_matrix(r_matrix: np.ndarray, h_matrix: np.ndarray, m_matrix: np.ndarray,
                        e_matrix: np.ndarray, chosen_check: int) -> None:
    check_node_row = get_row(h_matrix, chosen_check)
    check_row_one = get_one_pos(check_node_row)

    for i_column in check_row_one:
        check_node_column = get_column(h_matrix, int(i_column))
        check_column_one = get_one_pos(check_node_column)
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

def modify_l_matrix(p_matrix: np.ndarray, l_matrix: np.ndarray, h_matrix: np.ndarray, e_matrix: np.ndarray) -> None:
    column_size = h_matrix.shape[1]

    for i_column in range(column_size):
        h_column_slice = get_column(h_matrix, i_column)
        h_column_one_pos = get_one_pos(h_column_slice)
        sum_e = 0.0

        for j_row_e in h_column_one_pos:
            sum_e += e_matrix[j_row_e, i_column]

        l_matrix[i_column] = p_matrix[i_column] + sum_e

def determine_reward(state: np.ndarray) -> float:
    correct_bits = 0

    for i_column in range(np.shape(state)[0]):
        if state[i_column] == 0:
            correct_bits += 1
    
    return correct_bits / float(np.shape(state)[0])

def global_to_local_state_index(h_matrix: np.ndarray, global_state: np.ndarray, check_node: int, max_check_degree: int) -> int:
    check_node_connections = get_one_pos(h_matrix[check_node])
    temp_local_state = np.zeros(max_check_degree, dtype = np.int32)
    counter = max_check_degree - np.shape(check_node_connections)[0]

    for index in check_node_connections:
        temp_local_state[counter] = global_state[index]
        counter += 1

    return int("".join(temp_local_state.astype(str)), 2)

def update_q_table(h_matrix: np.ndarray, q_table: np.ndarray, current_state: np.ndarray, next_state: np.ndarray, current_action: int, max_check_degree: int, alpha: float, beta: float, reward: float) -> None:
    current_state_index = global_to_local_state_index(h_matrix, current_state, current_action, max_check_degree)
    next_state_index = global_to_local_state_index(h_matrix, next_state, current_action, max_check_degree)
    next_action = action_choice(h_matrix, next_state, q_table, -1)
    future_action_value = q_table[next_action][next_state_index]

    q_table[current_action][current_state_index] = (1 - alpha) * q_table[current_action][current_state_index] + alpha * (reward + beta * future_action_value)

def reinforcement_learning_training(training_llrs: np.ndarray, h_matrix: np.ndarray) -> np.ndarray:
    # Q table will be filled out throughout the whole training
    max_check_degree = highest_check_degree(h_matrix)
    state_space_size = 2 ** max_check_degree
    action_space_size = np.shape(h_matrix)[0]

    # The states are the bits the check node "can see", and the actions are all check nodes
    q_table = np.zeros((action_space_size, state_space_size) , dtype = np.float64)

    iteration_max = 50
    learning_rate = 0.1
    reward_discount_rate = 0.9
    exploration_probability = 0.6
    reward = 0.0

    for training_llr in training_llrs:
        # Basic matrices and a counter to start each episode of training
        p_matrix = training_llr
        l_matrix = np.zeros(np.shape(p_matrix), dtype = np.float64)
        z_matrix = np.zeros(np.shape(p_matrix), dtype = np.int32)
        m_matrix = np.zeros(np.shape(h_matrix), dtype = np.float64)
        e_matrix = np.zeros(np.shape(h_matrix), dtype = np.float64)

        # IMPORTANT: For comparing to a flood scheduler: # global iterations = (# of iterations) * (1 / # of check nodes)
        iteration_count = 0

        # Initial states of check nodes
        initiate_m_matrix(p_matrix, h_matrix, m_matrix)

        # Modifying Z for an action for the first iteration
        modify_z_matrix(p_matrix, z_matrix)
    
        # Start of an episode
        while iteration_count < iteration_max:
            # Current state before chosen check node propogation
            current_z_state = z_matrix

            # Agent chooses an action based on the state, the table, and exploration
            chosen_check_node = action_choice(h_matrix, z_matrix, q_table, exploration_probability)

            # Propogate message from the chosen check node, the E matrix will now be changed
            sequential_e_matrix(h_matrix, m_matrix, e_matrix, chosen_check_node)

            # Propogate what the variable nodes have to say to the neighbor check nodes
            sequential_m_matrix(p_matrix, h_matrix, m_matrix, e_matrix, chosen_check_node)

            # Modified LLRs
            modify_l_matrix(p_matrix, l_matrix, h_matrix, e_matrix)

            # Turn LLR into 0s and 1s
            modify_z_matrix(l_matrix, z_matrix)

            # Determine reward
            reward = determine_reward(z_matrix)

            # State after chosen check node propogation
            next_z_state = z_matrix

            # Update Q table
            update_q_table(h_matrix, q_table, current_z_state, next_z_state, chosen_check_node, max_check_degree, learning_rate, reward_discount_rate, reward)

            iteration_count += 1

    # Return optimized policy / Q table
    return q_table

# test_bits = np.zeros(128, dtype = np.int32)
# test_snr = np.array([1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0], dtype = np.float64)
# true_training = training_set(test_bits, test_snr, 180, 4./6)
# true_h_matrix = return_h_matrix("H_Matrix2.txt").astype(np.int32)
# start_time = time.time()
# q_table = reinforcement_learning_training(true_training, true_h_matrix)
# end_time = time.time()
# print(q_table)
# print("Time for execution: ", end_time - start_time, "seconds")

# # Tracker for small H matrix: For training set sizes: (180 llrs, 3 seconds), (1800 llrs, 30 seconds), (18000 llrs, 250 seconds)
# # Tracker for big H matrix: For training set sizes: (180 llrs, 20 seconds)
