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
