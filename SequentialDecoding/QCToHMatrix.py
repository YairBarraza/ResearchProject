import numpy as np
# When you print an numpy matrix it might condense it to make it readable
# This makes it so it never condenses
np.set_printoptions(threshold = np.inf)

def create_matrix(offset: int, size: int) -> np.ndarray:
    returned_matrix = np.zeros((size, size), dtype = int)

    if offset == -1:
        return returned_matrix

    for j in range(size):
        i = (j + offset) % size
        returned_matrix[j, i] = 1

    return returned_matrix


def add_two_matrices(matrix_1: np.ndarray, matrix_2: np.ndarray) -> np.ndarray:
    matrix_1_size = np.shape(matrix_1)
    matrix_2_size = np.shape(matrix_2)

    if (matrix_1_size[0] != matrix_2_size[0]) or (matrix_1_size[1] != matrix_2_size[1]):
        raise ValueError("Function (add_two_matrices): Matrices must match sizes...")

    returned_matrix = np.zeros(matrix_1_size, dtype = int)

    for j in range(matrix_1_size[0]):
        for i in range(matrix_1_size[0]):
            returned_matrix[j, i] = (matrix_1[j, i] + matrix_2[j, i]) % 2

    return returned_matrix


def conjoin_matrix_sides(matrix_1: np.ndarray, matrix_2: np.ndarray) -> np.ndarray:
    matrix_1_size = np.shape(matrix_1)
    matrix_2_size = np.shape(matrix_2)

    if matrix_1_size[0] != matrix_2_size[0]:
        raise ValueError("Function (conjoin_matrix_sides): Matrices must match row sizes...")

    rows = matrix_1_size[0]
    columns = matrix_1_size[1] + matrix_2_size[1]
    returned_matrix = np.zeros((rows, columns), dtype = int)

    for j in range(rows):
        for i in range(columns):
            if (i - matrix_1_size[1]) < 0:
                returned_matrix[j, i] = matrix_1[j, i]
            else:
                returned_matrix[j, i] = matrix_2[j, i - matrix_1_size[1]]

    return returned_matrix

def conjoin_matrix_tops(matrix_1: np.ndarray, matrix_2: np.ndarray) -> np.ndarray:
    matrix_1_size = np.shape(matrix_1)
    matrix_2_size = np.shape(matrix_2)

    if matrix_1_size[1] != matrix_2_size[1]:
        raise ValueError("Function (conjoin_matrix_tops): Matrices must match column sizes...")

    rows = matrix_1_size[0] + matrix_2_size[0]
    columns = matrix_1_size[1]
    returned_matrix = np.zeros((rows, columns), dtype = int)

    for i in range(columns):
        for j in range(rows):
            if (j - matrix_1_size[0]) < 0:
                returned_matrix[j, i] = matrix_1[j, i]
            else:
                returned_matrix[j, i] = matrix_2[j - matrix_1_size[0], i]

    return returned_matrix


def user_input_sequence(user_in_string: str) -> np.ndarray:
    user_in_string_list = user_in_string.split(", ")

    if len(user_in_string_list) > 2 or len(user_in_string_list) <= 0:
        raise ValueError("Function (user_input_sequence): Must only be 1 or 2 #s...")

    user_in_matrix = np.zeros([len(user_in_string_list)], dtype = int)

    for index_num in range(len(user_in_string_list)):
        user_in_matrix[index_num] = int(user_in_string_list[index_num])

    print()

    return user_in_matrix


def create_qc_matrix(super_rows: int, super_columns: int) -> np.ndarray:
    qc_matrix = np.zeros((super_rows, super_columns), dtype = np.ndarray)
    print("Input your QC values...")
    print("Example: 0  Example: 0, 2\n")

    for super_j in range(super_rows):
        for super_i in range(super_columns):
            input_string = input(f"Insert sub - matrix at index ({super_j}, {super_i}): ")
            input_matrix = user_input_sequence(input_string)
            qc_matrix[super_j, super_i] = input_matrix

    return qc_matrix


def create_h_matrix_from_qc() -> np.ndarray:
    super_rows = int(input("Input your QC row amount: "))
    super_columns = int(input("Input your QC column amount: "))
    sub_matrix_size = int(input("Input your sub matrix size: "))

    qc_matrix = create_qc_matrix(super_rows, super_columns)

    storage_for_matrices = np.zeros([super_rows, super_columns], dtype = np.ndarray)

    for j in range(super_rows):
        for i in range(super_columns):
            current_qc = qc_matrix[j, i]

            if len(current_qc) > 2 or len(current_qc) <= 0:
                raise ValueError("Function (user_input_sequence): Must only be 1 or 2 #s...")
            elif len(current_qc) == 1:
                new_matrix = create_matrix(current_qc[0], sub_matrix_size)
                storage_for_matrices[j, i] = new_matrix
            else:
                matrix_one = create_matrix(current_qc[0], sub_matrix_size)
                matrix_two = create_matrix(current_qc[1], sub_matrix_size)
                new_matrix = add_two_matrices(matrix_one, matrix_two)
                storage_for_matrices[j, i] = new_matrix

    storage_for_row_matrices = np.zeros(super_rows, dtype = np.ndarray)

    for j in range(super_rows):
        for i in range(super_columns):
            if not (i + 1 == super_columns):
                current_matrix = storage_for_matrices[j, i]
                next_matrix = storage_for_matrices[j, i + 1]
                storage_for_matrices[j, i + 1] = conjoin_matrix_sides(current_matrix, next_matrix)
            else:
                storage_for_row_matrices[j] = storage_for_matrices[j, i]

    last_matrix = np.ndarray

    for j in range(super_rows):
        if not (j + 1 == super_rows):
            current_matrix = storage_for_row_matrices[j]
            next_matrix = storage_for_row_matrices[j + 1]
            storage_for_row_matrices[j + 1] = conjoin_matrix_tops(current_matrix, next_matrix)
        else:
            last_matrix = storage_for_row_matrices[j]

    string_matrix = np.empty(super_rows * sub_matrix_size, dtype = object)

    for j in range(super_rows * sub_matrix_size):
        string_row = ""
        for i in range(super_columns * sub_matrix_size):
            string_row += str(last_matrix[j, i])

        if j + 1 != super_rows * sub_matrix_size:
            string_matrix[j] = string_row + "\n"
        else:
            string_matrix[j] = string_row

    return string_matrix


def qc_to_h_matrix() -> None:
    string_matrix = create_h_matrix_from_qc()
    file_path = "QCToMatrixWriter.txt"
    string_list = string_matrix.tolist()

    try:
        with open(file_path, 'w') as file:
            for line in string_list:
                file.write(line)
        file.close()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found [qc_to_h_matrix()].")

    except Exception as e:
        print(f"An error appeared [qc_to_h_matrix()]: {e}")

#qc_to_h_matrix()
