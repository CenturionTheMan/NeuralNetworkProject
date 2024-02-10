import random
import numpy as np
import xml.etree.ElementTree as ET


__biases_for_layers: list
__weights_for_layers: list
__layers: list
__layers_sizes: list
__epoch_amount: int
__expected_max_error: float
__learning_rate: float


def initialize(layers_sizes: list[int], learning_rate: float = 0.1, expected_max_error: float = 0.001,
               epoch_amount: int = 100):
    """
    This class is used for creating, testing and exploiting standard (classic) Neural Network.
    Its supports network with any amount of layers as well as neurons in each layer.
    Training is being done using backpropagation.
    Set-up neural network

    :param layers_sizes:
        List of layers, each (int) value represents amount of neurons in given layer.
        First layer (index = 0) represent input layer, last (index = len - 1) - output layer.
        List elements in between creates hidden layers
    :param learning_rate:
        It controls rate of learning. Value below 0 will rise an exception.
        It is suggested to do not exceed value 1.
    :param expected_max_error:
        While training MSE (mean square error) is being calculated.
        Learning will stop if value of any training point is lower than this threshold.
    :param epoch_amount:
        This int value controls how many iteration of learning should be performed.
    """

    global __learning_rate, __expected_max_error, __epoch_amount
    global __layers_sizes, __layers, __weights_for_layers, __biases_for_layers

    if len(layers_sizes) < 2:
        raise Exception("There must be no less than 2 layers!")
    if learning_rate < 0:
        raise Exception("Learning rate must be positive value")

    __learning_rate = learning_rate
    __expected_max_error = expected_max_error
    __epoch_amount = epoch_amount

    __layers_sizes = layers_sizes

    # set up layers (fill with zeros)
    __layers = [np.zeros((x, 1)) for x in layers_sizes]

    # declare lists of matrices
    __weights_for_layers = []  # for weights
    __biases_for_layers = []  # for biases

    # set up matrices
    for i in range(len(layers_sizes) - 1):
        current_size = layers_sizes[i]  # from layer (index)
        next_size = layers_sizes[i + 1]  # to layer (index)

        # setup weights matrix between from/to (index) layers
        __weights_for_layers.append(np.array([(x - 0.5) / 2 for x in np.random.rand(next_size, current_size)]))

        # setup biases matrix for (next index - skipping input layer) layers
        __biases_for_layers.append(np.array([(x - 0.5) / 5 for x in np.random.rand(next_size, 1)]))


def load_from_xml_file(file_path: str):
    global __learning_rate, __expected_max_error, __epoch_amount
    global __layers_sizes, __layers, __weights_for_layers, __biases_for_layers

    tree = ET.parse(file_path)
    root = tree.getroot()

    __learning_rate = float(root.find('learning_rate').text)
    __expected_max_error = float(root.find('expected_max_error').text)
    __epoch_amount = int(root.find('epoch_amount').text)

    __layers_sizes = [int(x.text) for x in root.find('layers_sizes').findall('layer_size')]

    # set up layers (fill with zeros)
    __layers = [np.zeros((x, 1)) for x in __layers_sizes]

    # declare lists of matrices
    __weights_for_layers = []  # for weights
    __biases_for_layers = []  # for biases

    for weights_matrix_xml in root.find('all_weights').findall('weights_matrix'):
        shape = (int(weights_matrix_xml.attrib['row_amount']), int(weights_matrix_xml.attrib['column_amount']))
        weights_matrix = np.zeros(shape)
        for weight_xml in weights_matrix_xml.findall('weight'):
            row, column = int(weight_xml.attrib['row_index']), int(weight_xml.attrib['column_index'])
            weights_matrix[row][column] = float(weight_xml.text)
        __weights_for_layers.append(weights_matrix)

    for biases_matrix_xml in root.find('all_biases').findall('biases_matrix'):
        shape = (int(biases_matrix_xml.attrib['row_amount']), 1)
        biases_matrix = np.zeros(shape)
        for bias_xml in biases_matrix_xml.findall('bias'):
            row = int(bias_xml.attrib['row_index'])
            biases_matrix[row][0] = float(bias_xml.text)
        __biases_for_layers.append(biases_matrix)
    print("DONE")


def predict(inputs: list) -> list:
    """
    This function will give prediction about outputs for given input.
    IMPORTANT: inputs should be normalized (between 0 and 1).
    Note: network should be first trained.
    :param inputs: Data point to predict results for
    :return: Neural network predictions
    """
    __feed_forward(inputs)
    raw = __layers[-1].transpose()[0]

    if len(raw) == 1:
        return list(raw)
    else:
        summarized = sum(raw)
        return [x / summarized for x in raw]


def train_with_tuple_data(data: list[tuple[list, list]]) -> None:
    """
    This function will initiate neural network training. This may take a while.
    IMPORTANT: inputs should be normalized (between 0 and 1).
    :param data:
        List of tuples. Each tuple is single training point.
        Each tuple should be formatted as follows:
        Index 0: input layer data
        Index 1: expected output layer data
    """

    for epoch in range(__epoch_amount):
        random.shuffle(data)
        rep_in_epoch = 0
        for single in data:
            input_point = single[0]
            expected = single[1]

            __feed_forward(input_point)

            error = __calculate_mse_cost(expected, __layers[-1])

            if rep_in_epoch % 100 == 0:
                print(f'Epoch: {epoch + 1}\n'
                      f'Epoch percent finish: {round(100 * rep_in_epoch / len(data), 2)}%\n'
                      f'Current mean error: {round(error, 4)}\n')

            if error < __expected_max_error:
                return

            __backpropagation(expected)
            rep_in_epoch += 1


def save_to_xml_file(file_path: str) -> None:
    xml_root = ET.Element('root')

    ET.SubElement(xml_root, 'epoch_amount').text = str(__epoch_amount)
    ET.SubElement(xml_root, 'expected_max_error').text = str(__expected_max_error)
    ET.SubElement(xml_root, 'learning_rate').text = str(__learning_rate)

    xml_layer_sizes = ET.SubElement(xml_root, 'layers_sizes')

    xml_all_weights = ET.SubElement(xml_root, 'all_weights')
    xml_all_biases = ET.SubElement(xml_root, 'all_biases')

    for index in range(len(__layers_sizes)):
        ET.SubElement(xml_layer_sizes, 'layer_size', column_index=f'{index}').text = str(__layers_sizes[index])

    for index in range(len(__weights_for_layers)):
        weights = __weights_for_layers[index]
        xml_single_weights_matrix = ET.SubElement(xml_all_weights, 'weights_matrix',
                                                  index=f'{index}',
                                                  row_amount=f'{weights.shape[0]}',
                                                  column_amount=f'{weights.shape[1]}')
        for row in range(weights.shape[0]):
            for column in range(weights.shape[1]):
                ET.SubElement(xml_single_weights_matrix, 'weight',
                              row_index=f'{row}', column_index=f'{column}').text = str(weights[row][column])

    for index in range(len(__biases_for_layers)):
        biases = __biases_for_layers[index]
        xml_single_biases_matrix = ET.SubElement(xml_all_biases, 'biases_matrix',
                                                 index=f'{index}',
                                                 row_amount=f'{biases.shape[0]}')
        for bias_index in range(len(biases)):
            ET.SubElement(xml_single_biases_matrix, 'bias',
                          row_index=f'{bias_index}').text = str(biases[bias_index][0])

    tree = ET.ElementTree(xml_root)
    ET.indent(tree, space="\t", level=0)
    tree.write(file_path, encoding='UTF-8', xml_declaration=True)


def __feed_forward(inputs: list):
    if np.shape(inputs) != (__layers_sizes[0],):
        raise Exception(f'Wrong input array size! Should be {(__layers_sizes[0],)} and was {np.shape(inputs)}')

    # assign input layer values
    __layers[0] = np.array(inputs).reshape(len(inputs), 1)

    # calculate values across layers, skipping layer at index 0 (input layer)
    for index in range(len(__layers) - 1):
        # multiply layer's 'i' weights with its values
        multiplied_by_weights_layer = np.matmul(__weights_for_layers[index], __layers[index])

        # add biases
        layer_with_added_biases = np.add(multiplied_by_weights_layer, __biases_for_layers[index])

        # apply activation function
        activated_layer = __sigmoid(layer_with_added_biases)

        # save results in next layer
        __layers[index + 1] = activated_layer


def __backpropagation(expected_results: list):
    if np.shape(expected_results) != (__layers_sizes[-1],):
        raise Exception(f'Wrong result array size! Should be {(__layers_sizes[-1],)} and was '
                        f'{np.shape(expected_results)}')

    # Preparing expected_results list
    expected_results_transposed = np.array(expected_results).reshape(len(expected_results), 1)

    # error matrix initialized with output layer error
    errors_matrix = expected_results_transposed - __layers[-1]

    # for each weight / bias matrix
    for index in reversed(range(len(__weights_for_layers))):
        # get each layer weighted input in derivative of activation function
        # since derivative of sigmoid is sig(x) * (1 - sig(x)) and layer are
        # already passed through sig(x) function only (1 - sig(x)) correction is needed.
        # Have that in mind if activation function should be changed.
        sigmoid_derivative_layer = __layers[index + 1] * (1 - __layers[index + 1])

        # calculate gradient
        gradient_matrix = sigmoid_derivative_layer * errors_matrix * __learning_rate

        # calculate matrix with delta weights (values to change weights in given layer)
        delta_weights_matrix = np.matmul(gradient_matrix, __layers[index].transpose())

        # adjust weights and biases
        __weights_for_layers[index] = __weights_for_layers[index] + delta_weights_matrix
        __biases_for_layers[index] = __biases_for_layers[index] + gradient_matrix

        # calculate error for next layer in respect for its weight
        errors_matrix = np.matmul(__weights_for_layers[index].transpose(), errors_matrix)


def __calculate_mse_cost(expected_values, real_values):
    val_sum = 0
    for expected, real in zip(expected_values, real_values):
        val_sum = val_sum + pow(expected - real[0], 2)
    return val_sum / len(real_values)


def __sigmoid(x):
    return 1 / (1 + np.exp(-x))


def __sigmoid_derivative(x):
    sig = __sigmoid(x)
    return sig * (1 - sig)
