import math
import random
import numpy as np
import xml.etree.ElementTree as ET


__biases_for_layers: list
__weights_for_layers: list
__layers_before_activation: list
__layers_sizes: list
__learning_rate: float
__activation_function: str


def initialize(*layers_sizes: int, activation_function: str = 'ReLU'):
    """
    This class is used for creating, testing and exploiting standard (classic) Neural Network.
    Its supports network with any amount of layers as well as neurons in each layer.
    Training is being done using backpropagation.
    Set-up neural network

    :param activation_function:
        Activation function used for layers. Every layer shall be activated by it except last one which is activated
        using softmax. 'ReLU' and 'sigmoid' are supported.

    :param layers_sizes:
        List of layers, each (int) value represents amount of neurons in given layer.
        First layer (index = 0) represent input layer, last (index = len - 1) - output layer.
        List elements in between creates hidden layers
    """

    global __layers_sizes, __layers_before_activation, __weights_for_layers, __biases_for_layers, __activation_function

    if len(layers_sizes) < 2:
        raise Exception("There must be no less than 2 layers!")

    activation_function = activation_function.lower()
    if not (activation_function == 'relu' or activation_function == 'sigmoid'):
        raise Exception("Provided activation function is not supported!")

    __layers_sizes = layers_sizes
    __activation_function = activation_function

    # set up layers (fill with zeros)
    __layers_before_activation = [np.zeros((x, 1)) for x in layers_sizes]

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
    global __layers_sizes, __layers_before_activation, __weights_for_layers, __biases_for_layers, __activation_function

    tree = ET.parse(file_path)
    root = tree.getroot()

    __layers_sizes = [int(x.text) for x in root.find('layers_sizes').findall('layer_size')]

    __activation_function = root.find('activation_function').text

    # set up layers (fill with zeros)
    __layers_before_activation = [np.zeros((x, 1)) for x in __layers_sizes]

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
    predictions = __feed_forward(inputs)
    raw = predictions.transpose()[0]

    return list(raw)


def train_with_mini_batch_gradient_descent(data: list[tuple[list, list]], learning_rate: float = 0.1,
                                           epoch_amount: int = 40, batch_size: int = 10,
                                           expected_max_error: float = 0.01) -> None:
    """
        This function will initiate neural network training. This may take a while.
        IMPORTANT: inputs should be normalized (between 0 and 1).
        :param data:
            List of tuples. Each tuple is single training point.
            Each tuple should be formatted as follows:
            Index 0: input layer data
            Index 1: expected output layer data
        :param learning_rate:
            It controls rate of learning. Value below 0 will rise an exception.
            It is suggested to do not exceed value 1.
        :param epoch_amount:
            This int value controls how many iteration of learning should be performed.
        :param batch_size
            The size of single batch
        :param expected_max_error:
            While training MSE (mean square error) is being calculated.
            Learning will stop if value of any training point is lower than this threshold.
        """
    global __learning_rate

    if learning_rate < 0:
        raise Exception("Learning rate must be positive value")
    if batch_size > len(data):
        raise Exception('Batch size must be smaller than data length')

    __learning_rate = learning_rate

    for epoch in range(epoch_amount):
        random.shuffle(data)
        batch_begin_index = 0

        while batch_begin_index < len(data):
            if batch_begin_index + batch_size < len(data):
                batch_samples = data[batch_begin_index:batch_begin_index+batch_size]
            else:
                batch_samples = data[batch_begin_index:]

            input_points = [x[0] for x in batch_samples]
            expected_points = [x[1] for x in batch_samples]

            error = __perform_learning_iteration(input_points, expected_points)

            print(f'Epoch: {epoch + 1}\n'
                  f'Epoch percent finish: {round(100 * batch_begin_index / len(data), 2)}%\n'
                  f'Batch error: {round(error, 4)}\n')

            if error <= expected_max_error:
                return

            batch_begin_index += batch_size


def save_to_xml_file(file_path: str) -> None:
    xml_root = ET.Element('root')

    xml_layer_sizes = ET.SubElement(xml_root, 'layers_sizes')

    xml_all_weights = ET.SubElement(xml_root, 'all_weights')
    xml_all_biases = ET.SubElement(xml_root, 'all_biases')

    for index in range(len(__layers_sizes)):
        ET.SubElement(xml_layer_sizes, 'layer_size', column_index=f'{index}').text = str(__layers_sizes[index])

    ET.SubElement(xml_root, 'activation_function').text = __activation_function

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


def __feed_forward(inputs: list) -> np.array:
    if np.shape(inputs) != (__layers_sizes[0],) and np.shape(inputs) != (__layers_sizes[0], 1):
        raise Exception(f'Wrong input array size! Should be {(__layers_sizes[0],)} or {(__layers_sizes[0], 1)} '
                        f'and was {np.shape(inputs)}')

    # assign input layer values
    __layers_before_activation[0] = np.array(inputs).reshape(len(inputs), 1)
    current_layer_value = __layers_before_activation[0]

    # calculate values across layers, skipping layer at index 0 (input layer)
    for index in range(len(__layers_before_activation) - 1):
        # multiply layer weights with its values
        multiplied_by_weights_layer = np.matmul(__weights_for_layers[index], current_layer_value)

        # add biases
        layer_with_added_biases = np.add(multiplied_by_weights_layer, __biases_for_layers[index])

        # apply activation function
        if index == len(__layers_before_activation) - 2:
            activated_layer = __softmax(layer_with_added_biases)
        elif __activation_function == 'sigmoid':
            activated_layer = __sigmoid(layer_with_added_biases)
        elif __activation_function == 'relu':
            activated_layer = __ReLU(layer_with_added_biases)
        else:
            raise Exception('Not supported activation function!')

        # save results in next layer
        __layers_before_activation[index + 1] = layer_with_added_biases
        current_layer_value = activated_layer
    return current_layer_value


def __backpropagation(expected_results: list, predictions: list):
    if np.shape(expected_results) != (__layers_sizes[-1],):
        raise Exception(f'Wrong result array size! Should be {(__layers_sizes[-1],)} and was '
                        f'{np.shape(expected_results)}')

    # Preparing expected_results list
    expected_results_transposed = np.array(expected_results).reshape(len(expected_results), 1)

    # error matrix initialized with output layer error
    errors_matrix = expected_results_transposed - predictions

    change_for_weights = [np.array([]) for x in range(len(__weights_for_layers))]
    change_for_biases = [np.array([]) for x in range(len(__biases_for_layers))]

    # for each weight / bias matrix
    for index in reversed(range(len(__weights_for_layers))):
        # get each layer weighted input in derivative of activation function
        if index == len(__weights_for_layers) - 1:
            activation_derivative_layer = __softmax_derivative(__layers_before_activation[index + 1])
        elif __activation_function == 'sigmoid':
            activation_derivative_layer = __sigmoid_derivative(__layers_before_activation[index + 1])
        elif __activation_function == 'relu':
            activation_derivative_layer = __ReLU_derivative(__layers_before_activation[index + 1])
        else:
            raise Exception('Not supported activation function!')

        # calculate gradient
        gradient_matrix = activation_derivative_layer * errors_matrix * __learning_rate

        # calculate matrix with delta weights (values to change weights in given layer)
        # NOTE: should be multiplied by unactivated layers??
        delta_weights_matrix = np.matmul(gradient_matrix, __layers_before_activation[index].transpose())

        # adjust weights and biases
        change_for_weights[index] = delta_weights_matrix
        change_for_biases[index] = gradient_matrix

        # calculate error for next layer in respect for its weight
        errors_matrix = np.matmul(__weights_for_layers[index].transpose(), errors_matrix)
    return change_for_weights, change_for_biases


def __perform_learning_iteration(data_samples: list, expected_results: list):
    global __weights_for_layers, __biases_for_layers

    all_change_for_weights = [[] for x in range(len(__weights_for_layers))]
    all_change_for_biases = [[] for x in range(len(__biases_for_layers))]
    error_sum = 0
    for data_sample, expected_result in zip(data_samples, expected_results):
        predictions = __feed_forward(data_sample)
        change_for_weights, change_for_biases = __backpropagation(expected_result, predictions)

        error_sum += __calculate_cross_entropy_cost(expected_result, predictions)

        for index in range(len(__weights_for_layers)):
            all_change_for_weights[index].append(change_for_weights[index])
            all_change_for_biases[index].append(change_for_biases[index])

    for index in range(len(__weights_for_layers)):
        delta_weights = np.mean(all_change_for_weights[index], axis=0)
        delta_biases = np.mean(all_change_for_biases[index], axis=0)
        __weights_for_layers[index] = __weights_for_layers[index] + delta_weights
        __biases_for_layers[index] = __biases_for_layers[index] + delta_biases

    return error_sum/len(data_samples)


def __calculate_cross_entropy_cost(expected_values, real_values):
    val_sum = 0
    for expected, real in zip(expected_values, real_values):
        val_sum += expected * math.log(real)
    return -val_sum


def __softmax(x):
    tmp = np.exp(x)
    return tmp / np.sum(tmp)


def __sigmoid(x):
    return 1 / (1 + np.exp(-x))


def __ReLU(x):
    return x * (x > 0)


def __sigmoid_derivative(x):
    sig = __sigmoid(x)
    return sig * (1 - sig)


def __softmax_derivative(x):
    tmp = __softmax(x)
    return tmp * (1 - tmp)


def __ReLU_derivative(x):
    return 1. * (x >= 0)
