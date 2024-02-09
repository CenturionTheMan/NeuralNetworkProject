import random
import numpy as np
import xml.etree.ElementTree as ET
import os


class NeuralNetwork:
    def __init__(self):
        """
        This class is used for creating, testing and exploiting standard (classic) Neural Network.
        Its supports network with any amount of layers as well as neurons in each layer.
        Training is being done using backpropagation.
        """
        self.__biases_for_layers = None
        self.__weights_for_layers = None
        self.__layers = None
        self.__layers_sizes = None
        self.__epoch_amount = None
        self.__expected_max_error = None
        self.__learning_rate = None

    def create(self, layers_sizes: list[int], learning_rate: float = 0.1, expected_max_error: float = 0.001,
               epoch_amount: int = 100):
        """
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

        if len(layers_sizes) < 2:
            raise Exception("There must be no less than 2 layers!")
        if learning_rate < 0:
            raise Exception("Learning rate must be positive value")

        self.__learning_rate = learning_rate
        self.__expected_max_error = expected_max_error
        self.__epoch_amount = epoch_amount

        self.__layers_sizes = layers_sizes

        # set up layers (fill with zeros)
        self.__layers = [np.zeros((x, 1)) for x in layers_sizes]

        # declare lists of matrices
        self.__weights_for_layers = []  # for weights
        self.__biases_for_layers = []  # for biases

        # set up matrices
        for i in range(len(layers_sizes) - 1):
            current_size = layers_sizes[i]  # from layer (index)
            next_size = layers_sizes[i + 1]  # to layer (index)

            # setup weights matrix between from/to (index) layers
            self.__weights_for_layers.append(np.array([(x - 0.5) / 2 for x in np.random.rand(next_size, current_size)]))

            # setup biases matrix for (next index - skipping input layer) layers
            self.__biases_for_layers.append(np.array([(x - 0.5) / 5 for x in np.random.rand(next_size, 1)]))

    def create_from_xml_file(self, file_path: str):
        tree = ET.parse(file_path)
        root = tree.getroot()

        self.__learning_rate = float(root.find('learning_rate').text)
        self.__expected_max_error = float(root.find('expected_max_error').text)
        self.__epoch_amount = int(root.find('epoch_amount').text)

        self.__layers_sizes = [int(x.text) for x in root.find('layers_sizes').findall('layer_size')]

        # set up layers (fill with zeros)
        self.__layers = [np.zeros((x, 1)) for x in self.__layers_sizes]

        # declare lists of matrices
        self.__weights_for_layers = []  # for weights
        self.__biases_for_layers = []  # for biases

        for weights_matrix_xml in root.find('all_weights').findall('weights_matrix'):
            shape = (int(weights_matrix_xml.attrib['row_amount']), int(weights_matrix_xml.attrib['column_amount']))
            weights_matrix = np.zeros(shape)
            for weight_xml in weights_matrix_xml.findall('weight'):
                row, column = int(weight_xml.attrib['row_index']), int(weight_xml.attrib['column_index'])
                weights_matrix[row][column] = float(weight_xml.text)
            self.__weights_for_layers.append(weights_matrix)

        for biases_matrix_xml in root.find('all_biases').findall('biases_matrix'):
            shape = (int(biases_matrix_xml.attrib['row_amount']), 1)
            biases_matrix = np.zeros(shape)
            for bias_xml in biases_matrix_xml.findall('bias'):
                row = int(bias_xml.attrib['row_index'])
                biases_matrix[row][0] = float(bias_xml.text)
            self.__biases_for_layers.append(biases_matrix)
        print("DONE")

    def predict(self, inputs: list) -> list:
        """
        This function will give prediction about outputs for given input.
        IMPORTANT: inputs should be normalized (between 0 and 1).
        Note: network should be first trained.
        :param inputs: Data point to predict results for
        :return: Neural network predictions
        """
        self.__feed_forward(inputs)
        raw = self.__layers[-1].transpose()[0]

        if len(raw) == 1:
            return list(raw)
        else:
            summarized = sum(raw)
            return [x / summarized for x in raw]

    def train_with_tuple_data(self, data: list[tuple[list, list]]) -> None:
        """
        This function will initiate neural network training. This may take a while.
        IMPORTANT: inputs should be normalized (between 0 and 1).
        :param data:
            List of tuples. Each tuple is single training point.
            Each tuple should be formatted as follows:
            Index 0: input layer data
            Index 1: expected output layer data
        """

        for epoch in range(self.__epoch_amount):
            random.shuffle(data)
            rep_in_epoch = 0
            for single in data:
                input_point = single[0]
                expected = single[1]

                self.__feed_forward(input_point)

                error = calculate_mse_cost(expected, self.__layers[-1])

                if rep_in_epoch % 100 == 0:
                    print(f'Epoch: {epoch + 1}\n'
                          f'Epoch percent finish: {round(100 * rep_in_epoch / len(data), 2)}%\n'
                          f'Current mean error: {round(error, 4)}\n')

                if error < self.__expected_max_error:
                    return

                self.__backpropagation(expected)
                rep_in_epoch += 1

    def train(self, inputs: list[list], expected_results: list[list]) -> None:
        """
        This function will initiate neural network training. This may take a while.
        IMPORTANT: inputs should be normalized (between 0 and 1).
        :param inputs: List of data points. Each data point contains values for input layer.
        :param expected_results:
            List of expected results for given data points.
            Each entry contains expected values for output layer.
            Indexes must match with inputs list data.
        """
        if len(inputs) != len(expected_results):
            raise Exception(f"Inputs length was {len(inputs)} and expected results length was {len(expected_results)}!"
                            f" Those values must be the same.")

        for epoch in range(self.__epoch_amount):
            for x in range(len(inputs)):
                current_point = random.randint(0, len(inputs) - 1)
                self.__feed_forward(inputs[current_point])
                error = calculate_mse_cost(expected_results, self.__layers[-1])
                print(f'Epoch: {epoch}\nCurrent mean error: {round(error, 3)}\n')
                if error < self.__expected_max_error:
                    return
                self.__backpropagation(expected_results[current_point])

    def save_to_xml_file(self, file_path: str) -> bool:
        xml_root = ET.Element('root')

        ET.SubElement(xml_root, 'epoch_amount').text = str(self.__epoch_amount)
        ET.SubElement(xml_root, 'expected_max_error').text = str(self.__expected_max_error)
        ET.SubElement(xml_root, 'learning_rate').text = str(self.__learning_rate)

        xml_layer_sizes = ET.SubElement(xml_root, 'layers_sizes')

        xml_all_weights = ET.SubElement(xml_root, 'all_weights')
        xml_all_biases = ET.SubElement(xml_root, 'all_biases')

        for index in range(len(self.__layers_sizes)):
            ET.SubElement(xml_layer_sizes, 'layer_size', column_index=f'{index}').text = str(self.__layers_sizes[index])

        for index in range(len(self.__weights_for_layers)):
            weights = self.__weights_for_layers[index]
            xml_single_weights_matrix = ET.SubElement(xml_all_weights, 'weights_matrix',
                                                      index=f'{index}',
                                                      row_amount=f'{weights.shape[0]}',
                                                      column_amount=f'{weights.shape[1]}')
            for row in range(weights.shape[0]):
                for column in range(weights.shape[1]):
                    ET.SubElement(xml_single_weights_matrix, 'weight',
                                  row_index=f'{row}', column_index=f'{column}').text = str(weights[row][column])

        for index in range(len(self.__biases_for_layers)):
            biases = self.__biases_for_layers[index]
            xml_single_biases_matrix = ET.SubElement(xml_all_biases, 'biases_matrix',
                                                     index=f'{index}',
                                                     row_amount=f'{biases.shape[0]}')
            for bias_index in range(len(biases)):
                ET.SubElement(xml_single_biases_matrix, 'bias',
                              row_index=f'{bias_index}').text = str(biases[bias_index][0])

        tree = ET.ElementTree(xml_root)
        ET.indent(tree, space="\t", level=0)
        tree.write(file_path, encoding='UTF-8', xml_declaration=True)

    def __feed_forward(self, inputs: list):
        if np.shape(inputs) != (self.__layers_sizes[0],):
            raise Exception(f'Wrong input array size! Should be {(self.__layers_sizes[0],)} and was {np.shape(inputs)}')

        # assign input layer values
        self.__layers[0] = np.array(inputs).reshape(len(inputs), 1)

        # calculate values across layers, skipping layer at index 0 (input layer)
        for index in range(len(self.__layers) - 1):
            # multiply layer's 'i' weights with its values
            multiplied_by_weights_layer = np.matmul(self.__weights_for_layers[index], self.__layers[index])

            # add biases
            layer_with_added_biases = np.add(multiplied_by_weights_layer, self.__biases_for_layers[index])

            # apply activation function
            activated_layer = sigmoid(layer_with_added_biases)

            # save results in next layer
            self.__layers[index + 1] = activated_layer

    def __backpropagation(self, expected_results: list):
        if np.shape(expected_results) != (self.__layers_sizes[-1],):
            raise Exception(f'Wrong result array size! Should be {(self.__layers_sizes[-1],)} and was '
                            f'{np.shape(expected_results)}')

        # Preparing expected_results list
        expected_results_transposed = np.array(expected_results).reshape(len(expected_results), 1)

        # error matrix initialized with output layer error
        errors_matrix = expected_results_transposed - self.__layers[-1]

        # for each weight / bias matrix
        for index in reversed(range(len(self.__weights_for_layers))):
            # get each layer weighted input in derivative of activation function
            # since derivative of sigmoid is sig(x) * (1 - sig(x)) and layer are
            # already passed through sig(x) function only (1 - sig(x)) correction is needed.
            # Have that in mind if activation function should be changed.
            sigmoid_derivative_layer = self.__layers[index + 1] * (1 - self.__layers[index + 1])

            # calculate gradient
            gradient_matrix = sigmoid_derivative_layer * errors_matrix * self.__learning_rate

            # calculate matrix with delta weights (values to change weights in given layer)
            delta_weights_matrix = np.matmul(gradient_matrix, self.__layers[index].transpose())

            # adjust weights and biases
            self.__weights_for_layers[index] = self.__weights_for_layers[index] + delta_weights_matrix
            self.__biases_for_layers[index] = self.__biases_for_layers[index] + gradient_matrix

            # calculate error for next layer in respect for its weight
            errors_matrix = np.matmul(self.__weights_for_layers[index].transpose(), errors_matrix)


def calculate_mse_cost(expected_values, real_values):
    val_sum = 0
    for expected, real in zip(expected_values, real_values):
        val_sum = val_sum + pow(expected - real[0], 2)
    return val_sum / len(real_values)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)
