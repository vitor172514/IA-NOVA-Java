import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

public class NOVA {

    // Classe interna estática para operações de matriz
    static class Matrix {
        double[][] data;
        int rows, cols;

        public Matrix(int rows, int cols) {
            this.rows = rows;
            this.cols = cols;
            data = new double[rows][cols];
        }

        public static Matrix fromArray(double[] arr) {
            Matrix m = new Matrix(arr.length, 1);
            for (int i = 0; i < arr.length; i++) {
                m.data[i][0] = arr[i];
            }
            return m;
        }

        public double[] toArray() {
            double[] arr = new double[rows * cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    arr[i * cols + j] = data[i][j];
                }
            }
            return arr;
        }

        public void randomize() {
            Random rand = new Random();
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    data[i][j] = rand.nextDouble() * 2 - 1; // Valores entre -1 e 1
                }
            }
        }

        public Matrix add(Matrix other) {
            if (rows != other.rows || cols != other.cols) {
                throw new IllegalArgumentException("Dimensões de matrizes incompatíveis para adição.");
            }
            Matrix result = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result.data[i][j] = data[i][j] + other.data[i][j];
                }
            }
            return result;
        }

        public Matrix subtract(Matrix other) {
            if (rows != other.rows || cols != other.cols) {
                throw new IllegalArgumentException("Dimensões de matrizes incompatíveis para subtração.");
            }
            Matrix result = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result.data[i][j] = data[i][j] - other.data[i][j];
                }
            }
            return result;
        }

        public Matrix multiply(Matrix other) {
            if (cols != other.rows) {
                throw new IllegalArgumentException("Dimensões de matrizes incompatíveis para multiplicação.");
            }
            Matrix result = new Matrix(rows, other.cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < other.cols; j++) {
                    double sum = 0;
                    for (int k = 0; k < cols; k++) {
                        sum += data[i][k] * other.data[k][j];
                    }
                    result.data[i][j] = sum;
                }
            }
            return result;
        }

        public Matrix multiply(double scalar) {
            Matrix result = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result.data[i][j] = data[i][j] * scalar;
                }
            }
            return result;
        }

        public Matrix transpose() {
            Matrix result = new Matrix(cols, rows);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result.data[j][i] = data[i][j];
                }
            }
            return result;
        }

        public Matrix applyFunction(Function<Double, Double> func) {
            Matrix result = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result.data[i][j] = func.apply(data[i][j]);
                }
            }
            return result;
        }

        public static Matrix hadamard(Matrix a, Matrix b) {
            if (a.rows != b.rows || a.cols != b.cols) {
                throw new IllegalArgumentException("Dimensões de matrizes incompatíveis para produto de Hadamard.");
            }
            Matrix result = new Matrix(a.rows, a.cols);
            for (int i = 0; i < a.rows; i++) {
                for (int j = 0; j < a.cols; j++) {
                    result.data[i][j] = a.data[i][j] * b.data[i][j];
                }
            }
            return result;
        }

        public void print() {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    System.out.print(data[i][j] + " ");
                }
                System.out.println();
            }
        }
    }

    // Classe interna estática para funções de ativação
    static enum ActivationFunction {
        SIGMOID(
            x -> 1 / (1 + Math.exp(-x)),
            y -> y * (1 - y) // Derivada da Sigmoid, onde y é a saída da função
        ),
        RELU(
            x -> Math.max(0, x),
            y -> (y > 0) ? 1.0 : 0.0 // Derivada da ReLU, onde y é a saída da função
        ),
        TANH(
            x -> Math.tanh(x),
            y -> 1 - (y * y) // Derivada da Tanh, onde y é a saída da função
        );

        private final Function<Double, Double> activation;
        private final Function<Double, Double> derivative;

        ActivationFunction(Function<Double, Double> activation, Function<Double, Double> derivative) {
            this.activation = activation;
            this.derivative = derivative;
        }

        public Function<Double, Double> getActivation() {
            return activation;
        }

        public Function<Double, Double> getDerivative() {
            return derivative;
        }
    }

    // Classe interna estática para uma camada da rede neural
    static class Layer {
        Matrix weights;
        Matrix bias;
        ActivationFunction activationFunction;

        Matrix input;
        Matrix output;

        public Layer(int inputNodes, int outputNodes, ActivationFunction activationFunction) {
            this.weights = new Matrix(outputNodes, inputNodes);
            this.weights.randomize();
            this.bias = new Matrix(outputNodes, 1);
            this.bias.randomize();
            this.activationFunction = activationFunction;
        }

        public Matrix feedForward(Matrix input) {
            this.input = input;
            Matrix weightedSum = weights.multiply(input).add(bias);
            this.output = weightedSum.applyFunction(activationFunction.getActivation());
            return this.output;
        }
    }

    // Classe principal da Rede Neural
    private List<Layer> layers;
    private double learningRate;

    public NeuralNetworkProject(int[] layerSizes, ActivationFunction activationFunction, double learningRate) {
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;

        for (int i = 0; i < layerSizes.length - 1; i++) {
            layers.add(new Layer(layerSizes[i], layerSizes[i+1], activationFunction));
        }
    }

    public Matrix predict(double[] input_array) {
        Matrix input = Matrix.fromArray(input_array);
        Matrix currentOutput = input;

        for (Layer layer : layers) {
            currentOutput = layer.feedForward(currentOutput);
        }
        return currentOutput;
    }

    public void train(double[] input_array, double[] target_array) {
        // 1. Feedforward
        Matrix input = Matrix.fromArray(input_array);
        Matrix currentOutput = input;
        for (Layer layer : layers) {
            currentOutput = layer.feedForward(currentOutput);
        }

        // 2. Calcular o erro da camada de saída
        Matrix targets = Matrix.fromArray(target_array);
        Matrix outputErrors = targets.subtract(currentOutput);

        // 3. Backpropagation
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer currentLayer = layers.get(i);
            Matrix layerOutput = currentLayer.output;

            // Calcular o gradiente da função de ativação
            Matrix gradients = layerOutput.applyFunction(currentLayer.activationFunction.getDerivative());
            gradients = Matrix.hadamard(gradients, outputErrors);
            gradients = gradients.multiply(learningRate);

            // Calcular deltas para pesos e bias
            Matrix weights_deltas = gradients.multiply(currentLayer.input.transpose());

            // Ajustar pesos e bias
            currentLayer.weights = currentLayer.weights.add(weights_deltas);
            currentLayer.bias = currentLayer.bias.add(gradients);

            // Calcular o erro para a próxima camada (anterior)
            Matrix transposedWeights = currentLayer.weights.transpose();
            outputErrors = transposedWeights.multiply(outputErrors);
        }
    }

    public static void main(String[] args) {
        // Exemplo de uso: XOR Gate
        // Definir a arquitetura da rede: 2 entradas, 4 neurônios na camada oculta, 1 saída
        int[] layerSizes = {2, 4, 1};
        NOVA nn = new NOVA(layerSizes, ActivationFunction.SIGMOID, 0.1);

        // Dados de treinamento para a porta XOR
        double[][] training_data = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        double[][] training_targets = {
            {0},
            {1},
            {1},
            {0}
        };

        // Treinar a rede
        System.out.println("Treinando a Rede Neural para XOR...");
        for (int i = 0; i < 50000; i++) { // 50.000 épocas
            int index = (int) (Math.random() * training_data.length);
            nn.train(training_data[index], training_targets[index]);
        }
        System.out.println("Treinamento concluído.\n");

        // Testar a rede
        System.out.println("Testando a Rede Neural:");
        for (int i = 0; i < training_data.length; i++) {
            Matrix prediction = nn.predict(training_data[i]);
            System.out.printf("Entrada: [%.0f, %.0f] -> Predição: %.4f (Esperado: %.0f)\n",
                              training_data[i][0], training_data[i][1],
                              prediction.data[0][0], training_targets[i][0]);
        }
    }
}
