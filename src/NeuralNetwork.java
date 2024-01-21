public class NeuralNetwork {

    Layer[] layers;
    double[][] bestWeights;
    double bestLoss = Double.MAX_VALUE;

    public NeuralNetwork(int ... nodes) {
        this.layers = new Layer[nodes.length - 1];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(nodes[i], nodes[i + 1]);
        }
    }

    public void train(double[] inputData, double[] expectedOutputData, double learnRate){
        double d = 0.0001;
        for (int n = 0; n < 1000; n++) {
            for (int x = 0; x < inputData.length; x++) {
                double initialLoss = totalLoss(inputData, expectedOutputData);
                for (Layer layer : layers){
                    for (int i = 0; i < layer.inputsAmount; i++) {
                        for (int j = 0; j < layer.outputsAmount; j++) {
                            layer.weights[j][i] += d;
                            double dLoss = totalLoss(inputData, expectedOutputData) - initialLoss;
                            layer.weights[j][i] -= d;
                            layer.lossGradientWeights[j][i] = dLoss / d;
                        }
                    }
                    for (int i = 0; i < layer.biases.length; i++) {
                        layer.biases[i] += d;
                        double dLoss = totalLoss(inputData, expectedOutputData) - initialLoss;
                        layer.biases[i] -= d;
                        layer.lossGradientBiases[i] = dLoss / d;
                    }
                }
                applyAllGradients(learnRate);
            }
            System.out.println("AVG LOSS: " + totalLoss(inputData, expectedOutputData));
        }

    }

    public void applyAllGradients(double learnRate){
        for (Layer layer : layers){
            layer.applyGradients(learnRate);
        }
    }

    public double[] predict(double[] inputs){
        for (Layer layer : layers){
            inputs = layer.feedForward(inputs);
        }
        return inputs;
    }

    public double totalLoss(double[] inputData, double[] expectedOutputData){
        double loss = 0;
        for (int i = 0; i < inputData.length; i++) {
            loss += loss(predict(new double[]{inputData[i]}), new double[]{expectedOutputData[i]});
        }
        return loss / inputData.length;
    }

    public double loss(double[] actualPredictions, double[] expectedOutputs){
        double loss = 0;
        for (int i = 0; i < expectedOutputs.length; i++) {
            loss += nodeLoss(actualPredictions[i], expectedOutputs[i]);
        }
        return loss;
    }

    private double nodeLoss(double actualPrediction, double expectedOutput){
        return Math.pow(actualPrediction - expectedOutput, 2);
    }

    private double nodeLossDerivative(double actualPrediction, double expectedOutput){
        return 2 * (actualPrediction - expectedOutput);
    }

}
