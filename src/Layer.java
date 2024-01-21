public class Layer {

    public int outputsAmount;
    public int inputsAmount;

    public double[][] weights;
    public double[] biases;

    public double[][] lossGradientWeights;
    public double[] lossGradientBiases;

    public Layer(int inputsAmount, int outputsAmount) {
        this.inputsAmount = inputsAmount;
        this.outputsAmount = outputsAmount;
        this.weights = new double[outputsAmount][this.inputsAmount];
        this.lossGradientWeights = new double[outputsAmount][this.inputsAmount];
        this.biases = new double[outputsAmount];
        this.lossGradientBiases = new double[outputsAmount];
        randomizeLayer();
    }

    public double[] feedForward(double[] inputs){
        double[] product = new double[this.outputsAmount];
        for (int i = 0; i < this.outputsAmount; i++) {
            product[i] = this.biases[i];
            for (int j = 0; j < this.inputsAmount; j++) {
                product[i] += inputs[j] * this.weights[i][j];
            }
            product[i] = activate(product[i]);
        }
        return product;
    }

    public void applyGradients(double learnRate){
        for (int i = 0; i < this.outputsAmount; i++) {
            this.biases[i] -= lossGradientBiases[i] * learnRate;
            for (int j = 0; j < this.inputsAmount; j++) {
                this.weights[i][j] -= lossGradientWeights[i][j] * learnRate;
            }
        }
    }

    private double activate(double value){
        if (false){
            return 1 / (1 + Math.exp(-value));
        } else {
            return Math.max(0, value);
        }
    }

    private double activateDerivative(double value){
        if (value <= 0){
            return 0;
        } else {
            return 1;
        }
    }

    private void randomizeWeights(){
        for (int i = 0; i < this.outputsAmount; i++) {
            for (int j = 0; j < this.inputsAmount; j++) {
                this.weights[i][j] = Math.random();
                this.lossGradientWeights[i][j] = Math.random();
            }
        }
    }

    private void randomizeBiases(){
        for (int i = 0; i < this.outputsAmount; i++) {
            this.biases[i] = Math.random();
            this.lossGradientBiases[i] = Math.random();
        }
    }

    public void randomizeLayer(){
        randomizeWeights();
        randomizeBiases();
    }

}
