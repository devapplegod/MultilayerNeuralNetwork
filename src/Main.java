import java.util.Scanner;

public class Main {

    public static void main(String[] args) {

        double[] inputs = getInputs(50);
        double[] expectedOutputs = getOutputs(inputs);

        NeuralNetwork network = new NeuralNetwork(1, 3, 5, 3, 1);

        network.train(inputs, expectedOutputs, 0.00001);

        while (true){
            Scanner scanner = new Scanner(System.in);
            System.out.println("Choose a number to square: ");
            double n = scanner.nextDouble();
            long start = System.nanoTime();
            double result = network.predict(new double[]{n})[0];
            long end = System.nanoTime();
            System.out.println(n + " squared = " + result + " (calculated in " + (end-start) + " nanoseconds)");
            long s = System.nanoTime();
            double r = Math.pow(n, 2);
            long e = System.nanoTime();
            System.out.println("Actual result = " + r + " (calculated in " + (e-s) + " nanoseconds)");
        }

    }

    public static void printArray(double[] array){
        for (int i = 0; i < array.length; i++) {
            System.out.println(array[i]);
        }
    }

    public static double[] getInputs(int size){
        double[] inputs = new double[size];
        for (int i = 0; i < size; i++) {
            inputs[i] = Math.random() * 20;
        }
        return inputs;
    }

    public static double[] getOutputs(double[] inputs){
        double[] outputs = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            outputs[i] = inputs[i] * inputs[i];
        }
        return outputs;
    }

}
