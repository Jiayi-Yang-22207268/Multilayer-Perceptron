import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * Training class for Multi-Layer Perceptron
 * Provides methods to train an MLP on a dataset with configurable parameters
 */
public class Training {
    
    private MLP network;
    private int maxEpochs;
    private int batchSize;  // "every now and then" parameter: how often to update weights
    private double learningRate;
    private String logFileName = "training_log.txt";  // Default log file name
    
    /**
     * Constructor for Training
     * @param network The MLP to train
     * @param maxEpochs Maximum number of training epochs
     * @param batchSize Number of examples between weight updates (1 = online, numExamples = batch)
     * @param learningRate Learning rate for gradient descent
     */
    public Training(MLP network, int maxEpochs, int batchSize, double learningRate) {
        this.network = network;
        this.maxEpochs = maxEpochs;
        this.batchSize = batchSize;
        this.learningRate = learningRate;
    }
    
    /**
     * Train the network on a dataset
     * @param inputs Array of input vectors
     * @param targets Array of target vectors
     * @param verbose If true, print error at each epoch
     * @return Final training error
     */
    public double train(double[][] inputs, double[][] targets, boolean verbose) {
        return train(inputs, targets, verbose, 1);  // Default: print every epoch if verbose
    }
    
    /**
     * Train the network on a dataset with configurable print interval
     * @param inputs Array of input vectors
     * @param targets Array of target vectors
     * @param verbose If true, print error at specified intervals
     * @param printInterval Print error every N epochs (only if verbose is true)
     * @return Final training error
     */
    public double train(double[][] inputs, double[][] targets, boolean verbose, int printInterval) {
        int numExamples = inputs.length;
        double error = 0;
        
        try (PrintWriter logWriter = new PrintWriter(new FileWriter(logFileName))) {
            for (int e = 0; e < maxEpochs; e++) {
                error = 0;
                
                for (int p = 0; p < numExamples; p++) {
                    // Forward pass
                    network.forward(inputs[p]);
                    
                    // Backward pass - accumulate gradients
                    error += network.backwards(inputs[p], targets[p]);
                    
                    // Update weights "every now and then" based on batch size
                    if ((p + 1) % batchSize == 0) {
                        network.updateWeights(learningRate);
                    }
                }
                
                // If batch size doesn't evenly divide numExamples, update at end of epoch
                if (numExamples % batchSize != 0) {
                    network.updateWeights(learningRate);
                }
                
                // Always write to log file
                logWriter.println("Error at epoch " + e + " is " + error);
                
                // Print to console based on verbose flag and print interval
                if (verbose && (e % printInterval == 0 || e == maxEpochs - 1)) {
                    System.out.println("Error at epoch " + e + " is " + error);
                }
            }
        } catch (IOException e) {
            System.err.println("Error writing to log file: " + e.getMessage());
        }
        
        return error;
    }
    
    /**
     * Train with default verbose setting (true)
     */
    public double train(double[][] inputs, double[][] targets) {
        return train(inputs, targets, true);
    }
    
    /**
     * Test the network on a dataset and return the total error
     * @param inputs Array of input vectors
     * @param targets Array of target vectors
     * @return Total error on the test set
     */
    public double test(double[][] inputs, double[][] targets) {
        double totalError = 0;
        int numExamples = inputs.length;
        
        for (int p = 0; p < numExamples; p++) {
            network.forward(inputs[p]);
            double[] output = network.getOutput();
            
            for (int k = 0; k < targets[p].length; k++) {
                double diff = targets[p][k] - output[k];
                totalError += diff * diff / 2.0;
            }
        }
        
        return totalError;
    }
    
    /**
     * Print predictions for all examples in a dataset
     * @param inputs Array of input vectors
     * @param targets Array of target vectors
     */
    public void printPredictions(double[][] inputs, double[][] targets) {
        int numExamples = inputs.length;
        
        System.out.println("\nPredictions:");
        System.out.println("============");
        
        for (int p = 0; p < numExamples; p++) {
            network.forward(inputs[p]);
            double[] output = network.getOutput();
            
            System.out.print("Input: [");
            for (int i = 0; i < inputs[p].length; i++) {
                System.out.print(inputs[p][i]);
                if (i < inputs[p].length - 1) System.out.print(", ");
            }
            System.out.print("] -> Target: [");
            for (int i = 0; i < targets[p].length; i++) {
                System.out.print(targets[p][i]);
                if (i < targets[p].length - 1) System.out.print(", ");
            }
            System.out.print("] -> Predicted: [");
            for (int i = 0; i < output.length; i++) {
                System.out.printf("%.4f", output[i]);
                if (i < output.length - 1) System.out.print(", ");
            }
            System.out.println("]");
        }
    }
    
    // Getters and setters
    public int getMaxEpochs() {
        return maxEpochs;
    }
    
    public void setMaxEpochs(int maxEpochs) {
        this.maxEpochs = maxEpochs;
    }
    
    public int getBatchSize() {
        return batchSize;
    }
    
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    
    public String getLogFileName() {
        return logFileName;
    }
    
    public void setLogFileName(String logFileName) {
        this.logFileName = logFileName;
    }
}
