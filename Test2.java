/**
 * Test2: Sin Function Approximation Test
 * Generate 500 vectors with 4 components each (random values between -1 and 1)
 * Output: sin(x1 - x2 + x3 - x4)
 * Train on 400 examples, test on 100 examples
 */
public class Test2 {
    
    public static void main(String[] args) {
        System.out.println("===========================================");
        System.out.println("Test2: Sin Function Approximation");
        System.out.println("===========================================\n");
        
        // Generate 500 random vectors
        int totalExamples = 500;
        int trainSize = 400;
        int testSize = 100;
        
        double[][] allInputs = new double[totalExamples][4];
        double[][] allTargets = new double[totalExamples][1];
        
        // Generate random data
        System.out.println("Generating " + totalExamples + " random examples...");
        for (int i = 0; i < totalExamples; i++) {
            // Generate random values between -1 and 1
            double x1 = Math.random() * 2 - 1;
            double x2 = Math.random() * 2 - 1;
            double x3 = Math.random() * 2 - 1;
            double x4 = Math.random() * 2 - 1;
            
            allInputs[i][0] = x1;
            allInputs[i][1] = x2;
            allInputs[i][2] = x3;
            allInputs[i][3] = x4;
            
            // Output: sin(x1 - x2 + x3 - x4)
            allTargets[i][0] = Math.sin(x1 - x2 + x3 - x4);
        }
        
        // Split into training and test sets
        double[][] trainInputs = new double[trainSize][4];
        double[][] trainTargets = new double[trainSize][1];
        double[][] testInputs = new double[testSize][4];
        double[][] testTargets = new double[testSize][1];
        
        for (int i = 0; i < trainSize; i++) {
            trainInputs[i] = allInputs[i];
            trainTargets[i] = allTargets[i];
        }
        for (int i = 0; i < testSize; i++) {
            testInputs[i] = allInputs[trainSize + i];
            testTargets[i] = allTargets[trainSize + i];
        }
        
        System.out.println("Training set: " + trainSize + " examples");
        System.out.println("Test set: " + testSize + " examples");
        
        // Create MLP: 4 inputs, 5 hidden units, 1 output
        // Using tanh for hidden (good for inputs in [-1,1]) and linear output (sin outputs in [-1,1])
        int numInputs = 4;
        int numHidden = 5;  // At least 5 as required
        int numOutputs = 1;
        boolean useTanhHidden = true;   // tanh works well for inputs in [-1,1]
        boolean useLinearOutput = true; // Linear output since sin outputs are in [-1,1]
        
        MLP network = new MLP(numInputs, numHidden, numOutputs, useTanhHidden, useLinearOutput);
        
        // Training parameters
        int maxEpochs = 5000;
        int batchSize = 20;  // Mini-batch learning
        double learningRate = 0.01;
        
        System.out.println("\nNetwork Configuration:");
        System.out.println("- Inputs: " + numInputs);
        System.out.println("- Hidden units: " + numHidden);
        System.out.println("- Outputs: " + numOutputs);
        System.out.println("- Hidden activation: " + (useTanhHidden ? "tanh" : "sigmoid"));
        System.out.println("- Output activation: " + (useLinearOutput ? "linear" : "sigmoid"));
        System.out.println("\nTraining Parameters:");
        System.out.println("- Max epochs: " + maxEpochs);
        System.out.println("- Batch size: " + batchSize);
        System.out.println("- Learning rate: " + learningRate);
        System.out.println("\n--- Training Started ---\n");
        
        // Train the network using Training class (logs every epoch to file)
        Training trainer = new Training(network, maxEpochs, batchSize, learningRate);
        trainer.setLogFileName("test2_training_log.txt");
        trainer.train(trainInputs, trainTargets, true, 500);
        
        System.out.println("\n--- Training Completed ---\n");
        
        // Calculate final training error
        double finalTrainError = 0;
        for (int p = 0; p < trainSize; p++) {
            network.forward(trainInputs[p]);
            double[] output = network.getOutput();
            double diff = trainTargets[p][0] - output[0];
            finalTrainError += diff * diff / 2.0;
        }
        
        // Calculate test error
        double testError = 0;
        for (int p = 0; p < testSize; p++) {
            network.forward(testInputs[p]);
            double[] output = network.getOutput();
            double diff = testTargets[p][0] - output[0];
            testError += diff * diff / 2.0;
        }
        
        // Calculate average errors
        double avgTrainError = finalTrainError / trainSize;
        double avgTestError = testError / testSize;
        
        System.out.println("===========================================");
        System.out.println("RESULTS");
        System.out.println("===========================================");
        System.out.println("\nTotal Training Error: " + finalTrainError);
        System.out.println("Total Test Error: " + testError);
        System.out.println("\nAverage Training Error per example: " + avgTrainError);
        System.out.println("Average Test Error per example: " + avgTestError);
        
        // Show some sample predictions from test set
        System.out.println("\n--- Sample Test Predictions (first 10) ---");
        for (int p = 0; p < Math.min(10, testSize); p++) {
            network.forward(testInputs[p]);
            double[] output = network.getOutput();
            System.out.printf("Input: [%.3f, %.3f, %.3f, %.3f] -> Target: %.4f, Predicted: %.4f, Error: %.4f%n",
                testInputs[p][0], testInputs[p][1], testInputs[p][2], testInputs[p][3],
                testTargets[p][0], output[0], Math.abs(testTargets[p][0] - output[0]));
        }
        
        // Analysis
        System.out.println("\n===========================================");
        System.out.println("ANALYSIS");
        System.out.println("===========================================");
        System.out.println("\nQ: What is the error on training at the end?");
        System.out.printf("A: Total training error: %.6f (Average: %.6f per example)%n", finalTrainError, avgTrainError);
        
        System.out.println("\nQ: How does it compare with the error on the test set?");
        double ratio = testError / finalTrainError;
        System.out.printf("A: Test error is %.2fx the training error%n", ratio);
        if (Math.abs(avgTestError - avgTrainError) < 0.01) {
            System.out.println("   The errors are very similar, indicating good generalization.");
        } else if (avgTestError > avgTrainError * 1.5) {
            System.out.println("   Test error is notably higher, which may indicate some overfitting.");
        } else {
            System.out.println("   Test error is reasonably close to training error.");
        }
        
        System.out.println("\nQ: Do you think you have learned satisfactorily?");
        if (avgTestError < 0.01) {
            System.out.println("A: YES - The network has learned the function very well!");
            System.out.println("   Average error per example is very small.");
        } else if (avgTestError < 0.05) {
            System.out.println("A: YES - The network has learned the function reasonably well.");
            System.out.println("   Predictions are close to the true sin values.");
        } else {
            System.out.println("A: The learning could be improved. Consider:");
            System.out.println("   - More hidden units");
            System.out.println("   - More training epochs");
            System.out.println("   - Different learning rate");
        }
    }
}
