/**
 * Test1: XOR Function Test
 * Train an MLP with 2 inputs, 4 hidden units and one output on the XOR function
 * XOR examples:
 * (0, 0) -> 0
 * (0, 1) -> 1
 * (1, 0) -> 1
 * (1, 1) -> 0
 */
public class Test1 {
    
    public static void main(String[] args) {
        System.out.println("===========================================");
        System.out.println("Test1: XOR Function Learning");
        System.out.println("===========================================\n");
        
        // XOR training data
        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        
        double[][] targets = {
            {0},
            {1},
            {1},
            {0}
        };
        
        // Create MLP: 2 inputs, 4 hidden units, 1 output
        // Using sigmoid for hidden units (false) and sigmoid for output (false)
        int numInputs = 2;
        int numHidden = 4;
        int numOutputs = 1;
        boolean useTanhHidden = false;  // Use sigmoid for hidden layer
        boolean useLinearOutput = false; // Use sigmoid for output layer
        
        MLP network = new MLP(numInputs, numHidden, numOutputs, useTanhHidden, useLinearOutput);
        
        // Training parameters
        int maxEpochs = 10000;
        int batchSize = 4;  // Update after all 4 examples (batch learning)
        double learningRate = 0.5;
        
        System.out.println("Network Configuration:");
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
        
        // Create trainer and train
        Training trainer = new Training(network, maxEpochs, batchSize, learningRate);
        trainer.setLogFileName("test1_training_log.txt");
        
        // Train with verbose output every 1000 epochs (logs every epoch to file)
        double error = trainer.train(inputs, targets, true, 1000);
        
        System.out.println("\n--- Training Completed ---");
        System.out.println("Final error: " + error);
        
        // Test predictions
        trainer.printPredictions(inputs, targets);
        
        // Check if predictions are correct (threshold at 0.5)
        System.out.println("\n--- Verification ---");
        boolean allCorrect = true;
        for (int p = 0; p < inputs.length; p++) {
            network.forward(inputs[p]);
            double[] output = network.getOutput();
            int predicted = (output[0] >= 0.5) ? 1 : 0;
            int expected = (int) targets[p][0];
            boolean correct = (predicted == expected);
            allCorrect = allCorrect && correct;
            
            System.out.printf("Input: (%.0f, %.0f) -> Expected: %d, Predicted: %d (%.4f) %s%n",
                inputs[p][0], inputs[p][1], expected, predicted, output[0],
                correct ? "✓" : "✗");
        }
        
        System.out.println("\n" + (allCorrect ? "SUCCESS: All predictions are correct!" : "FAILURE: Some predictions are incorrect."));
    }
}
