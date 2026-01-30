import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Test3: Letter Recognition Dataset Test
 * Train and test an MLP on the letter-recognition.csv dataset
 * - 16 input features
 * - 26 outputs (one for each letter A-Z)
 * - Split: 80% training, 20% testing
 */
public class Test3 {
    
    public static void main(String[] args) {
        System.out.println("===========================================");
        System.out.println("Test3: Letter Recognition");
        System.out.println("===========================================\n");
        
        // Load dataset
        String filename = "letter-recognition.csv";
        List<double[]> allInputs = new ArrayList<>();
        List<double[]> allTargets = new ArrayList<>();
        List<Character> allLetters = new ArrayList<>();
        
        System.out.println("Loading dataset from " + filename + "...");
        
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            boolean isHeader = true;
            
            while ((line = br.readLine()) != null) {
                // Skip header line
                if (isHeader) {
                    isHeader = false;
                    continue;
                }
                
                String[] parts = line.split(",");
                if (parts.length < 17) continue;
                
                // Parse 16 input features (columns 0-15)
                double[] input = new double[16];
                for (int i = 0; i < 16; i++) {
                    // Normalize features to [0, 1] range (original values are 0-15)
                    input[i] = Double.parseDouble(parts[i].trim()) / 15.0;
                }
                
                // Parse letter label (column 16)
                char letter = parts[16].trim().charAt(0);
                
                // Create one-hot encoded target (26 outputs for A-Z)
                double[] target = new double[26];
                int letterIndex = letter - 'A';
                target[letterIndex] = 1.0;
                
                allInputs.add(input);
                allTargets.add(target);
                allLetters.add(letter);
            }
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
            return;
        }
        
        int totalExamples = allInputs.size();
        System.out.println("Loaded " + totalExamples + " examples");
        
        // Shuffle the data for random split
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalExamples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices);
        
        // Split into training (80%) and testing (20%)
        int trainSize = (int) (totalExamples * 0.8);
        int testSize = totalExamples - trainSize;
        
        double[][] trainInputs = new double[trainSize][16];
        double[][] trainTargets = new double[trainSize][26];
        char[] trainLetters = new char[trainSize];
        
        double[][] testInputs = new double[testSize][16];
        double[][] testTargets = new double[testSize][26];
        char[] testLetters = new char[testSize];
        
        for (int i = 0; i < trainSize; i++) {
            int idx = indices.get(i);
            trainInputs[i] = allInputs.get(idx);
            trainTargets[i] = allTargets.get(idx);
            trainLetters[i] = allLetters.get(idx);
        }
        
        for (int i = 0; i < testSize; i++) {
            int idx = indices.get(trainSize + i);
            testInputs[i] = allInputs.get(idx);
            testTargets[i] = allTargets.get(idx);
            testLetters[i] = allLetters.get(idx);
        }
        
        System.out.println("Training set: " + trainSize + " examples");
        System.out.println("Test set: " + testSize + " examples");
        
        // Create MLP: 16 inputs, 10 hidden units (can be adjusted), 26 outputs
        int numInputs = 16;
        int numHidden = 30;  // Starting point as suggested
        int numOutputs = 26;
        boolean useTanhHidden = true;   // tanh often works well
        boolean useLinearOutput = false; // sigmoid for classification (output in [0,1])
        
        MLP network = new MLP(numInputs, numHidden, numOutputs, useTanhHidden, useLinearOutput);
        
        // Training parameters
        int maxEpochs = 5000;
        int batchSize = 100;  // Mini-batch learning
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
        long startTime = System.currentTimeMillis();
        Training trainer = new Training(network, maxEpochs, batchSize, learningRate);
        trainer.setLogFileName("test3_training_log.txt");
        double trainError = trainer.train(trainInputs, trainTargets, true, 500);
        
        long endTime = System.currentTimeMillis();
        System.out.println("\n--- Training Completed in " + (endTime - startTime) / 1000.0 + " seconds ---\n");
        
        // Evaluate on training set
        int trainCorrect = 0;
        for (int p = 0; p < trainSize; p++) {
            network.forward(trainInputs[p]);
            double[] output = network.getOutput();
            
            // Find predicted letter (index of max output)
            int predictedIndex = 0;
            double maxOutput = output[0];
            for (int k = 1; k < 26; k++) {
                if (output[k] > maxOutput) {
                    maxOutput = output[k];
                    predictedIndex = k;
                }
            }
            
            char predicted = (char) ('A' + predictedIndex);
            if (predicted == trainLetters[p]) {
                trainCorrect++;
            }
        }
        
        // Evaluate on test set
        int testCorrect = 0;
        int[] confusionMatrix = new int[26];  // Count correct predictions per letter
        int[] letterCounts = new int[26];      // Count total per letter
        
        for (int p = 0; p < testSize; p++) {
            network.forward(testInputs[p]);
            double[] output = network.getOutput();
            
            // Find predicted letter (index of max output)
            int predictedIndex = 0;
            double maxOutput = output[0];
            for (int k = 1; k < 26; k++) {
                if (output[k] > maxOutput) {
                    maxOutput = output[k];
                    predictedIndex = k;
                }
            }
            
            char predicted = (char) ('A' + predictedIndex);
            int actualIndex = testLetters[p] - 'A';
            letterCounts[actualIndex]++;
            
            if (predicted == testLetters[p]) {
                testCorrect++;
                confusionMatrix[actualIndex]++;
            }
        }
        
        // Calculate accuracies
        double trainAccuracy = (double) trainCorrect / trainSize * 100;
        double testAccuracy = (double) testCorrect / testSize * 100;
        
        System.out.println("===========================================");
        System.out.println("RESULTS");
        System.out.println("===========================================");
        System.out.println("\nFinal Training Error: " + String.format("%.4f", trainError));
        System.out.println("\nTraining Accuracy: " + trainCorrect + "/" + trainSize + 
                          " (" + String.format("%.2f", trainAccuracy) + "%)");
        System.out.println("Test Accuracy: " + testCorrect + "/" + testSize + 
                          " (" + String.format("%.2f", testAccuracy) + "%)");
        
        // Per-letter accuracy
        System.out.println("\n--- Per-Letter Test Accuracy ---");
        for (int i = 0; i < 26; i++) {
            char letter = (char) ('A' + i);
            if (letterCounts[i] > 0) {
                double letterAccuracy = (double) confusionMatrix[i] / letterCounts[i] * 100;
                System.out.printf("%c: %d/%d (%.1f%%)%n", letter, confusionMatrix[i], letterCounts[i], letterAccuracy);
            }
        }
        
        // Show some sample predictions
        System.out.println("\n--- Sample Test Predictions (first 20) ---");
        for (int p = 0; p < Math.min(20, testSize); p++) {
            network.forward(testInputs[p]);
            double[] output = network.getOutput();
            
            // Find predicted letter
            int predictedIndex = 0;
            double maxOutput = output[0];
            for (int k = 1; k < 26; k++) {
                if (output[k] > maxOutput) {
                    maxOutput = output[k];
                    predictedIndex = k;
                }
            }
            
            char predicted = (char) ('A' + predictedIndex);
            boolean correct = (predicted == testLetters[p]);
            
            System.out.printf("Actual: %c, Predicted: %c (confidence: %.3f) %s%n",
                testLetters[p], predicted, maxOutput, correct ? "✓" : "✗");
        }
        
        // Analysis
        System.out.println("\n===========================================");
        System.out.println("ANALYSIS");
        System.out.println("===========================================");
        
        System.out.println("\nQ: How well can you classify the test data?");
        System.out.printf("A: The network achieved %.2f%% accuracy on the test set.%n", testAccuracy);
        
        if (testAccuracy >= 80) {
            System.out.println("   This is excellent performance for letter recognition!");
        } else if (testAccuracy >= 60) {
            System.out.println("   This is good performance. Consider increasing hidden units or epochs.");
        } else if (testAccuracy >= 40) {
            System.out.println("   This is moderate performance. Try:");
            System.out.println("   - Increasing hidden units");
            System.out.println("   - More training epochs");
            System.out.println("   - Adjusting learning rate");
        } else {
            System.out.println("   Performance needs improvement. Suggestions:");
            System.out.println("   - Significantly more hidden units");
            System.out.println("   - Longer training");
            System.out.println("   - Different learning rate");
        }
        
        System.out.println("\nNote: Random baseline would be ~3.85% (1/26 chance)");
        System.out.println("The original research achieved ~80% accuracy.");
    }
}
