/**
 * Multi-Layer Perceptron implementation
 * A neural network with one hidden layer, supporting:
 * - Sigmoidal or tanh activation for hidden units
 * - Sigmoidal or linear activation for output units
 */
public class MLP {
    // Network architecture
    private int NI;  // Number of inputs
    private int NH;  // Number of hidden units
    private int NO;  // Number of outputs
    
    // Weights
    private double[][] W1;  // Weights from input to hidden layer (NH x NI+1, +1 for bias)
    private double[][] W2;  // Weights from hidden to output layer (NO x NH+1, +1 for bias)
    
    // Weight changes (gradients accumulated)
    private double[][] dW1;
    private double[][] dW2;
    
    // Activations (weighted sums before activation function)
    private double[] Z1;  // Activations for hidden layer
    private double[] Z2;  // Activations for output layer
    
    // Neuron outputs
    private double[] H;   // Hidden layer outputs
    private double[] O;   // Output layer outputs
    
    // Activation function options
    private boolean useTanhHidden;    // true = tanh, false = sigmoid for hidden layer
    private boolean useLinearOutput;  // true = linear, false = sigmoid for output layer
    
    /**
     * Constructor for MLP
     * @param numInputs Number of input neurons
     * @param numHidden Number of hidden neurons
     * @param numOutputs Number of output neurons
     * @param useTanhHidden true for tanh hidden activation, false for sigmoid
     * @param useLinearOutput true for linear output, false for sigmoid
     */
    public MLP(int numInputs, int numHidden, int numOutputs, boolean useTanhHidden, boolean useLinearOutput) {
        this.NI = numInputs;
        this.NH = numHidden;
        this.NO = numOutputs;
        this.useTanhHidden = useTanhHidden;
        this.useLinearOutput = useLinearOutput;
        
        // Initialize weight arrays (+1 for bias in each layer)
        W1 = new double[NH][NI + 1];
        W2 = new double[NO][NH + 1];
        
        dW1 = new double[NH][NI + 1];
        dW2 = new double[NO][NH + 1];
        
        Z1 = new double[NH];
        Z2 = new double[NO];
        
        H = new double[NH];
        O = new double[NO];
        
        // Initialize weights to small random values
        randomise();
    }
    
    /**
     * Initialize weights to small random values and reset weight changes to zero
     */
    public void randomise() {
        // Initialize W1 to small random values
        for (int j = 0; j < NH; j++) {
            for (int i = 0; i <= NI; i++) {
                W1[j][i] = (Math.random() - 0.5) * 0.5;  // Random values in [-0.25, 0.25]
            }
        }
        
        // Initialize W2 to small random values
        for (int k = 0; k < NO; k++) {
            for (int j = 0; j <= NH; j++) {
                W2[k][j] = (Math.random() - 0.5) * 0.5;
            }
        }
        
        // Reset weight changes to zero
        resetWeightChanges();
    }
    
    /**
     * Reset all weight changes to zero
     */
    private void resetWeightChanges() {
        for (int j = 0; j < NH; j++) {
            for (int i = 0; i <= NI; i++) {
                dW1[j][i] = 0.0;
            }
        }
        for (int k = 0; k < NO; k++) {
            for (int j = 0; j <= NH; j++) {
                dW2[k][j] = 0.0;
            }
        }
    }
    
    /**
     * Sigmoid activation function
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    /**
     * Derivative of sigmoid (given the output of sigmoid)
     */
    private double sigmoidDerivative(double sigmoidOutput) {
        return sigmoidOutput * (1.0 - sigmoidOutput);
    }
    
    /**
     * Derivative of tanh (given the output of tanh)
     */
    private double tanhDerivative(double tanhOutput) {
        return 1.0 - tanhOutput * tanhOutput;
    }
    
    /**
     * Forward pass - compute output for given input
     * @param I Input vector
     * @return Output vector
     */
    public double[] forward(double[] I) {
        // Compute hidden layer activations and outputs
        for (int j = 0; j < NH; j++) {
            Z1[j] = W1[j][NI];  // Bias term (last weight)
            for (int i = 0; i < NI; i++) {
                Z1[j] += W1[j][i] * I[i];
            }
            // Apply activation function
            if (useTanhHidden) {
                H[j] = Math.tanh(Z1[j]);
            } else {
                H[j] = sigmoid(Z1[j]);
            }
        }
        
        // Compute output layer activations and outputs
        for (int k = 0; k < NO; k++) {
            Z2[k] = W2[k][NH];  // Bias term
            for (int j = 0; j < NH; j++) {
                Z2[k] += W2[k][j] * H[j];
            }
            // Apply activation function
            if (useLinearOutput) {
                O[k] = Z2[k];  // Linear output
            } else {
                O[k] = sigmoid(Z2[k]);
            }
        }
        
        return O;
    }
    
    /**
     * Backward pass - compute weight updates based on target
     * @param I Input vector (needed for computing gradients)
     * @param t Target vector
     * @return Error on this example (sum of squared errors / 2)
     */
    public double backwards(double[] I, double[] t) {
        double error = 0.0;
        
        // Compute output layer deltas
        double[] deltaOutput = new double[NO];
        for (int k = 0; k < NO; k++) {
            double diff = t[k] - O[k];
            error += diff * diff;
            
            // Compute delta based on activation function
            if (useLinearOutput) {
                deltaOutput[k] = diff;  // Linear: derivative is 1
            } else {
                deltaOutput[k] = diff * sigmoidDerivative(O[k]);
            }
        }
        error /= 2.0;  // Standard squared error
        
        // Accumulate weight changes for W2 (hidden to output)
        for (int k = 0; k < NO; k++) {
            for (int j = 0; j < NH; j++) {
                dW2[k][j] += deltaOutput[k] * H[j];
            }
            dW2[k][NH] += deltaOutput[k];  // Bias update
        }
        
        // Compute hidden layer deltas
        double[] deltaHidden = new double[NH];
        for (int j = 0; j < NH; j++) {
            double sum = 0.0;
            for (int k = 0; k < NO; k++) {
                sum += deltaOutput[k] * W2[k][j];
            }
            // Compute delta based on activation function
            if (useTanhHidden) {
                deltaHidden[j] = sum * tanhDerivative(H[j]);
            } else {
                deltaHidden[j] = sum * sigmoidDerivative(H[j]);
            }
        }
        
        // Accumulate weight changes for W1 (input to hidden)
        for (int j = 0; j < NH; j++) {
            for (int i = 0; i < NI; i++) {
                dW1[j][i] += deltaHidden[j] * I[i];
            }
            dW1[j][NI] += deltaHidden[j];  // Bias update
        }
        
        return error;
    }
    
    /**
     * Update weights using accumulated gradients
     * @param learningRate Learning rate for gradient descent
     */
    public void updateWeights(double learningRate) {
        // Update W1
        for (int j = 0; j < NH; j++) {
            for (int i = 0; i <= NI; i++) {
                W1[j][i] += learningRate * dW1[j][i];
                dW1[j][i] = 0.0;  // Reset for next batch
            }
        }
        
        // Update W2
        for (int k = 0; k < NO; k++) {
            for (int j = 0; j <= NH; j++) {
                W2[k][j] += learningRate * dW2[k][j];
                dW2[k][j] = 0.0;  // Reset for next batch
            }
        }
    }
    
    /**
     * Get the current output
     * @return Output array
     */
    public double[] getOutput() {
        return O;
    }
    
    /**
     * Get number of inputs
     */
    public int getNumInputs() {
        return NI;
    }
    
    /**
     * Get number of hidden units
     */
    public int getNumHidden() {
        return NH;
    }
    
    /**
     * Get number of outputs
     */
    public int getNumOutputs() {
        return NO;
    }
}
