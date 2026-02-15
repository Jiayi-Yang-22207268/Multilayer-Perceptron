Multilayer Perceptron
=====================

Simple Java implementation of a multi-layer perceptron (MLP) with one hidden layer, plus three test programs:

- Test1: XOR learning (classification)
- Test2: Sin function approximation (regression)
- Test3: Letter recognition with the UCI letter dataset (classification)

Project layout
--------------

- MLP.java: Core MLP model (forward + backprop, sigmoid/tanh/linear options)
- Training.java: Training loop, batch or mini-batch updates, logging
- Test1.java: XOR experiment
- Test2.java: Sin approximation experiment
- Test3.java: Letter recognition experiment
- letter-recognition.csv: Converted dataset used by Test3
- letter-recognition.names: Dataset description
- convert_to_csv.py: Converts the original UCI dataset format to the CSV used here
- test*_training_log.txt: Example training logs

Requirements
------------

- Java 8+ (javac + java)
- Optional: Python 3 if you need to regenerate the CSV

How to compile
--------------

From the project folder:

```bash
javac MLP.java Training.java Test1.java Test2.java Test3.java
```

How to run
----------

Run one test at a time:

```bash
java Test1
```

```bash
java Test2
```

```bash
java Test3
```

Each test prints progress to the console and writes an epoch-by-epoch error log to a file:

- Test1 -> test1_training_log.txt
- Test2 -> test2_training_log.txt
- Test3 -> test3_training_log.txt

Dataset notes (Test3)
---------------------

Test3 expects letter-recognition.csv in the project root. The CSV format here is:

- 16 numeric feature columns (normalized to [0, 1] inside Test3)
- 1 label column at the end (A-Z)

If you have the original UCI file (letter-recognition.data), you can regenerate the CSV:

```bash
python convert_to_csv.py
```

Implementation details
----------------------

- Hidden activation: sigmoid or tanh
- Output activation: sigmoid (classification) or linear (regression)
- Training: squared error loss with gradient descent and configurable batch size

Suggested experiments
---------------------

- Increase hidden units in Test2/Test3 and compare training vs test error.
- Try smaller/larger learning rates to see convergence changes.
- Switch hidden activation to tanh or sigmoid and compare behavior.
