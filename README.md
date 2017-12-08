## AdaBoost Implementation with NumPy
AdaBoost is a machine learning meta-algorithm that combines weighted weak learners to create a more accurate label prediction. The weak learners in this algorithm are called stumps. The stump will split the data into two sections - a positive labeled side and a negative labeled side. As each weak learner is added, the weights are reassigned based on their effectiveness.

## Data
We borrow a US Postal Service zipcode data set filled with handwritten digits. Each tuple is made up of 256 features which represent the gray scale value of each pixel in a 16x16 image. Field 1 in the data set is the true label for that handwritten number (ranging from 0 to 9). This algorithm can compare any two labels in the data set specified on runtime.

## Running
Run AdaBoost using the following command

```
python run.py [Number of Weak Learners] [Path to Training Data] [Path to Test Data] [Positive Label (0-9)] [Negative Label (0-9)]
```

For example, this command will run until 50 weak learners are created and compare the handwritten digits 1 and 3:

```
python run.py 50 ../data/zip.train ../data/zip.test 1 3
```

## Results
In the "results" directory, there is a 1 vs 3 plot using 50 weak learners and a 3 vs 5 plot using 200 weak learners. These plots are done in MATLAB, but soon Python plotting will be added at the end of the algorithm.
