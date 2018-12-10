# Naive Bayes 

A Naive Basyes classifier algorithm based on specific dataset. We use n-bins to handle the continuous attributes. 

## Usage 


### Input format 
```bash
python nbayes.py spam 0 5 -1
```

1. Option 1 will be the path to the data (see problem 1).

2. Option 2 is a 0/1 option. If 0, use cross validation. If 1, run the algorithm on the full sample.

3. Option 3 (at least 2) is the number of bins for any continuous feature.

4. Option 4 is the value of m for the m-estimate. If this value is negative, use Laplace smoothing.

Note that m=0 is maximum likelihood estimation. The value of p in the m-estimate should be fixed to 1/v for a variable with v values.

### Output format

When either algorithm is run on any problem, it must produce output in exactly the following format:

```
Accuracy: 0.xyz 0.abc

Precision: 0.xyz 0.abc

Recall: 0.xyz 0.abc

Area under ROC: 0.xyz
```