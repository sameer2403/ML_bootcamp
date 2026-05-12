# Why Statistics Matters in ML?

# About Statistics
Statistics helps us understand, summarise, and
interpret data before modeling.
# Purpose in ML
Identify patterns in data
Detect issues like missing values or noise
Improve model performance by preparing clean features
# Example
Are most values centered tightly or widely spread?

--------------------------------------------

# Central Tendency
Shows the central or typical value in data. Represents
the “center of gravity” of your dataset.

# Purpose in ML
Helps understand where most values lie. 
Useful for summarising large datasets

# Example
Student test scores, daily temperatures

--------------------------------------------
# Mean, Median & Mode
Mean 
Average of all values.
(10 + 20 + 30) / 3 = 20
Use in ML: Used for imputing missing values

# Median 
Middle value when sorted.
Values: 5, 8, 11 -> 8
Use in ML: Robust when data has outliers.

# Mode
Most frequent value.
Values: 2, 2, 3 -> 2
Use in ML: Useful for categorical features.

--------------------------------------------
# Measure of Spread
Measure of Spread tells how far values vary from the center.

# Importance in ML
High spread indicates unstable (noisy) or inconsistent features.
Low spread implies stable and predictable data

# Example
Values between 20 and 40 have higher spread than
values between 28 and 32.

--------------------------------------------
# Range
Max minus min. Fast way to see spread.
# Variance
Shows how far values deviate from the mean on average.

# Standard deviation
Square root of variance.
Low SD means values are stable.
Use in ML: Feature scaling


# Inter Quartile Range
Middle 50 percent of the data (Q1 to Q3).
Use in ML: Outlier detection


--------------------------------------------
# Distribution
# Normal Distribution
Symmetrical bell curve
Example: Human heights
Use in ML: Scaling and assumptions in linear models.

# Skewed Distribution
More data on one side
Example: Salaries
Use in ML: Suggests using log transformations.

--------------------------------------------
# Covariance 
Shows how two variables move together.
Example: Height and weight often increase together.

# Correlation
Strength and direction of relationship.
Range: -1 to 1
Use in ML: Feature selection, removing multicollinearity


--------------------------------------------
# Percentiles 
Shows value below which a percentage of data lies.
Example: 90th percentile means better than 90 percent of values.

# Quartiles
• Q1: 25 percent
• Q2: 50 percent (median)
• Q3: 75 percent

# Use in ML
Understand distribution and detect outliers.


--------------------------------------------
# Outliers
Values far from most other values.

# Example
Salary 200000 in a dataset of 20000 to 40000.

# Detection
IQR method (1.5 times IQR rule)
IQR = Q3 - Q1

# Why Important in ML
• Skews regression line
• Affects mean and variance
• Can slow or confuse gradient descent


--------------------------------------------
# How Statistics Is Used in ML

# EDA
Find patterns, distributions, issues 
# SCALING 
Standardisation & normalisation

# OUTLIER TREATMENT
CLEAN Noiay data

# Evaluation
Understand error speed


------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------
## Why calculus matters in ML?

Calculus explains how things change. It helps measure
rates, slopes, and movement in functions.

# Purpose in ML
-> Understand how predictions react to small changes
-> Guide weight updates during training
-> Improve model accuracy by following the right direction

# Example
How does loss change when a weight increases slightly?


--------------------------------------------
# Function in ML
A function maps input to output (e.g., x → y).
Used in ML for predictions.

# Purpose in ML
-> Models behave like functions where small input changes produce new predictions.

# Example
Salary prediction based on years of experience.

--------------------------------------------
## Gradient (Multi Dimensional Slope)

# What It Is
-> Slope in many dimensions.
-> Each weight in a model has its own slope.
-> This full set of slopes is the gradient.

# Importance in ML
-> Shows fastest direction of change
-> Used to reduce loss
-> Helps tune all parameters together

# Why It Matters
Gradient tells the model how to update weights
• Big gradient → adjust weight strongly
• Small gradient → adjust weight gently
• Zero gradient → model has stopped improving

--------------------------------------------
## Loss Function (What We Minimize)

# What It Is
 -> Measure of error
-> Shows how far predictions are from the actual values.

# Example
-> U shaped curve (MSE)
# Sample Formula
-> Mean Squared Error (MSE)
# Why It Matters
-> Goal is to reduce loss
-> Lower loss means more accurate predictions.
-> Lower value means better performance.


--------------------------------------------
# Gradient Descent
Gradient Descent is a method that helps a model learn by moving its weights step by
step in the direction that reduces the loss. 
It follows the slope of the loss curve and keeps moving downhill until it reaches the lowest point.




--------------------------------------------
## How Models Learn
# Predict
Model makes an output.

# Calculate Loss
Compare prediction vs actual

# Update Weights

Use gradient descent to reduce the loss.

--------------------------------------------
## Why Calculus Matters
# Training
Use derivatives to understand change

# Optimisers
Use gradients to update weights

# Loss Reduction
Follow slopes to minimise error






--------------------------------------------
# Probability distribution
A distribution describes how likely each possible value of a random variable is.
It shows whether values are common, rare, or equally likely.

# Importance in ML
In Machine Learning, distributions help us understand data patterns,
noise, and uncertainty.
# Distribution Types
Uniform: All values equally likely
Normal: Bell-shaped curve (Gaussian)
Bernoulli: One Yes/No outcome
Binomial: Number of Yes/No outcomes across many trials

## Applications in ML

# Classification
Predicting the probability that an input belongs to a class.

# Naive Bayes
A powerful algorithm based entirely on conditional probability.

 # Sampling
Generating diverse synthetic data examples.

--------------------------------------------

## Conditional Probability

-- Conditional probability answers:
“What is the probability now that I know this extra fact?”

This idea powers models like Naive Bayes, which update
predictions using known features.

--------------------------------------------
## Probability Distributions: Deep Dive

### **1. Bernoulli Distribution**

**What it is:**
A single yes/no (1/0) outcome with probability `p`.

**Formula:**
- P(X = 1) = p
- P(X = 0) = 1 - p

**Why we need it:**
- Models a single binary event
- Foundation for all binary classification

**Where we use it:**
- Email: spam or not spam
- Medical test: positive or negative
- Classification: pass or fail
- Any single binary outcome

**Insight for applications:**
- If your model outputs a probability (0.7), that's a Bernoulli with p = 0.7
- Binary classification in ML is fundamentally Bernoulli events


### **2. Binomial Distribution**

**What it is:**
Count of successes in `n` independent Bernoulli trials.

**Example:** 
Flip a coin 10 times, count heads.

**Formula:**
- Number of successes in `n` trials with probability `p` each
- X can be any integer from 0 to n

**Why we need it:**
- Extends single event to multiple events
- Models cumulative success/failure counts

**Where we use it:**
- How many customers will buy out of 1000 prospects?
- How many patients will recover out of 50 treated?
- How many defective items in a batch of 100?

**Insight for applications:**
- If you have repeated binary events, use Binomial
- Small p: most outcomes near 0
- Large p: most outcomes near n
- p = 0.5: symmetric bell shape


### **3. Normal Distribution (Gaussian)**

**What it is:**
The classic bell curve. Most natural phenomena follow this shape.

**Example:** 
Heights of people, test scores, measurement errors

**Formula:**
- Defined by mean (μ) and standard deviation (σ)
- N(0, 1) means mean = 0, std = 1

**Why we need it:**
- Most common distribution in nature
- Central Limit Theorem: many random processes sum to Normal
- Easy to work with mathematically
- Many ML algorithms assume Normality

**Where we use it:**
- Heights, weights, test scores
- Sensor noise and measurement errors
- Stock returns (approximately)
- Machine learning: assume features are normally distributed
- Residuals in linear regression

**Insight for applications:**
- If data looks like a bell curve, it's likely Normal
- 68% of data falls within 1 std of mean
- 95% within 2 stds
- 99.7% within 3 stds (3-sigma rule)
- Use to detect outliers: values beyond 3 stds are rare


### **4. Uniform Distribution**

**What it is:**
All outcomes equally likely across a range.

**Example:** 
Random number between 0 and 1.

**Formula:**
- P(X) = constant for all X in [a, b]
- Outside [a, b]: P(X) = 0

**Why we need it:**
- Baseline assumption when you have no information
- Used for random sampling and initialization

**Where we use it:**
- Generate random noise (no bias)
- Shuffle data randomly
- Initialize neural network weights
- Random sampling from a range

**Insight for applications:**
- Use when outcomes have no reason to favor one value
- Often used to generate random data for testing
- Baseline for comparisons


### Quick Comparison

| Distribution | Type | Output | ML Use Case |
|---|---|---|---|
| Bernoulli | Discrete | 0 or 1 | Binary classification |
| Binomial | Discrete | 0 to n | Count successes |
| Uniform | Continuous | [a,b] | Random initialization |
| Normal | Continuous | Any real | Most phenomena |

**Key Insight:** The distribution you assume tells the model what to expect. Wrong assumption = bad predictions.

