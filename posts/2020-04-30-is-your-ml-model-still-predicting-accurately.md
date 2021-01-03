# Is Your Machine Learning Model Still Predicting Accurately?
## Use Direct Ratios for the measurement of the “model drift” — An example with Bitcoin price

When we build a machine learning model, we validate and test its accuracies/performance in various ways. Although the accuracy is quite high in the test set, the prediction will eventually drift by time. This is often caused by the future data “drift” to a different state from the training dataset. But how can we detect this drift effectively? This blogpost shows using direct ratio estimation to detect the change point with Bitcoin daily price data.
The codes for this post is available on my GitHub repo.
### Table of Contents
1. Change Point Detection Concept
2. Direct ratio (KL Divergence and Relative unconstrained Least-Squares Importance Fitting) Estimations
3. Example with Bitcoin Price Data
3.1. Introduction of Bitcoin and the Data
3.2. Applying the KL divergence estimation to detect the state change
3.3. State change threshold by using Bolinger bands
4. Conclusion and Next Steps
5. References