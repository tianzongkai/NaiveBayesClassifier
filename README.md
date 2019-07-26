Given a dataset X = {x1, . . . , xN } representing N emails and labels Y = {y1, . . . , yN}. Each email xn is a 54-dimensional vector having a label y. The label y = 0 indicates “non-spam” and y = 1 indicates “spam”. Each dimension x ∈ N is modeled as i.i.d. Poisson(λ). Unknow λ is modeled as λ ∼ Gamma(a, b). 

The labels are modeled as yn ∼ Bernoulli(π), the label bias assume the prior π ∼ Beta(e, f).

Let (x∗, y∗) be a new test pair. The goal is to predict y∗ given x∗.

The task it to code a naive Bayes classifier for distinguishing spam from non-spam emails.

Steps:
1. calculate the posterior of λ 
2. calculate the predictive distribution on a new observation (not label prediction, but probability distribution of
generating new observation from posterior Possion distribution) p(x*|x1...xn) = \int [p(x*|λ)p(λ|x1...xn)] dλ
3. compute probability of predicting lable y* = k ∈ {0, 1}: 
  p(y*=k|x*,X,Y) \propto p(x*|y*=k, {xi:yi=k}) * p(y*=k|Y)

(HW1 - Bayesian Machine Learnig)
