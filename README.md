Given a dataset {x1, . . . , xN }. Each data point xn is a 54-dimensional vector representing an email has a label 
y with y = 0 indicating “non-spam” and y = 1 indicating “spam”. Each dimension x ∈ N. You model it as i.i.d. 
Poisson(λ). Since you don’t know λ, you model it as λ ∼ Gamma(a, b). 

We model the labels as yn ∼ Bernoulli(π), the label bias assume the prior π ∼ Beta(e, f).

Let (x∗, y∗) be a new test pair. The goal is to predict y∗ given x∗, y∗ = y ∈ {0, 1}.

The task it to code a naive Bayes classifier for distinguishing spam from non-spam emails.


(HW1 - Bayesian Machine Learnig)
