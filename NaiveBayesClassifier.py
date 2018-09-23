import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

vocabulary = ["make", "address", "all", "3d", "our", "over", "remove", "internet", "order", "mail", "receive", "will", "people", "report", "addresses", "free", "business", "email", "you", "credit", "your", "font", "000", "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857", "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs", "meeting", "original", "project", "re", "edu", "table", "conference", ";", "(", "[", "!", "$", "#"]
a = 1
b = 1
e = 1
f = 1
training_x = pd.read_csv("data/X_train.csv", header=None)
training_label = pd.read_csv("data/label_train.csv", header=None)
training_label_array = training_label.values[:,0]
test_x = pd.read_csv("data/X_test.csv", header=None)
test_label = pd.read_csv("data/label_test.csv", header=None)
test_label_array = test_label.values[:,0]


training_total = training_label_array.shape[0]
training_one_total = np.sum(training_label_array)
training_zero_total = training_total - training_one_total

test_total = test_label.shape[0]

a_star = [] # shape (2,54)
b_star = [0,0] # = [N_0 + b = 2510, N_1 + b = 1632]
b_star[1] = b + training_one_total
b_star[0] = b + training_total - training_one_total
# print b_star
for y in [0,1]:
    a_star_y = []
    for d in range(training_x.shape[1]):
        a_star_y.append(a + np.sum(training_x.values[training_label_array == y,d]))
    a_star.append(a_star_y)
a_star = np.asarray(a_star)
# print a_star.shape


def gamma_division_log(numerator, denominator_1, denominator_2):
    """
    Calculate division of two gamma functions
        Gamma(numerator)/Gamma(denominator)/Gamma(exclusion)
    :param numerator: integer
    :param denominator_1: integer
    :param denominator_2: integer
    :return: float number
    """
    log_result = 0.0
    for i in range(denominator_1, numerator):
        try:
            log_result += math.log(i)
        except:
            print "!!! ERROR: log_result: %.2f; i: %.3f" % (log_result, i)
    for i in range(denominator_2):
        log_result -= math.log(denominator_2)
    return log_result

def firstTerm_log(x_star, y):
    """
    First term of predictive probability for label y*.
    :param x_star: a vector, input for test data
    :param y: a binary scalar label, 1/0
    :param a_star: a vector, posterior Gamma distribution G(a_star,b_star) for all independent lambda's
    :param b_star: posterior Gamma distribution G(a_star,b_star) for all independent lambda's
    :return: float number
    """
    # print "first term"
    log_result = 0.0
    for d in range(54):
            # print 'd, a_star[y][d]',d,a_star[y][d]
        if x_star[d] == 0:
            l_log = 0.0
        else:
            l_log = gamma_division_log(x_star[d] + a_star[y][d],
                           a_star[y][d],
                           x_star[d] + 1)
        m_log = a_star[y][d] * math.log(float(b_star[y])/(b_star[y] + 1))
        r_log = x_star[d] * math.log(1.0/(b_star[y]+1))
        try:
            log_result += l_log + m_log + r_log
        except:
            print 'l,m,r: ',l_log, m_log, r_log
        # print 'log_result', log_result
        # log_result *= (temp_result[0] / temp_result[1])
        # print 'first term result==', log_result
    return log_result

def secondTerm(y):
    if y == 1:
        return float(e + training_one_total)/(training_total + e + f)
    else:
        return float(f + training_zero_total)/(training_total + e + f)

second_term_result = [secondTerm(0), secondTerm(1)]

def predictive_probability_portion_log(x_star, y):
    first_term_log = firstTerm_log(x_star, y)
    # print 'first term:', first_term_log
    return  first_term_log + math.log(second_term_result[y])

def predict_test_set():
    log_probabilities = [] # shape(461,2)
    predict_labels = []
    for n in range(test_total):
        x_star = test_x.values[n,:]
        prob_one_log = predictive_probability_portion_log(x_star, 1)
        prob_zero_log = predictive_probability_portion_log(x_star, 0)
        log_probabilities.append([prob_zero_log, prob_one_log])
        if prob_one_log > prob_zero_log:
            predict_labels.append(1)
        else:
            predict_labels.append(0)
    return predict_labels, log_probabilities

def confusion_matrix():
    predict_labels = np.asarray(predict_test_set()[0])
    miss_classified_indxes = []
    print 'test set size:', test_label_array.shape[0]
    confusion_matrix = {'tp':0,'tn':0,'fp':0,'fn':0}
    correct_prediction = 0.0
    email_index = 0
    for true_label, predict_label in zip(test_label_array, predict_labels):
        if true_label == 1:
            if predict_label == true_label:
                correct_prediction += 1
                confusion_matrix['tp'] += 1
            else:
                confusion_matrix['fn'] += 1
                miss_classified_indxes.append(email_index)
        else: # true_label == 0
            if predict_label == true_label:
                correct_prediction += 1
                confusion_matrix['tn'] += 1
            else:
                confusion_matrix['fp'] += 1
                miss_classified_indxes.append(email_index)
        email_index += 1
    accuracy = correct_prediction / test_total
    print 'accuracy:', accuracy
    print 'confusion matrix:', confusion_matrix
    return miss_classified_indxes

def plot_4c():
    expected_values =[a_star[ind]/float(b_star[ind]) for ind in [0,1]]
    miss_classified_indxes = confusion_matrix()
    plot_index = miss_classified_indxes[3:6]
    predictive_prob = []
    plt.figure(figsize=(16, 9))
    for ind in plot_index:
        x_star = test_x.values[ind,:]
        prob_one_portion = math.exp(predictive_probability_portion_log(x_star, 1))
        prob_zero_portion = math.exp(predictive_probability_portion_log(x_star, 0))
        prob_one = prob_one_portion / (prob_one_portion + prob_zero_portion)
        prob_zero = prob_zero_portion / (prob_one_portion + prob_zero_portion)
        predictive_prob.append([prob_zero, prob_one])
        plt.plot(vocabulary, test_x.values[ind,:].tolist(),label='email index = %d, P(0) = %.2f'%(ind, prob_zero))
    for ind in [0,1]:
        plt.plot(vocabulary, expected_values[ind], label='expected lambda_%d' % ind)
    for idx, prob in zip(plot_index, predictive_prob):
        print "email %d, true label: %d,  p(0) = %.2f, p(1) = %.2f" % (idx, test_label_array[idx], prob[0], prob[1])

    plt.xticks(np.arange(len(vocabulary)), vocabulary, rotation= 'vertical', fontsize=11)
    plt.yticks(np.arange(0,25,2), fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Question 4c: feature plot of three miss-classied emails", fontsize=14)
    plt.show()


def plot_4d():
    expected_values =[a_star[idx]/float(b_star[idx]) for idx in [0,1]]
    predict_labels, log_probablities = predict_test_set() # log_probabilities: shape(461,2)
    log_prob_abs_difference = [abs(prob[0] - prob[1]) for prob in log_probablities]
    log_prob_abs_difference = np.asarray(log_prob_abs_difference)
    log_prob_abs_difference_sorted_idx = np.argsort(log_prob_abs_difference)
    predictive_prob = []
    plt.figure(figsize=(16, 9))
    for idx in log_prob_abs_difference_sorted_idx[:3]:
        x_star = test_x.values[idx, :]
        prob_one_portion = math.exp(predictive_probability_portion_log(x_star, 1))
        prob_zero_portion = math.exp(predictive_probability_portion_log(x_star, 0))
        prob_one = prob_one_portion / (prob_one_portion + prob_zero_portion)
        prob_zero = prob_zero_portion / (prob_one_portion + prob_zero_portion)
        predictive_prob.append([prob_zero, prob_one])
        plt.plot(vocabulary, test_x.values[idx, :].tolist(), label='email index = %d, P(0) = %.2f' % (idx, prob_zero))
    for idx in [0, 1]:
        plt.plot(vocabulary, expected_values[idx], label='expected lambda_%d' % idx)
    for idx, prob in zip(log_prob_abs_difference_sorted_idx[:3], predictive_prob):
        print "email %d, true label: %d,  p(0) = %.2f, p(1) = %.2f" % (idx, test_label_array[idx], prob[0], prob[1])
    plt.xticks(np.arange(len(vocabulary)), vocabulary, rotation= 'vertical', fontsize=11)
    plt.yticks(np.arange(0,33,2), fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Question 4d: feature plot of three most ambiguous predictions", fontsize=14)
    plt.show()

plot_4c()
plot_4d()