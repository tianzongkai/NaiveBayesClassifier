from scipy.special import gamma
import numpy as np
import math
from numpy import linalg as la
from matplotlib import cm
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
b_star[0] = training_total - training_one_total + b
# print b_star
for y in [0,1]:
    a_star_y = []
    for d in range(training_x.shape[1]):
        a_star_y.append(a + np.sum(training_x.values[training_label_array == y,d]))
    a_star.append(a_star_y)
a_star = np.asarray(a_star)
# print a_star.shape

def gamma_division(numerator, denominator_1, denominator_2):
    """
    Calculate division of two gamma functions
        Gamma(numerator)/Gamma(denominator)/Gamma(exclusion)
    :param numerator: integer
    :param denominator_1: integer
    :param denominator_2: integer
    :return: float number
    """
    result = 1.0
    for i in range(denominator_1, numerator):
        if denominator_2 > 2:
            denominator_2 -= 1
            result /= denominator_2
        try:
            result *= i
        except:
            print "!!! ERROR: result: %.2f; i: %.3f" % (result, i)
    if denominator_2 > 2:
        result /= math.factorial(denominator_2-1)
    return float(result)

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


def firstTerm(x_star, y):
    """
    First term of predictive probability for label y*.
    :param x_star: a vector, input for test data
    :param y: a binary scalar label, 1/0
    :param a_star: a vector, posterior Gamma distribution G(a_star,b_star) for all independent lambda's
    :param b_star: posterior Gamma distribution G(a_star,b_star) for all independent lambda's
    :return: float number
    """
    # print "first term"
    result = 1.0
    # print b_star[y]+1
    for d in range(54):
        # print 'd, a_star[y][d]',d,a_star[y][d]
        if x_star[d] == 0:
            l = 1.0
        else:
            l = gamma_division(x_star[d] + a_star[y][d],
                           a_star[y][d],
                           x_star[d] + 1)
        m =((float(b_star[y])/(b_star[y] + 1))**a_star[y][d])
        r = (1.0/(b_star[y]+1))**(x_star[d])
        print 'l,m,r: ',l,m,r
        result *= (l * m * r)
        print 'first term result==', result
    return result

def firstTerm_ratio(x_star):
    """
    First term of predictive probability for label y*.
    :param x_star: a vector, input for test data
    :param y: a binary scalar label, 1/0
    :param a_star: a vector, posterior Gamma distribution G(a_star,b_star) for all independent lambda's
    :param b_star: posterior Gamma distribution G(a_star,b_star) for all independent lambda's
    :return: float number
    """
    # print "first term"
    ratio_result = 1.0
    for d in range(54):
        temp_result = [0.0, 0.0]
        for y in [0,1]:
            # print 'd, a_star[y][d]',d,a_star[y][d]
            if x_star[d] == 0:
                l = 1.0
            else:
                l = gamma_division(x_star[d] + a_star[y][d],
                               a_star[y][d],
                               x_star[d] + 1)
            m =((float(b_star[y])/(b_star[y] + 1))**a_star[y][d])
            r = (1.0/(b_star[y]+1))**(x_star[d])
            # print 'l,m,r: ',l,m,r
            temp_result[y] = math.log(l) + math.log(m) + math.log(r)
            print 'temp_result', temp_result
        # ratio_result *= (temp_result[0] / temp_result[1])
        # print 'first term result==', ratio_result
    return ratio_result

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
second_term_ratio = second_term_result[0] / second_term_result[1]
print "second_term:", second_term_result

def calculate_predictive_probability(x_star, y):
    first_term = firstTerm(x_star, y)
    # print 'first term:', first_term
    return  first_term * second_term_result[y]

def calculate_predictive_probability_log(x_star, y):
    first_term_log = firstTerm_log(x_star, y)
    # print 'first term:', first_term_log
    return  first_term_log + math.log(second_term_result[y])

def calculate_predictive_probability_ratio(x_star):
    first_term_ratio = firstTerm_ratio(x_star)
    # print 'first term:', first_term
    return  first_term_ratio * second_term_ratio


def predict_test_set():
    predict_labels = []
    n_v = 254
    for n in range(test_total):
    # for n in range(n_v,n_v + 1):
    #     print 'n', n
        x_star = test_x.values[n,:]
        # first_term_ratio = firstTerm_ratio(x_star)
        prob_one_log = calculate_predictive_probability_log(x_star, 1)
        prob_zero_log = calculate_predictive_probability_log(x_star, 0)
        if prob_one_log > prob_zero_log:
            predict_labels.append(1)
        else:
            predict_labels.append(0)
        # print "n: %d, prob_1_log: %.2f; prob_0_log: %.2f" % (n, prob_one_log, prob_zero_log)
    return predict_labels

def confusion_matrix():
    predict_labels = np.asarray(predict_test_set())
    print predict_labels.shape
    print test_label_array.shape
    confusion_matrix = {'tp':0,'tn':0,'fp':0,'fn':0}
    correct_prediction = 0.0
    for true_label, predict_label in zip(test_label_array, predict_labels):
        if true_label == 1:
            if predict_label == true_label:
                correct_prediction += 1
                confusion_matrix['tp'] += 1
            else: confusion_matrix['fn'] += 1
        else: # true_label == 0
            if predict_label == true_label:
                correct_prediction += 1
                confusion_matrix['tn'] += 1
            else: confusion_matrix['fp'] += 1
    accuracy = correct_prediction / test_total
    print 'accuracy:', accuracy
    print 'confusion matrix:', confusion_matrix


confusion_matrix()
