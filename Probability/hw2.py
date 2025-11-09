import sys
import math
import os


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X = dict()

    for i in range(26):
        key = chr(ord('A') + i)
        X[key] = 0

    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        while True:
            ch = f.read(1)
            if not ch:
                break
            if ch.isalpha():
                ch = ch.upper()
                X[ch] += 1
    
    return X



# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!

def main():
    letterFile = sys.argv[1]
    ePrior = float(sys.argv[2])
    sPrior = float(sys.argv[3])

    e,s = get_parameter_vectors()
    X = shred(os.path.join( os.getcwd(),letterFile))

    #Q1
    print('Q1')
    for alphabet in X:
        print(alphabet, ' ', X[alphabet])
    

    #Q2
    if X['A'] > 0 and e[0] > 0:
        X1_logE1 = (X['A'] * (math.log(e[0])))
        X1_logS1 = (X['A'] * (math.log(s[0])))
    print('Q2')
    print("{:.4f}".format(X1_logE1))
    print("{:.4f}".format(X1_logS1))


    #Q3
    def logProducts(X, p):
        total = 0

        for i in range(26):
            key = chr(ord('A') + i)
            if X[key] > 0 and p[i] > 0:
                total += X[key] * math.log(p[i])

        return total


    F_e = math.log(ePrior) + logProducts(X, e)
    F_s = math.log(sPrior) + logProducts(X, s)
    print('Q3')
    print("{:.4f}".format(F_e))
    print("{:.4f}".format(F_s))
    

    #Q4
    if (F_s - F_e) >= 100:
        P_e_when_X = 0
    elif (F_s - F_e) <= -100:
        P_e_when_X = 1
    else:
        P_e_when_X = 1 / (1 + math.exp(F_s - F_e))
    print('Q4')
    print("{:.4f}".format(P_e_when_X))




if __name__ == "__main__":
    main()


