#!/usr/bin/env python
#coding:utf8
import numpy as np
import random
import pdb,sys
train_data = "../data/ml-1m/ratings.dat"
hidden_vector_size = 100
alpha = 0.01

def MatrixFactorization():
    #groupby id
    uniq_user = []
    for line in open(train_data):
        data = map(lambda y :int(y), line.strip().split("::")[:3])
        if not uniq_user or data[0] != uniq_user[-1][0]:
            uniq_user.append([data[0],[data[1:]]])
        else:
            uniq_user[-1][1].append(data[1:])

    users_num = len(uniq_user)
    #create nets
    #layer_0
    user_param_dict = {}
    for k,v in uniq_user:
        user_param_dict[k] = np.random.randn(hidden_vector_size) 
    
    #layer_1
    movie_param_dict = {}
    for k,vs in uniq_user:
        for v in vs:
            if v[0] in movie_param_dict:
                continue
            movie_param_dict[v[0]] = np.random.randn(hidden_vector_size) 
        
    print users_num
    #SGD
    max_loop = int(users_num * 60 )
    for loop in range(max_loop):
        index = random.randint(0,users_num - 1)
        userid,rates = uniq_user[index]
        w0 = user_param_dict[userid]
        w0_g = np.zeros(hidden_vector_size)
        #pdb.set_trace()
        for movieid,target in rates:
            w1 = movie_param_dict[movieid]  
            g = alpha * (np.dot(w0,w1) - target) 
            movie_param_dict[movieid] -= g * w0
            w0_g += g * w1
        user_param_dict[userid] -= w0_g /len(rates) 
        if loop % 500 == 0:
            print "loop:{0}/{1}".format(loop,max_loop)
            sys.stdout.flush()
    totoal_error = 0
    total_num = 0
    for userid,rates in uniq_user:
        w0 = user_param_dict[userid]
        for movieid,target in rates:
            w1 = movie_param_dict[movieid]  
            score = np.dot(w0,w1)
            totoal_error += np.abs((score - target))
            total_num += 1
    print "total sqrt error:",totoal_error / total_num

    fw = open("../data/user_vector.txt","w")
    for k,v in user_param_dict.iteritems():
        fw.write(str(k))
        for dim in v:
            fw.write(" "+str(dim))
        fw.write("\n")
    fw.close()
    
    fw = open("../data/movie_vector.txt","w")
    for k,v in movie_param_dict.iteritems():
        fw.write(str(k))
        for dim in v:
            fw.write(" "+str(dim))
        fw.write("\n")
    fw.close()

if __name__ == '__main__':
    MatrixFactorization()








