#/usr/bin/python
#coding:utf8

from pyspark import SparkContext
from math import log
import os


all_docs_num = 0
idf_dict_bc = None

def mapOneLineWordCount(line):
    data = line.strip().split()
    idf_dict = idf_dict_bc.value
    mydict = {}
    for word in data:
        if word not in mydict:
            mydict[word] = 1
        else:
            mydict[word] += 1
    word_size = len(data)
    result = []
    for k,v in mydict.iteritems():
        idf_count = idf_dict[k]
        tf_idf = (float(v)/word_size) * log(float(all_docs_num)/(idf_count))  
        result.append((k,tf_idf))
    result.sort(key = lambda x:-x[1])
    result = map(lambda x:x[0]+"|"+str(x[1]),result) 
    return ",".join(result) + "\n"

def main(sc):
    global idf_dict_bc,all_docs_num
    path = os.path.abspath('..') + "/data/"
    input_file = "file://" + path + "10000_lines.txt" 
    output_dir = "file://" + path + "result/" 
    
    data = sc.textFile(input_file)
    data.persist()
    all_docs_num = data.count()
    idf = data.flatMap(lambda x:list(set(x.strip().split())))\
               .map(lambda x:(x,1))\
               .reduceByKey(lambda x,y:x+y)
    idf_dict = dict(idf.collect())
    idf_dict_bc = sc.broadcast(idf_dict)
    final_result = data.map(mapOneLineWordCount) 
   
    final_result.saveAsTextFile(output_dir)

if __name__ == '__main__': 
    sc = SparkContext()
    main(sc)






