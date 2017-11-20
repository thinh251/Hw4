import conv_train
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
import util
import sklearn
import datetime

if not conv_train.validate_arguments(sys.argv):
    sys.exit("Invalid arguments")

results = []
start = datetime.datetime.now()
for i in range(1,int(sys.argv[4])):
    a = datetime.datetime.now()
    print "-------------------------------------------"
    print "-------------------------------------------"
    print " for max updates : ",i*5
    print "-------------------------------------------"
    sys.argv[4] = i*5
    results.append(conv_train.train(sys.argv[1], sys.argv[2],
                                    float(sys.argv[3]),
                                    int(sys.argv[4]), sys.argv[5].strip(),
          sys.argv[6], sys.argv[7]))
    b = start = datetime.datetime.now()
    print "Time required for this step: "+str(b-a)


x_value = []
y_value = []
z_value = []
for i in range(len(results)):
    x_value.append(results[i][0])
    y_value.append(results[i][1])
    z_value.append(results[i][2])
print x_value
print y_value
print z_value
plt.title("Q1. for '"+sys.argv[5].strip()+", cost function:"+sys.argv[1])
plt.xlabel("Max Updates")
plt.ylabel("Cost")
training_line, =plt.plot(x_value,y_value,color='blue',label="Training Cost",
         linestyle='dashed')
validation_line, = plt.plot(x_value,z_value,color='green',label="Validation "
                                                              "Cost",
         linestyle='dashed')
plt.legend(handles=[training_line,validation_line],loc=2)
end = datetime.datetime.now()
print "Total time required "+str(end-start)
plt.savefig("Q1. for '"+sys.argv[5].strip()+", cost function:"+sys.argv[1])
