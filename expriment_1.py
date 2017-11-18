import conv_train
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys

if not conv_train.validate_arguments(sys.argv):
    sys.exit("Invalid arguments")

results = []

for i in range(1,int(sys.argv[4])):
    print "-------------------------------------------"
    print "-------------------------------------------"
    print " for max updates : ",i*10
    print "-------------------------------------------"
    sys.argv[4] = i*10
    results.append(conv_train.train())


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
plt.title("Q1")
plt.xlabel("Max Updates")
plt.ylabel("Cost")
training_line, =plt.plot(x_value,y_value,color='blue',label="Training Cost",
         linestyle='dashed')
validation_line, = plt.plot(x_value,z_value,color='green',label="Validation "
                                                              "Cost",
         linestyle='dashed')
plt.legend(handles=[training_line,validation_line],loc=2)
plt.show()