import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
from matplotlib.pyplot import MultipleLocator

x1=['0', '0.6', '0.7', '0.8', '0.9']
y1=[0.557, 0.559, 0.678, 0.718, 0.709]
# x2=[0, 0.6, 0.7, 0.8, 0.9]
x2=['0', '0.6', '0.7', '0.8', '0.9']
y2=[0.628, 0.562, 0.766, 0.842, 0.854]
x3=['0', '0.6', '0.7', '0.8', '0.9']
y3=[0.437, 0.470, 0.447, 0.509, 0.432]

x4=['0', '0.6', '0.7', '0.8', '0.9']
y4=[0.624, 0.633, 0.711, 0.798, 0.698]
x5=['0', '0.6', '0.7', '0.8', '0.9']
y5=[0.629, 0.648, 0.738, 0.806, 0.750]

# x=np.arange(20,350)
l1=plt.plot(x1,y1,'b-',label='AUC')
l2=plt.plot(x2,y2,'g-',label='+Recall')
l3=plt.plot(x3,y3,'r--',label='-Recall')
l4=plt.plot(x4,y4,'orange',label='MAP')
l5=plt.plot(x5,y5,'purple',label='MRR')
# l4=plt.plot(x3,y3,'b--',label='-Recall')
# plt.plot(x1,y1,'ro-',x2,y2,'g+-',x3,y3,'b^-')
# plt.title('Change of performance under different similarity cut-offs')
plt.xlabel('similarity cut-off', fontsize=17)
plt.ylabel('Performance', fontsize=17)
plt.xticks(fontsize=15, )
plt.yticks(fontsize=15, )
plt.legend()

# ax=plt.gca()
# plt.gca().xaxis.set_major_locator(MultipleLocator(1))

# ax.xaxis.set_major_locator(MultipleLocator(1))
# plt.ylim(0, 1)
# plt.xticks(np.array([0, 0.6, 0.7, 0.8, 0.9]))
plt.show()
