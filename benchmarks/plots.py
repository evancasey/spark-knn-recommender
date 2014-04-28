import matplotlib.pyplot as plt
plt.plot([1,2,3,4,5,6,7,8],[1,1.89,2.4,3,3.2,3.35,3.43,3.51])
plt.ylabel('Relative speedup')
plt.xlabel('Number of partitions')
plt.axis([1,8,1,8])
plt.show()

# plt.plot([2,4,6,8,10,13,16,20],[1,2,2.92,3.86,5.1,6.98,8.08,9.6])
# plt.ylabel('Relative speedup')
# plt.xlabel('Number of x1.large instances')
# plt.axis([2,20,1,10])
# plt.show()

