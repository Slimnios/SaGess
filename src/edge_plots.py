
import matplotlib.pyplot as plt
import numpy as np





edge_numbers_train_unif_100ep = [0,1397,2588,3728,4773,5717,6746,7653,8464,9265,9879,10559,11103,11676,12203,12708,13131,13538,13981,14394,14806,15144,15546,15913,16236,16517,16708]
edge_numbers_train_unif_300ep = [0,1093,2113,3007,3800,4536,5243,5964,6698,7379,7975,8487,8950,9430,9913,10377,10755,11106,11483,
                                 11759,12090,12385,12658,12910,13146,13421,13669,13878,14082,14278,14469,14646,14815,14937,15089,
                                 15238,15404,15542,15657,15769,15889,16008,16098,16200,16308,16400,16474,16587,16669,16709]


edge_numbers_train_ego_100ep = [0,2805,5347,7266,8676,9951,10907,11818,12522,13241,13753,14270,14750,15246,15767,16124,16420,16697,16712]
edge_numbers_train_ego_300ep = [0,2961,5322,7257,8501,9777,10857,11682,12342,12901,13511,14064,14527,15001,15439,15772,16117,16396,16670,16709]

x_1 = range(0,32*len(edge_numbers_train_unif_100ep),32)
x_2 = range(0,32*len(edge_numbers_train_unif_300ep),32)
x_3 = range(0,32*len(edge_numbers_train_ego_100ep),32)
x_4 = range(0,32*len(edge_numbers_train_ego_300ep),32)

# plot lines
plt.plot(x_1, edge_numbers_train_unif_100ep, label = "Unif 100 epochs", color ='blue',linestyle = 'dashed')
plt.plot(x_2, edge_numbers_train_unif_300ep, label = "Unif 300 epochs", color ='blue')

plt.plot(x_3, edge_numbers_train_ego_100ep, label = "Ego 100 epochs", color = 'red',linestyle = 'dashed')
plt.plot(x_4, edge_numbers_train_ego_300ep, label = "Ego 300 epochs", color = 'red')


plt.plot(x_2,[16706 for i in range(len(x_2))], label = "Number of Real Edges", color='black')


plt.xlabel("Number of Graphs Generated")
plt.ylabel("Number of Edges")


plt.legend()
plt.show()