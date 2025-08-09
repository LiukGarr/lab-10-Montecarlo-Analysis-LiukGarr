import json
import random

import matplotlib.pyplot as plt
from pathlib import Path

# import numpy as np

from core.elements import Network

# Exercise Lab3: Network
ROOT = Path(__file__).parent.parent
INPUT_FOLDER = ROOT / 'resources'

# file_input = INPUT_FOLDER / 'nodes.json'
# file_input = INPUT_FOLDER / 'Base_topology' / 'nodes_full_fixed.json'
# f = open(file_input, 'r')
# data = json.load(f)

file_input1 = INPUT_FOLDER / 'Base_topology' / 'nodes_full_fixed.json'
# file_input1 = INPUT_FOLDER / 'Exam_topology' / 'full_network_fixed.json'
fixed_file = open(file_input1, 'r')
data1 = json.load(fixed_file)
file_input2 = INPUT_FOLDER / 'Base_topology' / 'nodes_full_flex.json'
# file_input2 = INPUT_FOLDER / 'Exam_topology' / 'full_network_flex.json'
flex_file = open(file_input2, 'r')
data2 = json.load(flex_file)
file_input3 = INPUT_FOLDER / 'Base_topology' / 'nodes_full_shannon.json'
# file_input3 = INPUT_FOLDER / 'Exam_topology' / 'full_network_shannon.json'
shan_file = open(file_input3, 'r')
data3 = json.load(shan_file)

nodes = []
vect_res_lat = []
vect_res_snr = []
total_capacity = 0.0
results = "SNR"
for nds in data1:
    nodes.append(nds)
net_fixed = Network(data1)
res_fixed = net_fixed.stream_w_matrix(results)
print(res_fixed)
net_flex = Network(data2)
res_flex = net_flex.stream_w_matrix(results)
print(res_flex)
net_shannon = Network(data3)
res_shannon = net_shannon.stream_w_matrix(results)
print(res_shannon)
#
xpoint = []
ypoint_fix = []
ypoint_flx = []
ypoint_shn = []
plt.figure(3)
for x in range(len(res_fixed)):
    xpoint.append(res_fixed[x][0])
    ypoint_fix.append(res_fixed[x][1])
    ypoint_flx.append(res_flex[x][1])
    ypoint_shn.append(res_shannon[x][1])
plt.plot(xpoint, ypoint_fix, xpoint, ypoint_flx, xpoint, ypoint_shn)
plt.title("Traffic matrix Saturation")
plt.xlabel("M")
plt.ylabel("Accepted connections")
plt.legend(['fixed-rate', 'flex-rate', 'shannon-rate'])
plt.grid()
plt.show()
#num_con = 100
# draw = net.draw()  # return the dataframe and the draw
# i = 1
# while i <= num_con:
#     node1 = random.choice(nodes)
#     node2 = random.choice(nodes)
#     while node2 == node1:
#         node2 = random.choice(nodes)
#     # print(f"Path between {node1} and {node2}")
#     dato_lat, dato_SNR, dato_Rb = net.stream(node1, node2, results)
#     if dato_lat != 0:
#         # print(f"{dato_lat}s")
#         vect_res_lat.append(float(dato_lat))
#         total_capacity += dato_Rb
#     if dato_SNR != "None":
#         # print(f"{dato_SNR}dB")
#         vect_res_snr.append(dato_SNR)
#     # if (dato_SNR != "None") and (dato_lat != 0):
#         # vect_res_lat.append(dato_lat)
#     i += 1
#
# print('Dataframe of all occupied channels between all possible nodes: \n', net.route_space)
# print(f"# of paths found for best {results}:", len(vect_res_lat))
# print(f"Average {results}:", round(sum(vect_res_snr)/len(vect_res_lat), 3), "dB")
# print(f"Tot. Capacity: {round(total_capacity,3)} Gbps; Average Cap: {round(total_capacity/len(vect_res_lat),3)} Gbps")
#
# fig, axs = plt.subplots(1, 2)
# fig.suptitle(f'Latency and SNR for best {results}')
# axs[0].hist(vect_res_lat, bins=i)
# axs[0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
# axs[0].set_title('Latency [s]')
# axs[0].grid()
# axs[1].hist(vect_res_snr, bins=i)
# axs[1].set_title('SNR [dB]')
# axs[1].grid()
#
# plt.show()
#
# file = open("Route_space.txt", "w")
# file.write(f"{net.weighted_paths().to_string()}")
# file.close()


fixed_file.close()
flex_file.close()
shan_file.close()
# Load the Network from the JSON file, connect nodes and lines in Network.
# Then propagate a Signal Information object of 1mW in the network and save the results in a dataframe.
# Convert this dataframe in a csv file called 'weighted_path' and finally plot the network.
# Follow all the instructions in README.md file
