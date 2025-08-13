import json
import random
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

# import numpy as np

from core.elements import Network

# Exercise Lab3: Network
ROOT = Path(__file__).parent.parent
INPUT_FOLDER = ROOT / 'resources'

# file_input = INPUT_FOLDER / 'nodes.json'
# file_input = INPUT_FOLDER / 'Base_topology' / 'nodes_full_fixed.json'
# f = open(file_input, 'r')
# data = json.load(f)

# file_input1 = INPUT_FOLDER / 'Base_topology' / 'nodes_full_fixed.json'
file_input1 = INPUT_FOLDER / 'Exam_topology' / 'not_full_network_fixed.json'
fixed_file = open(file_input1, 'r')
data1 = json.load(fixed_file)
# file_input2 = INPUT_FOLDER / 'Base_topology' / 'nodes_full_flex.json'
file_input2 = INPUT_FOLDER / 'Exam_topology' / 'not_full_network_flex.json'
flex_file = open(file_input2, 'r')
data2 = json.load(flex_file)
# file_input3 = INPUT_FOLDER / 'Base_topology' / 'nodes_full_shannon.json'
file_input3 = INPUT_FOLDER / 'Exam_topology' / 'not_full_network_shannon.json'
shan_file = open(file_input3, 'r')
data3 = json.load(shan_file)

nodes = []
vect_res_lat = []
vect_res_snr = []
total_capacity = 0.0
MC_runs = 50
scenario = "A"  # A = Single Traffic Matrix scenario; B = Network Congestion
results = "SNR"

xpoint = []
av_bit_rate = []
min_bit_rate = []
max_bit_rate = []
tot_bit_rate = []
av_GSNR = []
min_GSNR = []
max_GSNR = []
block_ev = []

net = Network(data3)

if scenario == "A":
    #   data1-EXAM-FIXED-RATE    --> M = 17
    #   data2-EXAM-FLEX-RATE     --> M = 30
    #   data3-EXAM-SHANNON       --> M = 45
    net.set_M_and_MC_runs(45, MC_runs)
    res = net.stream_w_matrix(results)

    for x in range(len(res)):
        xpoint.append(res[x][0])
        min_bit_rate.append(res[x][1])
        max_bit_rate.append(res[x][2])
        av_bit_rate.append(res[x][3])
        tot_bit_rate.append(res[x][4])
        min_GSNR.append(res[x][5])
        max_GSNR.append(res[x][6])
        av_GSNR.append(res[x][7])
        block_ev.append(res[x][8])

    print(f"Average value of total capacity: ", np.average(tot_bit_rate))
    plt.figure(1)
    plt.title("Total Capacity")
    plt.xlabel("M")
    plt.ylabel("Capacities [Gbps]")
    plt.plot(xpoint, tot_bit_rate)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.grid()

    plt.figure(2)
    plt.title(f'Per-link Average Capacity:')
    plt.plot(xpoint, av_bit_rate)
    plt.ylabel('[Gbps]')
    plt.xlabel("Cycle")
    plt.grid()

    plt.figure(3)
    plt.title(f'Per-link Average GNSR:')
    plt.plot(xpoint, av_GSNR)
    plt.ylabel('[dB]')
    plt.xlabel("Cycle")
    plt.grid()

    plt.figure(4)
    plt.title(f'Per-link Min Capacity:')
    plt.plot(xpoint, min_bit_rate)
    plt.ylabel('[Gbps]')
    plt.xlabel("Cycle")
    plt.grid()

    plt.figure(5)
    plt.title(f'Per-link Min GNSR:')
    plt.plot(xpoint, min_GSNR)
    plt.ylabel('[dB]')
    plt.xlabel("Cycle")
    plt.grid()

    plt.figure(6)
    plt.title(f'Per-link Max Capacity:')
    plt.plot(xpoint, max_bit_rate)
    plt.ylabel('[Gbps]')
    plt.xlabel("Cycle")
    plt.grid()

    plt.figure(7)
    plt.title(f'Per-link Max GSNR:')
    plt.plot(xpoint, max_GSNR)
    plt.ylabel('[dB]')
    plt.xlabel("Cycle")
    plt.grid()

    plt.figure(8)
    plt.title("Blocking event count")
    plt.xlabel("M")
    plt.ylabel("Occurrences")
    plt.plot(xpoint, block_ev)
    plt.grid()

    plt.show()
elif scenario == "B":
    res = net.stream_w_matrix(results)

    for x in range(len(res)):
        xpoint.append(res[x][0])
        min_bit_rate.append(res[x][1])
        max_bit_rate.append(res[x][2])
        av_bit_rate.append(res[x][3])
        tot_bit_rate.append(res[x][4])
        min_GSNR.append(res[x][5])
        max_GSNR.append(res[x][6])
        av_GSNR.append(res[x][7])
        block_ev.append(res[x][8])

    # print(min_max_bit_rate)
    plt.figure(1)
    plt.title("Total Capacity")
    plt.xlabel("M")
    plt.ylabel("Capacities [Gbps]")
    plt.plot(xpoint, tot_bit_rate)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.grid()

    plt.figure(2)
    plt.title(f'Per-link Average Capacity:')
    plt.plot(xpoint, av_bit_rate)
    plt.ylabel('[Gbps]')
    plt.xlabel("Cycle")
    plt.grid()

    plt.figure(3)
    plt.title(f'Per-link Average GNSR:')
    plt.plot(xpoint, av_GSNR)
    plt.ylabel('[dB]')
    plt.xlabel("Cycle")
    plt.grid()

    plt.figure(4)
    plt.title(f'Per-link Min Capacity:')
    plt.plot(xpoint, min_bit_rate)
    plt.ylabel('[Gbps]')
    plt.xlabel("Cycle")
    plt.grid()

    plt.figure(5)
    plt.title(f'Per-link Min GNSR:')
    plt.plot(xpoint, min_GSNR)
    plt.ylabel('[dB]')
    plt.xlabel("Cycle")
    plt.grid()

    plt.figure(6)
    plt.title(f'Per-link Max Capacity:')
    plt.plot(xpoint, max_bit_rate)
    plt.ylabel('[Gbps]')
    plt.xlabel("Cycle")
    plt.grid()

    plt.figure(7)
    plt.title(f'Per-link Max GSNR:')
    plt.plot(xpoint, max_GSNR)
    plt.ylabel('[dB]')
    plt.xlabel("Cycle")
    plt.grid()

    plt.figure(8)
    plt.title("Blocking event count")
    plt.xlabel("M")
    plt.ylabel("Occurrences")
    plt.plot(xpoint, block_ev)
    plt.grid()

    plt.show()
else:
    print("Unknown scenario")
# print(res_fixed)    # Constant after M = 15
# net_flex = Network(data2)
# res_flex = net_flex.stream_w_matrix(results)
# print(res_flex)     # Constant after M = 25
# net_shannon = Network(data3)
# res_shannon = net_shannon.stream_w_matrix(results)
# print(res_shannon)      # Constant after M = 40
#
#num_con = 100
# for nds in data1:
#     nodes.append(nds)
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
