s# import json
import math
import random
import pandas as pd
from math import *
from scipy.special import erfcinv
from scipy.constants import Planck as h
from core.parameters import *
from core.math_utils import *
import matplotlib.pyplot as plt

# pd.set_option("display.max_columns", )
list_availability = []
occupied = []
occupied_channels = []
ch_av_mat = []
arrow = '->'
channels = 10


class Signal_information(object):
    def __init__(self, sl, sn, path):
        self._signal_pow = 1e-3
        self._noise_pow = sn
        self._latency = sl
        self._path = path
        pass

    @property
    def signal_power(self):
        return self._signal_pow

    @signal_power.setter
    def signal_power(self, sig_pow):
        self._signal_pow = sig_pow

    def update_signal_power(self, increment_sp):
        self._signal_pow += increment_sp

    @property
    def noise_power(self):
        return self._noise_pow

    @noise_power.setter
    def noise_power(self, sig_np):
        self._noise_pow = sig_np

    def update_noise_power(self, increment_np):
        self._noise_pow += increment_np

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, nl):
        self._latency = nl

    def update_latency(self, increment_lat):
        self._latency += increment_lat

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, selected_path):
        self._path = selected_path

    def update_path(self):
        tmp_list_path = [self._path[0][1:]]
        self._path = tmp_list_path


class Lightpath(Signal_information):
    def __init__(self, sl, sn, path):
        super().__init__(sl, sn, path)
        self._channel = 0
        self._Rs = R_s
        self._df = df
        self._ISNR = 0.0
        self._GSNR = 0.0
        self._new_noise = 0.0
        pass

    def update_ISNR(self, ISNR):
        self._ISNR += ISNR

    @property
    def ISNR(self):
        return self._ISNR

    @property
    def GSNR(self):
        return self._GSNR

    @GSNR.setter
    def GSNR(self, GSNR):
        self._GSNR = GSNR

    @property
    def new_noise(self):
        return self._new_noise

    @new_noise.setter
    def new_noise(self, n_noise):
        self._new_noise = n_noise

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, ch_found):
        self._channel = ch_found
        pass


class Connection(object):
    def __init__(self, node1, node2, sign_pow):
        self._input = node1
        self._output = node2
        self._signal_power = sign_pow
        self._latency = 0.0
        self._snr = 0.0
        self._bit_rate = 0.0
        pass

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, conn_lat):
        self._latency = conn_lat

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, conn_snr):
        self._snr = conn_snr

    @property
    def bit_rate(self):
        return self._bit_rate

    @bit_rate.setter
    def bit_rate(self, conn_bit_r):
        self._bit_rate = conn_bit_r


class Node(object):
    def __init__(self, lab_nds, pos, connected, txrx):
        self._lab_nds = lab_nds
        self._pos = pos
        self._connected = connected
        self._next_nds = 0
        self._switching_mat = None
        self._copy_of_line = {}
        self._copy_of_info = {}
        if txrx == "":
            self._transceiver = "fixed-rate"
        else:
            self._transceiver = txrx
        pass

    @property
    def label(self):
        return self._lab_nds

    @property
    def position(self):
        return self._pos

    @property
    def connected_nodes(self):
        return self._connected

    @property
    def transceiver(self):
        return self._transceiver

    @property
    def successive(self):
        return self._next_nds

    @successive.setter
    def successive(self, next_line):
        self._next_nds = next_line
        pass

    @property
    def switching_matrix(self):
        return self._switching_mat

    @switching_matrix.setter
    def switching_matrix(self, sw_m):
        self._switching_mat = sw_m

    def copy_of_line(self, line_dict):
        self._copy_of_line = line_dict

    def copy_of_info(self, info_dict):
        self._copy_of_info = info_dict

    def propagate(self, signal_info, nds, mod_pow):
        # new_pow = signal_info.signal_power
        if len(signal_info.path) != 1:
            pos = signal_info.path.index(nds)
            id_line = signal_info.path[pos] + signal_info.path[pos + 1]
            return id_line
        else:
            signal_info.update_path()
            id_line = "XX"
            print(f"Node Probe: {id_line}")
            return id_line


class Line(object):
    def __init__(self, lab_line, pos1, pos2):
        self._length = 0.0
        self._next_lines = []
        self._arr_1 = []
        self._arr_2 = []
        self.occupied_lines = []
        self.occupied_channels = []
        self.all_lines = []
        self._lab_line = lab_line
        self._arr_1 = pos1
        self._arr_2 = pos2
        self.correct = 1
        self._n_amplifiers = 0      # length/80km
        self._noise_figure = db2lin(5.5)    # lin
        self._gain = db2lin(16)             # lin
        self._ASE = 0.0
        self._nli = 0.0
        self._eta_nli = 0.0
        self._Pxt = 0.0
        diff_x = pow((float(self._arr_2[0]) - float(self._arr_1[0])), 2)
        diff_y = pow((float(self._arr_2[1]) - float(self._arr_1[1])), 2)
        self._length = sqrt(diff_x + diff_y)
        self._n_amplifiers = ceil(self._length/Space_fact)
        pass

    @property
    def label(self):
        return self._lab_line

    @property
    def length(self):
        return self._length

    @property
    def successive(self):
        return self._next_lines

    @successive.setter
    def successive(self, next_node):
        self._next_lines = next_node
        pass

    @property
    def n_amplifiers(self):
        return self._n_amplifiers

    def latency_generation(self):
        latency_gen = self.length / (c * 2 / 3)
        return latency_gen

    def noise_generation(self, sign_pow):
        noise = 1e-9 * self.length * sign_pow
        # print(noise)
        return noise

    def new_noise_generation(self, ASE, NLI, XT):
        noise = ASE + NLI + XT
        # print(noise)
        return noise

    def propagate(self, sig_info, nds):
        latency = self.latency_generation()
        noise = self.noise_generation(sig_info.signal_power)
        sig_info.update_latency(latency)
        sig_info.update_noise_power(noise)
        pos = sig_info.path.index(nds)
        id_node = sig_info.path[pos + 1]
        return id_node

    def line_occupied(self):
        self.occupied_lines = occupied_channels
        return self.occupied_lines

    def line_correctness(self):
        return self.correct

    def line_state(self, state):
        if state == 0:
            i = True
            if len(occupied_channels) == 0:
                occupied.append(self._lab_line)
            #   line_occ = [line, num_of_occurrence]
                line_occ = [self._lab_line, 1]
                occupied_channels.append(line_occ)
            else:
                for occ_lines in occupied:
                    if self._lab_line == occ_lines:
                        occ_ind = occupied.index(occ_lines)
                        old_value = occupied_channels[occ_ind][1]
                        occupied_channels[occ_ind][1] = old_value + 1
                        if occupied_channels[occ_ind][1] > channels:
                            occupied_channels[occ_ind][1] = "FULL"
                        i = False
                if i:
                    occupied.append(self._lab_line)
                    line_occ = [self._lab_line, 1]
                    occupied_channels.append(line_occ)
            # print(f"{occupied_channels}")
        pass

    def ase_generation(self):
        self._ASE = self._n_amplifiers * (h * f * B_n * self._noise_figure * (self._gain - 1))
        return self._ASE

    def nli_generation(self, sign_pow):
        N_span = self._n_amplifiers - 1
        alpha_lin = alpha_dB / (20 * log10(e))
        L_eff = 1 / (2 * alpha_lin)
        f1 = 16 / (27 * pi)
        f2 = pow(pi, 2) / 2
        f3 = b2 * pow(R_s, 2) / alpha_lin
        f4 = pow(channels, 2 * R_s / df)
        f5 = (alpha_lin / b2) * pow((gam * L_eff), 2) / (pow(R_s, 3))
        eta_nli = f1 * log(f2 * f3 * f4) * f5
        # print(f1, f2*f3*f4, f5, eta_nli)
        # print(sign_pow, B_n, eta_nli, N_span)
        nli = pow(sign_pow, 3) * B_n * eta_nli * N_span
        # print(nli)
        return nli, eta_nli

    def crosstalk_generation(self, sign_pow, sel_ch):
        if sel_ch <= 8:
            if sel_ch >= 1:
                self._Pxt = 2*sign_pow * CT_coef
            else:
                self._Pxt = sign_pow * CT_coef
        else:
            self._Pxt = sign_pow * CT_coef

        return self._Pxt

    def optimized_launch_power(self, path_info):
        ASE = self.ase_generation()
        NLI, eta_NLI = self.nli_generation(path_info)
        power_opt = pow(ASE/(2*eta_NLI*(self._n_amplifiers-1)*B_n), (1/3))
        return power_opt


class Network(object):
    def __init__(self, data):
        self._line_occ = []
        self._nodes = {}
        self._lines = {}
        self._node2line = {}
        self._line2node = {}
        self.path = []
        self.lines_path = []
        self.path_mod = []
        self._all_paths = []
        self._paths_ch_available = []
        self._paths_chs_copy = []
        self._sign_info = {}
        self._route_space = 0
        self._path_conn = {}
        self._optimal_pow = 0.0
        self._traffic_matrix = []
        self._M = 1
        self._num_nodes = 0
        self._list_of_nodes = []
        self._num_stream = 0
        # self._switching_mat = None
        for nds in data:
            self._list_of_nodes.append(nds)
            self._nodes[nds] = Node(nds, data[str(nds)]['position'], data[str(nds)]['connected_nodes'],
                                    data[str(nds)]['transceiver'])
            self._num_nodes += 1
            # print(nds, self._nodes[nds].transceiver)
        # self._traffic_matrix = 100 * self._M * np.ones((self._num_nodes, self._num_nodes))
        # self._traffic_matrix[np.diag_indices_from(self._traffic_matrix)] = 0
        # print(self._traffic_matrix)

        for nds in self._nodes:
            for con_nds in self._nodes[nds].connected_nodes:
                line = nds + con_nds
                self._lines[line] = Line(line, self._nodes[nds].position, self._nodes[con_nds].position)
            self._nodes[nds].copy_of_line(self._lines.copy())
        self.connect(data)

        tabel = []
        column_list = ["path", "total latency [s]", "total noise [W]", "SNR [dB]"]
        for id_node1 in self._nodes:
            for id_node2 in self._nodes:
                if id_node1 != id_node2:
                    for path in self.find_paths(id_node1, id_node2):
                        self._all_paths.append(path)
                        self._sign_info[path] = Lightpath(0.0, 0.0, path)
                        self.probe(self._sign_info[path])
                        snr_evaluated = round(snr(self._sign_info[path].signal_power,
                                                  self._sign_info[path].noise_power), 3)
                        latency_eng = "{:.3e}".format(self._sign_info[path].latency)
                        noise_pow_eng = "{:.3e}".format(self._sign_info[path].noise_power)
                        row_list = [arrow.join(path), latency_eng, noise_pow_eng, snr_evaluated]
                        tabel.append(row_list)
            self._nodes[id_node1].copy_of_info(self._sign_info)
        self._weighted_paths = pd.DataFrame(tabel, columns=column_list)
        self._weighted_paths = self._weighted_paths.set_index("path", drop=False)
        # print('Dataframe of all possible paths between all possible nodes: \n', self.weighted_paths())
        # self.route_space("None", "None")
        self._column_list_probe = ["path", "Channel Availability"]
        channel_availability = [1] * channels
        for path in self._all_paths:
            ch_av_mat.append(channel_availability)
            dim = [path, channel_availability]
            self._paths_chs_copy.append(dim)
            # print(self._paths_chs_copy)
            self._paths_ch_available.append(dim)
        self.updt_route_space(self._paths_ch_available)
        self._ch_available = ch_av_mat.copy()
        # print(default_ch_av)
        # print('Dataframe of all non occupied paths between all possible nodes: \n', self.route_space)

    @property
    def traffic_matrix(self):
        return self._traffic_matrix
    @traffic_matrix.setter
    def traffic_matrix(self, tr_mat):
        self._traffic_matrix = tr_mat
    def calculate_bit_rate(self, sign_info, strategy):
        r_b = 0
        GSNR = sign_info.signal_power/sign_info.noise_power
        if strategy == "fixed-rate":
            upper_bound = 2*pow(erfcinv(2*BER_t), 2)*(R_s/B_n)
            # GSNR = self._path_conn[path].snr
            if GSNR >= upper_bound:
                r_b = 100
            else:
                r_b = 0
        else:
            if strategy == "flex-rate":
                lower_bound = 2 * pow(erfcinv(2 * BER_t), 2) * (R_s / B_n)
                intermediate_bound = (14/3) * pow(erfcinv((3/2) * BER_t), 2) * (R_s / B_n)
                upper_bound = 10 * pow(erfcinv((8/3) * BER_t), 2) * (R_s / B_n)
                # print(lower_bound, intermediate_bound, upper_bound)
                # GSNR = db2lin(self._path_conn[path].snr)
                # GSNR = self._path_conn[path].snr
                if GSNR < lower_bound:
                    r_b = 0
                elif GSNR < intermediate_bound:
                    if GSNR > lower_bound:
                        r_b = 100
                elif GSNR < upper_bound:
                    if GSNR > intermediate_bound:
                        r_b = 200
                else:
                    r_b = 400
            else:
                # GSNR = db2lin(self._path_conn[path].snr)
                # GSNR = self._path_conn[path].snr
                r_b = 2*R_s*log(1+GSNR*(R_s/B_n), 2) / 1e9
        return r_b
    def nodes(self):
        return self._nodes

    def weighted_paths(self):
        return self._weighted_paths

    def updt_route_space(self, paths_ch):
        tabel_route_space = []
        for path in paths_ch:
            # print(path)
            row_list = [arrow.join(path[0]), path[1]]
            # print(row_list)
            tabel_route_space.append(row_list)
        self._route_space = pd.DataFrame(tabel_route_space, columns=self._column_list_probe)
        self._route_space.set_index("path", drop=False)

    @property
    def route_space(self):
        return self._route_space

    @route_space.setter
    def route_space(self, db):
        self._route_space = db

    def probe(self, signal_information):
        # print("->", signal_information.path)
        id_node = signal_information.path[0]
        finish = False
        while not finish:
            id_line_probe = self._nodes[id_node].propagate(signal_information, id_node, 0)
            id_node = self._lines[id_line_probe].propagate(signal_information, id_node)
            # if len(signal_information.path) - 1 == 1:
            #     id_line_probe = self._nodes[id_node].propagate(signal_information, id_node)
            #     id_node = self._lines[id_line_probe].propagate(signal_information, id_node)
            # else:
            #     id_line_probe = self._nodes[id_node].propagate(signal_information, id_node)
            #     id_node = self._lines[id_line_probe].propagate(signal_information, id_node)
            if id_node == signal_information.path[-1]:
                # print(signal_information.path, '{:.3e}'.format(signal_information.signal_power))
                finish = True
        pass

    def find_best_snr(self, paths):
        if paths == "NF" or len(paths) == 0:
            best_snr = 0
            best_path, best_lat = "None", "None"
        else:
            list_noise = []
            list_lat = []
            for path in paths:
                list_lat.append(self._sign_info[path].latency)
                list_noise.append(self._sign_info[path].noise_power)
            # print(list_lat, "\n", list_noise)
            best_noise = min(list_noise)
            pos_best_noise = list_noise.index(best_noise)
            best_path = paths[pos_best_noise]
            best_snr = round(snr(self._sign_info[best_path].signal_power, best_noise), 3)
            best_lat = '{:.3e}'.format(list_lat[pos_best_noise])
        return best_lat, best_path, best_snr

    def find_best_latency(self, paths):
        if paths == "NF" or len(paths) == 0:
            best_lat, best_path, best_snr = "None", "None", "None"
        else:
            list_noise = []
            list_lat = []
            for path in paths:
                list_lat.append(self._sign_info[path].latency)
                list_noise.append(self._sign_info[path].noise_power)
            # print(f"Lat.:\n{list_lat}")
            best_lat = min(list_lat)
            pos_best_lat = list_lat.index(best_lat)
            best_lat = '{:.3e}'.format(min(list_lat))
            best_path = paths[pos_best_lat]
            best_snr = round(snr(self._sign_info[best_path].signal_power, list_noise[pos_best_lat]), 3)
        return best_lat, best_path, best_snr

    def matrix_connections(self):
        node1 = random.choice(self._list_of_nodes)
        node2 = random.choice(self._list_of_nodes)
        if node2 == node1:
            while node2 == node1:
                node2 = random.choice(self._list_of_nodes)
        return node1, node2

    def matrix_availability_slot(self, node1, node2):
        ind_node1 = self._list_of_nodes.index(node1)
        ind_node2 = self._list_of_nodes.index(node2)
        if self._traffic_matrix[ind_node1][ind_node2] == math.inf:
            return True
        else:
            return False

    def matrix_mgmt(self, node1, node2, bit_rate):
        ind_node1 = self._list_of_nodes.index(node1)
        ind_node2 = self._list_of_nodes.index(node2)
        self._traffic_matrix[ind_node1][ind_node2] -= bit_rate
        if self._traffic_matrix[ind_node1][ind_node2] <= 0:
            self._traffic_matrix[ind_node1][ind_node2] = math.inf
        self._num_stream += 1
        # print(self._traffic_matrix)

    def matrix_draw(self, M):
        matrix = np.array(self._traffic_matrix)
        nodes_label = self._list_of_nodes.copy()
        strategy = self._nodes[self._list_of_nodes[0]].transceiver
        plt.figure(2)
        for i in range(self._traffic_matrix.shape[0]):  # Righe
            for j in range(self._traffic_matrix.shape[1]):  # Colonne
                plt.text(j, i, str(matrix[i][j]), ha='center', va='center', color='white', fontsize=10)
        plt.imshow(matrix, cmap='viridis', interpolation='none')
        # plt.colorbar(label='Values')
        plt.title(f"Traffic Matrix for M = {M} {strategy} ")
        plt.xticks(ticks=range(len(nodes_label)), labels=nodes_label)  # Etichette asse X
        plt.yticks(ticks=range(len(nodes_label)), labels=nodes_label)
        plt.show()

    def matrix_saturated(self):
        cnt = 0
        for vec in self._traffic_matrix:
            for data in vec:
                if data == math.inf:
                    cnt += 1
        # print(cnt)
        if cnt == (self._num_nodes**2 - self._num_nodes):
            # print("Matrix saturated")
            return True
        else:
            return False

    def stream_w_matrix(self, label):
        result = []
        default_ch_av = []
        channel_availability = [1] * channels
        num_of_loops = 0
        for path in self._all_paths:
            dim = [path, channel_availability]
            default_ch_av.append(dim)
        max_val = self._num_nodes ** 2
        for M in range(1, 51):
            self._num_stream = 0
            num_of_loops = 0
            cnt = 0
            occupied.clear()
            occupied_channels.clear()
            self._line_occ.clear()
            self._paths_ch_available.clear()
            self._path_conn.clear()
            for vec in default_ch_av:
                self._paths_ch_available.append(vec.copy())
            self._traffic_matrix = 100 * M * np.ones((self._num_nodes, self._num_nodes))
            self._traffic_matrix[np.diag_indices_from(self._traffic_matrix)] = 0
            is_sat = False
            while not is_sat and cnt < max_val:
                node1, node2 = self.matrix_connections()
                is_inf = self.matrix_availability_slot(node1, node2)
                if not is_inf:
                    res = self.stream(node1, node2, label)
                    if res[2] != "Quit":
                        if res[2] != 0.0:
                            self.matrix_mgmt(node1, node2, res[2])
                            is_sat = self.matrix_saturated()
                            num_of_loops += 1
                        else:
                            cnt += 1
                    else:
                        cnt += 1
            # print(M, self._num_stream)
            dato = [M, self._num_stream]
            result.append(dato)
        return result

    def stream(self, node1, node2, label):
        paths = self.find_paths(node1, node2)
        paths_tmp = paths.copy()
        for lines in self._line_occ:
            for path in paths:
                for i in range(len(path) - 1):
                    lines_path = path[i] + path[i + 1]
                    self.lines_path.append(lines_path)
                for line_path in self.lines_path:
                    if lines[0] == line_path:
                        if lines[1] == "FULL" or lines[1] >= channels:
                            i = True
                            if i:
                                for mod in self.path_mod:
                                    if path == mod:
                                        i = False
                            if i:
                                self.path_mod.append(path)
                self.lines_path.clear()
        for not_path in self.path_mod:
            paths_tmp.remove(not_path)
        self.path_mod.clear()
        # print(paths_tmp)
        if label == "Latency":
            if len(paths_tmp) == 0:
                # print("No more paths available")
                best_lat, best_path, best_snr = self.find_best_latency("NF")
                # path_conn.conn_update(best_lat, best_snr)
                dato = [0, "None", "Quit"]
            else:
                best_lat, best_path, best_snr = self.find_best_latency(paths_tmp)
                self._path_conn[best_path] = Connection(node1, node2, Signal_information.signal_power)
                # path_conn.conn_update(best_lat, best_snr)
                dato = [best_lat, best_snr]
        else:
            if len(paths_tmp) == 0:
                # print("No more paths available")
                best_lat, best_path, best_snr = self.find_best_snr("NF")
                # path_conn.conn_update(best_lat, best_snr)
                dato = [0, "None", "Quit"]
            else:
                best_lat, best_path, best_snr = self.find_best_snr(paths_tmp)
                self._path_conn[best_path] = Connection(node1, node2, Signal_information.signal_power)
                # path_conn.conn_update(best_lat, best_snr)
                dato = [best_lat, best_snr]
        # paths_tmp.clear()
        ch_occ = 0
        if dato[1] != 0 and dato[0] != "None" and best_path != "None":
            flag_propagate = 1
            # print(f"{best_path}")
            size_best_path = len(best_path) - 1
            for paths in self._paths_ch_available:
                size_paths = len(paths[0]) - 1
                if size_best_path == 1:
                    ch_tmp = paths[1].copy()
                    to_check = self._all_paths.index(paths[0])
                    if size_paths == 1:
                        if best_path == paths[0]:
                            cnt = 0
                            found = False
                            while not found:
                                if cnt == channels:
                                    dato = [0, "None", 0.0]
                                    flag_propagate = 0
                                    found = True
                                else:
                                    if paths[1][cnt] == 1:
                                        found = True
                                        ch_occ = cnt
                                        ch_tmp[cnt] = 0
                                    if cnt < channels:
                                        cnt += 1
                            self._paths_ch_available[to_check][1] = ch_tmp
                    else:
                        paired = False
                        for x in range(len(paths) - 1):
                            check_line = paths[0][x] + paths[0][x + 1]
                            if best_path == check_line:
                                paired = True
                        if paired:
                            for x in range(len(paths[0]) - 1):
                                check_line = paths[0][x] + paths[0][x + 1]
                                to_list = self._all_paths.index(check_line)
                                # print(check_line, self._paths_ch_available[to_list][1])
                                list_availability.append(self._paths_ch_available[to_list][1])
                            cnt = 0
                            # print(paths)
                            while cnt < channels:
                                is_free = 0
                                for vec in list_availability:
                                    # print(cnt, vec)
                                    if vec[cnt] == 1:
                                        is_free += 1
                                if is_free == 0:
                                    ch_occ = cnt
                                    ch_tmp[cnt] = 0
                                cnt += 1
                            # print(best_path, paths[0], ch_tmp)
                            self._paths_ch_available[to_check][1] = ch_tmp
                            list_availability.clear()
                else:
                    list_lines_best_path = []
                    list_copy = []
                    if best_path == paths[0]:
                        ch_tmp = paths[1].copy()
                        for x in range(len(best_path)-1):
                            check_line = best_path[x] + best_path[x + 1]
                            list_lines_best_path.append(check_line)
                            to_list = self._all_paths.index(check_line)
                            list_availability.append(self._paths_ch_available[to_list][1])
                        finish = False
                        swm_cnt = 0
                        while not finish:
                            triple = best_path[swm_cnt] + best_path[swm_cnt + 1] + best_path[swm_cnt + 2]
                            list_availability.append(self._nodes[triple[1]].switching_matrix[triple[0]][triple[2]])
                            swm_cnt += 1
                            if triple[-1] == best_path[-1]:
                                finish = True
                        found = False
                        cnt = 0
                        ind = 0
                        while not found:
                            is_free = 0
                            if cnt == channels:
                                dato = [0, "None", 0.0]
                                flag_propagate = 0
                                found = True
                            else:
                                for vec in list_availability:
                                    if vec[cnt] == 1:
                                        is_free += 1
                                if is_free == len(list_availability):
                                    ch_tmp[cnt] = 0
                                    ind = cnt
                                    ch_occ = cnt
                                    found = True
                            cnt += 1
                        for vec in list_availability:
                            list_to_check = vec.copy()
                            list_to_check[ind] = 0
                            list_copy.append(list_to_check)
                        path_to_mod = self._all_paths.index(best_path)
                        self._paths_ch_available[path_to_mod][1] = ch_tmp
                        for ind, line_to_mod in enumerate(list_lines_best_path):
                            ch_to_mod = self._all_paths.index(line_to_mod)
                            self._paths_ch_available[ch_to_mod][1] = list_copy[ind]
                        list_availability.clear()
            if flag_propagate == 1:
                self.propagate(best_path, 0)
                self._sign_info[best_path].channel = ch_occ
                self._path_conn[best_path].latency = best_lat
                self._path_conn[best_path].snr = best_snr
                self._path_conn[best_path].bit_rate = (self.calculate_bit_rate(self._sign_info[best_path],
                                                                               self._nodes[best_path[0]].transceiver))
                dato.append(self._path_conn[best_path].bit_rate)
                # print(node1, "->", node2, best_path, ch_occ, self._path_conn[best_path].bit_rate)

        # ROUTE TABLE STANDARD
        # self.updt_rout_space(self._paths_ch_available)

        # ROUTE SPACE w/ SWITCHING MATRIX
        # self.updt_route_space(self._paths_chs_copy)
        return dato

    @property
    def lines(self):
        return self._lines

    def draw(self):
        for id_node in self._nodes:
            x0 = self._nodes[id_node].position[0]
            y0 = self._nodes[id_node].position[1]
            plt.plot(x0, y0, 'yo', markersize=10)
            plt.text(x0 + 20, y0 + 20, id_node)

            for con_node in self._nodes[id_node].connected_nodes:
                x1 = self._nodes[con_node].position[0]
                y1 = self._nodes[con_node].position[1]
                plt.plot([x0, x1], [y0, y1], 'r')

        plt.figure(1)
        plt.title('Network')
        plt.xlabel('X[m]')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.ylabel('Y[m]')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.grid()

    # find_paths: given two node labels, returns all paths that connect the 2 nodes
    # as a list of node labels. Admissible path only if cross any node at most once
    def find_paths(self, label1, label2):
        paths = []
        visited = []
        first_node = label1
        last_node = label2
        b_stop = False
        for nds in self._node2line:
            if nds == first_node:
                b_stop = True
        if not b_stop:
            print("Invalid node")
        else:
            visited.append(first_node)
            for next_lns in self._node2line[first_node]:
                next_nds = self._line2node[next_lns][0]
                visited.append(next_nds)
                if next_nds == last_node:
                    paths.append(first_node + next_nds)
                else:
                    for next_lns in self._node2line[next_nds]:
                        if next_lns[1] != first_node:
                            if next_lns[1] == last_node:
                                paths.append(first_node + next_nds + self._line2node[next_lns][0])
                            else:
                                next_nds1 = self._line2node[next_lns][0]
                                visited.append(next_nds1)
                                for next_lns1 in self._node2line[next_nds1]:
                                    if next_lns1[1] == last_node:
                                        paths.append(first_node + next_nds + self._line2node[next_lns][0] +
                                                     self._line2node[next_lns1][0])
                                    else:
                                        if (next_lns1[1] != next_nds) and (next_lns1[1] != first_node):
                                            next_nds2 = self._line2node[next_lns1][0]
                                            for next_lns2 in self._node2line[next_nds2]:
                                                if next_lns2[1] == last_node:
                                                    paths.append(first_node + next_nds + self._line2node[next_lns][0] +
                                                                 self._line2node[next_lns1][0] +
                                                                 self._line2node[next_lns2][0])
                                                else:
                                                    if (next_lns2[1] != next_nds) and (next_lns2[1] != first_node):
                                                        next_nds3 = self._line2node[next_lns2][0]
                                                        for next_lns3 in self._node2line[next_nds3]:
                                                            if next_lns3[1] == last_node:
                                                                tmp = first_node + next_nds + self._line2node[next_lns][
                                                                    0] + self._line2node[next_lns1][0]
                                                                if not self._line2node[next_lns2][0] in tmp:
                                                                    paths.append(first_node + next_nds +
                                                                                 self._line2node[next_lns][0] +
                                                                                 self._line2node[next_lns1][0] +
                                                                                 self._line2node[next_lns2][0] +
                                                                                 self._line2node[next_lns3][0])
            return paths

    # connect function set the successive attributes of all NEs as dicts
    # each node must have dict of lines and viceversa
    def connect(self, data):
        self._node2line = {}
        self._line2node = {}
        line2node = {}
        for nds in self._nodes:
            for lns in self._lines:
                char = lns[0]
                if nds == char:
                    self._node2line.setdefault(nds, []).append(lns)
        for lns in self._lines:
            char2 = lns[1]
            for nds in self._nodes:
                if nds == char2:
                    n = nds
                    self._line2node.setdefault(lns, []).append(n)
                    # print(self._line2node)
            line2node.setdefault(lns, []).append(self._line2node[lns])
            # print(line2node)
        for nds in data:
            self._nodes[nds].switching_matrix = data[str(nds)]['switching_matrix']
            # print(self._nodes[nds].switching_matrix)
        # print(self._nodes['A'].switching_matrix)
        pass

    def propagate_through_swm(self, sel_path):
        for paths in self._all_paths:
            if len(sel_path) == 2:
                paired = False
                for x in range(len(paths) - 1):
                    check_line = paths[x] + paths[x + 1]
                    if sel_path == check_line:
                        paired = True
                if paired:
                    to_check = self._all_paths.index(paths)
                    self._paths_chs_copy[to_check][1] = self._paths_ch_available[to_check][1]
            else:
                if sel_path == paths:
                    last_node = sel_path[-1]
                    finish = False
                    cnt = 0
                    arr_sup = [1]*channels
                    while not finish:
                        triple = sel_path[cnt] + sel_path[cnt + 1] + sel_path[cnt + 2]
                        arr2 = np.multiply(np.array(arr_sup),
                                           np.array(self._nodes[triple[1]].switching_matrix[triple[0]][triple[2]]))
                        arr_sup = arr2
                        cnt += 1
                        if triple[-1] == last_node:
                            to_check = self._all_paths.index(sel_path)
                            arr1 = np.array(self._paths_ch_available[to_check][1])
                            res = np.multiply(arr1, arr2)
                            self._paths_chs_copy[to_check][1] = res
                            finish = True
        pass

    # propagate signal_information through path specified in it
    # and returns the modified spectral information
    def propagate(self, sel_path, state):
        self.propagate_through_swm(sel_path)
        GSNR = 0.0
        ISNR = 0.0
        for x in range(len(sel_path) - 1):
            # print(sel_path)
            line = sel_path[x] + sel_path[x + 1]
            self._lines[line].line_state(state)
            self._line_occ = self._lines[line].line_occupied()
            self._optimal_pow = self._lines[line].optimized_launch_power(self._sign_info[line].signal_power)
        #     # print(self._sign_info[line].path, "Old Power:", '{:.3e}'.format(self._sign_info[line].signal_power))
            self._sign_info[line].signal_power = self._optimal_pow
            NLI, eta_nli = self._lines[line].nli_generation(self._sign_info[line].signal_power)
            ASE = self._lines[line].ase_generation()
            XT = self._lines[line].crosstalk_generation(self._sign_info[line].signal_power, self._sign_info[line]
                                                        .channel)
            self._sign_info[line].new_noise = self._lines[line].new_noise_generation(ASE, NLI, XT)
            GSNR = self._sign_info[line].signal_power / self._sign_info[line].new_noise
            tmp = pow(GSNR, -1)
        #   # ISNR += tmp
            self._sign_info[sel_path].update_ISNR(tmp)
        # print(sel_path, '{:.3e}'.format(self._sign_info[sel_path].signal_power), '{:.3e}'.format(GSNR),
        #       '{:.3e}'.format(self._sign_info[sel_path].ISNR))
        new_GSNR = -1*lin2db(self._sign_info[sel_path].ISNR)
        self._sign_info[sel_path].GSNR = new_GSNR
        # print(self._sign_info[sel_path].path, "GSNR [dB]:", round(self._sign_info[sel_path].GSNR, 3))
            # print(self._sign_info[line].path, "New Power:", '{:.3e}'.format(self._sign_info[line].signal_power))
        pass
