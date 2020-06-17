import roslibpy
import hmms
import time
import numpy as np
from itertools import combinations
from scipy.stats import entropy

# Load per-situation HMMs
pblHMM = hmms.DtHMM.from_file("Models/pblHMM-KL.npz")
pbrHMM = hmms.DtHMM.from_file("Models/pbrHMM-KL.npz")
rotlHMM = hmms.DtHMM.from_file("Models/rotlHMM-KL.npz")
rotrHMM = hmms.DtHMM.from_file("Models/rotrHMM-KL.npz")
pclHMM = hmms.DtHMM.from_file("Models/pclHMM-KL.npz")
pcrHMM = hmms.DtHMM.from_file("Models/pcrHMM-KL.npz")


# Configuration parameters
rejection_KL_thresh = 0.018
N_nodes = 81
random_state = 2

classes = ["PBL", "PBR", "ROTL", "ROTR", "PCL", "PCR"]


# Functions for mapping QTC_C states to integers
# Create list of QTC_C states so that indices can be used as integer state IDs compatible with HMM library
QTC_symbols = []
for i in range(0, 4):
    QTC_symbols.append("-")
    QTC_symbols.append("0")
    QTC_symbols.append("+")
# print("QTC symbols:", QTC_symbols[:3])
QTC_C_states = list(combinations(QTC_symbols, 4))
QTC_C_states = [state[0] + state[1] + state[2] + state[3] for state in QTC_C_states]
QTC_C_states = list(np.unique(QTC_C_states))
# print("QTC_C states:\n", QTC_C_states)
# print(len(QTC_C_states), "states total")

def QTC_C_to_num(QTC_C):
    return QTC_C_states.index(QTC_C)


def QTC_C_seq_to_num_seq(QTC_C_seq):
    num_seq = []
    for QTC_C in QTC_C_seq:
        num_seq.append(QTC_C_to_num(QTC_C))

    return num_seq


def num_to_QTC_C(num):
    return QTC_C_states[num]


def num_seq_to_QTC_C_seq(num_seq):
    QTC_C_seq = []
    for num in num_seq:
        QTC_C_seq.append(num_to_QTC_C(num))

    return QTC_C_seq

# print(QTC_C_to_num("++--"))
# print(num_to_QTC_C(8))
# print(num_seq_to_QTC_C_seq([0, 1, 2, 3]))
# print(QTC_C_seq_to_num_seq(num_seq_to_QTC_C_seq([0, 1, 2, 3])))

# Classify function
def classify_QTC_seqs(e_seqs):
    ll_pb_l = pblHMM.data_estimate(e_seqs)

    ll_pb_r = pbrHMM.data_estimate(e_seqs)

    ll_rotl = rotlHMM.data_estimate(e_seqs)

    ll_rotr = rotrHMM.data_estimate(e_seqs)

    ll_pcl = pclHMM.data_estimate(e_seqs)

    ll_pcr = pcrHMM.data_estimate(e_seqs)

    lls = [ll_pb_l, ll_pb_r, ll_rotl, ll_rotr, ll_pcl, ll_pcr]
    class_id = np.argmax(lls)
    KL = entropy(lls, [1 / len(lls) for ll in lls])
    if KL > rejection_KL_thresh:
        pred = classes[class_id]
    else:
        pred = "rejection"
        class_id = len(classes)
    # print("Classified as", pred)
    # print("KL divergence of likelihoods from uniform distribution:", KL)

    return pred


# Test imported models by generating QTC_C sequences
eg_seq = pcrHMM.generate(5)[0]
# print(num_seq_to_QTC_C_seq(eg_seq))
# print(classify_QTC_seqs(np.array([eg_seq])))

ros = roslibpy.Ros(host="localhost", port=9090)
ros.run()

print("ROS connected:", ros.is_connected)

qtcTopic = roslibpy.Topic(ros, "/points_to_qtc_c_state/qtc_c_state", "std_msgs/String")
sitTopic = roslibpy.Topic(ros, "/robot4/qsr/situation_predictions", "std_msgs/String")
qtc_seq = []
prev_time = time.time()


def qtc_update_callback(qtc_state_msg):
    global prev_time
    global qtc_seq
    qtc_state_str = qtc_state_msg['data']
    current_time = time.time()
    if current_time - prev_time > 3:
        qtc_seq = []

    if len(qtc_seq) == 0:
        qtc_seq.append(qtc_state_str)
    elif qtc_state_str != qtc_seq[-1]:
        qtc_seq.append(qtc_state_str)

    print(qtc_seq)
    sit = classify_QTC_seqs(np.array([QTC_C_seq_to_num_seq(qtc_seq)]))
    print(sit, "\n")
    sitTopic.publish(roslibpy.Message({'data': sit}))

    prev_time = current_time


# Current and recent human and then robot positions and angles:
# [xh0, yh0, hh0, xh1, yh1, hh1, xr0, yr0, hr0, xr1, yr1, hr1]
qtcTopic.subscribe(lambda message: qtc_update_callback(message))
