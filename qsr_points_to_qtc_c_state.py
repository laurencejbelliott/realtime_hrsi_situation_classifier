# Requires Python 2
import rospy
from std_msgs.msg import Float64MultiArray, String
from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import Object_State, World_Trace


def points_to_qtc_c(data):
    # Current and recent human and then robot positions and angles:
    # [xh0, yh0, hh0, xh1, yh1, hh1, xr0, yr0, hr0, xr1, yr1, hr1]
    qtc_points = list(data.data)
    xh0, yh0, hh0, xh1, yh1, hh1, xr0, yr0, hr0, xr1, yr1, hr1 = qtc_points
    rospy.loginfo(qtc_points)

    world = World_Trace()

    h_state_seq = []
    r_state_seq = []

    h_state_seq.append(Object_State(name="human", timestamp=1, x=xh0, y=yh0))
    h_state_seq.append(Object_State(name="human", timestamp=2, x=xh1, y=yh1))
    r_state_seq.append(Object_State(name="robot", timestamp=1, x=xr0, y=yr0))
    r_state_seq.append(Object_State(name="robot", timestamp=2, x=xr1, y=yr1))

    world.add_object_state_series(h_state_seq)
    world.add_object_state_series(r_state_seq)

    # make a QSRlib request message
    dynamic_args = {"qtccs": {"no_collapse": False, "quantisation_factor": quantisation_factor,
                              "validate": False, "qsrs_for": [("human", "robot")]}}

    qsrlib_request_message = QSRlib_Request_Message('qtccs', world, dynamic_args)

    # request your QSRs
    qsrlib_response_message = qsrlib.request_qsrs(req_msg=qsrlib_request_message)

    # Get QSR at each timestamp
    timestamps = qsrlib_response_message.qsrs.get_sorted_timestamps()
    #     print(timestamps)
    qtc_seq = []
    for t in timestamps:
        for val in qsrlib_response_message.qsrs.trace[t].qsrs.values():
            qtc_seq.append(val.qsr['qtccs'].replace(",", ""))

    rospy.loginfo(qtc_seq[0])
    qtc_c_state_publisher.publish(str(qtc_seq[0]))

if __name__ == "__main__":
    quantisation_factor = 0.003
    qsrlib = QSRlib()

    qtc_c_state_publisher = rospy.Publisher('points_to_qtc_c_state/qtc_c_state', String, queue_size=1)
    rospy.init_node("points_to_qtc_c_state")
    rospy.Subscriber("/robot4/qsr/qtc_points", Float64MultiArray, points_to_qtc_c)

    rospy.spin()
