# The exact observation vector the RL policy was trained on (24 floats for
# MujocoArm3Reach-v1: cos q, sin q, qdot/10, servo error, previous action,
# tip position, target position, 10*(target - tip)).
#
# `seq` increases every time a *new* observation exists (a physics step, a
# reset, or a new target). The simulator re-publishes the current observation
# while it waits for an action, so a consumer must act once per `seq`.
std_msgs/Header header
uint64 seq
float32[] data
