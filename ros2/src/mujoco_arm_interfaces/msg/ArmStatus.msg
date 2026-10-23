# Per-step task status, published once per physics/control step.
std_msgs/Header header
uint64 seq                    # matches ArmObservation.seq
uint32 episode                # increments on every /reset
uint32 episode_step           # steps since the last reset OR new target
uint32 episode_steps_max      # the env's episode length (50 for v1 at 25 Hz)
bool episode_done             # episode_step >= episode_steps_max
float64 sim_time              # seconds of simulated time since start-up
float32 distance              # tip-to-target distance [m]
float32 reward                # reward of the last step (0 right after a reset)
bool success                  # distance < env success threshold (5 cm)
bool success_2cm              # distance < 2 cm
float32 time_to_reach         # s from episode start to first success; -1 = not yet
geometry_msgs/Point ee        # tip position, world frame [m]
geometry_msgs/Point target    # target position, world frame [m]
