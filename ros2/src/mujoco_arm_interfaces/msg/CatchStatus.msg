# Ball-catch trial status, published once per control step by catch_sim_node.
std_msgs/Header header
uint64 seq                       # matches ArmObservation.seq
uint32 trial                     # increments on every /arm/catch/start
string phase                     # idle | pick | fall | rise | hold | failed | return
string gravity_name              # earth | mars | moon (or "<g> m/s^2")
float32 gravity                  # [m/s^2]
float32 restitution              # ball-floor restitution of this trial
float64 sim_time                 # [s] simulated time since the trial started
float32 time_since_release       # [s]; -1 before the release
uint32 bounces
bool holding                     # the cup has the ball
bool done                        # outcome decided
bool success                     # caught at the apex and held
string outcome                   # "" while running, e.g. "caught", "missed: second bounce"
geometry_msgs/Point ball         # ball centre, world frame [m]
geometry_msgs/Vector3 ball_velocity
geometry_msgs/Point ee           # cup (tip) position [m]
geometry_msgs/Point catch_point  # where the cup should meet the ball (green marker)
float32 catch_time_error         # [s] latch time - true apex time (NaN if no catch)
float32 catch_height_error       # [m] ball height at latch - true apex height (NaN if no catch)
float32 impact_speed             # [m/s] relative ball-cup speed at latch (NaN if no catch)
