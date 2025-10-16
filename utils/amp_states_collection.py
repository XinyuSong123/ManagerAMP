import torch

def collect_amp_states_from_obs(
        states: torch.Tensor,
        next_states: torch.Tensor,
)->  torch.Tensor:
    """Convert environment observations to AMP states.

    Args:
        states (torch.Tensor): current observations
        next_states (torch.Tensor): next observations

    Returns:
        amp_states (torch.Tensor): converted AMP states
    """
    '''
           [INFO] Observation Manager: <ObservationManager> contains 1 groups.
            +----------------------------------------------------------+
            | Active Observation Terms in Group: 'policy' (shape: (365,)) |
            +-----------+--------------------------------+-------------+
            |   Index   | Name                           |    Shape    |
            +-----------+--------------------------------+-------------+
            |     0     | base_lin_vel                   |     (3,)    |
            |     1     | base_ang_vel                   |     (3,)    |
            |     2     | projected_gravity              |     (3,)    |
            |     3     | velocity_commands              |     (3,)    |
            |     4     | joint_pos                      |    (115,)   |
            |     5     | joint_vel                      |    (115,)   |
            |     6     | actions                        |    (115,)   |
            |     7     | base_state_obs                 |     (7,)    |
            |     8     | timesteps                      |     (1,)    |
            +-----------+--------------------------------+-------------+
    '''
    joint_pos = states[:, 104:127]
    next_joint_pos = next_states[:, 104:127]
    amp_states = torch.cat([joint_pos, next_joint_pos
                                    ],dim=1)
    return amp_states.clone()