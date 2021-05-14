import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forwardMessages = [None] * num_time_steps
    forwardMessages[0] = prior_distribution
    backMessages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    # TODO: Compute the forward messages
    # initialize
    forwardMessages[0] = rover.Distribution({})
    for z0 in all_possible_hidden_states:
        initialPos = 1 if observations[0] == None else observation_model(z0)[observations[0]]
        prior_z0 = prior_distribution[z0]
        mes = initialPos * prior_distribution[z0]
        if (mes) != 0:
            forwardMessages[0][z0] = mes
    forwardMessages[0].renormalize()
    
    # for all other alpha_i's
    for i in range(1, num_time_steps):
        forwardMessages[i] = rover.Distribution({})
        observedPosition = observations[i]
        for zi in all_possible_hidden_states:
            probPosition_cur = 1 if observedPosition == None else observation_model(zi)[observedPosition]
            prob = 0
            for zi_minus_1 in forwardMessages[i-1]:
                prob += forwardMessages[i-1][zi_minus_1] * transition_model(zi_minus_1)[zi]
            if (probPosition_cur * prob) != 0: # only save non-zero values
                forwardMessages[i][zi] = probPosition_cur * prob

        forwardMessages[i].renormalize() # normalize forward messages
    
    # TODO: Compute the backward messages
    # initialization of backward message
    backMessages[num_time_steps-1] = rover.Distribution({})
    for zn_minus_1 in all_possible_hidden_states:
        backMessages[num_time_steps-1][zn_minus_1] = 1

    # fill the rest
    for i in range(1, num_time_steps):
        backMessages[num_time_steps-1-i] = rover.Distribution({})
        for zi in all_possible_hidden_states:
            prob = 0
            for zi_plus_1 in backMessages[num_time_steps-1-i+1]:
                observedPosition = observations[num_time_steps-1-i+1]
                probPosition_next = 1 if observedPosition == None else observation_model(zi_plus_1)[observedPosition]
                prob += backMessages[num_time_steps-1-i+1][zi_plus_1] * probPosition_next * transition_model(zi)[zi_plus_1]
            if prob != 0:
                backMessages[num_time_steps-1-i][zi] = prob
        backMessages[num_time_steps-1-i].renormalize()
    
    # TODO: Compute the marginals
    for i in range (0, num_time_steps): 
        marginals[i] = rover.Distribution({})    
        prob = 0
        for zi in all_possible_hidden_states:
            if forwardMessages[i][zi] * backMessages[i][zi] != 0:
                marginals[i][zi] = forwardMessages[i][zi] * backMessages[i][zi]
                prob += forwardMessages[i][zi] * backMessages[i][zi]
        for zi in marginals[i].keys():
            marginals[i][zi] /=  prob

    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    num_time_steps = len(observations)
    w = [None] * num_time_steps
    estimated_hidden_states = [None] * num_time_steps
    z_previous = [None] * num_time_steps

    # initialization
    w[0] = rover.Distribution({})
    initial_observedPosition = observations[0]
    for z0 in all_possible_hidden_states:
        initProbCur = 1 if initial_observedPosition == None else observation_model(z0)[initial_observedPosition]
        prior_z0 = prior_distribution[z0]
        if (initProbCur != 0) and (prior_z0 != 0):
            w[0][z0] = np.log(initProbCur) + np.log(prior_z0)
    
    # when i >= 1
    for i in range(1, num_time_steps):
        w[i] = rover.Distribution({})
        z_previous[i] = {}
        observedPosition = observations[i]
        for zi in all_possible_hidden_states:
            probPosition_cur = 1 if observedPosition == None else observation_model(zi)[observedPosition]
            
            max_term = -np.inf
            for zi_minus_1 in w[i-1]:
                if transition_model(zi_minus_1)[zi] != 0:
                    potential_max_term = np.log(transition_model(zi_minus_1)[zi]) + w[i-1][zi_minus_1]
                    if (potential_max_term > max_term) and (probPosition_cur != 0):
                        max_term = potential_max_term
                        # keep track of which zi_minus_1 can maximize w[i][zi]
                        z_previous[i][zi] = zi_minus_1 

            if probPosition_cur != 0:
                w[i][zi] = np.log(probPosition_cur) + max_term
            
    # back track to find z0 to zn
    # first, find zn* (the last)
    max_w = -np.inf
    for zi in w[num_time_steps-1]:
        potential_max_w = w[num_time_steps-1][zi]
        if potential_max_w > max_w:
            max_w = potential_max_w
            estimated_hidden_states[num_time_steps-1] = zi
    
    for i in range(1, num_time_steps):
        estimated_hidden_states[num_time_steps-1-i] = z_previous[num_time_steps-i][estimated_hidden_states[num_time_steps-i]]

    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


    timestep = num_time_steps - 1
#    timestep = 30
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
  
    # error calculation
    corrNum = 0
    for i in range(0, num_time_steps):
        if hidden_states[i] == estimated_states[i]:
            corrNum += 1
    print("viterbi's error is:", 1-corrNum/100)

    corrNum = 0
    for i in range(0, num_time_steps):
        #prediction = None
        max_prob = 0
        for zi in marginals[i]:
            if marginals[i][zi] > max_prob:
                prediction = zi
                max_prob = marginals[i][zi]
        print(i, ":", prediction)
        if hidden_states[i] == prediction:
            corrNum += 1
    print()
    print("forward & backward's error is:", 1-corrNum/100)

    




    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
