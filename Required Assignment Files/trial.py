import numpy as np
import matplotlib.pyplot as plt
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from generate_game import *
from Chess_env import *
import pandas as pd
from numba import jit, cuda

@jit(target_backend='cuda')
def EpsilonGreedy_Policy(Qvalues, epsilon, allowed):

    N_allowed = np.shape(allowed)[0]

    Qvalues_of_allowed = Qvalues[allowed]
    
    rand_value = np.random.uniform(0,1)

    rand_a = rand_value < epsilon

    if rand_a == True:

        temp = np.random.randint(0, N_allowed)
        a = allowed[temp]

    else:

        temp = np.argmax(Qvalues_of_allowed)
        a = allowed[temp]
            
    return a


@jit(target_backend='cuda')
def main():
    
    size_board = 4

    # %%
    ## INITIALISE THE ENVIRONMENT

    env=Chess_Env(size_board)


    # %%
    # INITIALISE THE PARAMETERS OF YOUR NEURAL NETWORK AND...
    # PLEASE CONSIDER TO USE A MASK OF ONE FOR THE ACTION MADE AND ZERO OTHERWISE IF YOU ARE NOT USING VANILLA GRADIENT DESCENT...
    # WE SUGGEST A NETWORK WITH ONE HIDDEN LAYER WITH SIZE 200. 

    S, X, allowed_a = env.Initialise_game()
    N_a = np.shape(allowed_a)[0]   # TOTAL NUMBER OF POSSIBLE ACTIONS
    N_in = np.shape(X)[0]    # INPUT SIZE
    N_h = 200                # NUMBER OF HIDDEN NODES

    ## INITALISE YOUR NEURAL NETWORK...

    # INITIALISE THE WEIGHTS OF THE NEURAL NETWORK
    W1 = np.random.randn(N_h, N_in) * 0.01
    b1 = np.zeros((N_h,)) * 0.01
    W2 = np.random.randn(N_a, N_h) * 0.01
    b2 = np.zeros((N_a,)) * 0.01

    # HYPERPARAMETERS SUGGESTED (FOR A GRID SIZE OF 4)

    epsilon_0 = 0.2     # STARTING VALUE OF EPSILON FOR THE EPSILON-GREEDY POLICY
    beta = 0.00005      # THE PARAMETER SETS HOW QUICKLY THE VALUE OF EPSILON IS DECAYING (SEE epsilon_f BELOW)
    gamma = 0.85        # THE DISCOUNT FACTOR
    eta = 0.0035        # THE LEARNING RATE

    N_episodes = 50000 # THE NUMBER OF GAMES TO BE PLAYED 

    # SAVING VARIABLES
    R_save = np.zeros([N_episodes, 1])
    N_moves_save = np.zeros([N_episodes, 1])



    # %%
    # TRAINING LOOP BONE STRUCTURE...

    eligibility_trace = True
    if eligibility_trace:    
        lamb = 0.3
        eta = 0.008

    for n in range(N_episodes):

        epsilon_f = epsilon_0 / (1 + beta * n)   ## DECAYING EPSILON
        Done = 0                                   ## SET DONE TO ZERO (BEGINNING OF THE EPISODE)
        i = 1                                    ## COUNTER FOR NUMBER OF ACTIONS
        
        S, X, allowed_a = env.Initialise_game()      ## INITIALISE GAME
        print(n)                                 ## REMOVE THIS OF COURSE, WE USED THIS TO CHECK THAT IT WAS RUNNING
        
        # Initialise eligibility traces 
        if eligibility_trace == True:
            
            e = np.zeros(N_a,)

        # Find the Qvalues corresponding to that state

        while Done == 0:                           ## START THE EPISODE
            
            # FORWARD PROPAGATION
            h1 = np.matmul(W1, X) + b1

            # Apply the ReLU activation function
            x1 = np.maximum(0, h1)

            # Compute the Qvalues (output of the network)
            Qvalues = np.matmul(W2, x1) + b2
            
            a,_=np.where(allowed_a==1)

            a_agent = EpsilonGreedy_Policy(Qvalues, epsilon_f, a)
                    
            S_next, X_next, allowed_a_next, R, Done = env.OneStep(a_agent)

            # Update the eligibility trace for the action made
            if eligibility_trace:

                e[a_agent] = e[a_agent] + 1

            if Done == 1:
                
                # BACKWARD PROPAGATION
                # update the weights of the output layer
                delta2 = R - Qvalues[a_agent]
                
                eta_delta2 = eta * delta2
                W2[a_agent] = W2[a_agent] + eta_delta2 * x1
                b2[a_agent] = b2[a_agent] + eta_delta2
                
                # update the weights of the hidden layer
                delta1 = np.dot(W2[a_agent], delta2) * (x1 > 0)

                W1 = W1 + eta * np.outer(delta1, X)
                b1 = b1 + eta * delta1

                # SAVE THE RESULTS
                R_save[n] = np.copy(R)
                N_moves_save[n] = np.copy(i)

                if eligibility_trace:
                    #Qvalues = Qvalues + eta_delta2 * e
                    W2 = W2 + eta_delta2 * np.outer(e, x1)

                break
            
            else:

                h1 = np.matmul(W1, X_next) + b1

                # Apply the ReLU activation function
                x1 = np.maximum(0, h1)

                # Compute the Qvalues (output of the network)
                Qvalues1 = np.matmul(W2, x1) + b2
                
                sarsa = True
                
                a1, _ = np.where(allowed_a_next == 1)

                if sarsa == True:
                    # implement the SARSA update rule
                    a1_agent = EpsilonGreedy_Policy(Qvalues1, epsilon_f, a1)
                else:
                    # implement the Q-learning update rule
                    a1_agent = EpsilonGreedy_Policy(Qvalues1, 0, a1)
                    
                # BACKWARD PROPAGATION
                # update the weights of the output layer
                delta2 = R + gamma * Qvalues1[a1_agent] - Qvalues[a_agent]

                eta_delta2 = eta * delta2
                
                W2[a_agent] = W2[a_agent] + eta_delta2 * x1
                b2[a_agent] = b2[a_agent] + eta_delta2

                # update the weights of the hidden layer
                delta1 = np.dot(W2[a_agent], delta2) * (x1 > 0)

                W1 = W1 + eta * np.outer(delta1, X)
                b1 = b1 + eta * delta1

                # Update the Qvalues for the case with eligibility traces
                if eligibility_trace:
                    # Qvalues = Qvalues + eta_delta2 * e
                    W2 = W2 + eta_delta2 * np.outer(e, x1)
                    e = gamma * lamb * e
                
                
            # NEXT STATE AND CO. BECOME ACTUAL STATE...
            S = np.copy(S_next)
            X = np.copy(X_next)
            allowed_a = np.copy(allowed_a_next)
            
            i += 1  # UPDATE COUNTER FOR NUMBER OF ACTIONS

    print('Random_Agent, Average reward:',np.mean(R_save),'Number of steps: ',np.mean(N_moves_save))

    
    print('Random_Agent, Average reward:',np.mean(R_save[0:25000]),'Number of steps: ',np.mean(N_moves_save[0:25000]))
    print('Random_Agent, Average reward:',np.mean(R_save[25000:50000]),'Number of steps: ',np.mean(N_moves_save[25000:50000]))
    print('Random_Agent, Average reward:',np.mean(R_save[50000:75000]),'Number of steps: ',np.mean(N_moves_save[50000:75000]))
    print('Random_Agent, Average reward:',np.mean(R_save[75000:100000]),'Number of steps: ',np.mean(N_moves_save[75000:100000]))
    print('Random_Agent, Average reward:',np.mean(R_save[100000:125000]),'Number of steps: ',np.mean(N_moves_save[100000:125000]))
    print('Random_Agent, Average reward:',np.mean(R_save[125000:150000]),'Number of steps: ',np.mean(N_moves_save[125000:150000]))
    print('Random_Agent, Average reward:',np.mean(R_save[150000:175000]),'Number of steps: ',np.mean(N_moves_save[150000:175000]))
    print('Random_Agent, Average reward:',np.mean(R_save[175000:200000]),'Number of steps: ',np.mean(N_moves_save[175000:200000]))

    plt.figure(figsize=(40, 12))
    cumsum = np.cumsum(np.insert(R_save, 0, 0))
    plt.plot((cumsum[100:] - cumsum[:-100]) / 100)


main()