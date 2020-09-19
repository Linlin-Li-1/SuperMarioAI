## Planned

#### Save the network
We do need save and load the model sometimes, in case we need to stop the learning and continue sometime later. What might be the best way to automatically save models (in frequency and memory)? Maybe save
- at particular intervals
- at best performance
- on quit (catch Exception)

We can find ways also to save episodes in video format, i.e. *.mp4, etc. Maybe only save the top 3 videos, and keep the newest 10 episodes? Needs to check performance reduction on saving videos.

**For Models**
- Save the hyperparameters only once
- Save the model once every certain steps (better in "h5" format? needs investigation)
- Calculate how large the experience replay buffer could be. (Save all / Save a portion / Do not save)
- Would be good if we can continue training once the process is terminated.

**For Logs**
- Print all the things in the notebook is not good for analysis and comparison.
- We need to save some metrics into logs so as to understand what is going on.
- If `verbose = 0`, there will be no training information shown on the screen. How to fetch these information and output to file?

## Researching
#### Mario is feeling tired?
I've noticed that Mario tends to refuse to jump as high after several hundred episodes, and the mean reward keeps lowering as a result of wasting much time in attempt to jump over the tall pipes. Maybe jumping is not encouraged enough? 

- Adding the output action last time may help?
- Extending the number of skipped frames? (Possible side effect: The Mario gets dumb).

#### Mario is not learning much compared to random.
Some better balanced reward function should be investigated. What are we encouraging the Mario to do?
Also might work if we try to improve our network.

## Finished
#### Inplement Double DQN
When calculating the loss, evaluate the Q value of the next step through the target network, however, at the best action given by the training network.

Suggestion: the copy rate should be every 30,000 steps.

*Finished on Apr 11 at 6:02pm*

#### Distinguish terminal and non-terminal states
We need to distinguish **terminal** states with **non-terminal** states. For terminal states, Q value should just be the instant reward.

*Finished on Apr 11 at 6:02pm*

#### Refactoring codes
On implementation of more various functions, there is a higher need to refactor the current codes. Some codes are copy-pasted and some codes are redundant. Should make the structure more clear and write the codes more elegantly.

*Finished on Apr 17 at 8:21pm*


## Deprecated

#### Add `mask` as an input of the network
Currently, the DQN neural network returns an output of shape `(batch_size, action_num)` indicating the expected optimal value under an input state. One should notice that this network is used for two purposes.

(1) For each step where the agent needs to make a decision, the neural network should return an output of shape `(1, action_num)`, and the agent selects the action with the highest value.

(2) At the end of each step, the neural network is trained with a minibatch. The loss compares $Q(s, a; \theta)$ with $R + \gamma \max_{a'}Q(s', a'; \theta^-)$. However, this should only update the parameters related to $Q(s, a)$ instead of $Q(s)$. (Only the action `a` is performed in this step.) Thus, we should mask $Q(s, \tilde{a})$ as 0 where $\tilde{a}\neq a$.

Therefore, we can add `mask` as an additional input of the network, and multiply the output with mask at the last layer of the network.

*Note: As an approach to Implement Double DQN*