from util import Counter

# utility functions for using q learning

def extractPolicy(qVals) -> []:
    """extracts the optimal policy from a qVal table
        Returns and prints list of optimal actions
            where each index is the state that corresponds to the action"""

    qValStateMapping = [{}, {}, {}, {}, {}, {}]
    policy = []

    if not qVals:
        return []

    for key, value in qVals.items():
        state = key[0]
        action = key[1]

        qValStateMapping[state][action] = value

    for i, a in enumerate(qValStateMapping):
        optAction = max(a, key=a.get, default=None)
        policy.append(optAction)
        print("State ", i, "Optimal Action ", optAction)

    return policy


def pickMaxAction(state, q_vals):
    """picks the best action for a given state and q value dictionary """

    bestQVal = float('-inf')
    bestAction = None

    for s, a in q_vals.keys():

        if s == state and q_vals[(s, a)] > bestQVal:
            bestAction = a
            bestQVal = q_vals[(s, a)]

    return bestAction


def maxValForState(state, q_vals):
    """picks the highest value for a state """

    bestQVal = float('-inf')
    flag = True

    for s, a in q_vals.keys():

        if s == state and q_vals[(s, a)] > bestQVal:
            bestQVal = q_vals[(s, a)]
            flag = False
    if flag:
        return 0
    return bestQVal
