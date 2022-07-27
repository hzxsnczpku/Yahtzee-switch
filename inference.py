import numpy as np
from utils import ACTION_INDEXES, encode, decode
from main import fill_slot, roll_dice


def infer(state, V):
    max_a, max_b, max_V = None, None, 0
    V_small = -1 * np.ones(1302527)
    i = encode(state)
    if state[12] == 2:
        for b in range(12):
            if state[b] == 1:
                continue
            new_V = fill_slot(i, b, V, V_small)
            if new_V > max_V:
                max_b, max_V = b, new_V
        return max_b
    else:
        for a in ACTION_INDEXES:
            new_V = roll_dice(i, a, V, V_small)
            if new_V > max_V:
                max_a, max_V = a, new_V
        return decode(i)[-5:][max_a]


if __name__ == '__main__':
    V = np.load('./value.npz')['V']
    state = np.array([1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 3, 5, 2, 1, 1, 5])

    action = infer(state, V)
    print(f'Slots: {state[:12]}')
    print(f'Stage: {state[12]}')
    print(f'Upper Score: {state[13]}')
    print(f'Dices: {state[14:]}')
    print('')

    print('Optimal Action: ')
    if state[12] == 2:
        print(f'Select Slots {action}')
    else:
        print(f'Reroll Dices {action}')
