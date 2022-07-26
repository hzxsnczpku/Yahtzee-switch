import numpy as np
from numba import njit
from utils import ACTION_INDEXES, encode, decode, ALL_POSS_5, ALL_POSS_4, ALL_POSS_3, ALL_POSS_2, ALL_POSS_1


@njit
def _k_of_a_kind(numbers, k):
    reward, counters = 0, np.zeros(6, dtype=np.int64)
    for i in numbers:
        counters[i] += 1
    if np.max(counters) >= k:
        reward = 50 if k == 5 else np.sum(numbers) + 5
    return reward


@njit
def _full_house(numbers):
    reward, counters = 0, np.zeros(6, dtype=np.int64)
    for i in numbers:
        counters[i] += 1
    flag1, flag2 = False, False
    for c in counters:
        if c == 2:
            flag1 = True
        if c == 3:
            flag2 = True
    if flag1 and flag2:
        reward = 25
    return reward


@njit
def _straight(numbers, k):
    counter, max_counter = 1, 1
    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:
            counter += 1
            max_counter = max(max_counter, counter)
        else:
            counter = 1
    if max_counter >= k:
        if k == 4:
            return 30
        if k == 5:
            return 40
        else:
            return 0
    else:
        return 0


@njit
def step(raw_state, indexes, slot, V):
    res = 0
    counter = 0

    state = decode(raw_state)
    slots, stage, upper_score, numbers = state[:12], state[12], state[13], state[14:]

    if 5 * slots[0] + 10 * slots[1] + 15 * slots[2] + 20 * slots[3] + 25 * slots[4] + 30 * slots[5] < upper_score or slots[slot] == 1:
        return V[raw_state]

    if stage <= 1:
        state[12] += 1
        if np.sum(indexes) == 0:
            res = V[encode(state)]
            counter = 1
        elif np.sum(indexes) == 1:
            for poss in ALL_POSS_1:
                numbers[indexes] = poss
                res += V[encode(state)]
                counter += 1
        elif np.sum(indexes) == 2:
            for poss in ALL_POSS_2:
                numbers[indexes] = poss
                res += V[encode(state)]
                counter += 1
        elif np.sum(indexes) == 3:
            for poss in ALL_POSS_3:
                numbers[indexes] = poss
                res += V[encode(state)]
                counter += 1
        elif np.sum(indexes) == 4:
            for poss in ALL_POSS_4:
                numbers[indexes] = poss
                res += V[encode(state)]
                counter += 1
        elif np.sum(indexes) == 5:
            for poss in ALL_POSS_5:
                numbers[indexes] = poss
                res += V[encode(state)]
                counter += 1
    else:
        reward = 0
        if slot <= 5:
            reward = np.sum(numbers[numbers == slot] + 1)
            upper_score += reward
            if upper_score >= 63 and upper_score - reward < 63:
                reward += 35
        elif slot == 6:  # total score
            reward = np.sum(numbers) + 5
        elif slot == 7:  # 4 of a kind
            reward = _k_of_a_kind(numbers, 4)
        elif slot == 8:  # full house
            reward = _full_house(numbers)
        elif slot == 9:
            reward = _straight(numbers, 4)
        elif slot == 10:
            reward = _straight(numbers, 5)
        elif slot == 11:
            reward = _k_of_a_kind(numbers, 5)

        state[12] = 0
        slots[slot] = 1
        for poss in ALL_POSS_5:
            numbers[:] = poss
            res += V[encode(state)]
            counter += 1
        res += counter * reward

    return res / counter


def infer(state):
    encoded_state = encode(state)
    max_a, max_b, max_V = None, None, 0
    for a in ACTION_INDEXES:
        for b in range(12):
            if state[b] == 1:
                continue
            new_V = step(encoded_state, a, b, V)
            if new_V > max_V:
                max_a, max_b, max_V = a, b, new_V

    return decode(encoded_state)[-5:][max_a], max_b


if __name__ == '__main__':
    V = np.load('./value.npz')['V']
    state = np.array([1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 3, 5, 2, 1, 1, 5])

    reroll_dices, slot = infer(state)
    print(f'Slots: {state[:12]}')
    print(f'Stage: {state[12]}')
    print(f'Upper Score: {state[13]}')
    print(f'Dices: {state[14:]}')
    print('')

    print('Optimal Action: ')
    if state[12] == 2:
        print(f'Select Slots {slot}')
    else:
        print(f'Reroll Dices {reroll_dices}')
