import numpy as np
from utils import ALL_POSS, POSS_POSITION, ACTION_INDEXES, decode, encode, encode_small
from numba import njit
import argparse


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
    reward, counter, max_counter = 0, 1, 1
    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:
            counter += 1
            max_counter = max(max_counter, counter)
        else:
            counter = 1
    if max_counter >= k:
        reward = 30 if k == 4 else 40
    return reward


@njit
def roll_dice(raw_state, indexes, V, V_small):
    res, counter = 0, 0
    state = decode(raw_state)
    slots, stage, upper_score, numbers = state[:12], state[12], state[13], state[14:]

    assert stage <= 1
    state[12] += 1
    num_change = np.sum(indexes)
    if num_change == 0:
        res = V[encode(state)]
    elif num_change <= 4:
        for poss in ALL_POSS[POSS_POSITION[num_change - 1]:POSS_POSITION[num_change], :]:
            numbers[indexes] = poss[-num_change:]
            res += poss[0] * V[encode(state)]
        res /= np.sum(ALL_POSS[POSS_POSITION[num_change - 1]:POSS_POSITION[num_change], 0])
    else:
        if V_small[encode_small(state)] != -1:
            res = V_small[encode_small(state)]
        else:
            for poss in ALL_POSS[POSS_POSITION[num_change - 1]:POSS_POSITION[num_change], :]:
                numbers[indexes] = poss[-num_change:]
                res += poss[0] * V[encode(state)]
            res /= np.sum(ALL_POSS[POSS_POSITION[num_change - 1]:POSS_POSITION[num_change], 0])
            V_small[encode_small(state)] = res
    return res


@njit
def fill_slot(raw_state, slot, V, V_small):
    res, reward = 0, 0
    state = decode(raw_state)
    slots, stage, upper_score, numbers = state[:12], state[12], state[13], state[14:]
    assert stage > 1

    if slots[slot] == 1:
        return 0.0
    elif slot <= 5:
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
    elif slot == 9:  # small straight
        reward = _straight(numbers, 4)
    elif slot == 10:  # large straight
        reward = _straight(numbers, 5)
    elif slot == 11:  # yahtzee
        reward = _k_of_a_kind(numbers, 5)

    state[12] = 0
    slots[slot] = 1
    if V_small[encode_small(state)] != -1:
        res = V_small[encode_small(state)]
    else:
        for poss in ALL_POSS[POSS_POSITION[4]:POSS_POSITION[5], :]:
            numbers[:] = poss[-5:]
            res += poss[0] * V[encode(state)]
        res /= np.sum(ALL_POSS[POSS_POSITION[4]:POSS_POSITION[5], 0])
        V_small[encode_small(state)] = res

    return reward + res


@njit
def single_step(V, index_list):
    V_small = -1 * np.ones(1302527)
    for index in range(len(index_list)):
        i = index_list[index]
        state = decode(i)
        if state[12] == 2:
            for b in range(12):
                V[i] = max(V[i], fill_slot(i, b, V, V_small))
        else:
            for a in ACTION_INDEXES:
                V[i] = max(V[i], roll_dice(i, a, V, V_small))
        if index % 10000 == 0:
            print(index, i, V[i], decode(i))


def process(cur_slots, cur_stage, cur_fold):
    total_fold = 8
    data = np.load('./indexes.npy')

    if cur_stage == 2:
        prev_slots = cur_slots + 1
        prev_stage = 0
    else:
        prev_slots = cur_slots
        prev_stage = cur_stage + 1

    V = np.load(f'./V_{prev_slots}_{prev_stage}.npy')
    data = data[(data[:, 1] == cur_slots) & (data[:, 2] == cur_stage), 0]

    fold_length = int(np.ceil(data.shape[0] / total_fold))
    print(fold_length)

    single_step(V, data[cur_fold * fold_length:(cur_fold + 1) * fold_length])
    np.save(f'./V_{cur_slots}_{cur_stage}_{cur_fold}.npy', V[data[cur_fold * fold_length:(cur_fold + 1) * fold_length]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--id', type=int)
    parser.add_argument('--slots', type=int)
    parser.add_argument('--stage', type=int)

    args = parser.parse_args()

    process(args.slots, args.stage, args.id)
