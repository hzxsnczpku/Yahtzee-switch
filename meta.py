import os
from multiprocessing import Process
import numpy as np
from utils import TOTAL_FOLD, decode
from numba import njit


def run_cal(slots, stage, id):
    os.system(f'python ./main.py --id {id} --slots {slots} --stage {stage}')


def merge_V(cur_slots, cur_stage):
    data = np.load('./indexes.npy')
    data = data[(data[:, 1] == cur_slots) & (data[:, 2] == cur_stage), 0]
    fold_length = int(np.ceil(data.shape[0] / TOTAL_FOLD))

    if cur_stage == 2:
        prev_slots = cur_slots + 1
        prev_stage = 0
    else:
        prev_slots = cur_slots
        prev_stage = cur_stage + 1

    V_total = np.load(f'./V_{prev_slots}_{prev_stage}.npy')
    for i in range(8):
        V_total[data[i * fold_length:(i + 1) * fold_length]] = np.maximum(
            V_total[data[i * fold_length:(i + 1) * fold_length]], np.load(f'./V_{cur_slots}_{cur_stage}_{i}.npy'))

    print(V_total, V_total.shape)
    np.save(f'./V_{cur_slots}_{cur_stage}.npy', V_total)


@njit
def get_index_list():
    indexes = []
    slot_sum = []
    stage = []
    for i in range(328237056):
        state = decode(i)
        if not 5 * state[0] + 10 * state[1] + 15 * state[2] + 20 * state[3] + 25 * state[4] + 30 * state[5] < state[13]:
            if np.sum(state[:12]) < 12:
                indexes.append(i)
                slot_sum.append(np.sum(state[:12]))
                stage.append(state[12])

    return np.array(indexes), np.array(slot_sum), np.array(stage)


def generate_index_list():
    indexes, slot_sum, stage = get_index_list()
    data = np.vstack([indexes, slot_sum, stage]).T
    np.save('./indexes.npy', data)


def main():
    np.save('./V_12_0.npy', np.zeros(328237056, dtype=np.float64))

    for slot in range(11, -1, -1):
        for stage in range(2, -1, -1):
            processes = [Process(target=run_cal, args=(slot, stage, id)) for id in range(8)]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            merge_V(slot, stage)
            for i in range(8):
                os.system(f'rm ./V_{slot}_{stage}_{i}.npy')

            if stage == 2:
                prev_slots = slot + 1
                prev_stage = 0
            else:
                prev_slots = slot
                prev_stage = stage + 1
            os.system(f'rm ./V_{prev_slots}_{prev_stage}.npy')

    np.savez_compressed('./value.npz', V=np.load('./V_0_0.npy'))


if __name__ == '__main__':
    if not os.path.exists('./indexes.npy'):
        print('indexes.npy not detected, generating ... please wait for 2 minutes')
        generate_index_list()
    main()
