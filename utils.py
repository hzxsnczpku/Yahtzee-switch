import numpy as np
from numba import njit


SLOT_LIMIT = 2
DICE_LIMIT = 6
STAGE_LIMIT = 3
SCORE_LIMIT = 106
SIZES = np.array([SLOT_LIMIT] * 12 + [STAGE_LIMIT, SCORE_LIMIT] + [DICE_LIMIT] * 5, dtype=np.int64)


def generate_index(k):
    if k == 1:
        return [[False], [True]]
    else:
        res_list = []
        temp = generate_index(k-1)
        for i in [False, True]:
            for t in temp:
                res_list.append([i] + t)
    return res_list


def number_decode_full(num):
    res = np.zeros(5, dtype=np.int64)
    for i in range(4, -1, -1):
        res[i] = num % 6
        num //= 6
    return res


@njit
def number_encode_full(vec):
    res = 0
    for i in range(len(vec)):
        res = res * 6 + vec[i]
    return res


def number_encoding_count(k):
    table = {}
    for i in range(6**k):
        encoded_num = number_encode_full(np.sort(number_decode_full(i)))
        if encoded_num not in table:
            table[encoded_num] = 0
        table[encoded_num] += 1

    output = np.zeros((len(table), 6))
    counter = 0
    for key in table.keys():
        output[counter, 1:] = number_decode_full(key)
        output[counter, 0] = table[key]
        counter += 1
    return output


def number_encoding():
    table = np.zeros(6**5, dtype=np.int64)
    decode_table = np.zeros(252, dtype=np.int64)
    counter = 0
    for i in range(6**5):
        table[i] = number_encode_full(np.sort(number_decode_full(i)))
        if table[i] == i:
            decode_table[counter] = i
            table[i] = counter
            counter += 1
        else:
            table[i] = table[table[i]]
    return table, decode_table


NUMBER_TABLE, NUMBER_DECODE_TABLE = number_encoding()


@njit
def number_encode_table(vec):
    return NUMBER_TABLE[number_encode_full(vec)]


@njit
def encode(vec):
    res = 0
    for i in range(len(vec) - 5):
        res = res * SIZES[i] + vec[i]
    res *= 252
    res += number_encode_table(vec[-5:])
    return res


@njit
def decode(num):
    res = np.zeros(len(SIZES), dtype=np.int32)
    num = num // 252 * 6**5 + NUMBER_DECODE_TABLE[num % 252]
    for i in range(len(res)-1, -1, -1):
        res[i] = num % SIZES[i]
        num //= SIZES[i]
    return res


@njit
def encode_small(vec):
    res = 0
    for i in range(len(vec) - 5):
        res = res * SIZES[i] + vec[i]
    return res

@njit
def decode_small(num):
    res = np.zeros(len(SIZES) - 5, dtype=np.int32)
    for i in range(len(res) - 1, -1, -1):
        res[i] = num % SIZES[i]
        num //= SIZES[i]
    return res


ALL_POSS = np.vstack([number_encoding_count(k+1) for k in range(5)])
POSS_POSITION = np.cumsum([0] + [len(number_encoding_count(k+1)) for k in range(5)])
ACTION_INDEXES = np.array(generate_index(5), dtype=bool)
TOTAL_FOLD = 8

if __name__ == '__main__':
    state = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 105, 5, 5, 5, 5, 5])
    num = encode(state)
    new_state = decode(num)

    print(num)
    print(new_state)

    print(encode_small(state))

    print(decode(32815918))

    print(ALL_POSS)
    print(len(ALL_POSS))
    print(POSS_POSITION)
