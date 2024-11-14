import numpy as np


def build_adjacency_list(edges):
    adj_list = {}
    for edge in edges:
        a, b = edge
        if a not in adj_list:
            adj_list[a] = []
        if b not in adj_list:
            adj_list[b] = []
        adj_list[a].append(b)
        adj_list[b].append(a)
    return adj_list

def invert_hash(hash):
    hash_inverted = {}
    for key, val in hash.items():
        if val not in hash:
            hash_inverted[val] = [key]
        else:
            hash_inverted[val].append(key)
    return hash_inverted

def get_key(dct, value):
    keys = []
    for key, val in dct.items():
        if val == value:
            keys.append(key)
    return keys

def direct_graph(edges):
    dg = []
    for edge in edges:
        dg.append(list(edge))
        dg.append(list(edge[::-1]))
    return dg

def custom_round(number, tolerance):
    k = int(-np.log10(tolerance))
    return round(number, k)

def list_rotate(lst):
    idx = lst.index(min(lst))
    return lst[idx:] + lst[:idx]