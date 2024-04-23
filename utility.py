import torch
import numpy as np
import os

import gzip
from Bio.Data.SCOPData import protein_letters_3to1
from Bio.PDB import PDBParser, MMCIFParser

def readname(file):
    i = 1
    name = []
    file = open(file, "r")
    for line in file:
        line = line.strip('\n')
        if i & 1:
            name.append(line)
        i += 1
    return name

def readfile(file1):
    i = 0

    file = open(file1, 'r')
    data = []
    for line in file:
        line = line.strip('\n')
        if i & 1:
            data.append(line)
        i += 1
    return data

def readstructure(data, file):
    strcuture = []
    for sequence in data:
        i = 0
        single = [0] * 500
        for ch in sequence:
            if ch == 'A':
                single[i] = 1
            elif ch == 'R':
                single[i] = 2
            elif ch == 'N':
                single[i] = 3
            elif ch == 'D':
                single[i] = 4
            elif ch == 'C':
                single[i] = 5
            elif ch == 'Q':
                single[i] = 6
            elif ch == 'E':
                single[i] = 7
            elif ch == 'G':
                single[i] = 8
            elif ch == 'H':
                single[i] = 9
            elif ch == 'I':
                single[i] = 10
            elif ch == 'L':
                single[i] = 11
            elif ch == 'K':
                single[i] = 12
            elif ch == 'M':
                single[i] = 13
            elif ch == 'F':
                single[i] = 14
            elif ch == 'P':
                single[i] = 15
            elif ch == 'S':
                single[i] = 16
            elif ch == 'T':
                single[i] = 17
            elif ch == 'W':
                single[i] = 18
            elif ch == 'Y':
                single[i] = 19
            elif ch == 'V':
                single[i] = 20
            i += 5
        strcuture.append(single)

    secondary = open(file, 'r')
    i = -1
    j = 0
    prev = ''
    for line in secondary:
        line = line.strip('\n')
        if len(line) < 1 or (line[0] != 'E' and line[0] != 'B'):
            continue
        vec = line.split('\t')
        if vec[2] != prev:
            i += 1
            j = 0
        prev = vec[2]
        strcuture[i][j + 1] = float(vec[-3])
        strcuture[i][j + 2] = float(vec[-2])
        strcuture[i][j + 3] = float(vec[-1])
        strcuture[i][j + 4] = float(vec[-6])
        j += 5
    return strcuture