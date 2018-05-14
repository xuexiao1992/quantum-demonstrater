# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:58:28 2017

@author: X.X
"""

import numpy as np
#from pycqed.measurement.randomized_benchmarking.clifford_group import (clifford_lookuptable)

#%%

# Clifford group decomposition maps

I = np.array([[1, 0,],
              [0, 1,],], dtype=complex)


Pauli_X = np.array([[0, 1,],
                    [1, 0,],], dtype=complex)

Pauli_Y = np.array([[0, -1j,],
                    [1j, 0,],], dtype=complex)

Pauli_Z = np.array([[1, 0,],
                    [0, -1,],], dtype=complex)


Xp = -1j*Pauli_X
Yp = -1j*Pauli_Y
Zp = -1j*Pauli_Z

mXp = -Xp
mYp = -Yp
mZp = -Zp

X9 = 1/np.sqrt(2)*np.array([[1, -1j,],
                            [-1j, 1,],], dtype=complex)

Y9 = 1/np.sqrt(2)*np.array([[1, -1,],
                            [1, 1,],], dtype=complex)

#Z9 = 1/np.sqrt(2)*np.array([[1-1j, 0,],
#                            [0, 1+1j,],], dtype=complex)

Z9 = 1/np.sqrt(2)*np.array([[1, 0,],
                            [0, 1j,],], dtype=complex)

mX9 = 1/np.sqrt(2)*np.array([[1, 1j,],
                             [1j, 1,],], dtype=complex)

mY9 = 1/np.sqrt(2)*np.array([[1, 1,],
                             [-1, 1,],], dtype=complex)

mZ9 = 1/np.sqrt(2)*np.array([[1, 0,],
                             [0, -1j,],], dtype=complex)


CZ = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, -1]], dtype=complex)

gates = {
        'I': I,
        'Xp': Xp,
        'Yp': Yp,
        'mXp': mXp,
        'mYp': mYp,
        'X9': X9,
        'Y9': Y9,
        'mX9': mX9,
        'mY9': mY9,
        'Zp': Zp,
        'mZp': mZp,
        'Z9': Z9,
        'mZ9': mZ9
        }

Clifford_gates = [
        ['I'], ['Xp'], ['Yp'], ['Yp', 'Xp'],
        ['X9', 'Y9'], ['X9', 'mY9'], ['mX9', 'Y9'], ['mX9', 'mY9'], ['Y9', 'X9'], ['Y9', 'mX9'], ['mY9', 'X9'], ['mY9', 'mX9'],
        ['X9'], ['mX9'], ['Y9'], ['mY9'], ['mX9', 'Y9', 'X9'], ['mX9', 'mY9', 'X9'],
        ['Xp', 'Y9'], ['Xp', 'mY9'], ['Yp', 'X9'], ['Yp', 'mX9'], ['X9', 'Y9', 'X9'], ['mX9', 'Y9', 'mX9']]

#%%     generate Clifford group for 1 Qubit

#Clifford_group = [np.empty([2, 2])]*(24)
Clifford_group = [{}]*(24)

Clifford_group[0] = np.linalg.multi_dot([I, I][::-1])
Clifford_group[1] = np.linalg.multi_dot([I, Xp][::-1])
Clifford_group[2] = np.linalg.multi_dot([I, Yp][::-1])
Clifford_group[3] = np.linalg.multi_dot([Yp, Xp][::-1])

Clifford_group[4] = np.linalg.multi_dot([X9, Y9][::-1])
Clifford_group[5] = np.linalg.multi_dot([X9, mY9][::-1])
Clifford_group[6] = np.linalg.multi_dot([mX9, Y9][::-1])
Clifford_group[7] = np.linalg.multi_dot([mX9, mY9][::-1])
Clifford_group[8] = np.linalg.multi_dot([Y9, X9][::-1])
Clifford_group[9] = np.linalg.multi_dot([Y9, mX9][::-1])
Clifford_group[10] = np.linalg.multi_dot([mY9, X9][::-1])
Clifford_group[11] = np.linalg.multi_dot([mY9, mX9][::-1])

Clifford_group[12] = np.linalg.multi_dot([I, X9][::-1])
Clifford_group[13] = np.linalg.multi_dot([I, mX9][::-1])
Clifford_group[14] = np.linalg.multi_dot([I, Y9][::-1])
Clifford_group[15] = np.linalg.multi_dot([I, mY9][::-1])
Clifford_group[16] = np.linalg.multi_dot([mX9, Y9, X9][::-1])
Clifford_group[17] = np.linalg.multi_dot([mX9, mY9, X9][::-1])

Clifford_group[18] = np.linalg.multi_dot([Xp, Y9][::-1])
Clifford_group[19] = np.linalg.multi_dot([Xp, mY9][::-1])
Clifford_group[20] = np.linalg.multi_dot([Yp, X9][::-1])
Clifford_group[21] = np.linalg.multi_dot([Yp, mX9][::-1])
Clifford_group[22] = np.linalg.multi_dot([X9, Y9, X9][::-1])
Clifford_group[23] = np.linalg.multi_dot([mX9, Y9, mX9][::-1])

#%%
S1_group = [{}]*(9)
S1_gates = [{}]*(9)

S1_group[0] = Clifford_group[0]
S1_group[1] = Clifford_group[8]
S1_group[2] = Clifford_group[7]

S1_group[3] = Clifford_group[12]
S1_group[4] = Clifford_group[22]
S1_group[5] = Clifford_group[15]

S1_group[6] = Clifford_group[14]
S1_group[7] = Clifford_group[20]
S1_group[8] = Clifford_group[17]


S1_gates[0] = Clifford_gates[0] 
S1_gates[1] = Clifford_gates[8]
S1_gates[2] = Clifford_gates[7]

S1_gates[3] = Clifford_gates[12]
S1_gates[4] = Clifford_gates[22]
S1_gates[5] = Clifford_gates[15]

S1_gates[6] = Clifford_gates[14]
S1_gates[7] = Clifford_gates[20]
S1_gates[8] = Clifford_gates[17]

#%%

C1_group = [{}]*(576)
C1_gates = [{}]*(576)

count = 0
for i in range(24):
    for j in range(24):
        C1_group[count] = np.kron(Clifford_group[i], Clifford_group[j])
        C1_gates[count] = ['Clifford_{}/{}'.format(i,j)]
        count+=1

#%%
CNOT_group = [{}]*(5184)
CNOT_gates = [{}]*(5184)

count = 0
for i in range(24):
    for j in range(24):
        for m in range(6,9):
            for n in range(3):
                CNOT_group[count] = np.linalg.multi_dot([np.kron(Clifford_group[i], Clifford_group[j]), CZ, 
                                                         np.kron(S1_group[m], S1_group[n])][::-1])
                CNOT_gates[count] = ['Clifford_{}/{}'.format(i,j)] + ['CZ'] + ['S1_{}/{}'.format(m,n)]
                count+=1

#%%

iSWAP_group = [{}]*(5184)
iSWAP_gates = [{}]*(5184)

count = 0
for i in range(24):
    for j in range(24):
        for m in range(6,9):
            for n in range(3,6):
                iSWAP_group[count] = np.linalg.multi_dot([np.kron(Clifford_group[i], Clifford_group[j]), CZ, 
                                                          np.kron(Y9, mX9), CZ, 
                                                          np.kron(S1_group[m], S1_group[n])][::-1])
                iSWAP_gates[count] = ['Clifford_{}/{}'.format(i,j)] + ['CZ'] + ['Clifford_14/13'] + ['CZ'] + ['S1_{}/{}'.format(m,n)]
                count+=1

#%%

SWAP_group = [{}]*(576)
SWAP_gates = [{}]*(576)

count = 0
for i in range(24):
    for j in range(24):
        SWAP_group[count] = np.linalg.multi_dot([np.kron(Clifford_group[i], Clifford_group[j]), CZ, 
                                            np.kron(Y9, mY9), CZ, 
                                            np.kron(mY9, Y9), CZ,
                                            np.kron(Y9, I)][::-1])
        SWAP_gates[count] = ['Clifford_{}/{}'.format(i,j)] + ['CZ'] + ['Clifford_14/15'] + ['CZ'] + ['Clifford_15/14'] + ['CZ'] + ['Clifford_14/0']
        count+=1

#%%
Clifford2_group = C1_group + CNOT_group + iSWAP_group + SWAP_group
Clifford2_gates = C1_gates + CNOT_gates + iSWAP_gates + SWAP_gates
#%%     convert to sequence

clifford_index = [6,3,0]

def convert_clifford_to_sequence(clifford_index, interleave = None):

    clifford_groups = []
    clifford_gates = []
    
    if len(clifford_index) != 0:
        for i in clifford_index:
            clifford_groups.append(Clifford2_group[i])
            clifford_gates.append(Clifford2_gates[i])
#        for gate in Clifford_gates[i]:
#            clifford_gates.append(gate)
            if interleave is not None:
                interleaved_gate = interleave if interleave != 'ZorI' else random.choice(['Zp', 'ZpI']) # specially used for a while
                clifford_groups.append(gates[interleaved_gate])
                clifford_gates.append([interleaved_gate])
    
    if len(clifford_groups) == 0:
        total_matrix = np.kron(I,I)
    elif len(clifford_groups) == 1:
        total_matrix = clifford_groups[0]
    else:
        total_matrix = np.linalg.multi_dot(clifford_groups[::-1])
    
#    return clifford_gates, total_matrix
#def calculate_recovery_clifford(total_matrix):
    m = 0
    for i in range(len(Clifford2_group)):
        mat = np.linalg.multi_dot([total_matrix, Clifford2_group[i]][::-1])
#        mat2 = np.linalg.multi_dot([total_matrix, CZ, Clifford2_group[i]][::-1])
#        mat3 = np.linalg.multi_dot([total_matrix, CZ, Clifford_group[i]][::-1])
        if abs(np.sum(abs(mat))-np.sum(abs(np.diag(mat)))) < 1e-5:
            m = 1
            clifford_gates.append(Clifford2_gates[i])
            return clifford_gates
#        elif abs(np.sum(abs(mat2))-np.sum(abs(np.diag(mat2)))) < 1e-5:
#            m = 2
#            ii = i
#            break
        elif i == 11519:
            if m == 2:
                clifford_gates.append(['CZ'])
                clifford_gates.append(Clifford_gates[ii])
                return clifford_gates
            else:
                print('not calculated rightly')
                return 0
#    if m == 2:
#        clifford_gates.append(['CZ'])
#    clifford_gates.append(Clifford_gates[i])
#    return clifford_gates
#    return i, np.around(total_matrix, decimals = 2), np.around(mat, decimals = 2)

#%%     generate randomized clifford sequence

def generate_randomized_clifford_sequence(interleave = None):
    
    clifford_sets = []
    
    sequence_length = 12
    
    sequence_number = 60
    
    for j in range(sequence_number):
        
        clifford_sets.append([])
        
        for i in range(sequence_length+1):
            
            if i in range(15, 30) and i%3 != 0:
                continue
            elif i in range(30, 101) and i%10 != 0:
                continue
            
            clifford_gates = 0
            while clifford_gates == 0:
                clifford_index = list((np.random.rand(i)*11520).astype(int))
                
                clifford_gates = convert_clifford_to_sequence(clifford_index, interleave)
            print('i=', i)
#            if clifford_gates == 0:
#                continue
            print(clifford_gates)
            
            clifford_sets[j].append(clifford_gates)
            
    return clifford_sets

#clifford_sets = generate_randomized_clifford_sequence(interleave = 'Zp')

#clifford_sets = generate_randomized_clifford_sequence(interleave = 'CZ')


clifford_sets = generate_randomized_clifford_sequence()
#









#%%
'''
H1 = 1/np.sqrt(2)*np.array([[1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [1, 0, -1, 0],
                            [0, 1, 0, -1]], dtype=complex)
H2 = 1/np.sqrt(2)*np.array([[1, 1, 0, 0],
                            [1, -1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 1, -1]], dtype=complex)

CZ = 1/np.sqrt(2)*np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, -1]], dtype=complex)

CRotX12 = 1/np.sqrt(2)*np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, -1j],
                                 [0, 0, -1j, 0]], dtype=complex)

CRotX21 = 1/np.sqrt(2)*np.array([[1, 0, 0, 0],
                                 [0, 0, 0, -1j],
                                 [0, 0, 1, 0],
                                 [0, -1j, 0, 0]], dtype=complex)

CRotY12 = 1/np.sqrt(2)*np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, -1],
                                 [0, 0, 1, 0]], dtype=complex)

CNotX12 = 1/np.sqrt(2)*np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0]], dtype=complex)

CNotX21 = 1/np.sqrt(2)*np.array([[1, 0, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0]], dtype=complex)

CNotY12 = 1/np.sqrt(2)*np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, -1j],
                                 [0, 0, 1j, 0]], dtype=complex)

ICRotX12 = np.linalg.inv(CRotX12)

ICRotX21 = np.linalg.inv(CRotX21)

ICRotY12 = np.linalg.inv(CRotY12)




Clifford_group1 = [np.empty([4, 4])]*(24)
# explictly reversing order because order of operators is order in time
Clifford_group1[0] = np.linalg.multi_dot([I, I][::-1])
Clifford_group1[1] = np.linalg.multi_dot([I, Xp1][::-1])
Clifford_group1[2] = np.linalg.multi_dot([I, Yp1][::-1])
Clifford_group1[3] = np.linalg.multi_dot([Yp1, Xp1][::-1])

Clifford_group1[4] = np.linalg.multi_dot([X91, Y91][::-1])
Clifford_group1[5] = np.linalg.multi_dot([X91, mY91][::-1])
Clifford_group1[6] = np.linalg.multi_dot([mX91, Y91][::-1])
Clifford_group1[7] = np.linalg.multi_dot([mX91, mY91][::-1])
Clifford_group1[8] = np.linalg.multi_dot([Y91, X91][::-1])
Clifford_group1[9] = np.linalg.multi_dot([Y91, mX91][::-1])
Clifford_group1[10] = np.linalg.multi_dot([mY91, X91][::-1])
Clifford_group1[11] = np.linalg.multi_dot([mY91, mX91][::-1])

Clifford_group1[12] = np.linalg.multi_dot([I, X91][::-1])
Clifford_group1[13] = np.linalg.multi_dot([I, mX91][::-1])
Clifford_group1[14] = np.linalg.multi_dot([I, Y91][::-1])
Clifford_group1[15] = np.linalg.multi_dot([I, mY91][::-1])
Clifford_group1[16] = np.linalg.multi_dot([mX91, Y91, X91][::-1])
Clifford_group1[17] = np.linalg.multi_dot([mX91, mY91, X91][::-1])

Clifford_group1[18] = np.linalg.multi_dot([Xp1, Y91][::-1])
Clifford_group1[19] = np.linalg.multi_dot([Xp1, mY91][::-1])
Clifford_group1[20] = np.linalg.multi_dot([Yp1, X91][::-1])
Clifford_group1[21] = np.linalg.multi_dot([Yp1, mX91][::-1])
Clifford_group1[22] = np.linalg.multi_dot([X91, Y91, X91][::-1])
Clifford_group1[23] = np.linalg.multi_dot([mX91, Y91, mX91][::-1])


Clifford_group2 = [np.empty([4, 4])]*(24)

Clifford_group2[0] = np.linalg.multi_dot([I, I][::-1])
Clifford_group2[1] = np.linalg.multi_dot([I, Xp2][::-1])
Clifford_group2[2] = np.linalg.multi_dot([I, Yp2][::-1])
Clifford_group2[3] = np.linalg.multi_dot([Yp2, Xp2][::-1])

Clifford_group2[4] = np.linalg.multi_dot([X92, Y92][::-1])
Clifford_group2[5] = np.linalg.multi_dot([X92, mY92][::-1])
Clifford_group2[6] = np.linalg.multi_dot([mX92, Y92][::-1])
Clifford_group2[7] = np.linalg.multi_dot([mX92, mY92][::-1])
Clifford_group2[8] = np.linalg.multi_dot([Y92, X92][::-1])
Clifford_group2[9] = np.linalg.multi_dot([Y92, mX92][::-1])
Clifford_group2[10] = np.linalg.multi_dot([mY92, X92][::-1])
Clifford_group2[11] = np.linalg.multi_dot([mY92, mX92][::-1])

Clifford_group2[12] = np.linalg.multi_dot([I, X92][::-1])
Clifford_group2[13] = np.linalg.multi_dot([I, mX92][::-1])
Clifford_group2[14] = np.linalg.multi_dot([I, Y92][::-1])
Clifford_group2[15] = np.linalg.multi_dot([I, mY92][::-1])
Clifford_group2[16] = np.linalg.multi_dot([mX92, Y92, X92][::-1])
Clifford_group2[17] = np.linalg.multi_dot([mX92, mY92, X92][::-1])

Clifford_group2[18] = np.linalg.multi_dot([Xp2, Y92][::-1])
Clifford_group2[19] = np.linalg.multi_dot([Xp2, mY92][::-1])
Clifford_group2[20] = np.linalg.multi_dot([Yp2, X92][::-1])
Clifford_group2[21] = np.linalg.multi_dot([Yp2, mX92][::-1])
Clifford_group2[22] = np.linalg.multi_dot([X92, Y92, X92][::-1])
Clifford_group2[23] = np.linalg.multi_dot([mX92, Y92, mX92][::-1])


#
S_group1 = [np.empty([4, 4])]*(9)

S_group1[0] = Clifford_group1[0]
S_group1[1] = Clifford_group1[8]
S_group1[2] = Clifford_group1[7]

S_group1[3] = Clifford_group1[12]
S_group1[4] = Clifford_group1[22]
S_group1[5] = Clifford_group1[15]

S_group1[6] = Clifford_group1[14]
S_group1[7] = Clifford_group1[20]
S_group1[8] = Clifford_group1[17]


S_group2 = [np.empty([4, 4])]*(9)

S_group2[0] = Clifford_group2[0]
S_group2[1] = Clifford_group2[8]
S_group2[2] = Clifford_group2[7]

S_group2[3] = Clifford_group2[12]
S_group2[4] = Clifford_group2[22]
S_group2[5] = Clifford_group2[15]

S_group2[6] = Clifford_group2[14]
S_group2[7] = Clifford_group2[20]
S_group2[8] = Clifford_group2[17]


'''



