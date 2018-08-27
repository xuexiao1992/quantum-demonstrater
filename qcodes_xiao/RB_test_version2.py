# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:31:43 2017

@author: X.X
"""
import numpy as np
#from pycqed.measurement.randomized_benchmarking.clifford_group import (clifford_lookuptable)


#%%
# Clifford group decomposition maps
#I = np.eye(2)
# Pauli group


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

CZ_dumy = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=complex)


Zp_prep = -1j*Pauli_Z


CZ00 = np.array([[-1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]], dtype=complex)
CZ01 = np.array([[1, 0, 0, 0],
                 [0, -1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]], dtype=complex)
CZ10 = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, -1, 0],
                 [0, 0, 0, 1]], dtype=complex)

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
        'mZ9': mZ9,
        'CZ': CZ,
        'CZ_dumy': CZ_dumy,
        'CZ_dumy_dumy': CZ_dumy,
        'Zp_prep': Zp_prep
        }

Clifford_gates = [
        ['I'],
#        ['X9', 'mX9'],
        ['Xp'], ['Yp'], ['Yp', 'Xp'],
        ['X9', 'Y9'], ['X9', 'mY9'], ['mX9', 'Y9'], ['mX9', 'mY9'], ['Y9', 'X9'], ['Y9', 'mX9'], ['mY9', 'X9'], ['mY9', 'mX9'],
        ['X9'], ['mX9'], ['Y9'], ['mY9'], ['mX9', 'Y9', 'X9'], ['mX9', 'mY9', 'X9'],
        ['Xp', 'Y9'], ['Xp', 'mY9'], ['Yp', 'X9'], ['Yp', 'mX9'], ['X9', 'Y9', 'X9'], ['mX9', 'Y9', 'mX9']
        ]
#%%     generate Clifford group for 1 Qubit

#Clifford_group = [np.empty([2, 2])]*(24)
Clifford_group = [{}]*(24)
# explictly reversing order because order of operators is order in time
#for i in range(24):
#    Clifford = single_qubit_Cliffords[i]
#    if len(Clifford) == 1:
#        matrix = gates[Clifford[0]]
#    else:
#        matrix =  np.linalg.multi_dot([gates[Clifford[i]] for i in range(len(Clifford))][::-1])
#    Clifford_group[i] = {
#            'gates': Clifford,
#            'matrix': matrix
#            }
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


Clifford_group_XY = Clifford_group
#%%
'''
Clifford_gates = [
        ['I'], ['Xp'], ['Yp'], ['Zp'],
        ['mZ9', 'X9'], ['Z9', 'X9'], ['Z9', 'mX9'], ['mZ9', 'mX9'], 
        ['X9', 'Z9'], ['mX9', 'mZ9'], ['X9', 'mZ9'], ['mX9', 'Z9'],
        ['X9'], ['mX9'], 
        ['Z9', 'mX9', 'mZ9'], ['Z9', 'X9', 'mZ9'], 
        ['Z9'], ['mZ9'],
        ['Y9', 'mZp'], ['mY9', 'Zp'], 
        ['X9', 'Zp'], ['mX9', 'mZp'], 
        ['Xp', 'Z9'], ['mXp', 'Z9']
        ]

'''


'''
Clifford_group = [{}]*(24)


Clifford_group[0] = np.linalg.multi_dot([I, I][::-1])
Clifford_group[1] = np.linalg.multi_dot([I, Xp][::-1])
Clifford_group[2] = np.linalg.multi_dot([mZ9, mXp, Z9][::-1])
Clifford_group[3] = np.linalg.multi_dot([I, Zp][::-1])

Clifford_group[4] = np.linalg.multi_dot([mZ9, X9][::-1])
Clifford_group[5] = np.linalg.multi_dot([Z9, X9][::-1])
Clifford_group[6] = np.linalg.multi_dot([Z9, mX9][::-1])
Clifford_group[7] = np.linalg.multi_dot([mZ9, mX9][::-1])
Clifford_group[8] = np.linalg.multi_dot([X9, Z9][::-1])
Clifford_group[9] = np.linalg.multi_dot([mX9, mZ9][::-1])
Clifford_group[10] = np.linalg.multi_dot([X9, mZ9][::-1])
Clifford_group[11] = np.linalg.multi_dot([mX9, Z9][::-1])

Clifford_group[12] = np.linalg.multi_dot([I, X9][::-1])
Clifford_group[13] = np.linalg.multi_dot([I, mX9][::-1])
Clifford_group[14] = np.linalg.multi_dot([Z9, mX9, mZ9][::-1])
Clifford_group[15] = np.linalg.multi_dot([Z9, X9, mZ9][::-1])
Clifford_group[16] = np.linalg.multi_dot([I, Z9][::-1])
Clifford_group[17] = np.linalg.multi_dot([I, mZ9][::-1])

Clifford_group[18] = np.linalg.multi_dot([Y9, mZp][::-1])
Clifford_group[19] = np.linalg.multi_dot([mY9, Zp][::-1])
Clifford_group[20] = np.linalg.multi_dot([X9, Zp][::-1])
Clifford_group[21] = np.linalg.multi_dot([mX9, mZp][::-1])
Clifford_group[22] = np.linalg.multi_dot([Xp, Z9][::-1])
Clifford_group[23] = np.linalg.multi_dot([mXp, mZ9][::-1])


'''


'''
H = XYZ = XZX = ZXZ = YZZ

'''


Clifford_gates = [
        ['X9', 'mX9'], ['Xp'], ['Yp'], ['X9', 'Zp', 'X9'],
        ['X9', 'Y9'], ['X9', 'mY9'], ['mX9', 'Y9'], ['mX9', 'mY9'], 
        ['Y9', 'X9'], ['Y9', 'mX9'], ['mY9', 'X9'], ['mY9', 'mX9'],
        ['Y9', 'mZ9', 'mY9'], ['Y9', 'Z9', 'mY9'],                         # X9, mX9
        ['X9', 'Z9', 'mX9'], ['X9', 'mZ9', 'mX9'],                          # Y9, mY9
        ['mX9', 'Z9', 'Y9'], ['mX9', 'mZ9', 'mY9'],
        ['X9', 'mZ9', 'X9'], ['X9', 'Z9', 'X9'],
        ['Y9', 'X9', 'Z9'], ['Y9', 'mX9', 'mZ9'],
        ['Xp', 'Z9'], ['mXp', 'mZ9']
        ]

Clifford_group = [{}]*(24)


#Clifford_group[0] = np.linalg.multi_dot([Z9, X9, mZ9, Y9][::-1])
Clifford_group[0] = np.linalg.multi_dot([X9, mX9][::-1])
Clifford_group[1] = np.linalg.multi_dot([I, Xp][::-1])
Clifford_group[2] = np.linalg.multi_dot([I, Yp][::-1])
Clifford_group[3] = np.linalg.multi_dot([X9, Zp, X9][::-1])

Clifford_group[4] = np.linalg.multi_dot([X9, Y9][::-1])
Clifford_group[5] = np.linalg.multi_dot([X9, mY9][::-1])
Clifford_group[6] = np.linalg.multi_dot([mX9, Y9][::-1])
Clifford_group[7] = np.linalg.multi_dot([mX9, mY9][::-1])
Clifford_group[8] = np.linalg.multi_dot([Y9, X9][::-1])
Clifford_group[9] = np.linalg.multi_dot([Y9, mX9][::-1])
Clifford_group[10] = np.linalg.multi_dot([mY9, X9][::-1])
Clifford_group[11] = np.linalg.multi_dot([mY9, mX9][::-1])

Clifford_group[12] = np.linalg.multi_dot([Y9, mZ9, mY9][::-1])
Clifford_group[13] = np.linalg.multi_dot([Y9, Z9, mY9][::-1])
Clifford_group[14] = np.linalg.multi_dot([X9, Z9, mX9][::-1])
Clifford_group[15] = np.linalg.multi_dot([X9, mZ9, mX9][::-1])
Clifford_group[16] = np.linalg.multi_dot([mX9, Z9, Y9][::-1])
Clifford_group[17] = np.linalg.multi_dot([mX9, mZ9, mY9][::-1])

Clifford_group[18] = np.linalg.multi_dot([X9, mZ9, X9][::-1])
Clifford_group[19] = np.linalg.multi_dot([X9, Z9, X9][::-1])
Clifford_group[20] = np.linalg.multi_dot([Y9, X9, Z9][::-1])
Clifford_group[21] = np.linalg.multi_dot([Y9, mX9, mZ9][::-1])
Clifford_group[22] = np.linalg.multi_dot([Xp, Z9][::-1])
Clifford_group[23] = np.linalg.multi_dot([mXp, mZ9][::-1])


#%%

for i in range(24):
#    if not np.array_equal(Clifford_group[i], Clifford_group_XY[i]):
    C1 = Clifford_group[i]
    C2 = Clifford_group_XY[i]
    C2 = np.matrix.getH(C2)
    matrix = np.linalg.multi_dot([C1, C2])
    if abs(matrix[0][0]-matrix[1][1])<1e-10 and abs(matrix[1][0])<1e-10 and abs(matrix[0][1])<1e-10:
        continue
    else:
        print('%dth Clifford is wrong\n'%i)



#%%

Pauli_group = [
        ['I', 'I'], 
        ['I', 'Zp_prep'], ['Zp_prep', 'I'], 
        ['Zp_prep', 'Zp_prep'],
        ['I', 'Xp'], ['I', 'Yp'], 
        ['Zp_prep', 'Xp'], ['Zp_prep', 'Yp'],
        ['Xp', 'I'], ['Yp', 'I'],
        ['Xp', 'Zp_prep'], ['Yp', 'Zp_prep'],
        ['Xp', 'Xp'], ['Xp', 'Yp'],
        ['Yp', 'Xp'], ['Yp', 'Yp'],
        ]

#%%

def convert_clifford_to_sequence(clifford_index_1, clifford_index_2, start_1 = 'I', start_2 = 'I', interleave = None):

    clifford_groups_1 = []
    clifford_groups_2 = []
    clifford_gates_1 = []
    clifford_gates_2 = []
    
    clifford_groups_all = []
    
    if len(clifford_index_1) != 0:
        for i in range(len(clifford_index_1)):
            index_1 = clifford_index_1[i]
            index_2 = clifford_index_2[i]
            
            clifford_groups_1.append(Clifford_group[index_1])
            clifford_groups_2.append(Clifford_group[index_2])
            
            clifford_gates_1.append(Clifford_gates[index_1])
            clifford_gates_2.append(Clifford_gates[index_2])
            
            clifford_groups_all.append(np.kron(Clifford_group[index_1], Clifford_group[index_2]))
            
            if interleave is not None:
                clifford_groups_1.append(gates[interleave])
                clifford_gates_1.append([interleave])
                clifford_groups_2.append(gates[interleave+'_dumy'])
                clifford_gates_2.append([interleave+'_dumy'])
                
                clifford_groups_all.append(gates[interleave])
                
    if len(clifford_groups_1) == 0:
        total_matrix = np.kron(I, I)
    
    elif len(clifford_groups_1) == 1:
        total_matrix = np.kron(clifford_groups_1[0], clifford_groups_2[0])
    
    else:
#        clifford_groups = [np.kron(clifford_groups_1[i], clifford_groups_2[i]) for i in range(len(clifford_index_1))]
        total_matrix = np.linalg.multi_dot(clifford_groups_all[::-1])
#        print('')
    
    if start_1 != 'I' and len(clifford_index_1) > 0:
        index_1 = clifford_index_1[0]
        first_random_Clifford = Clifford_group[index_1]
        Dice_gate = gates[start_1]
        first_real_Clifford_1 = np.linalg.multi_dot([Dice_gate, first_random_Clifford][::-1])
        inversed_first_real_Clifford_1 = np.matrix.getH(first_real_Clifford_1)
        for i in range(len(Clifford_group)):
            mat = np.linalg.multi_dot([inversed_first_real_Clifford_1, Clifford_group[i]][::-1])
            if abs(mat[1,0]) < 1e-10 and abs(mat[0,1]) < 1e-10 and abs(mat[0,0]-mat[1,1]) < 1e-10:
                break
            elif i == len(Clifford_group):
                raise KeyError('Cannot find the first Clifford for Qubit 1')
        clifford_gates_1[0] = Clifford_gates[i]
#    if start_1 != 'I' and len(clifford_index_1) == 0:
    if len(clifford_index_1) == 0:
        clifford_gates_1.append([start_1])
        
    if start_2 != 'I' and len(clifford_index_2) > 0:
        index_2 = clifford_index_2[0]
        first_random_Clifford = Clifford_group[index_2]
        Dice_gate = gates[start_2]
        first_real_Clifford_2 = np.linalg.multi_dot([Dice_gate, first_random_Clifford][::-1])
        inversed_first_real_Clifford_2 = np.matrix.getH(first_real_Clifford_2)
        for i in range(len(Clifford_group)):
            mat = np.linalg.multi_dot([inversed_first_real_Clifford_2, Clifford_group[i]][::-1])
            if abs(mat[1,0]) < 1e-10 and abs(mat[0,1]) < 1e-10 and abs(mat[0,0]-mat[1,1]) < 1e-10:
                break
            elif i == len(Clifford_group):
                raise KeyError('Cannot find the first Clifford for Qubit 2')
        clifford_gates_2[0] = Clifford_gates[i]
    
    if len(clifford_index_2) == 0:
        clifford_gates_2.append([start_2])
    
    if len(clifford_index_1) == 0 and len(clifford_index_2) == 0:
        return clifford_gates_1, clifford_gates_2
    
#    m = 0
    init_state = np.array([1,0,0,0])
    for i in range(len(Clifford_group)):
        for j in range(len(Clifford_group)):
#            print('i: %d\nj: %d\n'%(i,j))
            mat1 = np.linalg.multi_dot([init_state, total_matrix, np.kron(Clifford_group[i], Clifford_group[j])][::-1])
#            print('mat1: \n', mat1)
#            mat2 = np.linalg.multi_dot([total_matrix, CZ, np.kron(Clifford_group[i], Clifford_group[j])][::-1])
#            if abs(np.sum(abs(mat1))-np.sum(abs(np.diag(mat1)))) < 1e-5:# and abs(abs(np.sum(np.diag(mat1)))-np.sum(abs(np.diag(mat1)))) < 1e-5:
            if np.sum(abs(mat1[1:])) < 1e-8:
                m = 1
                print('mat1')
                clifford_gates_1.append(Clifford_gates[i])
                clifford_gates_2.append(Clifford_gates[j])
                print('C1:', clifford_gates_1)
                print('C2:', clifford_gates_2)
                return clifford_gates_1, clifford_gates_2
    
    for i in range(len(Clifford_group)):
        for j in range(len(Clifford_group)):
#            print('i: %d\nj: %d\n'%(i,j))
            mat2 = np.linalg.multi_dot([init_state, total_matrix, CZ, np.kron(Clifford_group[i], Clifford_group[j])][::-1])
#            if abs(np.sum(abs(mat2))-np.sum(abs(np.diag(mat2)))) < 1e-5 and abs(abs(np.sum(np.diag(mat2)))-np.sum(abs(np.diag(mat2)))) < 1e-5:
            if np.sum(abs(mat2[1:])) < 1e-8:
#                m = 2
                print('mat2')
                clifford_gates_1.append(['CZ'])
                clifford_gates_2.append(['CZ_dumy'])
                clifford_gates_1.append(Clifford_gates[i])
                clifford_gates_2.append(Clifford_gates[j])
#                print('C1:', clifford_gates_1)
#                print('C2:', clifford_gates_2)
                return clifford_gates_1, clifford_gates_2
    
    for i in range(len(Clifford_group)):
        for j in range(len(Clifford_group)):
#            print('i: %d\nj: %d\n'%(i,j))
            for G1 in ['X9', 'Y9', 'mX9', 'mY9',]:# 'Z9', 'mZ9',]:# 'I', 'Xp', 'Yp', 'Zp']:
                for G2 in ['X9', 'Y9', 'mX9', 'mY9',]:# 'Z9', 'mZ9',]:# 'I', 'Xp', 'Yp', 'Zp']:
                    mat3 = np.linalg.multi_dot([init_state, total_matrix, np.kron(gates[G1], gates[G2]), CZ, np.kron(Clifford_group[i], Clifford_group[j])][::-1])
#                    if abs(np.sum(abs(mat3))-np.sum(abs(np.diag(mat3)))) < 1e-5 and abs(abs(np.sum(np.diag(mat3)))-np.sum(abs(np.diag(mat3)))) < 1e-5:
                    if np.sum(abs(mat3[1:])) < 1e-8:    
                        print('mat3')
                        clifford_gates_1.append([G1])
                        clifford_gates_2.append([G2])
                        clifford_gates_1.append(['CZ'])
                        clifford_gates_2.append(['CZ_dumy'])
                        clifford_gates_1.append(Clifford_gates[i])
                        clifford_gates_2.append(Clifford_gates[j])
#                        print('C1:', clifford_gates_1)
#                        print('C2:', clifford_gates_2)
                        return clifford_gates_1, clifford_gates_2
                                
#    if i == 23 and j == 23 and m ==0:
    print('not calculated rightly')
    print('i: %d\nj: %d\n'%(i,j))
    print('C1:', clifford_gates_1)
    print('C2:', clifford_gates_2)
    print('Clifford_all:', clifford_groups_all)
    print('total_matrix:', total_matrix)
    raise ValueError('not calculated rightly')
    return 0, 0

#    clifford_gates.append(Clifford_gates[i])
    
#%%

def generate_randomized_clifford_sequence(start = 'I', interleave = None):
    
    clifford_sets_1 = []
    clifford_sets_2 = []
    
    sequence_length = 30
    
    rep_num = 40
    
    sequence_number = 16*rep_num
    
    sequence_number = 24
    start = start
    
    for j in range(sequence_number):
        
        clifford_sets_1.append([])
        clifford_sets_2.append([])
        
#        Pauli_index = j//rep_num
        Pauli_index = j%16
        
        print('j:', j)
        print('index:', Pauli_index)
        start_1 = Pauli_group[Pauli_index][0]
        start_2 = Pauli_group[Pauli_index][1]
        
        start_1 = 'I'
        start_2 = 'I'
#        
        for i in range(sequence_length+1):
            
            if i in range(15, 30) and i%3 != 0:
                continue
            elif i in range(30, 101) and i%10 != 0:
                continue
            
            clifford_gates_1 = 0
            while clifford_gates_1 == 0:
                
                clifford_index_1 = list((np.random.rand(i)*24).astype(int))
                clifford_index_2 = list((np.random.rand(i)*24).astype(int))
                print('i:', i)
                clifford_gates_1, clifford_gates_2 = convert_clifford_to_sequence(clifford_index_1, clifford_index_2, start_1, start_2, interleave)
            
#            print(clifford_gates_1)
#            print(clifford_gates_2)
            
            clifford_sets_1[j].append(clifford_gates_1)
            clifford_sets_2[j].append(clifford_gates_2)
            
    return clifford_sets_1, clifford_sets_2


clifford_sets_1, clifford_sets_2 = generate_randomized_clifford_sequence(interleave = 'CZ')

clifford_sets = clifford_sets_1
#clifford_sets_2 = clifford_sets_1


#%%
'''
mat = I
start = Xp
for i in range(len(clifford_sets_1)):
    for j in range(len(clifford_sets_1[i])):
        
        for k in range(len(clifford_sets_1[i][j])):
            for l in range(len(clifford_sets_1[i][j][k])):
                gate_name = clifford_sets_1[i][j][k][l]
                gate = gates[gate_name]
                mat = np.linalg.multi_dot([mat, gate][::-1])
        mat = np.linalg.multi_dot([mat, start][::-1])
        if abs(np.sum(abs(mat))-np.sum(abs(np.diag(mat)))) < 1e-8 and abs(abs(np.sum(np.diag(mat)))-np.sum(abs(np.diag(mat)))) < 1e-8:
            pass
        else:
            raise ValueError('Calculation not correct: clifford_%d_%d_%d_%d'%(i,j,k,l))
        
'''



#%%     convert to sequence
'''
#clifford_index = [6,3,8,0]

def convert_clifford_to_sequence(clifford_index, start = 'I', interleave = None):

    clifford_groups = []
    clifford_gates = []
    
    if len(clifford_index) != 0:
        for i in clifford_index:
            clifford_groups.append(Clifford_group[i])
            clifford_gates.append(Clifford_gates[i])
#        for gate in Clifford_gates[i]:
#            clifford_gates.append(gate)
            if interleave is not None:
                clifford_groups.append(gates[interleave])
                clifford_gates.append([interleave])
    
    if len(clifford_groups) == 0:
        total_matrix = I
    elif len(clifford_groups) == 1:
        total_matrix = clifford_groups[0]
    else:
        total_matrix = np.linalg.multi_dot(clifford_groups[::-1])
    
#    return clifford_gates, total_matrix
#def calculate_recovery_clifford(total_matrix):
    for i in range(len(Clifford_group)):
        mat = np.linalg.multi_dot([total_matrix, Clifford_group[i]][::-1])
#        if np.array_equal(np.around(mat, decimals = 1), np.around(Clifford_group[0],decimals = 1)):
        if abs(mat[1,0]) < 1e-5 and abs(mat[0,1]) < 1e-5:
            break
    
    clifford_gates.append(Clifford_gates[i])
    
    if start != 'I' and len(clifford_index)>0:
        index1 = clifford_index[0]
        first_random_Clifford = Clifford_group[index1]
        Dice_gate = gates[start]
        first_real_Clifford = np.linalg.multi_dot([Dice_gate, first_random_Clifford][::-1])
        for i in range(len(Clifford_group)):
            mat = np.linalg.multi_dot([first_real_Clifford, Clifford_group[i]][::-1])
            if abs(mat[1,0]) < 1e-5 and abs(mat[0,1]) < 1e-5:
                break
            
    return clifford_gates
#    return i, np.around(total_matrix, decimals = 2), np.around(mat, decimals = 2)

'''
#%%     generate randomized clifford sequence
'''
def generate_randomized_clifford_sequence(start = 'I', interleave = None):
    
    clifford_sets_1 = []
    clifford_sets_2 = []
    
    sequence_length = 100
    
    rep_num = 20
    
    sequence_number = 16*rep_num
    
    start = start
    
    for j in range(sequence_number):
        
        clifford_sets_1.append([])
        clifford_sets_2.append([])
        
        Pauli_index = j//rep_num
        
        start_1 = Pauli_group[Pauli_index][0]
        start_2 = Pauli_group[Pauli_index][0]
        
        for i in range(sequence_length+1):
            
            if i in range(15, 30) and i%3 != 0:
                continue
            elif i in range(30, 101) and i%10 != 0:
                continue
            
            clifford_index_1 = list((np.random.rand(i)*24).astype(int))
            clifford_index_2 = list((np.random.rand(i)*24).astype(int))
            
            clifford_gates_1 = convert_clifford_to_sequence(clifford_index_1, start_1, interleave)
            clifford_gates_2 = convert_clifford_to_sequence(clifford_index_2, start_2, interleave)
            
            print(clifford_gates_1)
            print(clifford_gates_2)
            
            clifford_sets_1[j].append(clifford_gates_1)
            clifford_sets_2[j].append(clifford_gates_2)
            
    return clifford_sets_1, clifford_sets_2

clifford_sets_1, clifford_sets_2 = generate_randomized_clifford_sequence(interleave = 'Zp')

#clifford_sets1 = generate_randomized_clifford_sequence(start = 'I')

#clifford_sets2 = generate_randomized_clifford_sequence(start = 'I')
'''


#%%
S1_group = [{}]*(9)

S1_group[0] = Clifford_group[0]
S1_group[1] = Clifford_group[8]
S1_group[2] = Clifford_group[7]

S1_group[3] = Clifford_group[12]
S1_group[4] = Clifford_group[22]
S1_group[5] = Clifford_group[15]

S1_group[6] = Clifford_group[14]
S1_group[7] = Clifford_group[20]
S1_group[8] = Clifford_group[17]

#%%
'''
pt = MatPlot()
ds = experiment.data_set
pt.add(y = ds.probability_data[:,0,:].mean(axis = 0))
'''