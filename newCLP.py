from collections.abc import Iterable
import random
import pandas as pd
import numpy as np
import torch


class SCLP:
    def __init__(self, r0, B, S_existing, S_new):
        self.r0 = r0                    #coverage Radius
        self.B = B                      #budget
        self.S_existing = S_existing    #fixed costs of upgrading existing facilities
        self.S_new = S_new              #fixed costs of building new facilities
        
    
    #Auslesen und Erstellen der Distanzmatrix 
    def get_n(self, filename):
        file = pd.read_csv(filename, header = None)
        n = int(file.iloc[0,0])
        return n
    
    def read_file(self, filename, n):
        file = pd.read_csv(filename, header = None)
        c = torch.full((n, n), float('inf'))
        
        for i in range(n):
            c[i][i] = 0
        
        for index, row in file.iterrows():
            if index > 0:
                i, j, k = int(row[0]), int(row[1]), int(row[2])
            
                c[i-1][j-1] = k
                c[j-1][i-1] = k
        return c 
    
    def floyd_warshall(self,c,n):
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    c[i][j] = min(c[i][j], c[i][k] + c[k][j])
        return c
    
    
    #Erstellen von Arrays für Radii, Fixkosten
    def create_r(self,p):
        r = torch.zeros(p, dtype=torch.int32)
        for i in range(10):
            r[i] = self.r0
        return r
    
    def create_S(self, p):
        s = np.zeros(p)
        for j in range(p):
            if j < 10:
                s[j] = self.S_existing
            else:
                s[j] = self.S_new
        return s
    
    #Kostenfuntkion
    def calc_f(self, r):
        f = r**2
        return f
    
    
    def calc_b(self, n, p, s, r, d_matrix):
        b_matrix = torch.zeros(n, p, dtype=torch.int32)
        for i in range(n):
            for k in range(p):
                if k < 10:
                    if d_matrix[i, k] > r[k]:
                        b_matrix[i, k] = int(self.calc_f(d_matrix[i, k]) - self.calc_f(r[k]) + s[k])
                    else:
                        b_matrix[i, k] = int(0)
                else:
                    b_matrix[i, k] = int((self.calc_f(d_matrix[i, k])) - (self.calc_f(r[k])) + s[k])
        return b_matrix
    # Remove tied values
    
    def create_B_dict(self, b_matrix, p):            
        B_dict = {}
        for k in range(p):
            sorted_values, _ = torch.sort(torch.unique(b_matrix[:, k]))
            if 0 not in sorted_values:
                sorted_values = torch.cat((torch.tensor([0.0]), sorted_values))
            B_dict[k] = sorted_values
        return B_dict
    
    def create_index_dict(self, b_matrix):
        sort_index_dict = {}
        for k in range(b_matrix.size(1)):  
            _, sorted_indices = torch.sort(b_matrix[:, k], stable=True)
            grouped_indices = self.group_tied_values(b_matrix[:, k], sorted_indices)
            # Fügen Sie NaN an erster Stelle in sort_index_dict für "nichts tun" hinzu
            sort_index_dict[k] = [float('nan')] + grouped_indices
        return sort_index_dict
    

    def group_tied_values(self, column, sorted_indices):
        grouped_indices = [[sorted_indices[0].item()]]
        for i in range(1, len(sorted_indices)):
            if column[sorted_indices[i]] == column[sorted_indices[i - 1]]:
                grouped_indices[-1].append(sorted_indices[i].item())
            else:
                grouped_indices.append([sorted_indices[i].item()])
        return grouped_indices
    
    
    def calc_new_marektshare (self, q, w, F, C, n):
        delta = 0
        for i in range (n):
            if F[i] + C[i] > 0:
                delta += w[i] * ((q[i] * C[i]) / ((F[i] + C[i]) * (F[i] + C[i] + q[i])))
            elif F[i] + C[i] == 0:
                if q[i] > 0:
                    delta += w[i]
        return delta
    
    
    '''
    Wenn die Distanz von i zu j größer als r0 ist für die ersten 10 facilites, aber kleiner als r0 für die zweiten 10 facilities, dann ist j in der pre_list enthalten
    Die Pre-List speichert nämlich die Indizes der Facilites die bereits 100 Prozent Market share haben (kompletten Demand abgreifen)
    
    def pre_analyisis(self, n, d_matrix):
        pre_list = []
        for i in range (n):
            for j in range (10):
                if d_matrix[i,j] < self.r0:
                    for k in range(10,19):
                        if d_matrix[i,k] > self.r0:
                            pre_list.append(i)
        return pre_list
    '''
    
    # e[i] is the market share added when demand point i is covered by a single additional chain facility
    '''Hier ist die Pre-Analysis schon mit drin, da wir nur die e[i] für die Demand Points berechnen, die noch nicht 100 Prozent Market Share haben'''
    def calc_e(self, n, F, C, w):
        e = np.zeros(n)
        for i in range(n):
            if F[i] + C[i] > 0:
                e[i] = (w[i] * C[i]) / ((F[i] + C[i]) * (F[i] + C[i] + 1))
            elif F[i] + C[i] == 0:
                e[i] = w[i]
        return e
   
    def save_tensor_as_csv(self, tensor, filename):
    # Erstellen eines DataFrame aus dem Tensor
        df = pd.DataFrame(tensor.numpy())
    # Speichern des DataFrame als CSV-Datei
        df.to_csv(filename, index=False)

    '''
    def calc_upper_bound(self, n, p, H, b_matrix, e):
        h = np.ceil(self.B / H)
        U = torch.zeros((H, p))  # U[h,k] is the upper bound for a remaining budget of h(B/H) available for improving facilities k,...,p
        V = torch.zeros((H, p))  # V[h,k] is the additional market share that can be obtained by using a budget h to improve facility k


        for k in range(p):
            for i in range(n):
                if b_matrix[i, k] > 0:
                    h = int(np.ceil(b_matrix[i, k]))
                    for g in range (h, H):
                        V[g, k] += e[i]
        #self.save_tensor_as_csv(V, "/Users/marcelpflugfelder/Documents/02_Studium/Master/Semester 4/07_Seminar/V.csv")
        #print("V saved")

        # Step 2: Calculation of U
        for y in range (H):
            U[y, p-1] = V[y, p-1]
            
        for k in range (p-2, -1, -1):
            print (k)
            for h in range (H-1, -1, -1):
                U[h, k] = max(V[s, k] + U[h-s, k+1] for s in range(h+1))
                
            
        print (U)
        self.save_tensor_as_csv(U, "/Users/marcelpflugfelder/Documents/02_Studium/Master/Semester 4/07_Seminar/U.csv")
        #print("U saved")
        return U
    '''
    def calc_upper_bound(self, filepath):
        file = pd.read_csv(filepath, header = None, sep=",",decimal=".", dtype=float, skiprows=1)
        U_matrix = file.values
        
        return U_matrix
    
    
    def get_index_list(self, b_index, x, y):
        indices = []
        for j in range (0,y+1):
            if isinstance(b_index[x][j], Iterable):
                for val in b_index[x][j]:
                    indices.append(val)
        return indices
    
    # Check if Node is to fathom
    def BandB_2(self, n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, epsilon, U, h, F, C, w, fathomed_dict, current_best, delta_k, iterator, indices, q):
        print("BandB_2")
        print ("B_B2_h: ", h)
        print ("B_B2_k: ", k)
        print ("B_B2_delta_k: ",  delta_k)
        print ("B_B2_D_Star: ", D_Star)
        print ("B_B2_U[h-1, k]: ", U[h-1, k])
        if delta_k + U[h-1 , k] <= D_Star + epsilon:
            print ("Fathomed")
            self.BandB_4(n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, epsilon, U, h, F, C, w, fathomed_dict, current_best, delta_k, iterator, indices, q)
        else:
            print("Continue with 2 -> 3")
            self.BandB_3(n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, epsilon, U, h, F, C, w, fathomed_dict, current_best, delta_k, iterator, indices, q)
    
    def BandB_3(self, n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, epsilon, U, h, F, C, w, fathomed_dict, current_best, delta_k, iterator, indices, q):
        #if iterator == 0:
        print("\n", "BandB_3")
        k += 1
        print ("B_B3_k: ", k)
        if k == p-1:
            #iterator += 1
            #Find out which is the maximum possible invest for facility p which is smaller than the budget
            y = 0
            invest_p = True
            while invest_p == True:
                y += 1
                a = B_dict[p-1][y]
                if a <= (self.B - B_zero):
                    invest_p = True
                    invest_help = True
                else:
                    invest_p = False
                    y = y - 1
                
            if invest_help == True:
                B_zero = B_zero + B_dict[p-1][y]
                print ("B_B3_B_zero: ", B_zero, "\n")
                print ("Budget für p: ", B_dict[p-1][y], "\n")
                print ("Nächste Stufe :", B_dict[p-1][y+1], "\n")
                
                    #Get the indices of the values in b_matrix that are equal to the values in B_dict[p-1] from 0 to y
                indices_new = self.get_index_list(b_index, p-1, y)
                for i in indices_new:
                #x = int(indices_new[i])
                    indices.append(i)
                print ("Indices: ", indices, "\n")
                
                # Calculate of additional chain facilities attracting demand point i - > q[i]
                for i in indices:
                    q[i] += 1
                t[p-1] = y
                # Calculate the market share for each demand point i
                delta_k_2 = self.calc_new_marektshare(q, w, F, C, n)
                delta_k = delta_k + delta_k_2
                print (delta_k)
                if delta_k > D_Star:
                    D_Star = delta_k
                    current_best = t
                    print(D_Star)
                    print("New best solution: ", current_best)
                    k = p - 2
                    self.BandB_4(n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, epsilon, U, h, F, C, w, fathomed_dict, current_best, delta_k, iterator, indices, q)
                
            elif k < p:
                t[k] = 0
                self.BandB_2(n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, epsilon, U, h, F, C, w, fathomed_dict, current_best, delta_k, iterator, indices, q)
        else:
            self.BandB_4(n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, epsilon, U, h, F, C, w, fathomed_dict, current_best, delta_k, iterator, indices, q)   
    
    def BandB_4(self, n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, epsilon, U, h, F, C, w, fathomed_dict, current_best, delta_k, iterator, indices, q):
        print("\n","BandB_4")
        if t[k] == float('nan'):
            t[k] = t[k] + 2
            a = 1
            b = 0
        else:
            t[k] = t[k] + 1
            a = 0
            b = 1
        print ("t[" ,k, "]: ", t[k])
        print (B_dict[k][t[k]])
        B_zero = B_zero + B_dict[k][t[k]] - B_dict[k][t[k] - 1]
        print ("B_B4_B_zero: ", B_zero)
        if B_zero > self.B:
            print("Ich bin bei B_zero > B, setze t[k] = t[k] - 1")
            B_zero = B_zero - B_dict[k][t[k]] + B_dict[k][t[k] - 1]
            print ("B_B4_B_zero nach zurücksetzen: ", B_zero)
            if a == 1:
                t[k] = t[k] - 2
            elif b == 1:
                t[k] = t[k] - 1
            k = k - 1
            if k > 0:
                self.BandB_4(n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, epsilon, U, h, F, C, w, fathomed_dict, current_best, delta_k, iterator, indices, q)
            elif k == 0:
                current_best = t
                print("Optimale Lösung gefunden\n")
                #print("Optimales Delta: ", D_Star)
                self.solution(D_Star, current_best, B_dict)
                
        elif B_zero <= self.B:
            H = self.B
            print("Ich bin bei B_zero <= B")
            h = int(np.ceil(H * ((self.B - B_zero) / self.B)))
            print ("B_B4_h: ", h)
            print ("Indices: ", indices)
            #print (B_dict[k])
            #print (b_index[k])
            print ("q: ", q)
            print ("F: ", F)
            # Update indices
            indices_update = self.get_index_list(b_index, k, t[k])
            print ("Indices_update: ", indices_update)
            for i in indices_update:
                q[i] += 1
                indices.append(i)
            delta_k_2= self.calc_new_marektshare(q, w, F, C, n)
            delta_k = delta_k + delta_k_2
            if delta_k > D_Star:
                D_Star = delta_k
                current_best = t
            print ("Indices_new: ", indices)
            for i in range (n):
                F[i] = F[i] + q[i]
            print ("in before: ", indices)
            indices = []
            print ("in after: ", indices)
            print ("t: ", t)
            self.BandB_2(n, p - 1, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, epsilon, U, h, F, C, w, fathomed_dict, current_best, delta_k, iterator, indices, q)
        
    def solution(self, D_Star, current_best, B_dict):
        print("solution")
        print("Optimale Lösung: ", D_Star)
        '''print("Es werden folgende Facilites gebaut: ")
        for j in range(len(current_best)):
            if current_best[j] > 0:
                print("Facility ", j, " wird gebaut mit dem Budget ", B_dict[j][current_best[j]])'''
        
    
    def Branch_and_Bound(self, n, p, b_matrix, B_dict, b_index, D_Star, epsilon, U, H, F, C, w, fathomed_dict, current_best):
        # Step 1: Initialization
        t = np.zeros(p, dtype = int)    # t is the vector of decisions  t[0] = 0 means that B_dict[f'B_{0}'][0] is chosen
        k = 0                           # k is the index of the current facility
        #t[k] = 0                        # t[k] is the index of the current value of the k-th facility
        B_zero = 0                      # B_zero is the used budget so far
        delta_k = 0                     # delta_k is the market share so far
        iterator = 0
        indices = []
        q = np.zeros(n, dtype=int)
        h = int(np.ceil(H * ((self.B - B_zero) / self.B)))
        print("LetsGO!")
        self.BandB_2(n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, epsilon, U, h, F, C, w, fathomed_dict, current_best, delta_k, iterator, indices, q)
        
            
    
    def run(self, filepath):
        n = self.get_n(filepath)
        p = n
        
        # Initialize all needed matrices and vectors
        s = self.create_S(p)                                #s is the vector of fixed costs
        r = self.create_r(p)                                #r is the vector of radii
        w = np.zeros(n)                                     #w is the vector of buying power of demand point i
        for i in range(n):
            w[i] = 1/(i+1)
       
        init_matrix = self.read_file(filepath, n)
        d_matrix = self.floyd_warshall(init_matrix, n)      #d_matrix is the distance matrix
        
        b_matrix = self.calc_b(n, p, s, r, d_matrix)        #b_matrix is the cost matrix (how much budjet need to be spend to improve a facility j to include demand point i)
        B_dict = self.create_B_dict(b_matrix, n)            #B_dict is a dictionary that includes the possible (unique) values for each column of b_matrix in a sorted manner
        b_index = self.create_index_dict(b_matrix)          #b_index is a dictionary that includes the indices of the values in b_matrix that are equal to the values in B_dict
        '''
        print ("d_",99,": ",d_matrix[99])
        print ("b_(spalte)",99,": ",b_matrix[:,99])
        print ("B_dict_",99,": ",B_dict[99])
        print ("B_dict 99,4: ", B_dict[99][4].item())
        print("B_index: ", b_index[99])                               #b_index beinhaltet weitere arrays und nicht direct integers.
        print("B_index: 99,19 ", b_index[99][19][0])
           ''' 
        
        F = np.zeros(n, dtype=int)
        C = np.zeros(n, dtype=int)
        for i in range (n):
            for j in range (20):
                if j < 10:
                    if d_matrix[i,j] < self.r0:
                        F[i] += 1                   #if the distance from i to j is smaller than r0, then the demand point i is covered by own facility j -> F[i] += 1
                else:
                    if d_matrix[i,j] < self.r0:
                        C[i] += 1                   #if the distance from i to j is smaller than r0, then the demand point i is covered by competitor facility j -> C[i] += 1
        
        
        
        
        e = self.calc_e(n, F, C, w)
   
        H = self.B              # Since all distances are integers, the cost for improving a facility is integer thus we set H = B
        epsilon = 0.00001       # the tolerance eplsilon is set to 0.00001 accoring to the paper
        
        #Step 0 Branch and Bound
        fathomed_dict= {}       #the fathomed dict dictionary will include combinations that won't lead to a better solution
        current_best = []       #the current best list will include the best solution so far
        D_Star = 0              #D_Star is the best solution so far 'measured in' extra market share
        #U = self.calc_upper_bound(n, p, self.B, b_matrix, e)
        U = self.calc_upper_bound("/Users/marcelpflugfelder/Documents/02_Studium/Master/Semester 4/07_Seminar/U.csv")
        #print (U)
        #print("starte B and B")
        self.Branch_and_Bound(n, p, b_matrix, B_dict, b_index, D_Star, epsilon, U, H, F, C, w, fathomed_dict, current_best)
        '''
        print (d_matrix[4])
        print (b_matrix[:,4])
        print (B_dict[4])
        print ("b_ind_4",b_index[4], "\n")
        print ("b_ind_11" ,b_index[11], "\n")
        print ("b_ind_99" ,b_index[99], "\n")
        '''
        
        
    
        
       


SCLP(20, 5000, 0, 500).run(filepath= "/Users/marcelpflugfelder/Documents/02_Studium/Master/Semester 4/07_Seminar/files/pmed1.csv")