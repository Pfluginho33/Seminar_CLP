from collections.abc import Iterable
import time
import os
import psutil
import csv
import math

class SCLP:
    def __init__(self, r0, B, S_existing, S_new):
        self.r0 = r0                    # coverage Radius
        self.B = B                      # budget
        self.S_existing = S_existing    # fixed costs of upgrading existing facilities
        self.S_new = S_new              # fixed costs of building new facilities
        self.nodes = 0                  # number of nodes visited
        self.H=B
        
    def get_n(self, filename):
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                n = int(row[0])
                break
        return n

    def read_file(self, filename, n):
        c = [[float('inf') for _ in range(n)] for _ in range(n)]
        for i in range(n):
            c[i][i] = 0

        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the first line
            for row in reader:
                i, j, k = int(row[0]), int(row[1]), int(row[2])
                c[i-1][j-1] = k
                c[j-1][i-1] = k

        return c

    def floyd_warshall(self, c, n):
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    c[i][j] = min(c[i][j], c[i][k] + c[k][j])
        return c

    def create_r(self, p):
        r = [0] * p
        for i in range(10):
            r[i] = self.r0
        return r

    def create_S(self, p):
        s = [0] * p
        for j in range(p):
            if j < 10:
                s[j] = self.S_existing
            else:
                s[j] = self.S_new
        return s

    def calc_f(self, r):
        return [x**2 for x in r]

    def calc_b(self, n, p, s, r, d_matrix):
        b_matrix = [[0 for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                if k < 10:
                    if d_matrix[i][k] > r[k]:
                        b_matrix[i][k] = int(self.calc_f([d_matrix[i][k]])[0] - self.calc_f([r[k]])[0] + s[k])
                    else:
                        b_matrix[i][k] = 0
                else:
                    b_matrix[i][k] = int(self.calc_f([d_matrix[i][k]])[0] - self.calc_f([r[k]])[0] + s[k])
        return b_matrix

    def create_B_dict(self, b_matrix, p):
        B_dict = {}
        for k in range(p):
            sorted_values = sorted(set(b_matrix[i][k] for i in range(len(b_matrix))))
            if 0 not in sorted_values:
                sorted_values = [0] + sorted_values
            B_dict[k] = sorted_values
        return B_dict

    def create_index_dict(self, b_matrix):
        sort_index_dict = {}
        for k in range(len(b_matrix[0])):
            sorted_indices = sorted(range(len(b_matrix)), key=lambda i: b_matrix[i][k])
            grouped_indices = self.group_tied_values([b_matrix[i][k] for i in sorted_indices], sorted_indices)
            sort_index_dict[k] = [float('nan')] + grouped_indices
        return sort_index_dict

    def group_tied_values(self, column, sorted_indices):
        grouped_indices = [[sorted_indices[0]]]
        for i in range(1, len(sorted_indices)):
            if column[i] == column[i - 1]:
                grouped_indices[-1].append(sorted_indices[i])
            else:
                grouped_indices.append([sorted_indices[i]])
        return grouped_indices
    
    def calc_new_marektshare (self, q, w, F, C, k):
        delta = 0
        for i in range (k):
            if F[i] + C[i] > 0:
                delta += w[i] * ((q[i] * C[i]) / ((F[i] + C[i]) * (F[i] + C[i] + q[i])))
            elif F[i] + C[i] == 0:
                if q[i] > 0:
                    delta += w[i]
        return delta
    
    def calc_deltak(self, t, k , w, F, C, n, b_index):
        q_help = [0 for _ in range(n)]
        ind_help = []
        for j in range (k+1):
            ind_new = self.get_index_list(b_index, j, t[j])
            if t[j] > 0:
                if isinstance(ind_new, Iterable):
                    for val in ind_new:
                        ind_help.append(val)
        for i in ind_help:
            q_help[i] += 1
        delta = self.calc_new_marektshare(q_help, w, F, C, n)
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
        e = [0 for _ in range(n)]
        for i in range(n):
            if F[i] + C[i] > 0:
                e[i] = (w[i] * C[i]) / ((F[i] + C[i]) * (F[i] + C[i] + 1))
            elif F[i] + C[i] == 0:
                e[i] = w[i]
        return e
        
        
    '''def calc_upper_bound(self, n, p, H, b_matrix, e):
        U = [[0 for _ in range(p)] for _ in range(H+1)]
    

        for k in range(p):
            for i in range(n):
                if b_matrix[i][k]> 0:
                    h = int(math.ceil(b_matrix[i][k]))
                    for g in range (h, H+1):
                        U[g][k] += e[i]
        print ("Done")
        print(len(U))

            
        for k in range (p-2, -1, -1):
            print (k)
            for h in range (H, -1, -1):
                U[h][k] = max(U[s][k] + U[h-s][k+1] for s in range(h+1))
        with open('/Users/marcelpflugfelder/Documents/02_Studium/Master/Semester 4/07_Seminar/U-Tabellen/U-8.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(U)
        
        
        return U
    
    '''
    def calc_upper_bound(self, filepath):
        U_matrix = []
        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                U_matrix.append([float(item) for item in row])
        return U_matrix
    
    
    def get_index_list(self, b_index, x, y):
        indices = []
        for j in range (0,y+1):
            if isinstance(b_index[x][j], Iterable):
                for val in b_index[x][j]:
                    indices.append(val)
        return indices
    
    # Check if Node is to fathom
    def BandB_2(self, n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star,  U, h, F, C, w, current_best, delta_k, indices, q):
        print("BandB_2 mit k: ", k)
        print ("delta_k + U[h][k+1]: ", delta_k + U[h][k+1])
        if delta_k + U[h][k+1] <= D_Star + 0.00001 :
            #print ("Fathomed")
            self.BandB_4(n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, U, h, F, C, w, current_best, delta_k, indices, q)
        else:
            self.BandB_3(n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, U, h, F, C, w, current_best, delta_k, indices, q)
    
    def BandB_3(self, n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, U, h, F, C, w, current_best, delta_k, indices, q):
        print("BandB_3", "k: ", k, "h: ", h, "B_zero: ", B_zero, "D_Star: ", D_Star, "delta_k: ", delta_k)
        self.nodes += 1
        '''Set k = k + 1'''
        k = k + 1
        if k == p - 1:
            '''If k = p - 1, calculate the extra market share by using B_B0 to expand facility p'''
            B_zero = sum(B_dict[k][t[k]] for k in range(p))   #Calculate the used budget so far
            b = self.B - B_zero
            print ("Für p zur Verfügung: ", b)
            x = 0
            for y in range(len(B_dict[k])):
                if B_dict[k][y] <= b:
                    x = y
                elif B_dict[k][y] > b:
                    break
            t[k] = x
            B_zero = B_zero + B_dict[k][t[k]]
            print("That will leave us with a B_zero: ", B_zero)
            new_share = self.calc_deltak(t, k, w, F, C, n, b_index)
            print("new_share: ", new_share)
            print("D_Star: ", D_Star)
            
            '''Update D_star if necessary'''
            
            if new_share > D_Star:
                print("We would improve the solution by using the rest budget on facality p")
                D_Star = new_share
                current_best = t.copy()
                
            '''Set k=p-1 and go to Step 4'''
            k = p - 2
            self.BandB_4(n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, U, h, F, C, w, current_best, new_share, indices, q)
        elif k < p - 1:
            '''If k < p, set t(k) = 0'''
            t[k] = 0
            self.BandB_2(n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, U, h, F, C, w, current_best, delta_k, indices, q)
    
    def BandB_4(self, n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, U, h, F, C, w, current_best, delta_k, indices, q):
        print("BandB_4", "k: ", k, "h: ", h, "B_zero: ", B_zero, "D_Star: ", D_Star, "delta_k: ", delta_k)
        self.nodes += 1
        '''Set t(k) = t(k) + 1. B_zero is changed to B_zero + B_dict[k][t[k]] - B_dict[k][t[k]-1]'''
        t[k] = t[k] + 1
        B_zero = B_zero + B_dict[k][t[k]] - B_dict[k][t[k]-1]
        
        if B_zero > self.B:
            print("Zurück")
            B_zero = B_zero - B_dict[k][t[k]] + B_dict[k][t[k]-1]
            t[k] = t[k] - 1
            k = k - 1
            if k >= 0:
                self.BandB_4(n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, U, h, F, C, w, current_best, delta_k, indices, q)
            elif k == -1:
                self.solution(D_Star, current_best, B_dict, b_index, q, F, w, C, n, delta_k, t)
        
        elif B_zero <= self.B:
            print(t)
            h = math.ceil(self.H * ((self.B - B_zero) / self.B))
            print ("B_zero: ", B_zero)
            delta_k = self.calc_deltak(t, k, w, F, C, n, b_index)
            if delta_k > D_Star:
                D_Star = delta_k
                print ("current_best: ", current_best)
            print (delta_k)
            self.BandB_2(n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, U, h, F, C, w, current_best, delta_k, indices, q)
                               
        
    def solution(self, D_Star, current_best, B_dict, b_index, q, F, w, C, n, delta_k, t):
        print("solution")
        indices = []
        for j in range (len(t)):
            indices = self.get_index_list(b_index, j, t[j])
            for i in indices:
                q[i] = 1
        delta = self.calc_new_marektshare(q, w, F, C, n)
        print(delta)
        print(self.nodes)
        print("Optimale Lösung: ", delta)
        print("Es werden folgende Facilites gebaut: ")
        for j in range(len(t)):
            if t[j] > 0:
                print("Facility ", j, " wird gebaut mit dem Budget ", B_dict[j][t[j]])
        
    
    def Branch_and_Bound(self, n, p, b_matrix, B_dict, b_index, D_Star, U, F, C, w, current_best):
        # Step 1: Initialization
        #print ("Branch_and_Bound" , p)
        t = [0 for _ in range(p)]           # t is the vector of decisions  t[0] = 0 means that B_dict[f'B_{0}'][0] is chosen
        k = 0                               # k is the index of the current facility
        B_zero = 0                          # B_zero is the used budget so far
        delta_k = 0                         # delta_k is the market share so far
        indices = []
        q = [0 for _ in range(n)]
        h = self.H
        print("LetsGO!")
        self.BandB_2(n, p, t, k, B_zero, b_matrix, B_dict, b_index, D_Star, U, h, F, C, w, current_best, delta_k, indices, q)
        
            
    
    def run(self, filepath):
        start_time = time.time()
        n = self.get_n(filepath)
        p = n
        
        # Initialize all needed matrices and vectors
        s = self.create_S(p)                                #s is the vector of fixed costs
        r = self.create_r(p)                                #r is the vector of radii
        w = [0 for _ in range(n)]                           #w is the vector of buying power of demand point i
        for i in range(n):
            w[i] = 1/(i+1)
       
        init_matrix = self.read_file(filepath, n)
        d_matrix = self.floyd_warshall(init_matrix, n)      #d_matrix is the distance matrix
        
        b_matrix = self.calc_b(n, p, s, r, d_matrix)        #b_matrix is the cost matrix (how much budjet need to be spend to improve a facility j to include demand point i)
        B_dict = self.create_B_dict(b_matrix, n)            #B_dict is a dictionary that includes the possible (unique) values for each column of b_matrix in a sorted manner
        b_index = self.create_index_dict(b_matrix)          #b_index is a dictionary that includes the indices of the values in b_matrix that are equal to the values in B_dict
        
        F = [0 for _ in range(n)] 
        C = [0 for _ in range(n)] 
        for i in range (n):
            for j in range (20):
                if j < 10:
                    if d_matrix[i][j] < self.r0:
                        F[i] += 1                   #if the distance from i to j is smaller than r0, then the demand point i is covered by own facility j -> F[i] += 1
                else:
                    if d_matrix[i][j] < self.r0:
                        C[i] += 1                   #if the distance from i to j is smaller than r0, then the demand point i is covered by competitor facility j -> C[i] += 1
        
        e = self.calc_e(n, F, C, w)
   
        #Step 0 Branch and Bound
        current_best = [0 for _ in range(n)]       #the current best list will include the best solution so far
        D_Star = 0              #D_Star is the best solution so far 'measured in' extra market share
        #U = self.calc_upper_bound(n, p, self.B, b_matrix, e)
        U = self.calc_upper_bound("/Users/marcelpflugfelder/Documents/02_Studium/Master/Semester 4/07_Seminar/U-Tabellen/U-1.csv")
        self.Branch_and_Bound(n, p, b_matrix, B_dict, b_index, D_Star, U, F, C, w, current_best)
        end_time = time.time()
        execution_time = end_time - start_time
        
        pid = os.getpid()
        current_process = psutil.Process(pid)
        cpu_usage = current_process.cpu_percent(interval=1)
        memory_usage = current_process.memory_info().rss / (1024 * 1024)  # in Megabytes

        print(f"Ausführungszeit: {execution_time} Sekunden")
        print(f"CPU-Nutzung: {cpu_usage}%")
        print(f"Speichernutzung: {memory_usage} MB")

SCLP(20, 5000, 0, 500).run(filepath= "/Users/marcelpflugfelder/Documents/02_Studium/Master/Semester 4/07_Seminar/files/pmed1.csv")