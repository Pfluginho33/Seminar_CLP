import pandas as pd
import numpy as np


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
        c = np.full((n, n), np.inf)
        np.fill_diagonal(c, 0)

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
        r = np.zeros(p)
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
        b_matrix = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                if k < 10:
                    if d_matrix[i, k] > r[k]:
                        b_matrix[i, k] = self.calc_f(d_matrix[i, k]) - self.calc_f(r[k]) + s[k]
                    else:
                        b_matrix[i, k] = 0
                else:
                    b_matrix[i, k] = self.calc_f(d_matrix[i, k]) - self.calc_f(r[k]) + s[k]
        return b_matrix
    
    # Remove tied values
    def create_B_dict(self, b_matrix, p):            
        B_dict = {}
        for k in range(p):
            unique_values = np.unique(b_matrix[:, k])
            if 0 not in unique_values:
                unique_values = np.append(0)
            sorted_values = np.sort(unique_values)
            B_dict[f'B_{k}'] = sorted_values
        return B_dict
    
    def get_i(b_matrix, B_dict, x, k):
        # Überprüfen, ob k in B_dict und in den Grenzen von b_matrix liegt
        if f'B_{k}' not in B_dict or k >= b_matrix.shape[1]:
            return "Ungültiger Spaltenindex"

        # Überprüfen, ob x im entsprechenden Array in B_dict enthalten ist
        if x not in B_dict[f'B_{k}']:
            return "Wert nicht im B_dict gefunden"

        # Suche nach dem Wert x in der k-ten Spalte der b_matrix
        for i in range(b_matrix.shape[0]):
            if b_matrix[i, k] == x:
                return i

        return "Wert nicht in der b_matrix gefunden"

    
    #Preliminary Analysis 
    '''
    Wenn die Distanz von i zu j größer als r0 ist für die ersten 10 facilites, aber kleiner als r0 für die zweiten 10 facilities, dann ist j in der pre_list enthalten
    Die Pre-List speichert nämlich die Indizes der Facilites die bereits 100 Prozent Market share haben (kompletten Demand abgreifen)
    '''
    def pre_analyisis(self, n, d_matrix):
        pre_list = []
        for i in range (n):
            for j in range (10):
                if d_matrix[i,j] < self.r0:
                    for k in range(10,19):
                        if d_matrix[i,k] > self.r0:
                            pre_list.append(i)
        return pre_list
    
    
    # e[i] is the market share added when demand point i is covered by a single additional chain facility
    '''Hier ist die Pre-Analysis schon mit drin, da wir nur die e[i] für die Demand Points berechnen, die noch nicht 100 Prozent Market Share haben'''
    def calc_e(self, n, F, C, w):
        e = np.zeros(n)
        for i in range(n):
            if F[i] + C[i] > 0:
                e[i] = (w[i] * C[i]) / ((F[i] + C[i]) * (F[i] + C[i]))
            elif F[i] + C[i] == 0:
                e[i] = w[i]
        return e
    
    '''Branch and Bound Algorithmus'''
    
    def calc_upper_bound(self, n, p, H, b_matrix, e):
        h = self.B / H
        U = np.zeros((H, p))  # U[h,k] is the upper bound for a remaining budget of h(B/H) available for improving facilities k,...,p
        V = np.zeros((H, p))  # V[h,k] is the additional market share that can be obtained by using a budget h to improve facility k

        # Step 1: Calculation of V
        for k in range(p):
            #for h in range(H):
                #V[h, k] = 0
            for i in range(n):
                if b_matrix[i, k] > 0:
                    if h >= b_matrix[i, k]:
                        V[:, k] += e[i]

        # Step 2: Calculation of U
        U[:, p - 1] = V[:, p - 1]
        for k in range((p - 2), 0, -1):
            for h in range(H):
                U[h, k] = max([V[s, k] + U[h - s, k + 1] for s in range(h)])

        return U
    
    def Branch_and_Bound(self, k, p, t, B_zero, b_matrix, B_dict, delta, D_Star, epsilon, U, h, e):
        
        fathomed_nodes = []  # Initialize the fathomed_nodes list

        # Check if Node is to fathom
        def BandB_2():
            if delta[k] + U[h, (k + 1)] <= D_Star + epsilon:
                print("Fathomed")  
                fathomed_nodes.append(k)
                BandB_4()
            else:
                BandB_3()
        
        def BandB_3():
            k += 1
            if k == p:
                # calc market share
                b_hilf = sum(b_matrix[t[j], j] for j in range(p))
                B_zero = self.B - b_hilf
                #Calculate extra marketshare
                #check e[t[j]] for j in range(p)
                #update D_Star if necessary
                k = p - 1
                BandB_4()
            elif k < p:
                t[k] = 1
                BandB_2()
            
        def BandB_4():
            t[k] = t[k] + 1
            B_zero = B_zero + b_matrix[t[k], k] - b_matrix[t[k] - 1, k]
    
            if B_zero > self.B:
                k = k - 1
                if k > 0:
                    BandB_4()
                elif k == 0:
                    print("Optimale Lösung gefunden\n")
                    print("Optimales Delta: ", D_Star)
            elif B_zero <= self.B:
                h = np.ceil(self.H * ((self.B - B_zero) / self.B))
                # Aktualisierung von delta[k]
                # Hier sollten Sie die spezifische Logik einfügen, wie delta[k] basierend auf den aktuellen Entscheidungen berechnet wird.
                # Zum Beispiel könnte delta[k] basierend auf der Veränderung des Marktanteils aktualisiert werden.
                # delta[k] = ...

                # Aktualisierung von D_Star, falls notwendig
                if delta[k] > D_Star:
                    D_Star = delta[k]
                    print("Neue beste Lösung: ", D_Star)
                BandB_2()

            
    
    def run(self, filepath):
        n_0 = self.get_n(filepath)
        n = [None] * n_0
        p = [None] * n_0
        for i in range(n_0):
            n[i] = i
            p[i] = i
        
        s = self.create_S(p)
        r = self.create_r(p)
        w = np.zeros(n_0)
        for i in range(n_0):
            w[i] = 1/(i+1)
            
        init_matrix = self.read_file(filepath, n_0)
        d_matrix = self.floyd_warshall(init_matrix, n_0)
        
        # Hier arbeiten mit der pre_analysis -> n und d verkleinern und dann mit den neuen Matrizen arbeiten -> Laufzeit verbessern?
        pre_list = self.pre_analyisis(len(n), d_matrix)
        n = [x for x in n if x not in pre_list]
        # Überlegen was mit d passiert
        #d_matrix = np.delete(d_matrix, pre_list, 0)
        b_matrix = self.calc_b(n, p, s, r, d_matrix)
        B_dict = self.create_B_dict(b_matrix, n_0)
        F = np.zeros(n_0)
        C = np.zeros(n_0)
        for i in range (n_0):
            for j in range (20):
                if j < 10:
                    if d_matrix[i,j] < self.r0:
                        F[i] += 1
                else:
                    if d_matrix[i,j] < self.r0:
                        C[i] += 1
            
        e = self.calc_e(n, F, C, w)
        
        H = self.B  # Since all distances are integers, the cost for improving a facility is integer thus we set H = B
        
        epsilon = 0.00001
        
        #First Step Branch and Bound
        t = np.zeros(len((p)))
        B_zero = 0
        delta = np.zeros(len((p)))
        D_Star = 0
        h = np.ceil(H * ((self.B - B_zero) / self.B))
        
        U = self.calc_upper_bound(self, n, p, H, b_matrix, e)
        k = 1
        t[k] = 1
       
        
        #Branch and Bound Steps mit Querverlinkung
        #If delta[k] + U[h, (k + 1)] <= D_Star + epsilon, the rest of the tree from this node is fathomed. Go to BandB_4
        self.Branch_and_Bound(self, k, p, t, B_zero, b_matrix, B_dict, delta, D_Star, epsilon, U, h, e)
        
        
        
        



SCLP(20,5000,500,0).run(filepath= "/Users/marcelpflugfelder/Documents/02_Studium/Master/Semester 4/07_Seminar/files/pmed1.csv")