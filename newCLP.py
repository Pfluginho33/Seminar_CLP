import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
# Gurobi benötigt keinen direkten Ersatz für LinearRelaxer
# Konfliktlösung in Gurobi erfolgt anders, daher wird dieser Import entfernt
# Gurobi hat eigene Methoden für den Fortschrittslistener


class SCLP:
    def __init__(self,r0, B, S_existing, S_new):
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

        b_sort = np.full((p, n), (self.B * 2))                #Verwernde B * 2 als Dummy Wert; B * 2 > B -> passt zu Budget Constraint
        # Remove tied values
        for k in range(p):
            unique_values = np.unique(b_matrix[:, k])
            b_sort[k, :len(unique_values)] = unique_values

        return b_sort
    
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
                    if d_matrix[i,(j+10)] > self.r0:
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
            for h in range(H):
                V[h, k] = 0
            for i in range(n):
                if b_matrix[i, k] > 0:
                    if h >= b_matrix[i, k]:
                        V[:, k] += e[i]

        # Step 2: Calculation of U
        for k in range((p - 1), 0, -1):
            for h in range(H):
                U[h, k] = max([V[s, k] + U[h - s, k + 1] for s in range(h)])

        return U
        
        
    '''
    def branch_and_bound(self, n, d_matrix, b_matrix, w, F, C, e, U, epsilon):
        p = n
        t = np.zeros(p)
        B_zero = 0
        delta = np.zeros(p)
        D_Star = 0
        H = self.B
        h = H * ((self.B - B_zero) / self.B)
        
        #Step 1 of Branch and Bound Algorithm
        
        U = self.calc_upper_bound(n, p, b_matrix, e)
        k = 0
        t[k] = 1
        delta[k] = 0
        
        #Step 2 of Branch and Bound Algorithm - Check if Node is to fathom
        
       # if delta[k] + U[h,(k+1)] <= D_Star + epsilon:
        
        #Step 3 of Branch and Bound Algorithm
        k += 1
        if k == p:
            # calc market share
            a = 0
        elif k < p:
            t[k] = 1
        
        #Step 4 of Branch and Bound Algorithm
        t[k] = t[k] + 1
        B_zero = B_zero + b_matrix[t[k], k] - b_matrix[t[k] - 1, k]
        if B_zero > self.B:
            k = k - 1
            if k > 0:
                a=0
                # Go to Step 4
            elif k == 0:
                return D_Star
        elif B_zero <= self.B:
            h = H * ((self.B - B_zero) / self.B)
            # The Fi and Dk are updated
            '''
    
    def build_model(self, n, p, s, r, w, F, C, d_matrix, b_matrix):
        # Initialisieren des gurobi - Modells
        mdl = gp.Model("SCLP")
        
        # ---Variables---
        x = mdl.addVars(n, p, vtype=GRB.BINARY, name="x")
        y = mdl.addVars(n, p, vtype=GRB.BINARY, name="y")
        
        #Indicator Set I; Überarbeiten 
        I = [np.nan]*p
        for j in range(p):
            I[j]=[i for i in range(len(d_matrix)) if d_matrix[i, j] > r[j]]
        
        # ---Objective---
        mdl.setObjective(gp.quicksum((w[i] * C[i] * gp.quicksum(mdl.y[i, j] for j in range(p))) /
                        ((F[i] + C[i]) * (F[i] + C[i] + gp.quicksum(mdl.y[i, j] for j in range(p)))) 
                        for i in range(n)), GRB.MAXIMIZE)

        # ---Constraints---
        #A.2
        for j in range(p):
            mdl.addConstr(sum(x[i, j] for i in range(n)) <= 1)
        
        #A.3
        for i in range(n):
            for j in range(p):
                mdl.addConstr(
                    d_matrix[i, j] * y[i, j] <= gp.quicksum(d_matrix[k, j] * x[k, j] for k in range(n)),
                    f"constraint_{i}_{j}")

        #A.4
        mdl.addConstr(gp.quicksum(b_matrix[i, j] * y[i, j] for i in range(n) for j in range(p)) <= self.B)
        #TODO: define b_matrix function
        
        #A.5
        for i in range(n):
            for j in range(p):
                if I[j] != i:
                    mdl.addConstr(y[i, j] == 0)
        
        return mdl
    
    
    def solve_model(self, mdl):
        # Das Modell lösen
        mdl.optimize()

        # Prüfen, ob eine Lösung gefunden wurde
        if mdl.status == GRB.OPTIMAL:
            # Ergebnisse ausgeben
           # for v in mdl.getVars():
            #    print(f"{v.varName} = {v.x}")

            # Objektivwert ausgeben
            print(f"Optimaler Zielfunktionswert: {mdl.objVal}")
        elif mdl.status == GRB.INF_OR_UNBD:
            print("Modell ist unbeschränkt oder unzulässig.")
        elif mdl.status == GRB.INFEASIBLE:
            print("Modell ist unzulässig.")
        elif mdl.status == GRB.UNBOUNDED:
            print("Modell ist unbeschränkt.")
        else:
            print(f"Optimierung wurde mit Status {mdl.status} beendet.")

        return mdl 
    
    
    def get_solution(self, mdl):
        #n = mdl.n
        if mdl.status == GRB.OPTIMAL:
            # Erstellen eines Wörterbuchs für die Lösungswerte
            solution = {}
            for v in mdl.getVars():
                solution[v.varName] = v.x  # Speichern des Wertes der Variablen

            # Ausgabe des Lösungswörterbuchs
            return solution
        else:
            # Keine gültige Lösung gefunden
            return None
    
    
    def get_objective(self, mdl):
        return mdl.objVal
    
    
    def get_kpis(self, mdl):
        return mdl.report_kpis()
    
    
    def run(self, filepath):
        n = self.get_n(filepath)
        p = n
        
        s = self.create_S(p)
        r = self.create_r(p)
        w = np.zeros(n)
        for i in range(n):
            w[i] = 1/(i+1)
            
        init_matrix = self.read_file(filepath, n)
        d_matrix = self.floyd_warshall(init_matrix, n)
        
        # Hier arbeiten mit der pre_analysis -> n und d verkleinern und dann mit den neuen Matrizen arbeiten -> Laufzeit verbessern?
        pre_list = self.pre_analyisis(n, d_matrix)
        n = n - len(pre_list)
        d_matrix = np.delete(d_matrix, pre_list, 0)
        
        b_matrix = self.calc_b(n, p, s, r, d_matrix)
        
        F = np.zeros(n)
        C = np.zeros(n)
        for i in range (n):
            for j in range (20):
                if j < 10:
                    if d_matrix[i,j] < self.r0:
                        F[i] += 1
                else:
                    if d_matrix[i,j] < self.r0:
                        C[i] += 1
            
        e = self.calc_e(n, F, C, w)
        
        H = self.B  # Since all distances are integers, the cost for improving a facility is integer thus we set H = B
        
        mdl = self.build_model(n, p, s, r, w, F, C, d_matrix, b_matrix)
        mdl = self.solve_model(mdl)
        #x, y = self.get_solution(mdl)
        obj = self.get_objective(mdl)
        #kpis = self.get_kpis(mdl)
        #print("x: ", x)
        #print("y: ", y)
        print("obj: ", obj)
        #print("kpis: ", kpis)
        #return x, y, obj, kpis
        return obj



SCLP(20,5000,500,0).run(filepath= "/Users/marcelpflugfelder/Documents/02_Studium/Master/Semester 4/07_Seminar/files/pmed1.csv")