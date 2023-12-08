import pandas as pd
import numpy as np
from docplex.mp.model import Model
from docplex.mp.relax_linear import LinearRelaxer
from docplex.mp.conflict_refiner import ConflictRefiner, VarUbConstraintWrapper, VarLbConstraintWrapper
from docplex.mp.progress import TextProgressListener


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
    def create_r(self,n):
        r = np.zeros(n)
        a = 20
        for i in range(a):
            r[i] = self.r0
        return r
    
    def create_S(self, n):
        s = np.zeros(n)
        for i in range(n):
            if i < 10:
                s[i] = self.S_existing
            # i bewteen 10 and 20
            #if i >= 10 and i < 20:
            #    s[i] = 0
            else:
                s[i] = self.S_new
        return s
    
    #Kostenfuntkion
    def calc_f(self, r):
        f = r**2
        return f
    
    
    def calc_b(self, n, p, d_matrix):
        #TODO: Sortierung implementieren
        s = self.create_S(n)
        b_matrix = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                if d_matrix[i, k] > self.r0:
                    b_matrix[i, k] = self.calc_f(d_matrix[i, k]) - self.calc_f(self.r0) + s[i]
                else:
                    b_matrix[i, k] = 0
        return b_matrix
    
    
    def build_model(self, filepath):
        # Initialisieren des CPLEX-Modells
        mdl = Model("SCLP")
        
        n = self.get_n(filepath)
        
        #p = n-10   #p are first 10 nodes and last 10 nodes
        #----------------------------------------------------------------
        #TODO:
        #Idee: p als "Set" aufbauen mit ersten 10 und letzten 10 Knoten
        #----------------------------------------------------------------
        
        init_matrix = self.read_file(filepath, n)
        d_matrix = self.floyd_warshall(init_matrix, n)
        b_matrix = self.calc_b(n, p, d_matrix)
        
        # ---Variables---
        mdl.x=mdl.binary_var_matrix(n, p, name="x")
        mdl.y=mdl.binary_var_matrix(n, p, name="y")
        
        
        #Demand
        w = []
        a = abs(n-20)
        for i in range(a):
            x = i + 20
            w[x] = 1/x
        
        #Indicator Set I
        I = {}
        for j in range(p):
            I[j]=[i for i in range(len(d_matrix)) if d_matrix[i, j] > self.r0]
        
        F=[]
        C=[]
        
        for z in range(10):
            F[z] = z
            C[z] = z + 10
        
        
        # ---Objective---
        mdl.maximize(mdl.sum((w[i] * self.C[i] * mdl.sum(mdl.y[i, j] for j in range(p))) /
                        ((self.F[i] + self.C[i]) * (self.F[i] + self.C[i] + mdl.sum(mdl.y[i, j] for j in range(p)))) 
                        for i in range(n)))
        
        # ---Constraints---
        #A.2
        for j in range(p):
            mdl.add_constraint(mdl.sum(mdl.x[i, j] for i in range(n) <= 1))
        
        #A.3
        for i in range(n):
            for j in range(p):
                mdl.add_constraint(d_matrix[i, j] * mdl.y[i, j] <= mdl.sum(d_matrix[k, j] * mdl.x[k, j] for k in enumerate(I[j])))
            
        #A.4
        mdl.add_constraint(mdl.sum(b_matrix[i, j] * mdl.y[i, j] for i in range(self.n) for j in range(self.p)) <= self.B)
        #TODO: define b_matrix function
        
        #A.5
        for i in range(n):
            for j in range(p):
                if I[j] != i:
                    mdl.add_constraint(mdl.y[i, j] == 0)
        
        return mdl
    
    def solve_model(self, mdl):
        mdl.solve()
        mdl.print_solution()
        mdl.report_kpis()
        return mdl  
    
    def get_solution(self, mdl):
        x = np.zeros((self.n, self.p))
        y = np.zeros((self.n, self.p))
        for i in range(self.n):
            for j in range(self.p):
                x[i, j] = mdl.solution.get_value(mdl.x[i, j])
                y[i, j] = mdl.solution.get_value(mdl.y[i, j])
        return x, y
    
    def get_objective(self, mdl):
        return mdl.objective_value
    
    
    def get_kpis(self, mdl):
        return mdl.report_kpis()
    
    
    def run(self):
        mdl = self.build_model(filepath= "/Users/marcelpflugfelder/Documents/02_Studium/Master/Semester 4/07_Seminar/files/pmed1.csv")
        mdl = self.solve_model(mdl)
        x, y = self.get_solution(mdl)
        obj = self.get_objective(mdl)
        kpis = self.get_kpis(mdl)
        print("x: ", x)
        print("y: ", y)
        print("obj: ", obj)
        print("kpis: ", kpis)
        return x, y, obj, kpis
    

SCLP(20,5000,500,0).run()
       
    