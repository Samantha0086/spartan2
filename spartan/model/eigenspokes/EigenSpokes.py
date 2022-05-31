import numpy as np
import scipy.sparse.linalg as slin
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix
from scipy.sparse import coo_matrix




class EigenSpokes(_model.DMmodel):
    def __init__(self, data_mat):
        self.data = data_mat

    def run(self, k = 3, is_directed = False): 
        
     
        sparse_matrix = self.data.to_scipy()
        sparse_matrix = sparse_matrix.asfptype()
        # eigenvalue and eigenvectors 
        
        # RU: Unitary matrix having left singular vectors as columns.
        # RS: singular value
        # RVt: Unitary matrix having right singular vectors as rows.
        RU, RS, RVt = slin.svds(sparse_matrix, k) # RU: m*k 
        RV = np.transpose(RVt) 
        U, S, V = np.flip(RU, axis=1), np.flip(RS), np.flip(RV, axis=1) 

        m = U.shape[0] 
        n = V.shape[0] 

        # adjacency matrix
        aj_matrix = sparse_matrix * np.transpose(sparse_matrix)

        
        for v_index in range(U.shape[1]):
            # Initialization (initialize output set using seed) 

            
            
            x = U[:,v_index] * -1 if np.abs(np.min(U[:,v_index])) > np.abs(np.max(U[:,v_index])) else U[:,v_index]
            #y = V[:,v_index] * -1 if np.abs(np.min(V[v_index])) > np.abs(np.max(V[v_index])) else V[k]
            
            node = np.argmax(x)
            
            # Construct graph G
            graph_x = [index for index in range(len(x)) if x[index] > 1 / np.sqrt(m)]
            #y_outliers = [index for index in range(len(y)) if y[index] > 1 / np.sqrt(n)]
            
           
            
            sm = aj_matrix.tocsr()
            
            total_edge = 0
            for i in graph_x:
          
                for j in graph_x:
                    total_edge= total_edge + sm[i,j]
                  

            #aj_matrix = aj_matrix.flip() ???
            total_edge = total_edge/2
            
           
            c2 = graph_x
            c2.remove(node)
            
           
            
            k = [np.sum(r) for r in sm]
                
            s = np.full((m), 0) 
            s[graph_x] = -1
            s[node] = 1
            s = np.transpose(s)     
            # s[something] = 1
            
            bx = sm*s - k* (np.dot(np.transpose(k),s))/(2*m)
            modularity_new = np.sum((1/(4*m))*np.transpose(s) * bx)
            
            modularity = -np.inf
            output=[]
            while (modularity_new > modularity):
                modularity = modularity_new
                output.append(node)
                # Expand output set
                # see discovery
                
                max = -np.inf
                node = 0
                Nc=[]
                
                for i in output:
                    Nc=[index for index in c2 if sm[i,index]>0]
                    for n in Nc:
                        if x[n]>max :
                            max = x[n]
                            node = n
                
               
                s[node] =1
                
                c2.remove(node)
                bx = sm*s - k* (np.dot(np.transpose(k),s))/(2*m)
                
                modularity_new = np.sum((1/(4*m))*np.transpose(s) * bx)
               
                
                
            # termination 
            
            # calculation of conductance
            nominator =0
            denominator1 =0
            denominator2 =0
            for i in output:
                for j in c2:
                    nominator = nominator +sm[i,j]
                denominator1 = denominator1+ k[i]
            for i in graph_x:
                denominator2 = denominator2+ k[i]
            conductance = nominator / min(denominator1, denominator2)
                
        return output
    
    
    def anomaly_detection(self):
        return self.run()

    def save(self, outpath):
        pass

