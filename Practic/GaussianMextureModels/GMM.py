# Importarea bibliotecilor
import numpy as np
import pandas as pd

class GMM:
    '''
    Această clasă reprezintă implementarea modelelor de amestec Gaussian (Gaussian Mixture Models),
    inspirată de implementarea scikit-learn.
    '''
    def __init__(self, n_components, max_iter=100, comp_names=None):
        '''
        Această funcție inițializează modelul stabilind următoarele parametri:
        :param n_components: int
            Numărul de clustere în care algoritmul trebuie să împartă setul de date
        :param max_iter: int, default=100
            Numărul de iterații pe care algoritmul le va parcurge pentru a găsi clusterele
        :param comp_names: listă de șiruri de caractere, default=None
            În cazul în care este setat ca o listă de șiruri de caractere, va fi folosit pentru a numi clusterele
        '''
        self.n_components = n_components
        self.max_iter = max_iter
        if comp_names == None:
            self.comp_names = [f"comp{index}" for index in range(self.n_components)]
        else:
            self.comp_names = comp_names
        # Lista pi conține fracțiunea setului de date pentru fiecare cluster
        self.pi = [1/self.n_components for comp in range(self.n_components)]

    def multivariate_normal(self, X, mean_vector, covariance_matrix):
        '''
        Această funcție implementează formula derivată normală multivariată,
        distribuția normală pentru vectori.
        Necesită următorii parametri:
        :param X: 1-d numpy array
            Vectorul linie pentru care dorim să calculăm distribuția
        :param mean_vector: 1-d numpy array
            Vectorul linie care conține media pentru fiecare coloană
        :param covariance_matrix: 2-d numpy array (matrice)
            Matricea 2-d care conține covarianțele pentru caracteristici
        '''
        return (2*np.pi)**(-len(X)/2) * np.linalg.det(covariance_matrix)**(-1/2) * np.exp(-np.dot(np.dot((X-mean_vector).T, np.linalg.inv(covariance_matrix)), (X-mean_vector))/2)

    def fit(self, X):
        '''
        Funcția pentru antrenarea modelului.
        :param X: 2-d numpy array
            Datele trebuie să fie transmise algoritmului ca o matrice 2-d,
            în care coloanele sunt caracteristicile și rândurile sunt mostrele
        '''
        # Împărțirea datelor în subseturi de n_componets
        new_X = np.array_split(X, self.n_components)
        # Calculul inițial al vectorului de medie și a matricei de covarianță
        self.mean_vector = [np.mean(x, axis=0) for x in new_X]
        self.covariance_matrixes = [np.cov(x.T) for x in new_X]
        # Ștergerea matricei new_X deoarece nu o vom mai avea nevoie
        del new_X

        for iteration in range(self.max_iter):
            '''
            -------------------------- PASUL E (E-STEP) --------------------------
            '''
            # Inițializarea matricei r, fiecare rând conține probabilitățile
            # pentru fiecare cluster pentru acest rând
            self.r = np.zeros((len(X), self.n_components))
            # Calculul matricei r
            for n in range(len(X)):
                for k in range(self.n_components):
                    self.r[n][k] = self.pi[k] * self.multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k])
                    self.r[n][k] /= sum([self.pi[j]*self.multivariate_normal(X[n], self.mean_vector[j], self.covariance_matrixes[j]) for j in range(self.n_components)])
            # Calculul lui N
            N = np.sum(self.r, axis=0)

            '''
            -------------------------- PASUL M (M-STEP) --------------------------
            '''
            # Inițializarea vectorului mediu ca un vector zero
            self.mean_vector = np.zeros((self.n_components, len(X[0])))
            # Actualizarea vectorului mediu
            for k in range(self.n_components):
                for n in range(len(X)):
                    self.mean_vector[k] += self.r[n][k] * X[n]
                self.mean_vector[k] = [1/N[k] * self.mean_vector[k] for k in range(self.n_components)]
            # Inițializarea listei de matrice de covarianță
            self.covariance_matrixes = [np.zeros((len(X[0]), len(X[0]))) for k in range(self.n_components)]
            # Actualizarea matricelor de covarianță
            for k in range(self.n_components):
                self.covariance_matrixes[k] = np.cov(X.T, aweights=(self.r[:, k]), ddof=0)
                self.covariance_matrixes = [1/N[k] * self.covariance_matrixes[k] for k in range(self.n_components)]
            # Actualizarea listei pi
            self.pi = [N[k] / len(X) for k in range(self.n_components)]

    def predict(self, X):
        '''
        Funcția de prezicere.
        :param X: 2-d array numpy array
            Datele pe care trebuie să le prezicem clusterele
        '''
        probas = []
        for n in range(len(X)):
            probas.append([self.multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k]) for k in range(self.n_components)])
        clusters = []
        for proba in probas:
            clusters.append(self.comp_names[proba.index(max(proba))])
        return clusters
