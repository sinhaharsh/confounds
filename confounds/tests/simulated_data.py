import itertools
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from pyradigm import ClassificationDataset
from scipy.special import expit
from scipy.stats import norm

from deconfounding.utils import import_string, make_save_path


class SimulationData:
    def __init__(self,
                 num_covariates=10,
                 num_unmeasured=1,
                 adjacency_list=None,
                 num_samples=1000,
                 rho: float = 0.2,
                 seed=42):
        if adjacency_list is None:
            raise ValueError('Adjacency list cannot be None')
        self.adjacency_list = adjacency_list
        self.num_samples = num_samples
        self.rho = rho
        self.num_covariates = num_covariates
        self.num_unmeasured = num_unmeasured

        self.data = None
        self.rng = np.random.default_rng(seed)

    def generate_col_names(self):
        """Generate column names for the synthetic dataset """
        pC = self.num_covariates  # number of covariates
        pU = self.num_unmeasured  # number of unmeasured confounders
        col_names = itertools.chain(['A', 'Y'],
                                    self.make_names(pC, 'c'),
                                    self.make_names(pU, 'u'))
        return list(col_names)

    def get_covariate_names(self):
        return self.make_names(self.num_covariates, 'c')

    @staticmethod
    def make_names(count, subscript):
        return ['{1}{0}'.format(i, subscript.upper())
                for i in range(1, count + 1)]

    def generate_normal_data(self, return_cov_matrix=False):
        # while True:
        total_variables = self.num_covariates + self.num_unmeasured + 2
        cov_x = np.eye(total_variables) + \
                ~np.eye(total_variables, dtype=bool) * self.rho

        column_names = self.generate_col_names()
        for var in self.adjacency_list:
            for adj_var, value in self.adjacency_list[var].items():
                index_var = column_names.index(var)
                index_adj_var = column_names.index(adj_var)
                cov_x[index_var, index_adj_var] = value
                cov_x[index_adj_var, index_var] = value

        psd_cov_x = self.get_psd_matrix(cov_x)
        # print(np.round(psd_cov_x, 2))
        if np.iscomplex(psd_cov_x).any():
            raise ValueError('Covariance matrix is complex.'
                             'Please check the input coefficients')
        data = self.rng.multivariate_normal(
            mean=np.zeros(total_variables),
            cov=psd_cov_x,
            size=self.num_samples)
        return data

    def generate_linear_dataset(self, return_cov_matrix=False):
        cov_matrix = None
        data = self.generate_normal_data(return_cov_matrix=return_cov_matrix)
        column_names = self.generate_col_names()
        df = pd.DataFrame(data=data, columns=column_names)
        binned_df, cov_matrix = self.discretize_data(df, return_cov_matrix)
        self.data = binned_df
        return cov_matrix

    def discretize_data(self, df, return_cov_matrix=False):
        binned_df = df.copy()
        column_names = self.generate_col_names()
        for col in column_names:
            if col.startswith('C') or col.startswith('U'):
                binned_df[col] = pd.qcut(df[col], 3, labels=False)
        binned_df['Y'] = np.round(expit(binned_df['Y']), 0)
        self.data = binned_df
        if return_cov_matrix:
            return binned_df, binned_df.corr()
        return binned_df, None

    @staticmethod
    def get_psd_matrix(A):
        # computing the nearest correlation matrix
        # (i.e., enforcing unit diagonal). Higham 2000
        # https://www.maths.manchester.ac.uk/~higham/narep/narep369.pdf
        # https://stackoverflow.com/a/10940283/3140172

        def _get_a_plus(matrix):
            eigval, eigvec = np.linalg.eig(matrix)
            Q = np.matrix(eigvec)
            xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
            return Q * xdiag * Q.T

        def _get_ps(matrix, W=None):
            W05 = np.matrix(W ** .5)
            return W05.I * _get_a_plus(W05 * matrix * W05) * W05.I

        def _get_pu(matrix, W=None):
            Aret = np.array(matrix.copy())
            Aret[W > 0] = np.array(W)[W > 0]
            return np.matrix(Aret)

        def near_pd(matrix, nit=10):
            n = matrix.shape[0]
            W = np.identity(n)
            # W is the matrix used for the norm
            # (assumed to be Identity matrix here)
            # the algorithm should work for any diagonal W
            deltaS = 0
            Yk = matrix.copy()

            for k in range(nit):
                Rk = Yk - deltaS
                Xk = _get_ps(Rk, W=W)
                deltaS = Xk - Rk
                Yk = _get_pu(Xk, W=W)
            return Yk

        while True:
            C = near_pd(A, nit=10)
            if np.all(np.linalg.eigvals(C) > 0):
                return C
            else:
                # It is possible that eigenvalues are -0.000 due to floating
                # point error. Need to correct for it.
                min_eigvalue = min(np.linalg.eigvals(C))
                C -= 10 * min_eigvalue * np.eye(*C.shape)
                if np.all(np.linalg.eigvals(C) > 0):
                    return C
            warnings.warn("regenerating random correlation matrix")

    def save_to_pyradigm(self, output_path):
        """
        Save data to pyradigm format for classification and regression

        Parameters
        ----------
        output_path: Optional[str]
            Output path to save pyradigm dataset
        """
        covariate_names = []
        for col in self.data.columns:
            if col.startswith('C') or col.startswith('U'):
                covariate_names.append(col)

        # Create pyradigm datasets
        clf_dataset = ClassificationDataset()

        for i, row in enumerate(self.data.to_dict('records')):
            subject_id = str(i)
            target_clf = row['Y']
            features = row['A']
            covariates = {k: v for k, v in row.items() if k in covariate_names}

            clf_dataset.add_samplet(samplet_id=subject_id,
                                    features=features,
                                    target=target_clf,
                                    feature_names=['treatment'],
                                    attr_names=list(covariates.keys()),
                                    attr_values=list(covariates.values()))
        if output_path is not None:
            clf_dataset.save(output_path)
        return clf_dataset


def generate_data():
    num_covariates = 3
    num_unmeasured = 0
    num_samples = 5000
    rho = 0
    adjacency_list = {
        'A' : {'Y': 0.7, 'C1': 0.3, 'C2': 0.5},
        'C1': {'Y': 0.5, 'A': 0.5},
        'C2': {'A': 0.6, 'Y': 0.0},
        'Y' : {'A': 0.7, 'C1': 0.5},
    }
    dataset = SimulationData(num_covariates=num_covariates,
                             num_unmeasured=num_unmeasured,
                             num_samples=num_samples, seed=42,
                             adjacency_list=adjacency_list,
                             rho=rho)
    dataset.generate_linear_dataset()
    pydm_ds = dataset.save_to_pyradigm(output_path=None)

    X, y, _ = pydm_ds.data_and_targets()  # noqa
    Y = y.astype(float)
    covariates = []
    for i in ['C1']:  # pydm_ds.attr.keys():
        covariates.append(pydm_ds.get_attr(i))
    C = np.array(covariates).T
    return X, Y, C
