import numpy as np
from sklearn.frozen import FrozenEstimator
from sklearn.mixture import GaussianMixture as SK_GaussianMixture
from sklearn.mixture._gaussian_mixture import (
    _compute_precision_cholesky, _estimate_gaussian_covariances_diag,
    _estimate_gaussian_covariances_full,
    _estimate_gaussian_covariances_spherical,
    _estimate_gaussian_covariances_tied)


# TODO: fit with fixed covariances
# TODO: more test to make sure it works as expected
class GaussianMixture(SK_GaussianMixture):

    def __init__(
        self,
        n_components=1,
        *,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        fix_weights=False,
        weights_init=None,
        fix_means=False,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        if fix_weights and weights_init is None:
            raise ValueError(
                "weights_init must be provided when fix_weights is True")
        self.fix_weights = fix_weights

        if fix_means and means_init is None:
            raise ValueError(
                "means_init must be provided when fix_means is True")
        self.fix_means = fix_means

        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(
            X,
            np.exp(log_resp),
            self.reg_covar,
            self.covariance_type,
            self.means_,
            fix_means=self.fix_means)
        if not self.fix_weights:
            self.weights_ = weights_ / weights_.sum()
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features = self.means_.shape
        if self.covariance_type == "full":
            cov_params = self.n_components * n_features * (n_features +
                                                           1) / 2.0
        elif self.covariance_type == "diag":
            cov_params = self.n_components * n_features
        elif self.covariance_type == "tied":
            cov_params = n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "spherical":
            cov_params = self.n_components
        else:
            raise ValueError(
                f"covariance_type must be one of 'full', 'tied', 'diag','spherical', got {self.covariance_type}"
            )
        mean_params = 0 if self.fix_means else n_features * self.n_components
        weights_params = 0 if self.fix_weights else self.n_components - 1
        return int(cov_params + mean_params + weights_params)

    def posterior_GMM(self, X_obs, Sigma_obs, A):
        """
        P(X_obs|Z) ~ N(A @ Z, Sigma_obs)
        P(Z) = GMM(Z)
        
        P(Z|X_obs) = P(X_obs|Z) * P(Z) / P(X_obs)
        """
        # TODO: other type
        if self.covariance_type != "full":
            raise NotImplementedError("Only Support full covariance type")

        n_samples, _ = X_obs.shape
        n_components, n_features = self.means_.shape

        post_covariances_ = np.zeros(
            (n_samples, n_components, n_features, n_features))
        post_means_ = np.zeros((n_samples, n_components, n_features))

        GMM_lst = []

        for i in range(n_samples):
            this_X_obs = X_obs[i]
            this_Sigma_obs = Sigma_obs[i]
            AT_Sigmainv = A.T @ np.linalg.pinv(this_Sigma_obs)
            AT_Sigmainv_A = AT_Sigmainv @ A
            for j in range(self.n_components):
                post_covariances_[i, j] = np.linalg.pinv(self.precisions_[j] +
                                                         AT_Sigmainv_A)
                post_means_[i, j] = post_covariances_[i, j] @ (
                    self.precisions_[j] @ self.means_[j] +  # type: ignore
                    AT_Sigmainv @ this_X_obs)

            GMM_post = self.new_GMM(post_covariances_[i], post_means_[i])
            GMM_lst.append(FrozenEstimator(GMM_post))

        return GMM_lst

    def new_GMM(self, post_covariances_, post_means_):
        # pylint: disable=attribute-defined-outside-init
        GMM_post = GaussianMixture(**self.get_params())
        GMM_post.covariances_ = post_covariances_
        GMM_post.means_ = post_means_
        GMM_post.weights_ = self.weights_
        GMM_post.precisions_cholesky_ = _compute_precision_cholesky(
            GMM_post.covariances_, GMM_post.covariance_type)
        GMM_post._set_parameters(GMM_post._get_parameters())
        return GMM_post


def _estimate_gaussian_parameters(X,
                                  resp,
                                  reg_covar,
                                  covariance_type,
                                  means,
                                  fix_means=False):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array.

    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    nk : array-like of shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    if not fix_means:
        means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = {
        "full": _estimate_gaussian_covariances_full,
        "tied": _estimate_gaussian_covariances_tied,
        "diag": _estimate_gaussian_covariances_diag,
        "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)
    return nk, means, covariances
