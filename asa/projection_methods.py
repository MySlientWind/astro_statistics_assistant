from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .utils import remove_bad


def get_LDA_projection(x,
                       y,
                       n_components=1,
                       use_transform=True,
                       return_more=False,
                       LDA_kwargs=None):
    """Get the projection of the data onto the first n_components of LDA.

    Args:
        x (np.ndarray): The data to project.
        y (np.ndarray): The labels to use for training the LDA.
        n_components (int): The number of components to project onto.
        return_more (bool): Whether to return more information.

    Returns:
        np.ndarray: The projection of the data onto the first n_components of
            LDA.
    """
    if LDA_kwargs is None:
        LDA_kwargs = {}

    lda = LinearDiscriminantAnalysis(n_components=n_components, **LDA_kwargs)
    lda.fit(x, y)
    if use_transform:
        x_projected = lda.transform(x)
    else:
        x_projected = x @ lda.scalings_

    # sourcery skip: assign-if-exp
    if return_more:
        return x_projected, lda
    else:
        return x_projected
