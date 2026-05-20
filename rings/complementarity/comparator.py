"""This module contains classes that compare metric spaces (i.e. pairwise distance matrices)."""

import numpy as np


class MatrixNormComparator:
    """
    A class to compare matrices based on specified norms and calculate complementarity.

    This class computes complementarity between two distance matrices by calculating
    their difference using a matrix norm. It provides a quantitative measure of how
    similar two metric spaces are.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments to specify the norm to be used. Supported norms are:

        - norm : str, default="L11"
            The matrix norm to use. Options include:
            - "L11": The L1,1 norm (sum of absolute values)
            - "frobenius": The Frobenius norm (square root of sum of squared values)
        - Additional parameters are ignored.

    Attributes
    ----------
    norm : str
        The norm to be used for comparison.

    Methods
    -------
    __call__(x, y, **kwargs)
        Compare two matrices using the specified norm.
    L11_norm(M)
        Calculate the L1,1 norm of a matrix.
    frobenius_norm(M)
        Calculate the Frobenius norm of a matrix.
    invalid_data
        Return a dictionary with NaN score and other standardized fields.

    Examples
    --------
    >>> import numpy as np
    >>> from rings.complementarity.comparator import MatrixNormComparator
    >>>
    >>> # Create two distance matrices
    >>> D1 = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    >>> D2 = np.array([[0, 2, 4], [2, 0, 2], [4, 2, 0]])  # D1 scaled by 2
    >>>
    >>> # Compare using L11 norm (default)
    >>> comparator1 = MatrixNormComparator()
    >>> result1 = comparator1(D1, D2)
    >>> print(f"L11 complementarity: {result1['score']}")
    L11 complementarity: 0.6666666666666666
    >>>
    >>> # Compare using Frobenius norm
    >>> comparator2 = MatrixNormComparator(norm="frobenius")
    >>> result2 = comparator2(D1, D2)
    >>> print(f"Frobenius complementarity: {result2['score']}")
    Frobenius complementarity: 0.5773502691896257
    """

    def __init__(self, **kwargs):
        """
        Initialize the MatrixNormComparator with the specified norm.
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to specify the norm to be used. Supported norms are:
            - "L11" (default)
            - "frobenius"
        """
        self.norm = kwargs["norm"] if "norm" in kwargs else "L11"

    def __call__(self, x, y, **kwargs):
        """
        Compare two matrices using the specified norm.

        Parameters
        ----------
        x : numpy.ndarray
            The first matrix for comparison.
        y : numpy.ndarray
            The second matrix for comparison.
        **kwargs : dict, optional
            Additional keyword arguments (not used).

        Returns
        -------
        dict
            Dictionary with keys:
            - score: The computed norm-based complementarity score
            - pvalue: None (p-values not computed for norm-based comparisons)
            - pvalue_adjusted: None
            - significant: None
            - method: The norm used (e.g., "L11" or "frobenius")
        """
        n = len(x)
        divisor = n**2 - n if n > 1 else 1

        if self.norm == "L11":
            complementarity_score = self.L11_norm(x - y)
        elif self.norm == "frobenius":
            complementarity_score = self.frobenius_norm(x - y)
            divisor = np.sqrt(divisor)
        else:
            raise RuntimeError(f"Unexpected norm '{self.norm}'")

        complementarity_score /= divisor

        return {
            "score": complementarity_score,
            "pvalue": None,
            "pvalue_adjusted": None,
            "significant": None,
            "method": self.norm,
        }

    @staticmethod
    def L11_norm(M):
        """
        Calculate the L1,1 norm of a matrix.
        Parameters
        ----------
        M : numpy.ndarray
            The matrix for which to calculate the L1,1 norm.
        Returns
        -------
        float
            The L1,1 norm of the matrix.
        """
        return abs(M).sum()

    @staticmethod
    def frobenius_norm(M):
        """
        Calculate the Frobenius norm of a matrix.
        Parameters
        ----------
        M : numpy.ndarray
            The matrix for which to calculate the Frobenius norm.
        Returns
        -------
        float
            The Frobenius norm of the matrix.
        """
        return np.linalg.norm(M, ord="fro")

    @property
    def invalid_data(self):
        """
        Return a dictionary with NaN score and other standardized fields.

        Returns
        -------
        dict
            Dictionary with keys:
            - score: NaN (invalid data)
            - pvalue: None
            - pvalue_adjusted: None
            - significant: None
            - method: The norm used
        """
        return {
            "score": np.nan,
            "pvalue": None,
            "pvalue_adjusted": None,
            "significant": None,
            "method": self.norm,
        }


#  ╭──────────────────────────────────────────────────────────╮
#  │ Factory Functions                                        |
#  ╰──────────────────────────────────────────────────────────╯


def L11MatrixNormComparator(**kwargs):
    """
    Factory function that returns a MatrixNormComparator with L11 norm.

    This is a convenience function that creates a MatrixNormComparator
    pre-configured to use the L1,1 norm (sum of absolute values).

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments to pass to the MatrixNormComparator constructor.

    Returns
    -------
    MatrixNormComparator
        A comparator configured to use the L11 norm.

    Examples
    --------
    >>> import numpy as np
    >>> from rings.complementarity.comparator import L11MatrixNormComparator
    >>>
    >>> # Create two distance matrices
    >>> D1 = np.array([[0, 1], [1, 0]])
    >>> D2 = np.array([[0, 2], [2, 0]])
    >>>
    >>> # Create comparator and compare matrices
    >>> comparator = L11MatrixNormComparator()
    >>> result = comparator(D1, D2)
    >>> print(f"Complementarity score: {result['score']}")
    Complementarity score: 0.5
    """
    return MatrixNormComparator(norm="L11", **kwargs)


def FrobeniusMatrixNormComparator(**kwargs):
    """
    Factory function that returns a MatrixNormComparator with Frobenius norm.

    This is a convenience function that creates a MatrixNormComparator
    pre-configured to use the Frobenius norm (square root of sum of squared values).

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments to pass to the MatrixNormComparator constructor.

    Returns
    -------
    MatrixNormComparator
        A comparator configured to use the Frobenius norm.

    Examples
    --------
    >>> import numpy as np
    >>> from rings.complementarity.comparator import FrobeniusMatrixNormComparator
    >>>
    >>> # Create two distance matrices
    >>> D1 = np.array([[0, 1], [1, 0]])
    >>> D2 = np.array([[0, 2], [2, 0]])
    >>>
    >>> # Create comparator and compare matrices
    >>> comparator = FrobeniusMatrixNormComparator()
    >>> result = comparator(D1, D2)
    >>> print(f"Complementarity score: {result['score']}")
    Complementarity score: 0.5
    """
    return MatrixNormComparator(norm="frobenius", **kwargs)
