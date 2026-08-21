"""Interferometer analysis with a stable figure of merit for LBFGS."""

import logging

import autofit as af
import autolens as al
import numpy as np

logger = logging.getLogger(__name__)

FIGURE_OF_MERIT_CHOICES = frozenset(
    {"log_evidence", "log_likelihood_with_regularization", "log_evidence_safe"}
)


class DeconvAnalysisInterferometer(al.AnalysisInterferometer):
    """
    Interferometer analysis for pixelized image deconvolution.

    ``figure_of_merit`` options:

    - ``log_evidence``
    - ``log_likelihood_with_regularization`` (recommended default for LBFGS)
    - ``log_evidence_safe``: try evidence, fall back on Cholesky failure
    """

    def __init__(self, figure_of_merit="log_likelihood_with_regularization", **kwargs):
        if figure_of_merit not in FIGURE_OF_MERIT_CHOICES:
            raise ValueError(
                f"Unsupported figure_of_merit: {figure_of_merit!r}. "
                f"Choose from {sorted(FIGURE_OF_MERIT_CHOICES)}."
            )
        self.figure_of_merit_mode = figure_of_merit
        super().__init__(**kwargs)

    def _figure_of_merit_from_fit(self, fit):
        if self.figure_of_merit_mode == "log_likelihood_with_regularization":
            return fit.log_likelihood_with_regularization

        if self.figure_of_merit_mode == "log_evidence":
            return fit.figure_of_merit

        try:
            return fit.figure_of_merit
        except np.linalg.LinAlgError:
            logger.debug(
                "log_evidence Cholesky failed; using log_likelihood_with_regularization"
            )
            return fit.log_likelihood_with_regularization

    def log_likelihood_function(self, instance):
        log_likelihood_penalty = self.log_likelihood_penalty_from(instance=instance)

        try:
            fit = self.fit_from(instance=instance)
            figure_of_merit = self._figure_of_merit_from_fit(fit)
        except (af.exc.FitException, np.linalg.LinAlgError):
            raise af.exc.FitException from None

        if figure_of_merit is None:
            raise af.exc.FitException

        try:
            if np.isnan(figure_of_merit):
                raise af.exc.FitException
        except TypeError:
            pass

        return figure_of_merit - log_likelihood_penalty
