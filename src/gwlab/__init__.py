"""GWLab — Research-Grade Gravitational-Wave Interactive Analysis Laboratory.

GWLab provides an interactive scientific interface to explore, audit, and visualize
the verified outputs of the gw-inference parameter estimation engine.

Strict Safety Constraint:
GWLab initial releases are strictly read-only with respect to scientific inference.
The web application does not launch nested samplers, evaluate expensive likelihoods,
or mutate pre-computed experimental artifacts.
"""

__version__ = "0.1.0"
__author__ = "Gravitational-Wave Inference & Parameter Estimation Research Project"
__license__ = "Proprietary / Research"
