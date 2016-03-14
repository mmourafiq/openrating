from __future__ import division

import numpy as np
import pandas as pd


class Frequencies(object):
    MONTH = 'M'
    YEAR = 'A'

    _frequencies = {MONTH, YEAR}

    @classmethod
    def is_valid(cls, frequency, supported_frequencies=_frequencies):
        return frequency in supported_frequencies

    @classmethod
    def get_periods_num(cls, frequency, supported_frequencies=_frequencies):
        if not cls.is_valid(frequency, supported_frequencies):
            raise ValueError('Frequency not supported.')

        if frequency == cls.MONTH:
            return 12
        if frequency == cls.YEAR:
            return 1

    @classmethod
    def get_periods_length(cls, start_date, end_date, frequency, supported_frequencies=_frequencies):
        if not cls.is_valid(frequency, supported_frequencies):
            raise ValueError('Frequency not supported.')

        return len(pd.date_range(start_date, end_date, freq=frequency))

    @staticmethod
    def get_conditional_probabilities(conditional_probabilities, periods_num):
        """
        Returns an interpolated conditional probabilities
        :param conditional_probabilities:
        :param periods_num:
        """
        if periods_num == 1:
            return conditional_probabilities

        if periods_num == 12:
            conditional_probabilities = conditional_probabilities.reindex(np.arange(0, 10, 1.0/12))
            return conditional_probabilities.interpolate(method='cubic', downcast='infer')


def year_frac(start_date, end_date, basis=3):
    denominator = 360 if basis == 2 else 365
    return (end_date - start_date).days / denominator


def check_tranches_prorata(tranches):
    for i in xrange(1, len(tranches)):
        if tranches[i].prorata and not tranches.protata[i - 1]:
            raise ValueError('All tranches after the tranche {}, must pay sequential'.find(i))
