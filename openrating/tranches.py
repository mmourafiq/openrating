import numpy as np

from .utils import year_frac


class Tranche(object):
    """
    The `Tranche` class represents a note or a tranche, to calculate and pay the interest
    and the notional of the tranches, to pay the recoveries, and to calculate the price value and
    the weighted average life.

    Attributes:
        * balances: array of left to pay balance of the tranche for each period (initial value is
                    equal to the initial balance times the size of the tranche)
        * paid_interest_rates: array of paid interest rates for each period
        * repayments: array of repayments for each period
        * prepayments: array of prepayments for each period
        * pdls: array of pdls for each period
        * recoveries: array of recoveries for each period
        * spread: the payed spread above LIBOR
    Properties:
        * rating: the rating of the tranche
    """
    def __init__(self, periods_length, initial_balance, size, spread, prorata=False):
        self.prorata = prorata
        self.balances = np.zeros(periods_length)
        self.balances[0] = initial_balance * size
        self.paid_interest_rates = np.zeros(periods_length)
        self.expected_interest_rates = np.zeros(periods_length)
        self.repayments = np.zeros(periods_length)
        self.prepayments = np.zeros(periods_length)
        self.pdls = np.zeros(periods_length)
        self.recoveries = np.zeros(periods_length)
        self.spread = spread
        self.shortfall = 0

    @property
    def rating(self):
        return ''

    def discounts(self, periods_length, periods_num, libors, flat=False):
        """
        Return the calculated discounts for each period, flat discounts don't include the spread:
            if flat:
                period_rate = (libors[i - 1]) / periods_num
            else:
                period_rate = (libors[i - 1] + spread) / periods_num

            discounts[i] = discounts[i - 1] / (1 + period_rate)
        :param periods_length:
        :param periods_num:
        :param libors:
        :param flat: whether to calculate a flat discounts or not
        """
        discounts = np.zeros(periods_length)
        discounts[0] = 1
        if flat:
            rates = libors[:-1] + self.spread
        else:
            rates = libors[:-1]

        discounts[1:] = np.cumprod(1 / (1 + (rates / periods_num)), dtype=float)
        return discounts

    def get_expected_interest(self, periods_num, period, libors):
        """
        Calculates the expected interest rate for the given period.
        The interest on the rest of the balance to pay and the shortfall from last
        period, plus the shortfall.
        :param periods_num:
        :param period:
        :return:
        """
        interest_rate = libors[period - 1] + self.spread
        interest = (self.balances[period - 1] + self.shortfall) * (interest_rate / periods_num)
        interest += self.shortfall
        self.expected_interest_rates[period] = interest
        return interest

    def pay_notional(self, period, available_cash):
        """
        Pays the notional in the waterfall.
        :param period:
        :param available_cash:
        :return:
        """
        balance_state = self.balances[period - 1]
        amount = min(available_cash, balance_state)
        available_cash -= amount

        self.balances[period] = balance_state - amount
        self.repayments[period] = amount

        assert available_cash

    def pay_interest(self, periods_num, period, available_cash, libors):
        """
        Pays the interest in the waterfall
        :param periods_num:
        :param period:
        :param available_cash:
        :param libors:
        :return:
        """
        expected_interest = self.get_expected_interest(periods_num, period, libors)
        amount = min(available_cash, expected_interest)
        available_cash -= amount
        self.paid_interest_rates[period] = amount
        self.shortfall = max(expected_interest - amount, 0)
        return available_cash

    def pay_recovery(self, period, available_cash):
        """

        :param period:
        :param available_cash:
        :return:
        """
        amount = min(available_cash, self.pdls[period - 1])
        available_cash -= amount
        self.pdls[period] = self.pdls[period - 1] - amount
        self.recoveries[period] = amount
        return available_cash

    def account_pdl(self, period, losses):
        """
        Calculates the principal deficiency ledger, the pdls provide a mechanism to distribute
        the risk of principal losses among noteholders in reverse order of seniority.
        :param period:
        :param losses:
        :return:
        """
        max_pdl = self.balances[period - 1] - self.pdls[period - 1]
        amount = min(losses, max_pdl)
        losses -= amount
        self.pdls[period] = self.pdls[period - 1] + amount
        return losses

    def get_price_value(self, periods_length, periods_num, libors, flat=False):
        """
        Calculates the price value of the tranche
        :param periods_length:
        :param periods_num:
        :param libors:
        :param flat: whether to calculate a flat price value or not
        :return:
        """
        discounts = self.discounts(periods_length, periods_num, libors, flat)
        price_value = discounts * (self.paid_interest_rates + self.repayments + self.recoveries)
        return (price_value.sum() / self.balances[0]) * 100

    def get_wal(self, periods_length, periods_num, start_date, end_date):
        """
        Calculates the average weighted life of the tranche
        :param periods_length:
        :param periods_num:
        :param start_date:
        :param end_date:
        :return:
        """
        wal = (self.recoveries + self.repayments) * np.arange(periods_length) / periods_num
        wal = wal.sum() + self.balances[-1] * year_frac(start_date, end_date)
        wal /= self.balances[0]
        return wal
