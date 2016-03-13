import numpy as np


class CashAccount(object):
    """
    The `CashAccount` class calculate the excess spread in our cash flow and
    put It in an available cash.

    Attributes:
        * value: the current value of the cash account
        * available_cash: array of available cash in the account for each period
        * excess_spread: array of excess spread for each period
    """
    def __init__(self, periods_length):
        self.value = 0
        self.available_cash = None
        self.excess_spread = None
        self.clear_cash(periods_length)

    def clear_cash(self, periods_length):
        self.value = 0
        self.available_cash = np.zeros(periods_length)
        self.excess_spread = np.zeros(periods_length)

    def add_cash(self, period, periods_num, amount, spread, libors):
        """
        Adds the amount of this period to the cash account (excess) and the interest earned on the
        cash from the previous period.
        :param periods_num:
        :param amount:
        :param period:
        :param spread:
        :param libors:
        :return:
        """
        self.value *= 1 + ((libors[period - 1] + spread) / periods_num)
        self.value += amount
        self.excess_spread[period] = amount
        self.available_cash[period] = self.value

    def pay_cash(self, amount, period):
        """
        Pays the amount of money from the cash account.
        :param amount:
        :param period:
        :return:
        """
        self.value -= amount
        self.available_cash[period] = self.value

    def use(self, period):
        """
        Returns the available cash in the account.
        :param period:
        :return:
        """
        amount = self.available_cash[period]
        self.value -= amount
        self.available_cash[period] = 0
        return amount

    def get_avg_excess_spread(self):
        return self.excess_spread.mean()


