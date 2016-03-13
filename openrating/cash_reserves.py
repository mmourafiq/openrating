import numpy as np


class CashReserve(object):
    """
    The `CashReserve` represents an internal credit enhancement, it provides cash to cover
    interest or principal shortfalls.
    The reserve fund is usually a percentage of the initial or outstanding aggregate
    principal amount of the notes (or assets).
    The reserve fund can be funded at closing by proceeds and reimbursed via the waterfall.

    Attributes:
        * available_cash: array of available cash in the account for each period (the first period
                          the cash reserve is filled with an initial balance)
        * interest_rate: interest rate used to calculate interest on the cash reserve each period
        * target_pct: the percentage of the outstanding balance, used to calculate the amount of
                      cash needed o fill the reserve
    """
    def __init__(self, periods_length, interest_rate, target_pct, initial_balance):
        self.available_cash = np.zeros(periods_length)
        self.available_cash[0] = initial_balance
        self.interest_rate = interest_rate
        self.target_pct = target_pct

    def earn_interest(self, periods_num, period):
        """
        Calculates the earned interest on the reserve fund.
        :param periods_num:
        :param period:
        """
        previous_cash = self.available_cash[period - 1]
        self.available_cash[period] = previous_cash * (1 + (self.interest_rate / periods_num))

    def get_expected_reserve_account(self, period, outstanding_balance):
        """
        Returns the target amount based on the outstanding balance if there's enough available cash,
        otherwise returns what left as available cash.
        :param period:
        :param outstanding_balance:
        :return:
        """
        return max(0, (self.target_pct * outstanding_balance) - self.available_cash[period])

    def reimburse(self, period, interest, outstanding_balance):
        """
        Reimburse the reserve account, and return the rest of the interest after reimbursement.
        Each period we look at the reserve account and we try to reach to target percentage.
        :param period:
        :param interest: interest collected in the current period
        :param outstanding_balance:
        :return:
        """
        if outstanding_balance <= 0:
            return
        amount = min(interest, self.get_expected_reserve_account(period, outstanding_balance))
        self.available_cash[period] = self.available_cash[period - 1] + amount
        return interest - amount

    def use(self, period):
        """
        Returns the available cash reserve of the period to be used.
        :param period:
        :return:
        """
        self.available_cash[period], amount = 0, self.available_cash[period]
        return amount
