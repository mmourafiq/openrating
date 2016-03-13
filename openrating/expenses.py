import numpy as np


class Expense(object):
    """
    The `Expense` class represents an expense that needs to be paid in the priority of payments.
    In general we need to instances of this class:
        * Senior expenses:
            - shortfall: the senior shortfall
            - shortfall_rate: the senior shortfall_rate
            - rate: the issuer expenses rate
            - paid: the paid issuer expenses in each period
        * Servicing fees
            - shortfall: the servicing fee shortfall
            - shortfall_rate: the servicing fee shortfall rate
            - rate: servicing fee rate
            - paid: the paid servicing fees
    """
    def __init__(self, periods_length, rate, shortfall_rate):
        self.shortfall = 0
        self.shortfall_rate = shortfall_rate
        self.rate = rate
        self.paid = np.zeros(periods_length)

    def get_expected_expense(self, periods_num, outstanding_balance):
        """
        Calculates the expected expense.
        :param periods_num:
        :param outstanding_balance:
        :return:
        """
        current_issuer_expenses = outstanding_balance * (self.rate / periods_num)
        senior_shortfall_expenses = self.shortfall * (1 + self.shortfall_rate / periods_num)
        return current_issuer_expenses + senior_shortfall_expenses

    def pay_expenses(self, periods_num, period, interest, outstanding_balance):
        """
        Pays the expense in the waterfall (priority of payment).
        :param periods_num:
        :param period:
        :param interest:
        :param outstanding_balance:
        :return: left of the interest
        """
        if outstanding_balance <= 0:
            return interest

        expense = self.get_expected_expense(periods_num, outstanding_balance)
        if interest < expense:
            interest, amount = 0, interest
            self.shortfall = expense - amount
        else:
            interest -= expense
            amount = expense
            self.shortfall = 0

        self.paid[period] = amount
        return interest
