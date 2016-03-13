import numpy as np


class CashFlow(object):
    """
    The `CashFlow` represents the allocation of interest and principal collections from the asset
    pool to the transaction parties is described by the priority of payments.
    The transaction parties that keep the structure functioning (originator, servicer, and issuer)
    have the highest priorities.
    After these senior fees and expenses.
    Waterfalls can be classified either as combined waterfalls or as separate waterfalls.
     * combined waterfall: all cash collections from the asset pool are combined into available
            funds and the allocation is described in a single waterfall. There is, thus no
            distinction made between interest collections and principal collections.
     * separate waterfall: interest collections and principal collections are kept separated and
                distributed according to an interest waterfall and a principal waterfall,
                respectively.
     We should keep in mind that we can start with a pro rata allocation which means a proportional
     allocation of the note redemption amount, such that the redemption amount due to each note
     is an amount proportional to the note's fraction of the total outstanding principal amount of
     the notes on the closing date.
    """
    SEPARATE = 'separate'
    COMBINED = 'combined'

    def __init__(self, triggers, tranches, pool, cash_account, cash_reserve, senior_expenses,
                 servecing_fees, waterfall=COMBINED):
        self.waterfall = waterfall
        self.triggers = triggers
        self.tranches = tranches
        self.pool = pool
        self.cash_account = cash_account
        self.cash_reserve = cash_reserve
        self.senior_expenses = senior_expenses
        self.servicing_fees = servecing_fees

    @property
    def has_prorata(self):
        return any([tranche.prorata for tranche in self.tranches])

    @property
    def is_separate(self):
        return self.waterfall == self.SEPARATE

    def run(self, periods_length, periods_num, periods, assets, defaults_matrix, libors):
        has_prorata = self.has_prorata
        self.pool.calculate_cash_flows(periods_length, periods, assets, defaults_matrix, libors)

        for period in xrange(periods_length):
            if has_prorata and self.pool.cumulative_losses[period] < self.triggers[period]:
                self.prorata_waterfall(period, periods_num, periods, defaults_matrix, libors)
            else:
                if self.is_separate:
                    self.separate_waterfall(period, periods_num, periods, defaults_matrix, libors)
                else:
                    self.combined_waterfall(period, periods_num, periods, defaults_matrix, libors)

    def separate_waterfall(self, period, periods_num, periods, defaults_matrix, libors):
        """
        Collects interest and principal separately and distributes them according to
        the interest waterfall and the principal waterfall, respectively.

        :param periods_num:
        :param periods:
        :param defaults_matrix:
        :param libors:
        :return:
        """

        # earning interest on the cashReserve
        self.cash_reserve.earn_interest(periods_num, period)
        # collect cashflows and defaults from the pool
        interest = self.pool.available_interests[period]
        notional = (self.pool.repayments[period] +
                    self.pool.recoveries[period] +
                    self.cash_reserve.available_cash[period - 1] +
                    self.cash_account.available_cash[period - 1])
        losses = self.pool.losses[period]

        # pay first the senior expenses and the servicing fees
        self.senior_expenses.pay_expanses(periods_num, period, interest, self.pool.balances[period])
        self.servicing_fees.pay_expenses(periods_num, period, interest, self.pool.balances[period])

        # perform payments of interest and capital
        for tranche in self.tranches:
            # iterate over the tranches distribution interest and notional by priority
            # after interest/notional has been paid to a class, the corresponding variable is
            # automatically decreased by this amount.
            interest = tranche.pay_interest(periods_num, period, interest, libors)
            notional = tranche.pay_notional(period, notional)

        # after paying the issuer expenses, servicing fees, the interest and the notional
        # if something left we use it to reimburse the reserve account
        interest = self.cash_reserve.reimburse(period, interest, self.pool.balances[period])

        # what left is considred as an excess spread
        self.cash_account.add_cash(period, periods_num, interest, self.pool.spreads[period])

        # calculate the PDLs - assign losses in reverse order
        for tranche in reversed(self.tranches):
            losses = tranche.account_pdl(period, losses)

    def combined_waterfall(self, period, periods_num, periods, defaults_matrix, libors):
        # earning interest on the cashReserve
        self.cash_reserve.earn_interest(periods_num, period)
        # collect cashflows and defaults from the pool
        interest = self.pool.available_interests[period]
        notional = (self.pool.repayments[period] +
                    self.pool.recoveries[period] +
                    self.cash_reserve.available_cash[period - 1] +
                    self.cash_account.available_cash[period - 1])
        total_pool = interest + notional
        losses = self.pool.losses[period]

        # pay first the senior expenses and the servicing fees
        self.senior_expenses.pay_expanses(periods_num, period, total_pool, self.pool.balances[period])
        self.servicing_fees.pay_expenses(periods_num, period, total_pool, self.pool.balances[period])

        # perform payments of interest
        for tranche in self.tranches:
            # iterate over the tranches distribution interest by priority
            total_pool = tranche.pay_interest(periods_num, period, total_pool, libors)

        # recalculate the losses
        losses -= total_pool - notional

        # perform payments of notional
        for tranche in self.tranches:
            # iterate over the tranches distribution notional by priority
            total_pool = tranche.pay_notional(period, total_pool)

        # after paying the issuer expenses, servicing fees, the interest and the notional
        # if something left we use it to reimburse the reserve account
        total_pool = self.cash_reserve.reimburse(period, total_pool, self.pool.balances[period])

        # what left is considred as an excess spread
        self.cash_account.add_cash(period, periods_num, total_pool, self.pool.spreads[period])

        # calculate the PDLs - assign losses in reverse order
        for tranche in reversed(self.tranches):
            losses = tranche.account_pdl(period, losses)

    def prorata_waterfall(self, period, periods_num, periods, defaults_matrix, libors):
        # earning interest on the cashReserve
        self.cash_reserve.earn_interest(periods_num, period)
        # collect cashflows and defaults from the pool
        interest = self.pool.available_interests[period]
        notional = (self.pool.repayments[period] +
                    self.pool.recoveries[period] +
                    self.cash_reserve.available_cash[period - 1] +
                    self.cash_account.available_cash[period - 1])
        total_pool = interest + notional
        losses = self.pool.losses[period]

        # proportions of notional to distribute
        proportions = [(tranche.balances[0] / self.pool.balances[0]) * notional for tranche in self.tranches]

        # pay first the senior expenses and the servicing fees
        self.senior_expenses.pay_expanses(periods_num, period, interest, self.pool.balances[period])
        self.servicing_fees.pay_expenses(periods_num, period, interest, self.pool.balances[period])

        # perform payments of interest no prorata
        last_tranche = 0
        for i, tranche in enumerate(self.tranches):
            # iterate over the tranches distribution interest and notional by priority
            interest = tranche.pay_interest(periods_num, period, interest, libors)

            if tranche.prorata:
                # iterate over the tranches distribution interest by priority
                if i > 0:
                    proportions[i] += proportions[i - 1]
                    proportions[i - 1] = 0

                proportions[i] = tranche.pay_notional(period, proportions[i])
            else:
                break

        rest_notional = np.sum([proportions])

        for i in xrange(last_tranche, len(self.tranches)):
            # iterate over the tranches distribution notional by priority
            rest_notional = self.tranches[i].pay_notional(period, rest_notional)
