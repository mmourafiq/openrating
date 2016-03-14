import numpy as np
import pandas as pd

from scipy import stats


class Pool:
    """
    The `Pool` model represents the pool of assets, it allows to initialize the assets,
    to calculate the  delinquencies, the recoveries, the available interest rate.
        .Delinquencies: to calculate delinquent assets
        .Recoveries: to recover delinquent assets

    Asset definition:
        * id: id the of the asset
        * issuer: issuer of the asset (unique identifier)
        * value: par value of the asset
        * maturity: maturity date of the asset
        * rating: Fitch rating of the asset
        * dp_multiplier: defaulting probability multiplier of the asset
        * industry: Fitch industry of the asset
        * country: country of the asset
        * seniority: seniority of the asset
        * rr_multiplier: recovery rate multiplier of the asset
        * fixed_rr: fixed recovery rate of the asset
        * spread: spread generate by the asset
    """
    def __init__(self, periods_length, initial_balance, spread=0, amortizations=None,
                 recovery_step=None, recovery_rate=None, repayment_rates=None, timing_vector=None):
        self.balances = np.zeros(periods_length)
        self.balances[0] = initial_balance
        self.available_interests = np.zeros(periods_length)
        self.losses = np.zeros(periods_length)
        self.delinquencies = np.zeros(periods_length)
        self.recoveries = np.zeros(periods_length)
        self.repayments = np.zeros(periods_length)
        self.spreads = np.empty(periods_length)
        self.spreads.fill(spread)
        self.amortizations = amortizations
        self.recovery_step = recovery_step
        self.recovery_rate = recovery_rate
        self.repayment_rates = repayment_rates  # for MC
        self.timing_vector = timing_vector  # for MC

    @property
    def cumulative_losses(self):
        return self.losses.cumsum()


class CopulaPool(Pool):
    """
    The `CopulaPool` model represents the pool of assets used in case of copula calculation
    to calculate the cash flows.
    """
    def __init__(self, assets, periods_length, initial_balance, spread=0, recovery_step=2):
        self.assets = assets
        super(CopulaPool, self).__init__(periods_length=periods_length,
                                         initial_balance=initial_balance,
                                         spread=spread,
                                         recovery_step=recovery_step)

    def get_cholesky_correlation(self, correlation):
        correlation_mat = correlation.get_correlation_matrix(self.assets)
        return np.linalg.cholesky(correlation_mat)

    def get_asset_conditional_probabilities(self, conditional_probabilities):
        asset_conditional_probabilities = pd.DataFrame(index=conditional_probabilities['maturities'],
                                                       columns=self.assets['id'],
                                                       data=np.nan)
        for asset in self.assets.iterrows():
            asset_conditional_probabilities.ix[asset['id'], :] = conditional_probabilities[asset['maturity']]
        return asset_conditional_probabilities

    def get_default_matrix(self, correlation, conditional_probabilities):
        cholesky_correlation = self.get_cholesky_correlation(correlation)
        asset_conditional_probabilities = self.get_asset_conditional_probabilities(conditional_probabilities)
        rand_loans = np.random.uniform(low=0.0, high=1.0, size=len(self.assets))
        rand_loans = stats.norm.ppf(rand_loans, loc=0, scale=1)
        correlated_loans = np.matmul(cholesky_correlation, rand_loans)
        correlated_loans = stats.norm.cdf(correlated_loans, loc=0, scale=1)
        return (1 - asset_conditional_probabilities) >= correlated_loans

    def _periodic_cash_flows(self, periods_length, periods_num, period, period_date,
                             previous_period_date, defaults_matrix, libors):
        """
        Calculates the cash flows of a single period.
        :param periods_length:
        :param periods_num:
        :param period:
        :param period_date:
        :param previous_period_date:
        :param defaults_matrix:
        :param libors:
        :return:
        """
        def calculate_balance(mask):
            self.balances[period] = self.assets['value'][mask].sum()

        def calculate_losses(mask):
            self.losses[period] = self.assets['value'][mask].sum()

        def calculate_spread(mask):
            spread = (self.assets['value'][mask] * self.assets['spread'][mask] / self.balances[0])
            self.spreads[period] = spread.sum()

        def calculate_recovery(mask):
            if period + self.recovery_step < periods_length:
                recovery = (self.assets['value'][mask] *
                            self.assets['rr_multiplier'][mask] *
                            self.assets['fixed_rr'][mask])
                self.recoveries[period + self.recovery_step] = recovery.sum()

        def calculate_repayments(mask):
            self.repayments[period] = self.assets['value'][mask].sum()

        def calculate_available_interests():
            interest = (self.balances[period - 1] *
                        periods_num *
                        (libors[period - 1] + self.spreads[period - 1]))
            self.available_interests[period] = max(0, interest)

        recent_maturing_assets_mask = ((self.assets['maturity'] <= period_date) &
                                       (self.assets['maturity'] > previous_period_date))
        non_maturing_assets_mask = self.assets['maturity'] > period_date
        defaulting_assets_mask = defaults_matrix[period, :]
        performing_assets_mask = ~defaulting_assets_mask
        # newly defaulting assets are assets that defaulted only this period, but performed last one
        new_defaults_mask = ~defaults_matrix[period - 1, :] & defaulting_assets_mask

        if period == 1:
            # first period we need to do the calculation for all assets to include also assets
            # maturing before the first period
            calculate_balance(performing_assets_mask)
            calculate_recovery(new_defaults_mask)
        else:
            calculate_balance(performing_assets_mask & non_maturing_assets_mask)
            calculate_recovery(new_defaults_mask & non_maturing_assets_mask)

        calculate_spread(performing_assets_mask &
                         (recent_maturing_assets_mask | non_maturing_assets_mask))
        calculate_losses(new_defaults_mask &
                         (recent_maturing_assets_mask | non_maturing_assets_mask))
        calculate_repayments(performing_assets_mask & recent_maturing_assets_mask)
        calculate_available_interests()

    def _period_amortizing_cash_flows(self, periods_length, periods_num, period, period_date,
                                      previous_period_date, defaults_matrix,
                                      libors):
        """
        Calculates the cash flows of a single period.
        :param periods_length:
        :param periods_num:
        :param period:
        :param period_date:
        :param previous_period_date:
        :param defaults_matrix:
        :param libors:
        :return:
        """
        def calculate_balance(mask):
            self.balances[period] = self.assets['value'][mask].sum() * self.amortizations[period - 1][mask]

        def calculate_losses(mask):
            self.losses[period] = self.assets['value'][mask].sum() * self.amortizations[period - 1][mask]

        def calculate_spread(mask):
            spread = (self.assets['value'][mask] * self.assets['spread'][mask] *
                      self.amortizations[period - 1][mask] / self.balances[0])
            self.spreads[period] = spread.sum()

        def calculate_recovery(mask):
            if period + self.recovery_step < periods_length:
                recovery = (self.assets['value'][mask] *
                            self.assets['rr_multiplier'][mask] *
                            self.assets['fixed_rr'][mask] *
                            self.amortizations[period - 1][mask])
                self.recoveries[period + self.recovery_step] = recovery.sum()

        def calculate_repayments(mask):
            self.repayments[period] = (self.assets['value'][mask].sum() *
                                       self.amortizations[period - 1][mask] -
                                       self.amortizations[period][mask])

        def calculate_available_interests():
            interest = (self.balances[period - 1] *
                        periods_num *
                        (libors[period - 1] + self.spreads[period - 1]))
            self.available_interests[period] = max(0, interest)

        recent_maturing_assets_mask = ((self.assets['maturity'] <= period_date) &
                                       (self.assets['maturity'] > previous_period_date))
        non_maturing_assets_mask = self.assets['maturity'] > period_date
        defaulting_assets_mask = defaults_matrix[period, :]
        performing_assets_mask = ~defaulting_assets_mask
        # newly defaulting assets are assets that defaulted only this period, but performed last one
        new_defaults_mask = ~defaults_matrix[period - 1, :] & defaulting_assets_mask

        if period == 1:
            # first period we need to do the calculation for all assets to include also assets
            # maturing before the first period
            calculate_balance(performing_assets_mask)
            calculate_recovery(new_defaults_mask)
        else:
            calculate_balance(performing_assets_mask & non_maturing_assets_mask)
            calculate_recovery(new_defaults_mask & non_maturing_assets_mask)

        calculate_spread(performing_assets_mask &
                         (recent_maturing_assets_mask | non_maturing_assets_mask))
        calculate_losses(new_defaults_mask &
                         (recent_maturing_assets_mask | non_maturing_assets_mask))
        calculate_repayments(performing_assets_mask)
        calculate_available_interests()

    def calculate_cash_flows(self, periods_length, periods, defaults_matrix, libors):
        """
        Calculate the cash flows.
        :param periods_length:
        :param periods:
        :param assets:
        :param defaults_matrix:
        :param libors:
        :return:
        """
        if self.amortizations:
            for period, period_date in enumerate(periods):
                self._period_amortizing_cash_flows(periods_length, len(periods_length), period,
                                                   period_date, periods[period - 1], self.assets,
                                                   defaults_matrix, libors)
        else:
            for period, period_date in enumerate(periods):
                self._periodic_cash_flows(periods_length, len(periods_length), period, period_date,
                                          periods[period - 1], self.assets, defaults_matrix, libors)


class MCPool(Pool):
    """
    The `MCPool` model represents the pool of assets used in case of monte carlo simulation
    to calculate the cash flows.
    """
    LOG_NORMAL = 'log_normal'
    INVERSE_GAUSSIAN = 'inverse_gaussian'

    def __init__(self, periods_length, initial_balance, distribution, mu, sigma, spread=0,
                 recovery_step=None, recovery_rate=None, repayment_rates=None, timing_vector=None):

        if distribution == self.LOG_NORMAL:
            self.default_rate = np.random.lognormal(mu, sigma)
        elif distribution == self.INVERSE_GAUSSIAN:
            self.default_rate = np.random.wald(mu, sigma)

        super(MCPool, self).__init__(periods_length=periods_length,
                                     initial_balance=initial_balance,
                                     spread=spread,
                                     recovery_step=recovery_step,
                                     recovery_rate=recovery_rate,
                                     repayment_rates=repayment_rates,
                                     timing_vector=timing_vector)

    def calculate_cash_flows(self, periods_length, periods_num):
        """
        Calculate the cash flows.
        :param periods_length:
        :param periods_num:
        :return:
        """
        total_repayments = 0
        total_loss = 0
        for period in xrange(1, periods_length):
            if self.balances[period - 1] > 0:
                expected_loss = self.balances[0] * self.default_rate * self.timing_vector[period - 1]
                self.losses[period] = max(0, expected_loss)
                total_loss += expected_loss

            total_repayments += self.repayment_rates[period - 1] * self.balances[0]
            available_interest = ((self.balances[period - 1] - self.losses[period]) *
                                  (self.libors[period - 1] + self.spreads[period - 1]) /
                                  periods_num)
            self.available_interests[period] = max(0, available_interest)

            if self.balances[period - 1]:
                balance = (self.balances[0] - total_loss) * (100 - total_repayments) / 100
                self.balances[period] = max(0, balance)
                repayment = ((self.balances[period - 1] - self.balances[period]) *
                             (self.default_rate * self.timing_vector[period - 1] * self.balances[period]))
                self.repayments[period] = max(0, repayment)
