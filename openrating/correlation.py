class Correlation(object):
    """
    An asset correlation estimated across a range of asset types that are consistent with the
    long-run historical performance and risk profile of these assets.
    """
    def __init__(self, base, regions, countries, industry_sectors, industries):
        """
        :param base: the global base correlation value
        :param regions: (dict) regions correlation values by ids
        :param countries: (dict) countries correlation values by ids
        :param industry_sectors: (dict) industry sectors correlation values by ids
        :param industries: (dict) industries correlation values by ids
        :return:
        """
        self.base = base
        self.regions = regions
        self.countries = countries
        self.industry_sectors = industry_sectors
        self.industries = industries

    def get_correlation(self, issuer1, issuer2):
        """
        Returns the correlation between two issuers.
        :param issuer1: (namedtuple) (region, country, industry_sector, industry)
        :param issuer2: (namedtuple) (region, country, industry_sector, industry)
        :return:
        """
        correlation = self.base
        if issuer1.region == issuer2.region:
            correlation += self.regions[issuer1.region]
        if issuer1.country == issuer2.country:
            correlation += self.countries[issuer1.country]
        if issuer1.industry_sector == issuer2.industry_sector:
            correlation += self.industry_sectors[issuer1.industry_sector]
        if issuer1.industry == issuer2.industry:
            correlation += self.industries[issuer1.industry]
        return correlation

    def get_correlation_matrix(self, assets):
        asset_len = assets.size
        correlation = []
        for i in xrange(asset_len):
            i_correlation = [1]
            for j in xrange(i + 1, asset_len):
                i_correlation.append(self.get_correlation(assets.ix[i, :], assets.ix[j, :]))
            correlation.append(i_correlation)

        return correlation
