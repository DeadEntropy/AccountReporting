from datetime import datetime

from bkanalysis.managers.data_manager import DataManager
from bkanalysis.managers.market_manager import MarketManager
from bkanalysis.managers.transformation_manager import TransformationManager


class TransformationManagerCache(TransformationManager):
    """an implementation of Transformation Manger that caches data"""

    def __init__(
        self,
        data_manager: DataManager,
        market_manager: MarketManager,
        year: str,
        account: str = None,
        hows: list[str] = None,
        include_iat: bool = False,
        include_full_types: bool = True,
    ):
        TransformationManager.__init__(self, data_manager, market_manager)

        self._account = account
        self._hows = hows if hows is not None else ["both"]
        self._include_iat = include_iat
        self._include_full_types = include_full_types

        self._total_flow_value = TransformationManager.get_flow_values(self)
        self._populate_cache(year)

    def _populate_cache(self, year):
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)

        self._year = year
        self._flow_values = {
            how: TransformationManager.get_flow_values(
                self, start_date, end_date, self._account, how, self._include_iat, self._include_full_types
            )
            for how in self._hows
        }

    def get_flow_values(
        self,
        date_start: datetime = None,
        date_end: datetime = None,
        account: str = None,
        how: str = "both",
        include_iat: bool = False,
        include_full_types: bool = True,
    ):
        if date_start is None and date_end is None and account is None and how == "both" and not include_iat and include_full_types:
            return self._total_flow_value.copy(deep=True)

        if date_start.year != date_end.year:
            raise ValueError(f"invalid start_date, date_end must be same year but was: {date_start.year} vs {date_end.year}")

        if date_end.year != self._year:
            self._populate_cache(date_end.year)
        if account != self._account:
            raise ValueError("invalid account")
        if how not in self._hows:
            raise ValueError(f"how={how} was not cached.")
        if include_iat != self._include_iat:
            raise ValueError("invalid include_iat")
        if include_full_types != self._include_full_types:
            raise ValueError("invalid include_full_types")

        return self._flow_values[how].copy(deep=True)
