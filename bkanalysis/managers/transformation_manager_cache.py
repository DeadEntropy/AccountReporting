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
        self._year = year
        self._account = account
        self._hows = hows if hows is not None else ["both"]
        self._include_iat = include_iat
        self._include_full_types = include_full_types

        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)

        self._total_flow_value = TransformationManager.get_flow_values(self)

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
            return self._total_flow_value

        assert date_start.year == self._year, f"invalid start_date: {date_start.year} vs {self._year}"
        assert date_end.year == self._year, f"invalid end_date: {date_start.year} vs {self._year}"
        assert account == self._account, "invalid account"
        assert how in self._hows, f"how={how} was not cached."
        assert include_iat == self._include_iat, "invalid include_iat"
        assert include_full_types == self._include_full_types, "invalid include_full_types"

        return self._flow_values[how]
