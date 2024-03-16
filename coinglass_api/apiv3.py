from typing import Optional

import pandas as pd
import requests

from .exceptions import (
    CoinglassAPIError,
    CoinglassRequestError,
    NoDataReturnedError,
    RateLimitExceededError,
)
from .parameters import CoinglassParameterValidation

class CoinglassAPIv3(CoinglassParameterValidation):
    """ Unofficial Python client for Coinglass API """

    def __init__(self, coinglass_secret: str):
        """
        Args:
            coinglass_secret: key from Coinglass, get one at
            https://www.coinglass.com/pricing
        """

        super().__init__()

        self.__coinglass_secret = coinglass_secret
        self._base_url = "https://open-api-v3.coinglass.com/api/futures/"
        self._session = requests.Session()

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        if params:
            self.validate_params(params)

        headers = {
            "accept": "application/json",
            "CG-API-KEY": self.__coinglass_secret
        }
        url = self._base_url + endpoint
        return self._session.request(
            method='GET',
            url=url,
            params=params,
            headers=headers,
            timeout=30
        ).json()

    @staticmethod
    def _create_dataframe(
            data: list[dict],
            time_col: str | None = None,
            unit: str | None = "s",
            cast_objects_to_numeric: bool = False
    ) -> pd.DataFrame:
        """
        Create pandas DataFrame from a list of dicts

        Args:
            data: list of dicts
            time_col: name of time column in dict
            unit: unit of time column, specify None to use auto-resolver (default: s)
            cast_objects_to_numeric: cast all object columns to numeric (default: False)

        Returns:
            pandas DataFrame
        """
        df = pd.DataFrame(data)

        if time_col:
            if time_col == "time":
                # Handle edge case of time column being named "time"
                df.rename(columns={"time": "t"}, inplace=True)
                time_col = "t"

            df["time"] = pd.to_datetime(df[time_col], unit=unit)
            df.drop(columns=[time_col], inplace=True)
            df.set_index("time", inplace=True, drop=True)

            if "t" in df.columns:
                # Drop additional "t" column if it exists
                df.drop(columns=["t"], inplace=True)

        if cast_objects_to_numeric:
            cols = df.columns[df.dtypes.eq('object')]
            df[cols] = df[cols].apply(pd.to_numeric)

        return df

    @staticmethod
    def _create_multiindex_dataframe(
            data: list[dict],
            list_key: str
    ) -> pd.DataFrame:
        """
        Create MultiIndex pandas DataFrame from a list of nested dicts

        Args:
            data: list of nested dicts
            list_key: key in dict that contains list of dicts

        Returns:
            dict of pandas DataFrame
        """
        flattened_data = {}

        # Flatten nested dicts
        for symbol_data in data:
            flattened_dict = {}
            for outer_key, outer_value in symbol_data.items():
                if isinstance(outer_value, list):
                    for exchange in outer_value:
                        ex = exchange["exchangeName"]
                        for inner_key, value in exchange.items():
                            flattened_dict[(outer_key, ex, inner_key)] = value
                else:
                    flattened_dict[outer_key] = outer_value

            # Remove non-tuple keys
            remove_keys = []
            for key in list(flattened_dict.keys()):
                if not isinstance(key, tuple):
                    remove_keys.append(key)

            for k in remove_keys:
                flattened_dict.pop(k, None)

            df = pd.DataFrame.from_dict(flattened_dict, orient="index")
            df.index = pd.MultiIndex.from_tuples(df.index)

            flattened_data[symbol_data[list_key]] = df

        return pd.concat(flattened_data, axis=1)

    @staticmethod
    def _flatten_dictionary(data: dict) -> dict:
        flattened_dict = {}

        for outer_key, outer_value in data.items():
            if isinstance(outer_value, dict):
                for inner_key, inner_value in outer_value.items():
                    if isinstance(inner_value, list):
                        flattened_dict[(outer_key, inner_key)] = inner_value
                    else:
                        flattened_dict[inner_key] = inner_value
            else:
                flattened_dict[(outer_key, 0)] = outer_value

        return flattened_dict

    @staticmethod
    def _check_for_errors(response: dict) -> None:
        """ Check for errors in response """

        # Handle case where unable to communicate with API
        if "success" not in response:
            raise CoinglassAPIError(
                status=response["status"],
                err=response["error"]
            )

        # Handle case where API response is unsuccessful
        if not response["success"]:
            code, msg = int(response["code"]), response["msg"]
            match code:
                case 50001:
                    raise RateLimitExceededError()
                case _:
                    raise CoinglassRequestError(code=code, msg=msg)

        # Handle case where API returns no data
        if "data" not in response:
            raise NoDataReturnedError()

    def supported_coins(self, )->list:
        """
        Args:

        Returns:
            list of acceptable coins
        """
        response = self._get(
            endpoint="supported-coins",
            params=None,
        )
        self._check_for_errors(response)
        data = response["data"]
        return data

    def supported_exchanges_pairs(self, )->dict:
        """
        Query an API to get the acceptable trading currency pairs for cryptocurrency exchanges.

        This function makes a call to an external API, which provides the available trading pairs in the form of a dict of dicts. Each key in the main dictionary is an exchange name, and the value is another dictionary with details about the trading pair, including the instrument ID, the base asset, and the quote asset.

        Returns:
            A dictionary of dictionaries containing the trading pairs information for the specified cryptocurrency exchange. The structure is as follows:
            {
                exchange_name: {
                    instrumentId: str,  # Unique identifier for the trading pair
                    baseAsset: str,     # The base currency in the trading pair
                    quoteAsset: str,    # The quote currency in the trading pair
                }
            }
            
            The `exchange_name` is a placeholder for the actual name of the exchange. Each `exchange_name` key maps to another dictionary that holds details about the trading pair.

        No arguments are required for this function as it queries for a pre-defined or dynamically determined target cryptocurrency exchange.

        Examples:
            To use this function, simply call it without any arguments:
            ```
            trading_pairs_info = query_trading_pairs()
            ```

            Then, `trading_pairs_info` will contain the dictionary structure as described in the Returns section.
        """
        response = self._get(
            endpoint="supported-exchange-pairs",
            params=None,
        )
        self._check_for_errors(response)
        data = response["data"]
        return data
    
    def ohlc_history(self, exchange:str, symbol:str, interval:str, limit:int=1000, startTime:int=None, endTime:int=None)->pd.DataFrame:
        """
        Query OHLC (Open, High, Low, Close) data for a specific financial instrument.

        This function fetches OHLC data from a designated source or API, structured to provide insights into the price movements within a specified time frame. The returned data is organized into a pandas DataFrame, facilitating further analysis and visualization. The columns of the DataFrame are as follows:

        - t: Timestamp representing the start of the time interval.
        - o: Open price at the start of the time interval.
        - h: Highest price during the time interval.
        - l: Lowest price during the time interval.
        - c: Close price at the end of the time interval.

        Returns:
            pandas.DataFrame: A DataFrame with columns ['t', 'o', 'h', 'l', 'c'] representing the timestamp, open, high, low, and close prices, respectively.

        Example:
            To query the OHLC data, you might use the function as follows:
            ```
            ohlc_data = ohlc_history()
            ```
            `ohlc_data` will then contain the OHLC data structured in a pandas DataFrame.
            
        Note:
            Depending on the implementation, this function might require additional arguments such as the instrument identifier, the time frame for the data, or the start and end dates for the query. Ensure these details are properly documented if applicable to your function's design.
        """        
        endpoint = "openInterest/ohlc-history?"
        response = self._get(
            endpoint=endpoint,
            params=dict(exchange=exchange,symbol=symbol,interval=interval, limit=limit, startTime=startTime, endTime=endTime),
        )
        self._check_for_errors(response)
        data = response["data"]
        df = self._create_dataframe(data, cast_objects_to_numeric=True)
        return df

    def liquidation_map(self, exchange:str, symbol: str, range: str) -> tuple[pd.DataFrame, float]:
        """
        Returns:
            pandas.DataFrame: A DataFrame with columns ['liqprice', 'liqusd', 'leverage', 'none'] representing the liquidation price, liquidation value in USD, Leverage Ratio and null respectively.
            float: last price
        """
        response = self._get(
            endpoint="liquidation/map?",
            params={"exchange":exchange, "symbol": symbol, "range": range}
        )
        self._check_for_errors(response)
        liq = response["data"]

        last_price, liq_map = liq['lastPrice'], liq['data']
        df_liq = pd.DataFrame([k for liq in liq_map.values() for k in liq])
        cols = ['liqprice', 'liqusd', 'leverage', 'none']
        df_liq.columns = cols        
        return df_liq, last_price
    
    def liquidation_heatmap(self, exchange:str, symbol: str, range: str) -> tuple[pd.DataFrame, float]:
        """
        Args:

        Returns:
            pandas.DataFrame: a dataframe for liquidation containing cols [t, price, luqusd]
            pandas.DataFrame: a dataframe for price containing cols ['t', 'o', 'h', 'l', 'c', 'NotSure']
        """
        response = self._get(
            endpoint="liquidation/heatmap?",
            params={"exchange":exchange, "symbol": symbol, "range": range}
        )
        self._check_for_errors(response)
        liqh = response["data"]
        TIME = 't'
        COLS_TICK = [TIME, 'o', 'h', 'l', 'c', 'NotSure']

        df_liq = pd.DataFrame(liqh['liq'])
        df_liq.columns = ['x', 'y', 'liqusd']
        df_ticks = pd.DataFrame(liqh['prices'])
        df_ticks.columns = COLS_TICK

        # convert x to time: merge x to tick time
        df_liq = df_liq.merge(df_ticks[['t']], left_on='x', right_index=True)

        # convert y to price
        df_price = pd.DataFrame(liqh['y'], columns=['price'])
        df_liq = df_liq.merge(df_price, left_on='y', right_index=True)
        df_liq = df_liq[[TIME, 'price', 'liqusd']]
        return df_liq, df_ticks
        

    # def perpetual_market(self, symbol: str) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="perpetual_market",
    #         params={"symbol": symbol}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"][symbol]
    #     return self._create_dataframe(data)

    # def futures_market(self, symbol: str) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="futures_market",
    #         params={"symbol": symbol}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"][symbol]
    #     return self._create_dataframe(data)

    # def funding_rate(self) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="funding",
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_multiindex_dataframe(data, list_key="symbol")

    # def funding_usd_history(self, symbol: str, time_type: str) -> list[dict]:
    #     """
    #     Get funding history in USD for a coin

    #     Args:
    #         symbol: Coin symbol (e.g. BTC)
    #         time_type: Time type (e.g. m1, m5, h8)

    #     Returns:
    #         List of dicts
    #     """
    #     response = self._get(
    #         endpoint="funding_usd_history",
    #         params={"symbol": symbol, "time_type": time_type}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return data

    # def funding_coin_history(self, symbol: str, time_type: str) -> list[dict]:
    #     """
    #     Get funding history in coin for a coin

    #     Args:
    #         symbol: Coin symbol (e.g. BTC)
    #         time_type: Time type (e.g. m1, m5, h8)

    #     Returns:
    #         List of dicts
    #     """
    #     response = self._get(
    #         endpoint="funding_coin_history",
    #         params={"symbol": symbol, "time_type": time_type}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return data

    # def open_interest(self, symbol: str) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="open_interest",
    #         params={"symbol": symbol}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data)

    # def open_interest_history(
    #         self,
    #         symbol: str,
    #         time_type: str,
    #         currency: str
    # ) -> pd.DataFrame:
    #     """
    #     Get open interest history

    #     Args:
    #         symbol: Coin symbol (e.g. BTC)
    #         time_type: Time type (e.g. m1, m5, h8, all)
    #         currency: Currency (e.g. USD or symbol)

    #     Returns:
    #         pandas DataFrame
    #     """
    #     response = self._get(
    #         endpoint="open_interest_history",
    #         params={"symbol": symbol, "time_type": time_type, "currency": currency}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]

    #     flattened_dict = {}

    #     for k, v in data.items():
    #         if isinstance(v, dict):
    #             for outer_key, outer_value in v.items():
    #                 if isinstance(outer_value, list):
    #                     flattened_dict[(k, outer_key)] = outer_value
    #                 else:
    #                     flattened_dict[outer_key] = outer_value
    #         else:
    #             flattened_dict[(k, 0)] = v

    #     df = pd.DataFrame(flattened_dict)
    #     df["time"] = pd.to_datetime(df["dateList"][0], unit="ms")
    #     df.drop(columns=["dateList"], inplace=True)
    #     df.set_index("time", inplace=True, drop=True)
    #     return df

    # def option(self, symbol: str) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="option",
    #         params={"symbol": symbol}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data)

    # def option_history(self, symbol: str, currency: str) -> pd.DataFrame:
    #     """
    #     Get option history

    #     Args:
    #         symbol: Coin symbol (e.g. BTC)
    #         currency: Currency (e.g. USD or symbol)

    #     Returns:
    #         pandas DataFrame
    #     """
    #     response = self._get(
    #         endpoint="option_history",
    #         params={"symbol": symbol, "currency": currency}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     df = pd.DataFrame(self._flatten_dictionary(data[0]))
    #     df["time"] = pd.to_datetime(df["dateList"][0], unit="ms")
    #     df.drop(columns=["dateList"], inplace=True, level=0)
    #     df.set_index("time", inplace=True, drop=True)
    #     return df

    # def option_vol_history(self, symbol: str, currency: str) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="option/vol/history",
    #         params={"symbol": symbol, "currency": currency}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     df = pd.DataFrame(self._flatten_dictionary(data))
    #     df["time"] = pd.to_datetime(df["dateList"][0], unit="ms")
    #     df.drop(columns=["dateList"], inplace=True, level=0)
    #     df.set_index("time", inplace=True, drop=True)
    #     return df

    # def top_liquidations(self, time_type: str) -> pd.DataFrame:
    #     """
    #     Get top liquidations

    #     Args:
    #         time_type: Time type (e.g. h1, h4, h12, h24)

    #     Returns:
    #         pandas DataFrame
    #     """
    #     response = self._get(
    #         endpoint="liquidation_top",
    #         params={"time_type": time_type}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data)


    # def liquidation_info(self, symbol: str, time_type: str) -> dict:
    #     response = self._get(
    #         endpoint="liquidation_info",
    #         params={"symbol": symbol, "time_type": time_type}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return data

    # def liquidation_order(
    #         self,
    #         ex_name: str,
    #         coin: str,
    #         vol_usd: str,
    #         start_time: int,
    #         end_time: int
    # ) -> dict:
    #     response = self._get(
    #         endpoint="liqMap",
    #         params={"ex_name": ex_name, "coin": coin, "vol_usd": vol_usd,
    #                 "start_time": start_time, "end_time": end_time}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return data

    # def exchange_liquidations(self, symbol: str, time_type: str) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="liquidation_ex",
    #         params={"symbol": symbol, "time_type": time_type}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data)

    # def liquidations_history(self, symbol: str, time_type: str) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="liquidation_history",
    #         params={"symbol": symbol, "time_type": time_type}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     # TODO: Improve formatting
    #     return self._create_multiindex_dataframe(data, list_key="createTime")

    # def exchange_long_short_ratio(self, symbol: str, time_type: str) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="long_short",
    #         params={"symbol": symbol, "time_type": time_type}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     # TODO: Improve formatting
    #     return self._create_multiindex_dataframe(data, list_key="symbol")

    # def long_short_ratio_history(self, symbol: str, time_type: str) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="long_short_history",
    #         params={"symbol": symbol, "time_type": time_type}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="dateList")

    # def futures_coins_markets(self) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="futures_coins_markets",
    #         params={}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data)

    # def futures_coins_price_change(self) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="futures_coins_price_change",
    #         params={}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data)

    # def futures_basis_chart(self, symbol: str) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="futures_basis_chart",
    #         params={"symbol": symbol}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]

    #     flattened_data = {}

    #     # Flatten nested dicts
    #     for symbol_data in data:
    #         flattened_dict = {}
    #         for outer_key, outer_value in symbol_data.items():
    #             if isinstance(outer_value, dict):
    #                 for inner_key, value in outer_value.items():
    #                     flattened_dict[(outer_key, inner_key)] = value
    #             else:
    #                 flattened_dict[outer_key] = outer_value

    #         # Remove non-tuple keys
    #         remove_keys = []
    #         for key in list(flattened_dict.keys()):
    #             if not isinstance(key, tuple):
    #                 remove_keys.append(key)

    #         for k in remove_keys:
    #             flattened_dict.pop(k, None)

    #         df = pd.DataFrame.from_dict(flattened_dict, orient="index")
    #         df.index = pd.MultiIndex.from_tuples(df.index)

    #         flattened_data[symbol_data["exName"]] = df

    #     return pd.concat(flattened_data, axis=1)

    # def futures_vol(self, symbol: str, time_type: str) -> pd.DataFrame:
    #     """
    #     Get futures volume

    #     Args:
    #         symbol: Coin symbol (e.g. BTC, ETH, LTC, etc.)
    #         time_type: Time type (e.g. h1, h4, h12, h24, all)

    #     Returns:
    #         pandas DataFrame
    #     """
    #     response = self._get(
    #         endpoint="futures_vol",
    #         params={"symbol": symbol, "time_type": time_type}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     df = pd.DataFrame(self._flatten_dictionary(data))
    #     df["time"] = pd.to_datetime(df["dateList"][0], unit="ms")
    #     df.drop(columns=["dateList"], inplace=True, level=0)
    #     df.set_index("time", inplace=True, drop=True)
    #     return df

    # def funding(
    #         self,
    #         ex: str,
    #         pair: str,
    #         interval: str,
    #         limit: int = 500,
    #         start_time: Optional[int] = None,
    #         end_time: Optional[int] = None
    # ) -> pd.DataFrame:
    #     """
    #     Funding rate for a given pair

    #     Args:
    #         ex: exchange to get funding rate (e.g. Binance, dYdX, etc.)
    #         pair: pair to get funding rate (e.g. BTCUSDT on Binance, BTC-USD on dYdX)
    #         interval: interval to get funding rate (e.g. m1, m5, m15, m30, h1, h4, etc.)
    #         limit: number of data points to return (default: 500)
    #         start_time: start time in milliseconds
    #         end_time: end time in milliseconds

    #     Returns:
    #         pandas DataFrame with funding rate
    #     """
    #     response = self._get(
    #         endpoint="indicator/funding",
    #         params={"ex": ex, "pair": pair, "interval": interval, "limit": limit,
    #                 "start_time": start_time, "end_time": end_time}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="createTime")

    # def funding_ohlc(
    #         self,
    #         ex: str,
    #         pair: str,
    #         interval: str,
    #         limit: int = 500,
    #         start_time: Optional[int] = None,
    #         end_time: Optional[int] = None
    # ) -> pd.DataFrame:
    #     """
    #     Funding rate in OHLC format for an exchange pair

    #     Args:
    #         ex: exchange to get funding rate (e.g. Binance, dYdX, etc.)
    #         pair: pair to get funding rate (e.g. BTCUSDT on Binance, BTC-USD on dYdX)
    #         interval: interval to get funding rate (e.g. m1, m5, m15, m30, h1, h4)
    #         limit: number of data points to return (default: 500)
    #         start_time: start time in milliseconds
    #         end_time: end time in milliseconds

    #     Returns:
    #         pandas DataFrame with funding rate in OHLC format for an exchange pair
    #     """
    #     response = self._get(
    #         endpoint="indicator/funding_ohlc",
    #         params={"ex": ex, "pair": pair, "interval": interval, "limit": limit,
    #                 "start_time": start_time, "end_time": end_time}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="t")

    # def funding_average(
    #         self,
    #         symbol: str,
    #         interval: str,
    #         limit: int = 500,
    #         start_time: Optional[int] = None,
    #         end_time: Optional[int] = None
    # ) -> pd.DataFrame:
    #     """
    #     Average funding rate for a symbol

    #     Args:
    #         symbol: symbol to get funding rate for
    #         interval: interval to get funding rate (e.g. m1, m5, m15, m30, h1, h4, etc.)
    #         limit: number of data points to return (default: 500)
    #         start_time: start time in milliseconds
    #         end_time: end time in milliseconds

    #     Returns:
    #         pandas DataFrame with funding rate
    #     """
    #     response = self._get(
    #         endpoint="indicator/funding_avg",
    #         params={"symbol": symbol, "interval": interval, "limit": limit,
    #                 "start_time": start_time, "end_time": end_time}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="createTime")

    # def open_interest_ohlc(
    #         self,
    #         ex: str,
    #         pair: str,
    #         interval: str,
    #         limit: int = 500,
    #         start_time: Optional[int] = None,
    #         end_time: Optional[int] = None
    # ) -> pd.DataFrame:
    #     """
    #     Open interest in OHLC format for an exchange pair

    #     Args:
    #         ex: exchange to get OI for (e.g. Binance, dYdX, etc.)
    #         pair: pair to get OI for (e.g. BTCUSDT on Binance, BTC-USD on dYdX, etc.)
    #         interval: interval to get OI for (e.g. m1, m5, m15, m30, h1, h4, etc.)
    #         limit: number of data points to return (default: 500)
    #         start_time: start time in milliseconds
    #         end_time: end time in milliseconds

    #     Returns:
    #         pandas DataFrame with open interest in OHLC format for an exchange pair
    #     """
    #     response = self._get(
    #         endpoint="indicator/open_interest_ohlc",
    #         params={"ex": ex, "pair": pair, "interval": interval, "limit": limit,
    #                 "start_time": start_time, "end_time": end_time}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="t")

    # def open_interest_aggregated_ohlc(
    #         self,
    #         symbol: str,
    #         interval: str,
    #         limit: int = 500,
    #         start_time: Optional[int] = None,
    #         end_time: Optional[int] = None
    # ) -> pd.DataFrame:
    #     """
    #     Aggregated open interest in OHLC format for a symbol

    #     Args:
    #         symbol: symbol to get OI for
    #         interval: interval to get OI for (e.g. m1, m5, m15, m30, h1, h4, etc.)
    #         limit: number of data points to return (default: 500)
    #         start_time: start time in milliseconds
    #         end_time: end time in milliseconds

    #     Returns:
    #         pandas DataFrame with aggregated open interest in OHLC format
    #     """
    #     response = self._get(
    #         endpoint="indicator/open_interest_aggregated_ohlc",
    #         params={"symbol": symbol, "interval": interval, "limit": limit,
    #                 "start_time": start_time, "end_time": end_time}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="t")

    # def liquidation_symbol(
    #         self,
    #         symbol: str,
    #         interval: str,
    #         limit: int = 500,
    #         start_time: Optional[int] = None,
    #         end_time: Optional[int] = None
    # ) -> pd.DataFrame:
    #     """
    #     Liquidation data for a symbol

    #     Args:
    #         symbol: symbol to get liquidation data for
    #         interval: interval to get liquidation data (e.g. m1, m5, m15, m30, h1, h4)
    #         limit: number of data points to return (default: 500)
    #         start_time: start time in milliseconds
    #         end_time: end time in milliseconds

    #     Returns:
    #         pandas DataFrame with liquidation data
    #     """
    #     response = self._get(
    #         endpoint="indicator/liquidation_symbol",
    #         params={"symbol": symbol, "interval": interval, "limit": limit,
    #                 "start_time": start_time, "end_time": end_time}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="createTime")

    # def liquidation_pair(
    #         self,
    #         ex: str,
    #         pair: str,
    #         interval: str,
    #         limit: int = 500,
    #         start_time: Optional[int] = None,
    #         end_time: Optional[int] = None
    # ) -> pd.DataFrame:
    #     """
    #     Liquidation data for an exchange pair

    #     Args:
    #         ex: exchange to get liquidation data for (e.g. Binance, dYdX, etc.)
    #         pair: pair to get liquidation data (e.g. BTCUSDT on Binance,BTC-USD on dYdX)
    #         interval: interval to get liquidation data (e.g. m1, m5, m15, m30, h1, h4)
    #         limit: number of data points to return (default: 500)
    #         start_time: start time in milliseconds
    #         end_time: end time in milliseconds

    #     Returns:
    #         pandas DataFrame with liquidation data for an exchange pair
    #     """
    #     response = self._get(
    #         endpoint="indicator/liquidation_pair",
    #         params={"ex": ex, "pair": pair, "interval": interval, "limit": limit,
    #                 "start_time": start_time, "end_time": end_time}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="t")

    # def long_short_accounts(
    #         self,
    #         ex: str,
    #         pair: str,
    #         interval: str,
    #         limit: int = 500,
    #         start_time: Optional[int] = None,
    #         end_time: Optional[int] = None
    # ) -> pd.DataFrame:
    #     """
    #     Long/short ratio for an exchange pair

    #     Args:
    #         ex: exchange to get liquidation data for (e.g. Binance, dYdX, etc.)
    #         pair: pair to get liquidation data (e.g. BTCUSDT on Binance,BTC-USD on dYdX)
    #         interval: interval to get liquidation data (e.g. m1, m5, m15, m30, h1, h4)
    #         limit: number of data points to return (default: 500)
    #         start_time: start time in milliseconds
    #         end_time: end time in milliseconds

    #     Returns:
    #         pandas DataFrame with long/short ratio for an exchange pair
    #     """
    #     response = self._get(
    #         endpoint="indicator/long_short_accounts",
    #         params={"ex": ex, "pair": pair, "interval": interval, "limit": limit,
    #                 "start_time": start_time, "end_time": end_time}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="createTime")

    # def long_short_symbol(
    #         self,
    #         symbol: str,
    #         interval: str,
    #         limit: int = 500,
    #         start_time: Optional[int] = None,
    #         end_time: Optional[int] = None
    # ) -> pd.DataFrame:
    #     """
    #     Long/short ratio for a symbol

    #     Args:
    #         symbol: symbol to get long/short ratio for
    #         interval: interval to get long/short ratio (e.g. m1, m5, m15, m30, h1, h4)
    #         limit: number of data points to return (default: 500)
    #         start_time: start time in milliseconds
    #         end_time: end time in milliseconds

    #     Returns:
    #         pandas DataFrame with long/short ratio
    #     """
    #     response = self._get(
    #         endpoint="indicator/long_short_symbol",
    #         params={"symbol": symbol, "interval": interval, "limit": limit,
    #                 "start_time": start_time, "end_time": end_time}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="t")

    # def top_long_short_account_ratio(
    #         self,
    #         ex: str,
    #         pair: str,
    #         interval: str,
    #         limit: int = 500,
    #         start_time: Optional[int] = None,
    #         end_time: Optional[int] = None
    # ) -> pd.DataFrame:
    #     """
    #     Top accounts long/short ratio for an exchange pair

    #     Args:
    #         ex: exchange to get liquidation data for (e.g. Binance, dYdX, etc.)
    #         pair: pair to get liquidation data (e.g. BTCUSDT on Binance,BTC-USD on dYdX)
    #         interval: interval to get liquidation data (e.g. m1, m5, m15, m30, h1, h4)
    #         limit: number of data points to return (default: 500)
    #         start_time: start time in milliseconds
    #         end_time: end time in milliseconds

    #     Returns:
    #         pandas DataFrame with top accounts long/short ratio for an exchange pair
    #     """
    #     response = self._get(
    #         endpoint="indicator/top_long_short_account_ratio",
    #         params={"ex": ex, "pair": pair, "interval": interval, "limit": limit,
    #                 "start_time": start_time, "end_time": end_time}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="createTime")

    # def top_long_short_position_ratio(
    #         self,
    #         ex: str,
    #         pair: str,
    #         interval: str,
    #         limit: int = 500,
    #         start_time: Optional[int] = None,
    #         end_time: Optional[int] = None
    # ) -> pd.DataFrame:
    #     """
    #     Top positions long/short ratio for an exchange pair

    #     Args:
    #         ex: exchange to get liquidation data for (e.g. Binance, dYdX, etc.)
    #         pair: pair to get liquidation data (e.g. BTCUSDT on Binance,BTC-USD on dYdX)
    #         interval: interval to get liquidation data (e.g. m1, m5, m15, m30, h1, h4)
    #         limit: number of data points to return (default: 500)
    #         start_time: start time in milliseconds
    #         end_time: end time in milliseconds

    #     Returns:
    #         pandas DataFrame with top positions long/short ratio for an exchange pair
    #     """
    #     response = self._get(
    #         endpoint="indicator/top_long_short_position_ratio",
    #         params={"ex": ex, "pair": pair, "interval": interval, "limit": limit,
    #                 "start_time": start_time, "end_time": end_time}
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="createTime")

    # def bitcoin_bubble_index(self) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="index/bitcoin_bubble_index",
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="time", unit=None)

    # def ahr999(self) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="index/ahr999",
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="date", unit=None)

    # def tow_year_ma_multiplier(self) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="index/tow_year_MA_multiplier",
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="createTime")

    # def tow_hundred_week_moving_avg_heatmap(self) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="index/tow_hundred_week_moving_avg_heatmap",
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="createTime")

    # def puell_multiple(self) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="index/puell_multiple",
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="createTime")

    # def stock_flow(self) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="index/stock_flow",
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="createTime", unit=None)

    # def pi(self) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="index/pi",
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="createTime",
    #                                   cast_objects_to_numeric=True)

    # def golden_ratio_multiplier(self) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="index/golden_ratio_multiplier",
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="createTime",
    #                                   cast_objects_to_numeric=True)

    # def bitcoin_profitable_days(self) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="index/bitcoin_profitable_days",
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="createTime")

    # def log_log_regression(self) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="index/log_log_regression",
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data)

    # def grayscale_market_history(self) -> pd.DataFrame:
    #     response = self._get(
    #         endpoint="index/grayscale_market_history",
    #     )
    #     self._check_for_errors(response)
    #     data = response["data"]
    #     return self._create_dataframe(data, time_col="dateList")
