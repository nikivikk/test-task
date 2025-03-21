import pandas as pd
import numpy as np


class Processor:

    def __init__(self,
                 data: pd.DataFrame,
                 applications: pd.DataFrame,
                 *args,
                 **kwargs):
        self.data = data
        self.applications = applications

    def _process_field(
            self,
            _id: int,
            field_name: str,
            year: int,
            prev: int = 0,
            last_available: int = 0,
            *args,
            **kwargs,
    ):

        target_year = year - prev

        if last_available == 0:
            return self.data.loc[_id].get(f"{target_year}, {field_name}",
                                          np.nan)

        columns_to_check = [
            f"{target_year - i}, {field_name}" for i in range(last_available + 1)
        ]
        fields_series = self.data.loc[_id].reindex(columns_to_check)
        if fields_series.empty:
            return np.nan
        return fields_series.bfill().iloc[0]

    def _calculate_profit_from_sales(self, *args, **kwargs):
        revenue = float(self._process_field(*args,
                                            field_name="Выручка, RUB",
                                            **kwargs)
                        )
        cost_of_sales = float(
            self._process_field(*args, field_name="Себестоимость продаж, RUB",
                                **kwargs)
        )
        management_costs = float(
            self._process_field(
                *args, field_name="Управленческие расходы, RUB", **kwargs
            )
        )
        commercial_costs = float(
            self._process_field(*args,
                                field_name="Коммерческие расходы, RUB",
                                **kwargs)
        )
        if pd.isna(revenue):
            return np.nan

        return revenue - cost_of_sales - management_costs - commercial_costs

    def _get_company_age(self, _id, year: int, *args, **kwargs):
        registration_date = pd.to_datetime(self.data.loc[_id, "Дата регистрации"])
        company_age = year - registration_date.year
        if company_age < 0:
            return np.nan
        return company_age

    def get_data(self, request: list) -> pd.DataFrame:
        output = pd.DataFrame()
        for query in request:
            field_name = query.get("field_name")
            prev = query.get("prev", 0)
            last_available = query.get("last_available", 0)

            postfix = (
                f"{('Prev' if prev > 0 else 'Next') * abs(prev)} "
                f"{f'LA{last_available}' if last_available != 0 else ''}"
            ).strip()
            output_column = f"{field_name} {postfix}".strip()

            if field_name == "Прибыль (убыток) от продажи, RUB":
                output[output_column] = self.applications.apply(
                    lambda row: self._calculate_profit_from_sales(
                        _id=row["_id"],
                        year=row["year"],
                        prev=prev,
                        last_available=last_available,
                    ),
                    axis=1,
                )
            elif field_name == "Возраст компании, years":
                output[output_column] = self.applications.apply(
                    lambda row: self._get_company_age(_id=row["_id"],
                                                      year=row["year"]
                                                      ),
                    axis=1,
                )
            else:
                output[output_column] = self.applications.apply(
                    lambda row: self._process_field(
                        _id=row["_id"],
                        year=row["year"],
                        field_name=field_name,
                        prev=prev,
                        last_available=last_available,
                    ),
                    axis=1,
                )

        return output
