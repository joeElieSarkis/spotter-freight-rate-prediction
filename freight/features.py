"""Features available at quoting time, with training-only city lookups."""

import numpy as np
import pandas as pd

from freight.data import validate_inputs

CATEGORICAL = ["pickup", "delivery", "equipment", "lane", "weekday"]
FEATURE_SETS = {"full", "no_quote", "core"}


class FeatureBuilder:
    def __init__(self, feature_set: str):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {feature_set}")
        self.feature_set = feature_set
        self.city_coordinates = {}

    def fit(self, frame: pd.DataFrame):
        locations = []
        for side in ["pickup", "delivery"]:
            columns = [side, f"{side}_lat", f"{side}_lon"]
            if set(columns) <= set(frame.columns):
                locations.append(frame[columns].set_axis(["city", "lat", "lon"], axis=1))
        if locations:
            # Supplied coordinates can be anonymized; do not replace them with geocoding.
            lookup = pd.concat(locations).groupby("city")[["lat", "lon"]].median()
            self.city_coordinates = lookup.to_dict(orient="index")
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        validate_inputs(frame)
        dates = pd.to_datetime(frame["date"])
        output = pd.DataFrame(index=frame.index)
        for column in ["pickup", "delivery", "equipment"]:
            output[column] = frame[column].astype(str)
        output["lane"] = output.pickup + " -> " + output.delivery
        output["weekday"] = dates.dt.dayofweek.astype(str)
        output["distance"] = pd.to_numeric(frame.distance).astype(float)
        output["log_distance"] = np.log1p(output.distance)
        output["inverse_distance"] = 1 / output.distance
        weight = pd.to_numeric(frame.weight, errors="coerce")
        weight = weight.where(np.isfinite(weight) & (weight > 0))
        output["weight"] = weight
        output["weight_missing"] = weight.isna().astype(int)
        output["weight_per_mile"] = weight / output.distance
        phase = 2 * np.pi * (dates.dt.dayofyear - 1) / 365.25
        output["year_sin"] = np.sin(phase)
        output["year_cos"] = np.cos(phase)
        output["weekend"] = (dates.dt.dayofweek >= 5).astype(int)
        for side in ["pickup", "delivery"]:
            for coordinate in ["lat", "lon"]:
                name = f"{side}_{coordinate}"
                lookup = frame[side].map({city: values[coordinate] for city, values in self.city_coordinates.items()})
                if self.feature_set != "core" and name in frame:
                    supplied = pd.to_numeric(frame[name], errors="coerce")
                    output[name] = supplied.where(np.isfinite(supplied), lookup)
                else:
                    output[name] = lookup
        output["latitude_change"] = output.delivery_lat - output.pickup_lat
        output["longitude_change"] = output.delivery_lon - output.pickup_lon
        if self.feature_set != "core":
            columns = ["market_index"] + (["quote_signal"] if self.feature_set == "full" else [])
            for column in columns:
                if column not in frame:
                    raise ValueError(f"{self.feature_set} model requires {column}; use the core model for the December scenario")
                values = pd.to_numeric(frame[column], errors="coerce")
                output[column] = values.where(np.isfinite(values) & (values > 0))
                output[f"{column}_missing"] = output[column].isna().astype(int)
        return output

    def to_dict(self) -> dict:
        return {"feature_set": self.feature_set, "city_coordinates": self.city_coordinates}

    @classmethod
    def from_dict(cls, state):
        builder = cls(state["feature_set"])
        builder.city_coordinates = state["city_coordinates"]
        return builder
