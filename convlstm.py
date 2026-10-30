"""ConvLSTM home-price model using monthly grids of prior same-type sales."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn

import features
from models import CommunityComparableXGBoostModel


class SpatialSaleHistory:
    """Monthly spatial grids, indexed relative to each home's listing month."""

    grid_height = 64
    grid_width = 80
    cell_size_km = 1.0
    origin_latitude = 43.4
    origin_longitude = -79.7
    longitude_scale = 0.72
    km_per_degree = 111.2
    lookback_months = 12
    patch_size = 7

    def fit(self, sales: pd.DataFrame) -> SpatialSaleHistory:
        rows = sales.copy()
        rows["date_sold"] = pd.to_datetime(rows["date_sold"], errors="coerce")
        rows["sold_price"] = pd.to_numeric(rows["sold_price"], errors="coerce")
        rows["area_sqft"] = features.square_feet(rows["house_area"])
        rows["latitude"] = pd.to_numeric(rows["latitude"], errors="coerce")
        rows["longitude"] = pd.to_numeric(rows["longitude"], errors="coerce")
        rows["house_type_name"] = rows["house_type_name"].fillna("").astype(str)
        rows = rows[
            rows["date_sold"].notna()
            & (rows["sold_price"] > 0)
            & np.isfinite(rows["latitude"])
            & np.isfinite(rows["longitude"])
            & (rows["house_type_name"] != "")
        ].copy()
        if rows.empty:
            raise ValueError("ConvLSTM needs at least one valid historical sale.")

        self.type_to_index = {
            kind: index for index, kind in enumerate(sorted(rows["house_type_name"].unique()))
        }
        rows["type_index"] = rows["house_type_name"].map(self.type_to_index)
        rows["month"] = rows["date_sold"].dt.to_period("M")
        rows["grid_x"] = np.floor(
            (rows["longitude"] - self.origin_longitude)
            * self.longitude_scale
            * self.km_per_degree
            / self.cell_size_km
        ).astype(int)
        rows["grid_y"] = np.floor(
            (rows["latitude"] - self.origin_latitude)
            * self.km_per_degree
            / self.cell_size_km
        ).astype(int)
        rows = rows[
            rows["grid_x"].between(0, self.grid_width - 1)
            & rows["grid_y"].between(0, self.grid_height - 1)
        ].copy()
        if rows.empty:
            raise ValueError("Historical sale coordinates fall outside the configured grid.")

        self.first_month_ordinal = int(rows["month"].map(lambda month: month.ordinal).min())
        last_month_ordinal = int(rows["month"].map(lambda month: month.ordinal).max())
        month_count = last_month_ordinal - self.first_month_ordinal + 1
        self.maps = np.zeros(
            (month_count + 1, len(self.type_to_index), 3, self.grid_height, self.grid_width),
            dtype=np.float32,
        )
        rows["cell"] = rows["grid_y"] * self.grid_width + rows["grid_x"]
        rows["log_price"] = np.log(rows["sold_price"])
        rows["price_per_sqft"] = rows["sold_price"] / rows["area_sqft"]
        rows["log_price_per_sqft"] = np.where(
            np.isfinite(rows["price_per_sqft"]) & (rows["price_per_sqft"] > 0),
            np.log(rows["price_per_sqft"]),
            np.nan,
        )

        for (month, type_index), group in rows.groupby(["month", "type_index"], sort=False):
            map_index = month.ordinal - self.first_month_ordinal + 1
            cell_counts = group.groupby("cell", sort=False).size()
            flat_map = self.maps[map_index, type_index].reshape(3, -1)
            cells = cell_counts.index.to_numpy(dtype=int)
            flat_map[0, cells] = np.log1p(cell_counts.to_numpy()) / 4.0

            median_price = group.groupby("cell", sort=False)["log_price"].median()
            flat_map[1, median_price.index.to_numpy(dtype=int)] = (
                median_price.to_numpy(dtype=np.float32) - 13.3
            )
            median_price_per_sqft = group.groupby("cell", sort=False)[
                "log_price_per_sqft"
            ].median().dropna()
            flat_map[2, median_price_per_sqft.index.to_numpy(dtype=int)] = (
                median_price_per_sqft.to_numpy(dtype=np.float32) - 6.2
            )
        return self

    def patches(self, homes: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
        batch = homes.iloc[positions]
        dates = pd.to_datetime(batch["date_listed"], errors="coerce")
        period_ordinals = np.where(
            dates.notna(),
            dates.dt.to_period("M").map(
                lambda period: period.ordinal if pd.notna(period) else -1
            ),
            -1,
        ).astype(np.int64)
        offsets = np.arange(self.lookback_months, 0, -1, dtype=np.int64)
        map_indices = (
            period_ordinals[:, np.newaxis] - self.first_month_ordinal - offsets + 1
        )
        valid_months = (map_indices > 0) & (map_indices < len(self.maps))
        map_indices = np.where(valid_months, map_indices, 0)

        longitude = pd.to_numeric(batch["longitude"], errors="coerce").to_numpy(dtype=float)
        latitude = pd.to_numeric(batch["latitude"], errors="coerce").to_numpy(dtype=float)
        valid_coords = np.isfinite(longitude) & np.isfinite(latitude)
        center_x = np.floor(
            (np.where(valid_coords, longitude, self.origin_longitude) - self.origin_longitude)
            * self.longitude_scale
            * self.km_per_degree
            / self.cell_size_km
        ).astype(np.int64)
        center_y = np.floor(
            (np.where(valid_coords, latitude, self.origin_latitude) - self.origin_latitude)
            * self.km_per_degree
            / self.cell_size_km
        ).astype(np.int64)
        half = self.patch_size // 2
        offsets_xy = np.arange(-half, half + 1, dtype=np.int64)
        grid_x = center_x[:, np.newaxis] + offsets_xy
        grid_y = center_y[:, np.newaxis] + offsets_xy
        valid_coords &= (
            (center_x >= 0)
            & (center_x < self.grid_width)
            & (center_y >= 0)
            & (center_y < self.grid_height)
        )
        valid_patch = (
            ((grid_x >= 0) & (grid_x < self.grid_width))[:, np.newaxis, np.newaxis, :]
            & ((grid_y >= 0) & (grid_y < self.grid_height))[
                :, np.newaxis, :, np.newaxis
            ]
        )
        safe_x = np.clip(grid_x, 0, self.grid_width - 1)
        safe_y = np.clip(grid_y, 0, self.grid_height - 1)

        type_names = batch["house_type_name"].fillna("").astype(str).to_numpy()
        type_indices = np.array(
            [self.type_to_index.get(kind, -1) for kind in type_names], dtype=np.int64
        )
        valid_homes = valid_coords & (type_indices >= 0)
        safe_types = np.maximum(type_indices, 0)
        selected_maps = self.maps[
            map_indices,
            safe_types[:, np.newaxis],
        ].transpose(0, 1, 3, 4, 2)

        batch_indices = np.arange(len(batch), dtype=np.int64)[:, np.newaxis, np.newaxis, np.newaxis]
        time_indices = np.arange(self.lookback_months, dtype=np.int64)[
            np.newaxis, :, np.newaxis, np.newaxis
        ]
        patches = selected_maps[
            batch_indices,
            time_indices,
            safe_y[:, np.newaxis, :, np.newaxis],
            safe_x[:, np.newaxis, np.newaxis, :],
        ].transpose(0, 1, 4, 2, 3)
        patches *= (
            valid_patch[:, np.newaxis, :, :]
            & valid_homes[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        )
        return np.nan_to_num(patches, copy=False)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, inputs, hidden, cell):
        gates = self.gates(torch.cat([inputs, hidden], dim=1))
        input_gate, forget_gate, output_gate, candidate = gates.chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


class ConvLSTMPriceNetwork(nn.Module):
    def __init__(self, numeric_count: int, category_sizes: list[int], hidden_channels=8):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvLSTMCell(3, hidden_channels)
        embedding_sizes = [min(16, max(4, int(np.ceil(size**0.25) * 4))) for size in category_sizes]
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, width) for size, width in zip(category_sizes, embedding_sizes)
        ])
        self.head = nn.Sequential(
            nn.Linear(numeric_count + sum(embedding_sizes) + hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, spatial, numeric, categories):
        batch_size, _, _, height, width = spatial.shape
        hidden = spatial.new_zeros(batch_size, self.hidden_channels, height, width)
        cell = torch.zeros_like(hidden)
        for step in range(spatial.shape[1]):
            hidden, cell = self.cell(spatial[:, step], hidden, cell)
        center = hidden[:, :, height // 2, width // 2]
        category_values = [
            embedding(categories[:, column])
            for column, embedding in enumerate(self.embeddings)
        ]
        combined = torch.cat([center, numeric, *category_values], dim=1)
        return self.head(combined).squeeze(1)


class ConvLSTMPriceModel:
    """Predict log price from property features and prior monthly spatial sale grids."""

    epochs = 2
    batch_size = 512
    categorical_columns = ["house_type_name", "house_style", "municipality", "community"]

    def __init__(self, track):
        if track not in {"off_market", "list_aware"}:
            raise ValueError(f"Unknown track {track!r}; choose 'off_market' or 'list_aware'.")
        self.track = track
        self.feature_builder = CommunityComparableXGBoostModel(track)
        self.history = SpatialSaleHistory()
        self.numeric_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.numeric_scaler = StandardScaler()
        self.category_maps = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _static_values(self, homes, fit=False):
        numeric, categories = self.feature_builder._values(homes)
        if fit:
            numeric_values = self.numeric_scaler.fit_transform(
                self.numeric_imputer.fit_transform(numeric)
            ).astype(np.float32)
            self.category_maps = {
                column: {
                    value: code + 1
                    for code, value in enumerate(
                        sorted(categories[column].dropna().astype(str).unique())
                    )
                }
                for column in self.categorical_columns
            }
        else:
            numeric_values = self.numeric_scaler.transform(
                self.numeric_imputer.transform(numeric)
            ).astype(np.float32)
        category_values = np.column_stack([
            categories[column]
            .fillna("(missing)")
            .astype(str)
            .map(self.category_maps[column])
            .fillna(0)
            .to_numpy(dtype=np.int64)
            for column in self.categorical_columns
        ])
        return numeric_values, category_values

    def fit(self, train: pd.DataFrame):
        train = train.loc[
            pd.to_numeric(train["sold_price"], errors="coerce").gt(0)
        ].copy()
        if train.empty:
            raise ValueError("ConvLSTM needs at least one training sale with a positive price.")
        torch.manual_seed(0)
        if self.device.type == "cpu":
            torch.set_num_threads(1)

        self.feature_builder.comparables.fit(train)
        self.history.fit(train)
        rng = np.random.default_rng(0)
        numeric, categories = self.feature_builder._values(train)
        numeric_values = self.numeric_scaler.fit_transform(
            self.numeric_imputer.fit_transform(numeric)
        ).astype(np.float32)
        self.category_maps = {
            column: {
                value: code + 1
                for code, value in enumerate(
                    sorted(categories[column].dropna().astype(str).unique())
                )
            }
            for column in self.categorical_columns
        }
        category_values = np.column_stack([
            categories[column]
            .fillna("(missing)")
            .astype(str)
            .map(self.category_maps[column])
            .fillna(0)
            .to_numpy(dtype=np.int64)
            for column in self.categorical_columns
        ])
        category_sizes = [
            len(self.category_maps[column]) + 1 for column in self.categorical_columns
        ]
        self.network = ConvLSTMPriceNetwork(
            numeric_count=numeric_values.shape[1],
            category_sizes=category_sizes,
        ).to(self.device)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=0.001, weight_decay=1e-4)
        loss_function = nn.SmoothL1Loss()
        targets = np.log(train["sold_price"].to_numpy(dtype=np.float32))
        self.network.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(train))
            for start in range(0, len(train), self.batch_size):
                positions = order[start : start + self.batch_size]
                spatial = torch.from_numpy(
                    self.history.patches(train, positions)
                ).to(self.device)
                numeric_tensor = torch.from_numpy(numeric_values[positions]).to(self.device)
                category_tensor = torch.from_numpy(category_values[positions]).to(self.device)
                target_tensor = torch.from_numpy(targets[positions]).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.network(spatial, numeric_tensor, category_tensor)
                loss = loss_function(prediction, target_tensor)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, homes: pd.DataFrame) -> np.ndarray:
        numeric_values, category_values = self._static_values(homes)
        predictions = []
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(homes), self.batch_size):
                positions = np.arange(start, min(start + self.batch_size, len(homes)))
                spatial = torch.from_numpy(
                    self.history.patches(homes, positions)
                ).to(self.device)
                numeric_tensor = torch.from_numpy(numeric_values[positions]).to(self.device)
                category_tensor = torch.from_numpy(category_values[positions]).to(self.device)
                predictions.append(
                    self.network(spatial, numeric_tensor, category_tensor).cpu().numpy()
                )
        if not predictions:
            return np.empty(0, dtype=float)
        return np.exp(np.concatenate(predictions))
