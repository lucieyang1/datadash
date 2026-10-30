# Validation report

Walk-forward validation, 2025-12 to 2026-05: each month was predicted by models trained only on the sales before it.
Errors are (predicted - sold) / sold. Brackets: bootstrap 95% interval.

| Track | Homes | MAPE | MdAPE | Median signed error |
|---|---|---|---|---|
| off_market | 10,069 | 12.55% (12.33 to 12.80) | 9.48% (9.27 to 9.72) | +1.43% |
| list_aware | 10,069 | 5.54% (5.45 to 5.63) | 4.40% (4.29 to 4.49) | +0.93% |

MAPE: mean absolute percentage error. MdAPE: median absolute percentage error.
A median signed error above 0 means the model tends to predict too high.
The intervals treat homes as independent. Homes sold in the same month share the same
market swings, so the real uncertainty is somewhat wider.

![Error by price range](../figures/error_by_price_range.png)
