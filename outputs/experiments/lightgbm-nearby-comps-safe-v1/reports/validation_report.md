# Validation report

Walk-forward validation, 2025-12 to 2026-05: each month was predicted by models trained only on the sales before it.
Errors are (predicted - sold) / sold. Brackets: bootstrap 95% interval.

| Track | Homes | MAPE | MdAPE | Median signed error |
|---|---|---|---|---|
| off_market | 10,069 | 12.22% (12.00 to 12.46) | 9.26% (9.05 to 9.50) | +0.91% |
| list_aware | 10,069 | 5.54% (5.46 to 5.64) | 4.42% (4.33 to 4.51) | +0.82% |

MAPE: mean absolute percentage error. MdAPE: median absolute percentage error.
A median signed error above 0 means the model tends to predict too high.
The intervals treat homes as independent. Homes sold in the same month share the same
market swings, so the real uncertainty is somewhat wider.

![Error by price range](../figures/error_by_price_range.png)
