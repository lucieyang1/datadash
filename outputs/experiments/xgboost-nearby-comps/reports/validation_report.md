# Validation report

Walk-forward validation, 2025-12 to 2026-05: each month was predicted by models trained only on the sales before it.
Errors are (predicted - sold) / sold. Brackets: bootstrap 95% interval.

| Track | Homes | MAPE | MdAPE | Median signed error |
|---|---|---|---|---|
| off_market | 10,069 | 12.51% (12.28 to 12.76) | 9.43% (9.21 to 9.70) | +1.28% |
| list_aware | 10,069 | 5.51% (5.43 to 5.61) | 4.43% (4.34 to 4.52) | +0.98% |

MAPE: mean absolute percentage error. MdAPE: median absolute percentage error.
A median signed error above 0 means the model tends to predict too high.
The intervals treat homes as independent. Homes sold in the same month share the same
market swings, so the real uncertainty is somewhat wider.

![Error by price range](../figures/error_by_price_range.png)
