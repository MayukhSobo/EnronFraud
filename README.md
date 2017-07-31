# EnronFraud
Fraud Detection by finding the Person of Interest (POI)

### Feature Importances

#### ----- Top 5 features -----

**K_best algorithm**
```python
['poi_interaction', 'total_stock_value', 'shared_receipt_with_poi', 'income_ratio', 'exercised_stock_options']
```
**XGBoost algorithm**
```python
['poi_interaction', 'expenses', 'shared_receipt_with_poi', 'restricted_stock', 'exercised_stock_options']
```

**Random Forest algorithm**
```python
['deferred_income', 'expenses', 'poi_interaction', 'shared_receipt_with_poi', 'income_ratio']
```