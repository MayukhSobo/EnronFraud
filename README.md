# EnronFraud
Fraud Detection by finding the Person of Interest (POI)

### Feature Importances

#### ----- Top 5 features -----

**K_best algorithm**
```python
['bonus', 'salary', 'exercised_stock_options', 'total_stock_value', 'shared_receipt_with_poi']
```
**XGBoost algorithm**
```python
['salary', 'to_messages', 'deferral_payments', 'total_payments', 'bonus']
```

**Random Forest algorithm**
```python
['to_messages', 'deferral_payments', 'bonus', 'total_stock_value', 'expenses']
```