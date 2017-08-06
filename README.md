# Fraud Analysis Using Machine Learning

![banner](img/enron_greed.jpg) ![banner](img/guilty.jpg)

## Introduction
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. These data have been combined with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity. These data have created a dataset of 21 features for 146 employees.

The scope of the project is the creation of an algorithm with the ability to identify Enron Employees who may have committed fraud. To achieve this goal, Exploratory Data Analysis and Machine Learning are deployed to clear the dataset from outliers, identify new parameters and classify the employees as potential Persons of Interest.

**Note**: *This is just an overview of the project. You may find mode detailed EDA [here](https://github.com/MayukhSobo/EnronFraud/blob/master/eda.ipynb).*)
## Data Exploration

The features included in the dataset can be divided in three categories, Payment Features, Stock Features and Email Features. Bellow you may find the full feature list with  brief definition of each one.

**Payment Features**

| Payments            | Definitions of Category Groupings                                                                                                                                                                                                                                                                                                                                                                                                |
|:--------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ***Salary***              | Reflects items such as base salary, executive cash allowances, and benefits payments.                                                                                                                                                                                                                                                                                                                                            |
| ***Bonus***               | Reflects annual cash incentives paid based upon company performance. Also may include other retention payments.                                                                                                                                                                                                                                                                                                                  |
| ***Long Term Incentive*** | Reflects long-term incentive cash payments from various long-term incentive programs designed to tie executive compensation to long-term success as measuredagainst key performance drivers and business objectives over a multi-year period, generally 3 to 5 years.                                                                                                                                                            |
| ***Deferred Income***     | Reflects voluntary executive deferrals of salary, annual cash incentives, and long-term cash incentives as well as cash fees deferred by non-employee directorsunder a deferred compensation arrangement. May also reflect deferrals under a stock option or phantom stock unit in lieu of cash arrangement.                                                                                                                     |
| ***Deferral Payments***   | Reflects distributions from a deferred compensation arrangement due to termination of employment or due to in-service withdrawals as per plan provisions.                                                                                                                                                                                                                                                                        |
| ***Loan Advances***       | Reflects total amount of loan advances, excluding repayments, provided by the Debtor in return for a promise of repayment. In certain instances, the terms of thepromissory notes allow for the option to repay with stock of the company.                                                                                                                                                                                       |
| ***Other***               | Reflects items such as payments for severence, consulting services, relocation costs, tax advances and allowances for employees on international assignment (i.e.housing allowances, cost of living allowances, payments under Enron’s Tax Equalization Program, etc.). May also include payments provided with respect toemployment agreements, as well as imputed income amounts for such things as use of corporate aircraft. |
| ***Expenses***            | Reflects reimbursements of business expenses. May include fees paid for consulting services.                                                                                                                                                                                                                                                                                                                                     |
| ***Director Fees***       | Reflects cash payments and/or value of stock grants made in lieu of cash payments to non-employee directors.                                                                                                                                                                                                                                                                                                                     |
| ***Total Payments***      | Sum of the above values                                                                                                                                                                                                                                                                                                                                                                                                         |
***

**Stock Features**

| Stock Value              | Definitions of Category Groupings                                                                                                                                                                                                                                                                                                                                                       |
|:--------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ***Exercised Stock Options***  | Reflects amounts from exercised stock options which equal the market value in excess of the exercise price on the date the options were exercised either throughcashless (same-day sale), stock swap or cash exercises. The reflected gain may differ from that realized by the insider due to fluctuations in the market price andthe timing of any subsequent sale of the securities. |
| ***Restricted Stock***         | Reflects the gross fair market value of shares and accrued dividends (and/or phantom units and dividend equivalents) on the date of release due to lapse of vestingperiods, regardless of whether deferred.                                                                                                                                                                             |
| ***Restricted StockDeferred*** | Reflects value of restricted stock voluntarily deferred prior to release under a deferred compensation arrangement.                                                                                                                                                                                                                                                                     |
| ***Total Stock Value***        | Sum of the above values                                                                                                                                                                                                                                                                                                                                                                 |
***

**email features**

| Variable                      | Definition                                                                    |
|:------------------------------|:------------------------------------------------------------------------------|
| ***to messages***             | Total number of emails received (person's inbox)                              |
| ***email address***           | Email address of the person                                                   |
| ***from poi to this person*** | Number of emails received by POIs                                             |
| ***from messages***           | Total number of emails sent by this person                                    |
| ***from this person to poi*** | Number of emails sent by this person to a POI.                                |
| ***shared receipt with poi*** | Number of emails addressed by someone else to a POI where this person was CC. |
## Algorithm Selection

The most appropriate algorithm for the specific case was **Nearest Centroid**. Below you may find all the evaluated algorithms and their performance.

|           Category           |        Algorithm       |   Accuracy  |  Precision  |    Recall   |      F1     |      F2     |
|:----------------------------:|:----------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
|    Support Vector Machine    |           SVC          | 0.84113 | 0.39051 |   0.34150   |   0.36436   |   0.35029   |
| **Nearest Neighbors**        | **NearestCentroid**    | 0.83333     | 0.42573     | **0.71650** | **0.53410** | **0.63039** |
| Naive Bayes | GaussianNB | **0.87113**     | **0.52238**     | 0.39100     | 0.44724     | 0.41171     |
| Decision Tree Classifier       | DecisionTreeClassifier    | 0.84847     | 0.41870    | 0.35150 | 0.38217 | 0.36316 |

## Project Report

**EDA:**  [eda](eda.ipynb)

**Submission Report:**  [report](https://github.com/MayukhSobo/EnronFraud/raw/master/Enron%20Submission%20Free.pdf)


**NOTE 1:** Please do not use any other ```tester.py``` file other than the one in the repo. This may cause error while running the code.

**NOTE 2:** Please use python 2 environment. This source code might not work with python 3

**NOTE 3:** For package compatibility, make sure the virtual environment is made using the given ```requirement.txt``` file
