import pandas
import numpy
import plotly.express as xpress
import plotly.graph_objects as graph
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
pio.templates.default = 'plotly_white'

class Helpers:
    def __init__(self, data):
        self.data = data

    def show_dataset_info(self):
        print('------Dataset------')
        print(self.data.head())
        print('------Categories------')
        print(self.data.info())
        print('------Null Entries------')
        print(self.data.isnull().sum())
        print('------Result Count------')
        print(self.data['Credit_Score'].value_counts())
        print('------Occupation does not influence credit score------')
        self.custom_map('Credit Score vs Occupation', self.data, 'Credit_Score', x='Occupation')

    def custom_map(self, title, data, color, x, y='', color_map={'Poor':'red', 'Standard':'yellow', 'Good':'green'}):
        if y:
            fig = xpress.box(data, x=x, color=color, title=title, y=y, color_discrete_map=color_map)
        else:
            fig = xpress.box(data, x=x, color=color, title=title, color_discrete_map=color_map)
        fig.show()


class Model:
    def __init__(self, data):
        self.model = self._train(data)

    def _train(self, data):
        x = numpy.array(data[['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
                     'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Credit_Mix', 'Outstanding_Debt',
                     'Credit_History_Age', 'Monthly_Balance']])
        y = numpy.array(data[['Credit_Score']])

        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=42)
        model = RandomForestClassifier()
        model.fit(xtrain, ytrain)
        return model

    def get_result(self, income_annual, salary_monthly, num_bank_acct, num_cc, interest_rate, num_loan, delay_from_date, 
                   num_delayed, cred_mix, debt_outstanding, age_credit_history, balance_monthly):
        parameters = numpy.array([[income_annual, salary_monthly, num_bank_acct, num_cc, interest_rate, num_loan, delay_from_date, 
                                num_delayed, cred_mix, debt_outstanding, age_credit_history, balance_monthly]])
        return self.model.predict(parameters)

    

if __name__ == '__main__':
    data = pandas.read_csv('CreditScoreData/train.csv')
    data['Credit_Mix'] = data['Credit_Mix'].map({'Standard':1, 'Good':2, 'Bad':0})
    # Helper to help visualize the dataset 
    '''
    helpers = Helpers(data)
    helpers.show_dataset_info()
    '''

    app = Model(data)

    ret = app.get_result(
        float(input('Annual Income ($): ')),
        float(input("Monthly Inhand Salary ($): ")),
        float(input("Number of Bank Accounts (int): ")),
        float(input("Number of Credit cards (int): ")),
        float(input("Interest rate (%): ")),
        float(input("Number of Loans (int): ")),
        float(input("Average number of days delayed from due date (float): ")),
        float(input("Number of delayed payments (int): ")),
        input("Credit Mix (Bad: 0, Standard: 1, Good: 3): "),
        float(input("Outstanding Debt ($): ")),
        float(input("Credit History Age (days): ")),
        float(input("Monthly Balance ($): "))
    )
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\nRESULT:')
    print(ret[0])
