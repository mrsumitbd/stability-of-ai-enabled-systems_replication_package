from atm import ATM

atm = ATM()

results = atm.run(train_path='atm/demos/pollution.csv')

print(results.describe())

print(results.get_best_classifier())
