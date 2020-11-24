import pandas as pd
from pyarc import CBA,TransactionDB
from sklearn.model_selection import train_test_split

print("")
print("Rule Generation")
data = pd.read_csv('total_data.csv')  #disc_data/Known_attack_data
print("Data Size : %s"%str(data.shape))

# 데이터 분할
train, test = train_test_split(data, test_size=0.2, random_state=123)

txns_train = TransactionDB.from_DataFrame(train, target="class")
txns_test = TransactionDB.from_DataFrame(test)

print("Association Rule Generation")
cba = CBA(support=0.1)
cba.fit(txns_train)
print(cba.fit(txns_train))

print("\nRULES : ({})".format(len(cba.clf.rules)))
for i in cba.clf.rules:
    print(i)

accuracy = cba.rule_model_accuracy(txns_test)
print("")
print(accuracy)