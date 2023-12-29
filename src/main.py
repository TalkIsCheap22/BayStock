import preprocessing
import model

stocks_file = "../dataset/all_stocks_5yr.csv"

labels, datas = preprocessing.read_stocks(stocks_file)
print(labels)
#training_set, testing_set = preprocessing.train_test_divide(datas, 0.7)
#print("training")
#print(training_set)
#print("testing")
#print(testing_set)

#single_model = model.SingleModel("prev_diff", "price_trend", 4)
#single_model.evaluate(testing_set)
#single_model.train(training_set)
#single_model.evaluate(testing_set)