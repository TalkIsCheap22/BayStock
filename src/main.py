import preprocessing
import model

stocks_file = "../dataset/all_stocks_5yr.csv"

labels, dataset = preprocessing.read_and_process(stocks_file)
print("read and process finished")
training_set, testing_set = preprocessing.train_test_divide(dataset, 0.7)
print("train test divide finished")

model = model.Model(labels, "avg_increase_5days", ["price_trend_1day", "price_trend_5days"])
model.train(training_set)
model.evaluate(testing_set)

#training_set, testing_set = preprocessing.train_test_divide(datas, 0.7)
#print("training")
#print(training_set)
#print("testing")
#print(testing_set)

#single_model = model.SingleModel("prev_diff", "price_trend", 4)
#single_model.evaluate(testing_set)
#single_model.train(training_set)
#single_model.evaluate(testing_set)