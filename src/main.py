import preprocessing
import model

stocks_file = "../dataset/all_stocks_5yr.csv"

labels, dataset = preprocessing.read_and_process(stocks_file)
print("read and process finished")
training_set, testing_set = preprocessing.train_test_divide(dataset, 0.6)
print("train test divide finished")

model = model.Model(labels, ["increase_1day", "avg_increase_5days"], ["price_trend_1day", "price_trend_5days"])
model.train(training_set)
model.evaluate(testing_set, {"inds":["increase_1day", "avg_increase_5days"], "deps":["price_trend_1day"]})