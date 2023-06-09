# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Create a SparkSession
spark = SparkSession.builder.appName("TelecomChurnPrediction").getOrCreate()

# Load the dataset
data = spark.read.csv("telecom_dataset.csv", header=True, inferSchema=True)

# Data preprocessing
# Drop unnecessary columns
data = data.drop("CustomerID")

# Handle missing values
data = data.na.drop()

# Calculate new features
data = data.withColumn("CallDuration", col("TotalCharges") / col("MonthlyCharges"))
data = data.withColumn("AverageMonthlySpend", col("MonthlyCharges"))
data = data.withColumn("CustomerTenure", col("Age"))
data = data.withColumn("CustomerSatisfactionScore", avg(col("Age")).over({}))

# Encode categorical variables
categorical_cols = ["Gender", "Contract"]

indexers = [StringIndexer(inputCol=col(column), outputCol=col(column + "_index")).fit(data) for column in categorical_cols]

pipeline = Pipeline(stages=indexers)
data = pipeline.fit(data).transform(data)

# Feature scaling
assembler = VectorAssembler(inputCols=["Age", "MonthlyCharges", "TotalCharges", "CallDuration",
                                       "AverageMonthlySpend", "CustomerTenure", "CustomerSatisfactionScore",
                                       "Gender_index", "Contract_index"],
                            outputCol="unscaled_features")

data = assembler.transform(data)

scaler = StandardScaler(inputCol="unscaled_features", outputCol="features")
scaler_model = scaler.fit(data)
data = scaler_model.transform(data)

# Split the data into training and testing sets
(train_data, test_data) = data.randomSplit([0.8, 0.2], seed=42)

# Model selection and training
classifiers = [
    LogisticRegression(labelCol="Churn", featuresCol="features"),
    RandomForestClassifier(labelCol="Churn", featuresCol="features", numTrees=10),
    GBTClassifier(labelCol="Churn", featuresCol="features", maxIter=10),
    LinearSVC(labelCol="Churn", featuresCol="features", maxIter=10)
]

best_model = None
best_accuracy = 0.0

for classifier in classifiers:
    model = classifier.fit(train_data)
    
    # Model evaluation on training dataset
    train_predictions = model.transform(train_data)
    train_evaluator = MulticlassClassificationEvaluator(labelCol="Churn")
    train_accuracy = train_evaluator.evaluate(train_predictions, {train_evaluator.metricName: "accuracy"})
    train_precision = train_evaluator.evaluate(train_predictions, {train_evaluator.metricName: "weightedPrecision"})
    train_recall = train_evaluator.evaluate(train_predictions, {train_evaluator.metricName: "weightedRecall"})
    train_f1_score = train_evaluator.evaluate(train_predictions, {train_evaluator.metricName: "f1"})
    
    print("Model:", classifier.__class__.__name__)
    print("Train Accuracy:", train_accuracy)
    print("Train Precision:", train_precision)
    print("Train Recall:", train_recall)
    print("Train F1-score:", train_f1_score)
    
    # Model evaluation on testing dataset
    test_predictions = model.transform(test_data)
    test_evaluator = MulticlassClassificationEvaluator(labelCol="Churn")
    test_accuracy = test_evaluator.evaluate(test_predictions, {test_evaluator.metricName: "accuracy"})
    test_precision = test_evaluator.evaluate(test_predictions, {test_evaluator.metricName: "weightedPrecision"})
    test_recall = test_evaluator.evaluate(test_predictions, {test_evaluator.metricName: "weightedRecall"})
    test_f1_score = test_evaluator.evaluate(test_predictions, {test_evaluator.metricName: "f1"})
    
    print("Test Accuracy:", test_accuracy)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)
    print("Test F1-score:", test_f1_score)
    print("-------------------------")
    
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model = model

# Save the best model
best_model.save("telecom_churn_model")

# Stop the SparkSession
spark.stop()
