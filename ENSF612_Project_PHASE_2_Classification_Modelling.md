```python


columns_to_drop = ["Album Name", "Release Date", "Added By", "Added At", "Genres", "Mode", "Spotify ID", "Artist IDs", "Track Name", "Artist Name(s)", "FirstArtist"]

# Dropping the columns
cleaned_df = cleaned_df.drop(*columns_to_drop)

```


```python
from pyspark.sql.functions import when

# Adding a new column 'pop_rating' based on the 'popularity' score
cleaned_df = cleaned_df.withColumn(
    'pop_rating',
    when(cleaned_df.Popularity <= 50, 'low')
    # .when((cleaned_df.Popularity > 50) & (cleaned_df.Popularity < 75), 'medium')
    .otherwise('high')
)

```


```python
from pyspark.sql.functions import when

# Assigning popularity levels based on the 'Popularity' score
data = cleaned_df.withColumn(
    'popularity_level',
    when(cleaned_df.Popularity <= 50, 1)
    # .when((cleaned_df.Popularity > 30) & (cleaned_df.Popularity <= 60), 2)
    .otherwise(2)
)

# Display the first 10 rows
data.show(10)

```

    +-------------+----------+------------+------+---+--------+-----------+------------+----------------+--------+-------+-------+--------------+----------+----------------+
    |Duration (ms)|Popularity|Danceability|Energy|Key|Loudness|Speechiness|Acousticness|Instrumentalness|Liveness|Valence|  Tempo|Time Signature|pop_rating|popularity_level|
    +-------------+----------+------------+------+---+--------+-----------+------------+----------------+--------+-------+-------+--------------+----------+----------------+
    |       230026|        75|       0.476| 0.718|  5|  -7.227|      0.149|       0.263|         0.00261|   0.109|  0.361|183.932|             4|      high|               2|
    |       170573|        68|       0.404| 0.564| 11|  -7.013|     0.0344|       0.915|         0.00252|   0.134|  0.371| 93.631|             4|      high|               2|
    |       230453|        94|       0.679| 0.587|  7|  -7.015|      0.276|       0.141|         6.35E-6|   0.137|  0.486|186.003|             4|      high|               2|
    |       227645|        61|       0.317|  0.31|  6|  -9.235|     0.0301|       0.152|         1.44E-4|   0.235| 0.0389| 82.606|             3|      high|               2|
    |       214186|        58|       0.571| 0.692|  0|  -6.656|     0.0545|      0.0245|             0.0|  0.0813|   0.22|127.937|             4|      high|               2|
    |       345251|        47|       0.425| 0.782|  5|   -6.52|     0.0892|       0.737|         1.81E-4|     0.5|   0.41|150.092|             4|       low|               1|
    |       258453|        65|       0.639| 0.633|  3|  -7.338|     0.0352|       0.574|         2.18E-4|   0.111|  0.243|129.996|             4|      high|               2|
    |       191298|        49|        0.48| 0.493|  0|  -8.311|     0.0512|       0.123|             0.0|  0.0995|  0.211| 87.343|             4|       low|               1|
    |       345160|        51|       0.464| 0.688|  1|  -7.444|     0.0549|       0.248|         2.61E-4|   0.629|  0.183| 75.054|             4|      high|               2|
    |       405213|        77|        0.65| 0.711|  0|  -5.417|     0.0377|      0.0123|         0.00943|   0.301|  0.319| 89.019|             4|      high|               2|
    +-------------+----------+------------+------+---+--------+-----------+------------+----------------+--------+-------+-------+--------------+----------+----------------+
    only showing top 10 rows
    



```python
from pyspark.sql.functions import col

# Counting the values in 'popularity_level'
popularity_level_counts = data.groupBy("popularity_level").count().orderBy("popularity_level")

# Display the counts
popularity_level_counts.show()

```

    +----------------+-----+
    |popularity_level|count|
    +----------------+-----+
    |               1|  654|
    |               2| 1252|
    +----------------+-----+
    


**Next we balanced out the data a bit since classification algorithms perform poorly with imbalanced data.**


```python
from pyspark.sql.functions import col

# Sample a subset of level 2 to match the count of level 1
level_1_count = 654
level_2_sample = data.filter(col('popularity_level') == 2).sample(withReplacement=False, fraction=level_1_count/1252)


# Combine the samples with level 1 data
balanced_data = data.filter(col('popularity_level') == 1).union(level_2_sample)

```


```python
balanced_data.head()

```




    Row(Duration (ms)=345251, Popularity=47, Danceability=0.425, Energy=0.782, Key=5, Loudness=-6.52, Speechiness=0.0892, Acousticness=0.737, Instrumentalness=0.000181, Liveness=0.5, Valence=0.41, Tempo=150.092, Time Signature=4, pop_rating='low', popularity_level=1)




```python
# Separate features and target
feature_columns = [col for col in balanced_data.columns if col != 'popularity_level']
X = balanced_data.select(*feature_columns)
y = balanced_data.select('popularity_level')

# Split the data into training and testing sets
train_data, test_data = balanced_data.randomSplit([0.75, 0.25], seed=42)

```


```python
from pyspark.sql.functions import when
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline

# Calculate the median or mean for 'Tempo' to replace zero values, if needed
median_tempo = balanced_data.approxQuantile('Tempo', [0.5], 0.01)[0]

# Replace zero values in 'Tempo' if necessary
balanced_data = balanced_data.withColumn('Tempo', when(col('Tempo') == 0, median_tempo).otherwise(col('Tempo')))

# Assemble numerical features into a vector
assembler = VectorAssembler(inputCols=['Duration (ms)', 'Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Time Signature'], outputCol='features_vector')

# Scale the features
scaler = MinMaxScaler(inputCol='features_vector', outputCol='scaled_features')

# Pipeline: Assemble and then scale
pipeline = Pipeline(stages=[assembler, scaler])




```


```python
from pyspark.sql.functions import col, isnan, when, count

# Check for null or NaN values in each column
nulls_in_each_column = train_data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in train_data.columns])
nulls_in_each_column.show()

```

    +-------------+----------+------------+------+---+--------+-----------+------------+----------------+--------+-------+-----+--------------+----------+----------------+
    |Duration (ms)|Popularity|Danceability|Energy|Key|Loudness|Speechiness|Acousticness|Instrumentalness|Liveness|Valence|Tempo|Time Signature|pop_rating|popularity_level|
    +-------------+----------+------------+------+---+--------+-----------+------------+----------------+--------+-------+-----+--------------+----------+----------------+
    |            0|         0|           1|     1|  1|       1|          1|           1|               1|       1|      1|    1|             1|         0|               0|
    +-------------+----------+------------+------+---+--------+-----------+------------+----------------+--------+-------+-----+--------------+----------+----------------+
    



```python
cleaned_train_data = train_data.dropna()
cleaned_test_data = test_data.dropna()

```


```python
# Fit and transform the data
# Fit the pipeline on the training data
fitted_pipeline = pipeline.fit(cleaned_train_data)

# Transform both training and test data
transformed_train_data = fitted_pipeline.transform(cleaned_train_data)
transformed_test_data = fitted_pipeline.transform(cleaned_test_data)


```

# **Modelling**:
## (1) Logistic Regression
## (2) Decision Tree
## (3) Random Forest Classifier


```python
from pyspark.ml.classification import LogisticRegression

# Train a Logistic Regression model
lr = LogisticRegression(featuresCol='scaled_features', labelCol='popularity_level')
lr_model = lr.fit(transformed_train_data)

# Make predictions on the test data
lr_predictions = lr_model.transform(transformed_test_data)

```


```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="popularity_level", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(lr_predictions)
print("Accuracy of Logistic Regression model: ", accuracy)

```

    Accuracy of Logistic Regression model:  0.6363636363636364



```python
from pyspark.ml.classification import DecisionTreeClassifier

# Train a Decision Tree model
dt = DecisionTreeClassifier(featuresCol='scaled_features', labelCol='popularity_level')
dt_model = dt.fit(transformed_train_data)

# Make predictions on the test data
dt_predictions = dt_model.transform(transformed_test_data)
```


```python
# Evaluate accuracy
dt_accuracy = evaluator.evaluate(dt_predictions)
print("Accuracy of Decision Tree model: ", dt_accuracy)
```

    Accuracy of Decision Tree model:  0.6201298701298701



```python
from pyspark.ml.classification import RandomForestClassifier

# Train a Random Forest model
rf = RandomForestClassifier(featuresCol='scaled_features', labelCol='popularity_level')
rf_model = rf.fit(transformed_train_data)

# Make predictions on the test data
rf_predictions = rf_model.transform(transformed_test_data)

```


```python
# Assuming your predictions from Random Forest model are stored in `rf_predictions`
rf_accuracy = evaluator.evaluate(rf_predictions)
print("Accuracy of Random Forest model: ", rf_accuracy)

```

    Accuracy of Random Forest model:  0.6558441558441559


# Hyper parameter tuning of the Random Forest **Model**


```python
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create an instance of the Random Forest Classifier
rf = RandomForestClassifier(featuresCol='scaled_features', labelCol='popularity_level')

# Simplified parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 15]) \
    .addGrid(rf.maxDepth, [5, 8]) \
    .build()
```


```python
# Create a cross-validator with fewer folds
crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(labelCol="popularity_level",
                          predictionCol="prediction", metricName="accuracy"),
                          numFolds=3)
```


```python
# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(transformed_train_data)
```


```python
# Make predictions on test data. cvModel uses the best model found.
cv_predictions = cvModel.transform(transformed_test_data)

# Evaluate the model
cv_accuracy = evaluator.evaluate(cv_predictions)
print("Accuracy after Hyperparameter Tuning: ", cv_accuracy)
```

    Accuracy after Hyperparameter Tuning:  0.6558441558441559



```python
from pyspark.mllib.evaluation import MulticlassMetrics

# Convert the predictions DataFrame to an RDD of (prediction, label) tuples
cv_predictions_and_labels = cv_predictions.select("prediction", "popularity_level").rdd.map(lambda row: (row[0], float(row[1])))

# Instantiate metrics object
cv_metrics = MulticlassMetrics(cv_predictions_and_labels)

# Confusion Matrix
cv_confusion_matrix = cv_metrics.confusionMatrix().toArray()

# Output the Confusion Matrix
print("Confusion Matrix:\n", cv_confusion_matrix)

# True Positives, False Positives, True Negatives, and False Negatives can be extracted from the confusion matrix
cv_tp = cv_confusion_matrix[1, 1]
cv_fp = cv_confusion_matrix[0, 1]
cv_tn = cv_confusion_matrix[0, 0]
cv_fn = cv_confusion_matrix[1, 0]

print(f"True Positives: {cv_tp}")
print(f"False Positives: {cv_fp}")
print(f"True Negatives: {cv_tn}")
print(f"False Negatives: {cv_fn}")
```

    Confusion Matrix:
     [[ 86.  52.]
     [ 54. 116.]]
    True Positives: 116.0
    False Positives: 52.0
    True Negatives: 86.0
    False Negatives: 54.0

