# Setup Spark in Google Colab
*reference: https://www.analyticsvidhya.com/blog/2020/11/a-must-read-guide-on-how-to-work-with-pyspark-on-google-colab-for-data-scientists/*


*to install other versions, get the download link from https://spark.apache.org/downloads.html*


```python
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
```


```python
!wget https://dlcdn.apache.org/spark/spark-3.3.3/spark-3.3.3-bin-hadoop3.tgz
```

    --2023-12-14 19:12:08--  https://dlcdn.apache.org/spark/spark-3.3.3/spark-3.3.3-bin-hadoop3.tgz
    Resolving dlcdn.apache.org (dlcdn.apache.org)... 151.101.2.132, 2a04:4e42::644
    Connecting to dlcdn.apache.org (dlcdn.apache.org)|151.101.2.132|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 299426263 (286M) [application/x-gzip]
    Saving to: ‘spark-3.3.3-bin-hadoop3.tgz.2’
    
    spark-3.3.3-bin-had 100%[===================>] 285.55M  23.2MB/s    in 5.6s    
    
    2023-12-14 19:12:14 (50.8 MB/s) - ‘spark-3.3.3-bin-hadoop3.tgz.2’ saved [299426263/299426263]
    



```python
!tar -xvf spark-3.3.3-bin-hadoop3.tgz
```

    spark-3.3.3-bin-hadoop3/
    spark-3.3.3-bin-hadoop3/LICENSE
    spark-3.3.3-bin-hadoop3/NOTICE
    spark-3.3.3-bin-hadoop3/R/
    spark-3.3.3-bin-hadoop3/R/lib/
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/DESCRIPTION
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/INDEX
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/Meta/
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/Meta/Rd.rds
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/Meta/features.rds
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/Meta/hsearch.rds
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/Meta/links.rds
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/Meta/nsInfo.rds
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/Meta/package.rds
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/Meta/vignette.rds
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/NAMESPACE
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/R/
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/R/SparkR
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/R/SparkR.rdb
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/R/SparkR.rdx
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/doc/
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/doc/index.html
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/doc/sparkr-vignettes.R
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/doc/sparkr-vignettes.Rmd
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/doc/sparkr-vignettes.html
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/help/
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/help/AnIndex
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/help/SparkR.rdb
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/help/SparkR.rdx
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/help/aliases.rds
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/help/paths.rds
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/html/
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/html/00Index.html
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/html/R.css
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/profile/
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/profile/general.R
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/profile/shell.R
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/tests/
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/tests/testthat/
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/tests/testthat/test_basic.R
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/worker/
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/worker/daemon.R
    spark-3.3.3-bin-hadoop3/R/lib/SparkR/worker/worker.R
    spark-3.3.3-bin-hadoop3/R/lib/sparkr.zip
    spark-3.3.3-bin-hadoop3/README.md
    spark-3.3.3-bin-hadoop3/RELEASE
    spark-3.3.3-bin-hadoop3/bin/
    spark-3.3.3-bin-hadoop3/bin/beeline
    spark-3.3.3-bin-hadoop3/bin/beeline.cmd
    spark-3.3.3-bin-hadoop3/bin/docker-image-tool.sh
    spark-3.3.3-bin-hadoop3/bin/find-spark-home
    spark-3.3.3-bin-hadoop3/bin/find-spark-home.cmd
    spark-3.3.3-bin-hadoop3/bin/load-spark-env.cmd
    spark-3.3.3-bin-hadoop3/bin/load-spark-env.sh
    spark-3.3.3-bin-hadoop3/bin/pyspark
    spark-3.3.3-bin-hadoop3/bin/pyspark.cmd
    spark-3.3.3-bin-hadoop3/bin/pyspark2.cmd
    spark-3.3.3-bin-hadoop3/bin/run-example
    spark-3.3.3-bin-hadoop3/bin/run-example.cmd
    spark-3.3.3-bin-hadoop3/bin/spark-class
    spark-3.3.3-bin-hadoop3/bin/spark-class.cmd
    spark-3.3.3-bin-hadoop3/bin/spark-class2.cmd
    spark-3.3.3-bin-hadoop3/bin/spark-shell
    spark-3.3.3-bin-hadoop3/bin/spark-shell.cmd
    spark-3.3.3-bin-hadoop3/bin/spark-shell2.cmd
    spark-3.3.3-bin-hadoop3/bin/spark-sql
    spark-3.3.3-bin-hadoop3/bin/spark-sql.cmd
    spark-3.3.3-bin-hadoop3/bin/spark-sql2.cmd
    spark-3.3.3-bin-hadoop3/bin/spark-submit
    spark-3.3.3-bin-hadoop3/bin/spark-submit.cmd
    spark-3.3.3-bin-hadoop3/bin/spark-submit2.cmd
    spark-3.3.3-bin-hadoop3/bin/sparkR
    spark-3.3.3-bin-hadoop3/bin/sparkR.cmd
    spark-3.3.3-bin-hadoop3/bin/sparkR2.cmd
    spark-3.3.3-bin-hadoop3/conf/
    spark-3.3.3-bin-hadoop3/conf/fairscheduler.xml.template
    spark-3.3.3-bin-hadoop3/conf/log4j2.properties.template
    spark-3.3.3-bin-hadoop3/conf/metrics.properties.template
    spark-3.3.3-bin-hadoop3/conf/spark-defaults.conf.template
    spark-3.3.3-bin-hadoop3/conf/spark-env.sh.template
    spark-3.3.3-bin-hadoop3/conf/workers.template
    spark-3.3.3-bin-hadoop3/data/
    spark-3.3.3-bin-hadoop3/data/graphx/
    spark-3.3.3-bin-hadoop3/data/graphx/followers.txt
    spark-3.3.3-bin-hadoop3/data/graphx/users.txt
    spark-3.3.3-bin-hadoop3/data/mllib/
    spark-3.3.3-bin-hadoop3/data/mllib/als/
    spark-3.3.3-bin-hadoop3/data/mllib/als/sample_movielens_ratings.txt
    spark-3.3.3-bin-hadoop3/data/mllib/als/test.data
    spark-3.3.3-bin-hadoop3/data/mllib/gmm_data.txt
    spark-3.3.3-bin-hadoop3/data/mllib/images/
    spark-3.3.3-bin-hadoop3/data/mllib/images/license.txt
    spark-3.3.3-bin-hadoop3/data/mllib/images/origin/
    spark-3.3.3-bin-hadoop3/data/mllib/images/origin/kittens/
    spark-3.3.3-bin-hadoop3/data/mllib/images/origin/kittens/29.5.a_b_EGDP022204.jpg
    spark-3.3.3-bin-hadoop3/data/mllib/images/origin/kittens/54893.jpg
    spark-3.3.3-bin-hadoop3/data/mllib/images/origin/kittens/DP153539.jpg
    spark-3.3.3-bin-hadoop3/data/mllib/images/origin/kittens/DP802813.jpg
    spark-3.3.3-bin-hadoop3/data/mllib/images/origin/kittens/not-image.txt
    spark-3.3.3-bin-hadoop3/data/mllib/images/origin/license.txt
    spark-3.3.3-bin-hadoop3/data/mllib/images/origin/multi-channel/
    spark-3.3.3-bin-hadoop3/data/mllib/images/origin/multi-channel/BGRA.png
    spark-3.3.3-bin-hadoop3/data/mllib/images/origin/multi-channel/BGRA_alpha_60.png
    spark-3.3.3-bin-hadoop3/data/mllib/images/origin/multi-channel/chr30.4.184.jpg
    spark-3.3.3-bin-hadoop3/data/mllib/images/origin/multi-channel/grayscale.jpg
    spark-3.3.3-bin-hadoop3/data/mllib/kmeans_data.txt
    spark-3.3.3-bin-hadoop3/data/mllib/pagerank_data.txt
    spark-3.3.3-bin-hadoop3/data/mllib/pic_data.txt
    spark-3.3.3-bin-hadoop3/data/mllib/ridge-data/
    spark-3.3.3-bin-hadoop3/data/mllib/ridge-data/lpsa.data
    spark-3.3.3-bin-hadoop3/data/mllib/sample_binary_classification_data.txt
    spark-3.3.3-bin-hadoop3/data/mllib/sample_fpgrowth.txt
    spark-3.3.3-bin-hadoop3/data/mllib/sample_isotonic_regression_libsvm_data.txt
    spark-3.3.3-bin-hadoop3/data/mllib/sample_kmeans_data.txt
    spark-3.3.3-bin-hadoop3/data/mllib/sample_lda_data.txt
    spark-3.3.3-bin-hadoop3/data/mllib/sample_lda_libsvm_data.txt
    spark-3.3.3-bin-hadoop3/data/mllib/sample_libsvm_data.txt
    spark-3.3.3-bin-hadoop3/data/mllib/sample_linear_regression_data.txt
    spark-3.3.3-bin-hadoop3/data/mllib/sample_movielens_data.txt
    spark-3.3.3-bin-hadoop3/data/mllib/sample_multiclass_classification_data.txt
    spark-3.3.3-bin-hadoop3/data/mllib/sample_svm_data.txt
    spark-3.3.3-bin-hadoop3/data/mllib/streaming_kmeans_data_test.txt
    spark-3.3.3-bin-hadoop3/data/streaming/
    spark-3.3.3-bin-hadoop3/data/streaming/AFINN-111.txt
    spark-3.3.3-bin-hadoop3/examples/
    spark-3.3.3-bin-hadoop3/examples/jars/
    spark-3.3.3-bin-hadoop3/examples/jars/scopt_2.12-3.7.1.jar
    spark-3.3.3-bin-hadoop3/examples/jars/spark-examples_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/examples/src/
    spark-3.3.3-bin-hadoop3/examples/src/main/
    spark-3.3.3-bin-hadoop3/examples/src/main/java/
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/JavaHdfsLR.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/JavaLogQuery.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/JavaPageRank.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/JavaSparkPi.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/JavaStatusTrackerDemo.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/JavaTC.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/JavaWordCount.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaAFTSurvivalRegressionExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaALSExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaBinarizerExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaBisectingKMeansExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaBucketedRandomProjectionLSHExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaBucketizerExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaChiSqSelectorExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaChiSquareTestExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaCorrelationExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaCountVectorizerExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaDCTExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaDecisionTreeClassificationExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaDecisionTreeRegressionExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaDocument.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaElementwiseProductExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaEstimatorTransformerParamExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaFMClassifierExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaFMRegressorExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaFPGrowthExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaFeatureHasherExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaGaussianMixtureExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaGeneralizedLinearRegressionExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaGradientBoostedTreeClassifierExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaGradientBoostedTreeRegressorExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaImputerExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaIndexToStringExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaInteractionExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaIsotonicRegressionExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaKMeansExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaLDAExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaLabeledDocument.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaLinearRegressionWithElasticNetExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaLinearSVCExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaLogisticRegressionSummaryExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaLogisticRegressionWithElasticNetExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaMaxAbsScalerExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaMinHashLSHExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaMinMaxScalerExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaModelSelectionViaCrossValidationExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaModelSelectionViaTrainValidationSplitExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaMulticlassLogisticRegressionWithElasticNetExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaMultilayerPerceptronClassifierExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaNGramExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaNaiveBayesExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaNormalizerExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaOneHotEncoderExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaOneVsRestExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaPCAExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaPipelineExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaPolynomialExpansionExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaPowerIterationClusteringExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaPrefixSpanExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaQuantileDiscretizerExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaRFormulaExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaRandomForestClassifierExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaRandomForestRegressorExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaRobustScalerExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaSQLTransformerExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaStandardScalerExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaStopWordsRemoverExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaStringIndexerExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaSummarizerExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaTfIdfExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaTokenizerExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaUnivariateFeatureSelectorExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaVarianceThresholdSelectorExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaVectorAssemblerExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaVectorIndexerExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaVectorSizeHintExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaVectorSlicerExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/ml/JavaWord2VecExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaALS.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaAssociationRulesExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaBinaryClassificationMetricsExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaBisectingKMeansExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaChiSqSelectorExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaCorrelationsExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaDecisionTreeClassificationExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaDecisionTreeRegressionExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaElementwiseProductExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaGaussianMixtureExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaGradientBoostingClassificationExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaGradientBoostingRegressionExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaHypothesisTestingExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaHypothesisTestingKolmogorovSmirnovTestExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaIsotonicRegressionExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaKMeansExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaKernelDensityEstimationExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaLBFGSExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaLatentDirichletAllocationExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaLogisticRegressionWithLBFGSExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaMultiLabelClassificationMetricsExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaMulticlassClassificationMetricsExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaNaiveBayesExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaPCAExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaPowerIterationClusteringExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaPrefixSpanExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaRandomForestClassificationExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaRandomForestRegressionExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaRankingMetricsExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaRecommendationExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaSVDExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaSVMWithSGDExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaSimpleFPGrowth.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaStratifiedSamplingExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaStreamingTestExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/mllib/JavaSummaryStatisticsExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/sql/
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/sql/JavaSQLDataSourceExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/sql/JavaSparkSQLExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/sql/JavaUserDefinedScalar.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/sql/JavaUserDefinedTypedAggregation.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/sql/JavaUserDefinedUntypedAggregation.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/sql/hive/
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/sql/hive/JavaSparkHiveExample.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/sql/streaming/
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/sql/streaming/JavaStructuredComplexSessionization.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/sql/streaming/JavaStructuredKafkaWordCount.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/sql/streaming/JavaStructuredKerberizedKafkaWordCount.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/sql/streaming/JavaStructuredNetworkWordCount.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/sql/streaming/JavaStructuredNetworkWordCountWindowed.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/sql/streaming/JavaStructuredSessionization.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/streaming/
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/streaming/JavaCustomReceiver.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/streaming/JavaDirectKafkaWordCount.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/streaming/JavaDirectKerberizedKafkaWordCount.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/streaming/JavaNetworkWordCount.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/streaming/JavaQueueStream.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/streaming/JavaRecord.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/streaming/JavaRecoverableNetworkWordCount.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/streaming/JavaSqlNetworkWordCount.java
    spark-3.3.3-bin-hadoop3/examples/src/main/java/org/apache/spark/examples/streaming/JavaStatefulNetworkWordCount.java
    spark-3.3.3-bin-hadoop3/examples/src/main/python/
    spark-3.3.3-bin-hadoop3/examples/src/main/python/__init__.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/als.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/avro_inputformat.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/kmeans.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/logistic_regression.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/__init__,py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/aft_survival_regression.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/als_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/binarizer_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/bisecting_k_means_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/bucketed_random_projection_lsh_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/bucketizer_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/chi_square_test_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/chisq_selector_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/correlation_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/count_vectorizer_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/cross_validator.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/dataframe_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/dct_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/decision_tree_classification_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/decision_tree_regression_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/elementwise_product_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/estimator_transformer_param_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/feature_hasher_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/fm_classifier_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/fm_regressor_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/fpgrowth_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/gaussian_mixture_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/generalized_linear_regression_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/gradient_boosted_tree_classifier_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/gradient_boosted_tree_regressor_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/imputer_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/index_to_string_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/interaction_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/isotonic_regression_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/kmeans_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/lda_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/linear_regression_with_elastic_net.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/linearsvc.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/logistic_regression_summary_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/logistic_regression_with_elastic_net.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/max_abs_scaler_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/min_hash_lsh_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/min_max_scaler_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/multiclass_logistic_regression_with_elastic_net.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/multilayer_perceptron_classification.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/n_gram_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/naive_bayes_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/normalizer_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/one_vs_rest_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/onehot_encoder_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/pca_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/pipeline_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/polynomial_expansion_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/power_iteration_clustering_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/prefixspan_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/quantile_discretizer_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/random_forest_classifier_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/random_forest_regressor_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/rformula_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/robust_scaler_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/sql_transformer.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/standard_scaler_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/stopwords_remover_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/string_indexer_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/summarizer_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/tf_idf_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/tokenizer_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/train_validation_split.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/univariate_feature_selector_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/variance_threshold_selector_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/vector_assembler_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/vector_indexer_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/vector_size_hint_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/vector_slicer_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/ml/word2vec_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/__init__.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/binary_classification_metrics_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/bisecting_k_means_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/correlations.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/correlations_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/decision_tree_classification_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/decision_tree_regression_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/elementwise_product_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/fpgrowth_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/gaussian_mixture_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/gaussian_mixture_model.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/gradient_boosting_classification_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/gradient_boosting_regression_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/hypothesis_testing_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/hypothesis_testing_kolmogorov_smirnov_test_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/isotonic_regression_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/k_means_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/kernel_density_estimation_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/kmeans.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/latent_dirichlet_allocation_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/linear_regression_with_sgd_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/logistic_regression.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/logistic_regression_with_lbfgs_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/multi_class_metrics_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/multi_label_metrics_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/naive_bayes_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/normalizer_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/pca_rowmatrix_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/power_iteration_clustering_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/random_forest_classification_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/random_forest_regression_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/random_rdd_generation.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/ranking_metrics_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/recommendation_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/regression_metrics_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/sampled_rdds.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/standard_scaler_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/stratified_sampling_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/streaming_k_means_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/streaming_linear_regression_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/summary_statistics_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/svd_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/svm_with_sgd_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/tf_idf_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/word2vec.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/mllib/word2vec_example.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/pagerank.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/parquet_inputformat.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/pi.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/sort.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/sql/
    spark-3.3.3-bin-hadoop3/examples/src/main/python/sql/__init__.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/sql/arrow.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/sql/basic.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/sql/datasource.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/sql/hive.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/sql/streaming/
    spark-3.3.3-bin-hadoop3/examples/src/main/python/sql/streaming/__init__,py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/sql/streaming/structured_kafka_wordcount.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/sql/streaming/structured_network_wordcount.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/sql/streaming/structured_network_wordcount_windowed.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/sql/streaming/structured_sessionization.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/status_api_demo.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/streaming/
    spark-3.3.3-bin-hadoop3/examples/src/main/python/streaming/__init__.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/streaming/hdfs_wordcount.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/streaming/network_wordcount.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/streaming/network_wordjoinsentiments.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/streaming/queue_stream.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/streaming/recoverable_network_wordcount.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/streaming/sql_network_wordcount.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/streaming/stateful_network_wordcount.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/transitive_closure.py
    spark-3.3.3-bin-hadoop3/examples/src/main/python/wordcount.py
    spark-3.3.3-bin-hadoop3/examples/src/main/r/
    spark-3.3.3-bin-hadoop3/examples/src/main/r/RSparkSQLExample.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/data-manipulation.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/dataframe.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/als.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/bisectingKmeans.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/decisionTree.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/fmClassifier.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/fmRegressor.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/fpm.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/gaussianMixture.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/gbt.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/glm.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/isoreg.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/kmeans.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/kstest.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/lda.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/lm_with_elastic_net.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/logit.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/ml.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/mlp.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/naiveBayes.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/powerIterationClustering.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/prefixSpan.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/randomForest.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/survreg.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/ml/svmLinear.R
    spark-3.3.3-bin-hadoop3/examples/src/main/r/streaming/
    spark-3.3.3-bin-hadoop3/examples/src/main/r/streaming/structured_network_wordcount.R
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/META-INF/
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/META-INF/services/
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/META-INF/services/org.apache.spark.sql.SparkSessionExtensionsProvider
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/META-INF/services/org.apache.spark.sql.jdbc.JdbcConnectionProvider
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/dir1/
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/dir1/dir2/
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/dir1/dir2/file2.parquet
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/dir1/file1.parquet
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/dir1/file3.json
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/employees.json
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/full_user.avsc
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/kv1.txt
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/people.csv
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/people.json
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/people.txt
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/user.avsc
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/users.avro
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/users.orc
    spark-3.3.3-bin-hadoop3/examples/src/main/resources/users.parquet
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/AccumulatorMetricsTest.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/BroadcastTest.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/DFSReadWriteTest.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/DriverSubmissionTest.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ExceptionHandlingTest.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/GroupByTest.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/HdfsTest.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/LocalALS.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/LocalFileLR.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/LocalKMeans.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/LocalLR.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/LocalPi.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/LogQuery.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/MiniReadWriteTest.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/MultiBroadcastTest.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/SimpleSkewedGroupByTest.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/SkewedGroupByTest.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/SparkALS.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/SparkHdfsLR.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/SparkKMeans.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/SparkLR.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/SparkPageRank.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/SparkPi.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/SparkRemoteFileTest.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/SparkTC.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/extensions/
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/extensions/AgeExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/extensions/SessionExtensionsWithLoader.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/extensions/SessionExtensionsWithoutLoader.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/extensions/SparkSessionExtensionsTest.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/graphx/
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/graphx/AggregateMessagesExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/graphx/Analytics.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/graphx/ComprehensiveExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/graphx/ConnectedComponentsExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/graphx/LiveJournalPageRank.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/graphx/PageRankExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/graphx/SSSPExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/graphx/SynthBenchmark.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/graphx/TriangleCountingExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/AFTSurvivalRegressionExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/ALSExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/BinarizerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/BisectingKMeansExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/BucketedRandomProjectionLSHExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/BucketizerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/ChiSqSelectorExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/ChiSquareTestExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/CorrelationExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/CountVectorizerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/DCTExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/DataFrameExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/DecisionTreeClassificationExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/DecisionTreeExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/DecisionTreeRegressionExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/DeveloperApiExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/ElementwiseProductExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/EstimatorTransformerParamExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/FMClassifierExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/FMRegressorExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/FPGrowthExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/FeatureHasherExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/GBTExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/GaussianMixtureExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/GeneralizedLinearRegressionExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/GradientBoostedTreeClassifierExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/GradientBoostedTreeRegressorExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/ImputerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/IndexToStringExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/InteractionExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/IsotonicRegressionExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/KMeansExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/LDAExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/LinearRegressionExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/LinearRegressionWithElasticNetExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/LinearSVCExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/LogisticRegressionExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/LogisticRegressionSummaryExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/LogisticRegressionWithElasticNetExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/MaxAbsScalerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/MinHashLSHExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/MinMaxScalerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/ModelSelectionViaCrossValidationExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/ModelSelectionViaTrainValidationSplitExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/MulticlassLogisticRegressionWithElasticNetExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/MultilayerPerceptronClassifierExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/NGramExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/NaiveBayesExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/NormalizerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/OneHotEncoderExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/OneVsRestExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/PCAExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/PipelineExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/PolynomialExpansionExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/PowerIterationClusteringExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/PrefixSpanExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/QuantileDiscretizerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/RFormulaExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/RandomForestClassifierExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/RandomForestExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/RandomForestRegressorExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/RobustScalerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/SQLTransformerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/StandardScalerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/StopWordsRemoverExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/StringIndexerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/SummarizerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/TfIdfExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/TokenizerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/UnaryTransformerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/UnivariateFeatureSelectorExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/VarianceThresholdSelectorExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/VectorAssemblerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/VectorIndexerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/VectorSizeHintExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/VectorSlicerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/ml/Word2VecExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/AbstractParams.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/AssociationRulesExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/BinaryClassification.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/BinaryClassificationMetricsExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/BisectingKMeansExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/ChiSqSelectorExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/Correlations.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/CorrelationsExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/CosineSimilarity.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/DecisionTreeClassificationExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/DecisionTreeRegressionExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/DecisionTreeRunner.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/DenseKMeans.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/ElementwiseProductExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/FPGrowthExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/GaussianMixtureExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/GradientBoostedTreesRunner.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/GradientBoostingClassificationExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/GradientBoostingRegressionExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/HypothesisTestingExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/HypothesisTestingKolmogorovSmirnovTestExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/IsotonicRegressionExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/KMeansExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/KernelDensityEstimationExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/LBFGSExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/LDAExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/LatentDirichletAllocationExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/LogisticRegressionWithLBFGSExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/MovieLensALS.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/MultiLabelMetricsExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/MulticlassMetricsExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/MultivariateSummarizer.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/NaiveBayesExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/NormalizerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/PCAOnRowMatrixExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/PCAOnSourceVectorExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/PMMLModelExportExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/PowerIterationClusteringExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/PrefixSpanExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/RandomForestClassificationExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/RandomForestRegressionExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/RandomRDDGeneration.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/RankingMetricsExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/RecommendationExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/SVDExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/SVMWithSGDExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/SampledRDDs.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/SimpleFPGrowth.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/SparseNaiveBayes.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/StandardScalerExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/StratifiedSamplingExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/StreamingKMeansExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/StreamingLinearRegressionExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/StreamingLogisticRegression.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/StreamingTestExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/SummaryStatisticsExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/TFIDFExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/TallSkinnyPCA.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/TallSkinnySVD.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/mllib/Word2VecExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/pythonconverters/
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/pythonconverters/AvroConverters.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/RDDRelation.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/SQLDataSourceExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/SimpleTypedAggregator.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/SparkSQLExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/UserDefinedScalar.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/UserDefinedTypedAggregation.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/UserDefinedUntypedAggregation.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/hive/
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/hive/SparkHiveExample.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/jdbc/
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/jdbc/ExampleJdbcConnectionProvider.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/streaming/
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/streaming/StructuredComplexSessionization.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/streaming/StructuredKafkaWordCount.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/streaming/StructuredKerberizedKafkaWordCount.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/streaming/StructuredNetworkWordCount.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/streaming/StructuredNetworkWordCountWindowed.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/sql/streaming/StructuredSessionization.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/streaming/
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/streaming/CustomReceiver.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/streaming/DirectKafkaWordCount.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/streaming/DirectKerberizedKafkaWordCount.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/streaming/HdfsWordCount.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/streaming/NetworkWordCount.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/streaming/QueueStream.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/streaming/RawNetworkGrep.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/streaming/RecoverableNetworkWordCount.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/streaming/SqlNetworkWordCount.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/streaming/StatefulNetworkWordCount.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/streaming/StreamingExamples.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/streaming/clickstream/
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/streaming/clickstream/PageViewGenerator.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scala/org/apache/spark/examples/streaming/clickstream/PageViewStream.scala
    spark-3.3.3-bin-hadoop3/examples/src/main/scripts/
    spark-3.3.3-bin-hadoop3/examples/src/main/scripts/getGpusResources.sh
    spark-3.3.3-bin-hadoop3/jars/
    spark-3.3.3-bin-hadoop3/jars/HikariCP-2.5.1.jar
    spark-3.3.3-bin-hadoop3/jars/JLargeArrays-1.5.jar
    spark-3.3.3-bin-hadoop3/jars/JTransforms-3.1.jar
    spark-3.3.3-bin-hadoop3/jars/RoaringBitmap-0.9.25.jar
    spark-3.3.3-bin-hadoop3/jars/ST4-4.0.4.jar
    spark-3.3.3-bin-hadoop3/jars/activation-1.1.1.jar
    spark-3.3.3-bin-hadoop3/jars/aircompressor-0.21.jar
    spark-3.3.3-bin-hadoop3/jars/algebra_2.12-2.0.1.jar
    spark-3.3.3-bin-hadoop3/jars/annotations-17.0.0.jar
    spark-3.3.3-bin-hadoop3/jars/antlr-runtime-3.5.2.jar
    spark-3.3.3-bin-hadoop3/jars/antlr4-runtime-4.8.jar
    spark-3.3.3-bin-hadoop3/jars/aopalliance-repackaged-2.6.1.jar
    spark-3.3.3-bin-hadoop3/jars/arpack-2.2.1.jar
    spark-3.3.3-bin-hadoop3/jars/arpack_combined_all-0.1.jar
    spark-3.3.3-bin-hadoop3/jars/arrow-format-7.0.0.jar
    spark-3.3.3-bin-hadoop3/jars/arrow-memory-core-7.0.0.jar
    spark-3.3.3-bin-hadoop3/jars/arrow-memory-netty-7.0.0.jar
    spark-3.3.3-bin-hadoop3/jars/arrow-vector-7.0.0.jar
    spark-3.3.3-bin-hadoop3/jars/audience-annotations-0.5.0.jar
    spark-3.3.3-bin-hadoop3/jars/automaton-1.11-8.jar
    spark-3.3.3-bin-hadoop3/jars/avro-1.11.0.jar
    spark-3.3.3-bin-hadoop3/jars/avro-ipc-1.11.0.jar
    spark-3.3.3-bin-hadoop3/jars/avro-mapred-1.11.0.jar
    spark-3.3.3-bin-hadoop3/jars/blas-2.2.1.jar
    spark-3.3.3-bin-hadoop3/jars/bonecp-0.8.0.RELEASE.jar
    spark-3.3.3-bin-hadoop3/jars/breeze-macros_2.12-1.2.jar
    spark-3.3.3-bin-hadoop3/jars/breeze_2.12-1.2.jar
    spark-3.3.3-bin-hadoop3/jars/cats-kernel_2.12-2.1.1.jar
    spark-3.3.3-bin-hadoop3/jars/chill-java-0.10.0.jar
    spark-3.3.3-bin-hadoop3/jars/chill_2.12-0.10.0.jar
    spark-3.3.3-bin-hadoop3/jars/commons-cli-1.5.0.jar
    spark-3.3.3-bin-hadoop3/jars/commons-codec-1.15.jar
    spark-3.3.3-bin-hadoop3/jars/commons-collections-3.2.2.jar
    spark-3.3.3-bin-hadoop3/jars/commons-collections4-4.4.jar
    spark-3.3.3-bin-hadoop3/jars/commons-compiler-3.0.16.jar
    spark-3.3.3-bin-hadoop3/jars/commons-compress-1.21.jar
    spark-3.3.3-bin-hadoop3/jars/commons-crypto-1.1.0.jar
    spark-3.3.3-bin-hadoop3/jars/commons-dbcp-1.4.jar
    spark-3.3.3-bin-hadoop3/jars/commons-io-2.11.0.jar
    spark-3.3.3-bin-hadoop3/jars/commons-lang-2.6.jar
    spark-3.3.3-bin-hadoop3/jars/commons-lang3-3.12.0.jar
    spark-3.3.3-bin-hadoop3/jars/commons-logging-1.1.3.jar
    spark-3.3.3-bin-hadoop3/jars/commons-math3-3.6.1.jar
    spark-3.3.3-bin-hadoop3/jars/commons-pool-1.5.4.jar
    spark-3.3.3-bin-hadoop3/jars/commons-text-1.10.0.jar
    spark-3.3.3-bin-hadoop3/jars/compress-lzf-1.1.jar
    spark-3.3.3-bin-hadoop3/jars/core-1.1.2.jar
    spark-3.3.3-bin-hadoop3/jars/curator-client-2.13.0.jar
    spark-3.3.3-bin-hadoop3/jars/curator-framework-2.13.0.jar
    spark-3.3.3-bin-hadoop3/jars/curator-recipes-2.13.0.jar
    spark-3.3.3-bin-hadoop3/jars/datanucleus-api-jdo-4.2.4.jar
    spark-3.3.3-bin-hadoop3/jars/datanucleus-core-4.1.17.jar
    spark-3.3.3-bin-hadoop3/jars/datanucleus-rdbms-4.1.19.jar
    spark-3.3.3-bin-hadoop3/jars/derby-10.14.2.0.jar
    spark-3.3.3-bin-hadoop3/jars/dropwizard-metrics-hadoop-metrics2-reporter-0.1.2.jar
    spark-3.3.3-bin-hadoop3/jars/flatbuffers-java-1.12.0.jar
    spark-3.3.3-bin-hadoop3/jars/generex-1.0.2.jar
    spark-3.3.3-bin-hadoop3/jars/gson-2.2.4.jar
    spark-3.3.3-bin-hadoop3/jars/guava-14.0.1.jar
    spark-3.3.3-bin-hadoop3/jars/hadoop-client-api-3.3.2.jar
    spark-3.3.3-bin-hadoop3/jars/hadoop-client-runtime-3.3.2.jar
    spark-3.3.3-bin-hadoop3/jars/hadoop-shaded-guava-1.1.1.jar
    spark-3.3.3-bin-hadoop3/jars/hadoop-yarn-server-web-proxy-3.3.2.jar
    spark-3.3.3-bin-hadoop3/jars/hive-beeline-2.3.9.jar
    spark-3.3.3-bin-hadoop3/jars/hive-cli-2.3.9.jar
    spark-3.3.3-bin-hadoop3/jars/hive-common-2.3.9.jar
    spark-3.3.3-bin-hadoop3/jars/hive-exec-2.3.9-core.jar
    spark-3.3.3-bin-hadoop3/jars/hive-jdbc-2.3.9.jar
    spark-3.3.3-bin-hadoop3/jars/hive-llap-common-2.3.9.jar
    spark-3.3.3-bin-hadoop3/jars/hive-metastore-2.3.9.jar
    spark-3.3.3-bin-hadoop3/jars/hive-serde-2.3.9.jar
    spark-3.3.3-bin-hadoop3/jars/hive-service-rpc-3.1.2.jar
    spark-3.3.3-bin-hadoop3/jars/hive-shims-0.23-2.3.9.jar
    spark-3.3.3-bin-hadoop3/jars/hive-shims-2.3.9.jar
    spark-3.3.3-bin-hadoop3/jars/hive-shims-common-2.3.9.jar
    spark-3.3.3-bin-hadoop3/jars/hive-shims-scheduler-2.3.9.jar
    spark-3.3.3-bin-hadoop3/jars/hive-storage-api-2.7.2.jar
    spark-3.3.3-bin-hadoop3/jars/hive-vector-code-gen-2.3.9.jar
    spark-3.3.3-bin-hadoop3/jars/hk2-api-2.6.1.jar
    spark-3.3.3-bin-hadoop3/jars/hk2-locator-2.6.1.jar
    spark-3.3.3-bin-hadoop3/jars/hk2-utils-2.6.1.jar
    spark-3.3.3-bin-hadoop3/jars/httpclient-4.5.13.jar
    spark-3.3.3-bin-hadoop3/jars/httpcore-4.4.14.jar
    spark-3.3.3-bin-hadoop3/jars/istack-commons-runtime-3.0.8.jar
    spark-3.3.3-bin-hadoop3/jars/ivy-2.5.1.jar
    spark-3.3.3-bin-hadoop3/jars/jackson-annotations-2.13.4.jar
    spark-3.3.3-bin-hadoop3/jars/jackson-core-2.13.4.jar
    spark-3.3.3-bin-hadoop3/jars/jackson-core-asl-1.9.13.jar
    spark-3.3.3-bin-hadoop3/jars/jackson-databind-2.13.4.2.jar
    spark-3.3.3-bin-hadoop3/jars/jackson-dataformat-yaml-2.13.4.jar
    spark-3.3.3-bin-hadoop3/jars/jackson-datatype-jsr310-2.13.4.jar
    spark-3.3.3-bin-hadoop3/jars/jackson-mapper-asl-1.9.13.jar
    spark-3.3.3-bin-hadoop3/jars/jackson-module-scala_2.12-2.13.4.jar
    spark-3.3.3-bin-hadoop3/jars/jakarta.annotation-api-1.3.5.jar
    spark-3.3.3-bin-hadoop3/jars/jakarta.inject-2.6.1.jar
    spark-3.3.3-bin-hadoop3/jars/jakarta.servlet-api-4.0.3.jar
    spark-3.3.3-bin-hadoop3/jars/jakarta.validation-api-2.0.2.jar
    spark-3.3.3-bin-hadoop3/jars/jakarta.ws.rs-api-2.1.6.jar
    spark-3.3.3-bin-hadoop3/jars/jakarta.xml.bind-api-2.3.2.jar
    spark-3.3.3-bin-hadoop3/jars/janino-3.0.16.jar
    spark-3.3.3-bin-hadoop3/jars/javassist-3.25.0-GA.jar
    spark-3.3.3-bin-hadoop3/jars/javax.jdo-3.2.0-m3.jar
    spark-3.3.3-bin-hadoop3/jars/javolution-5.5.1.jar
    spark-3.3.3-bin-hadoop3/jars/jaxb-runtime-2.3.2.jar
    spark-3.3.3-bin-hadoop3/jars/jcl-over-slf4j-1.7.32.jar
    spark-3.3.3-bin-hadoop3/jars/jdo-api-3.0.1.jar
    spark-3.3.3-bin-hadoop3/jars/jersey-client-2.36.jar
    spark-3.3.3-bin-hadoop3/jars/jersey-common-2.36.jar
    spark-3.3.3-bin-hadoop3/jars/jersey-container-servlet-2.36.jar
    spark-3.3.3-bin-hadoop3/jars/jersey-container-servlet-core-2.36.jar
    spark-3.3.3-bin-hadoop3/jars/jersey-hk2-2.36.jar
    spark-3.3.3-bin-hadoop3/jars/jersey-server-2.36.jar
    spark-3.3.3-bin-hadoop3/jars/jline-2.14.6.jar
    spark-3.3.3-bin-hadoop3/jars/joda-time-2.10.13.jar
    spark-3.3.3-bin-hadoop3/jars/jodd-core-3.5.2.jar
    spark-3.3.3-bin-hadoop3/jars/jpam-1.1.jar
    spark-3.3.3-bin-hadoop3/jars/json-1.8.jar
    spark-3.3.3-bin-hadoop3/jars/json4s-ast_2.12-3.7.0-M11.jar
    spark-3.3.3-bin-hadoop3/jars/json4s-core_2.12-3.7.0-M11.jar
    spark-3.3.3-bin-hadoop3/jars/json4s-jackson_2.12-3.7.0-M11.jar
    spark-3.3.3-bin-hadoop3/jars/json4s-scalap_2.12-3.7.0-M11.jar
    spark-3.3.3-bin-hadoop3/jars/jsr305-3.0.0.jar
    spark-3.3.3-bin-hadoop3/jars/jta-1.1.jar
    spark-3.3.3-bin-hadoop3/jars/jul-to-slf4j-1.7.32.jar
    spark-3.3.3-bin-hadoop3/jars/kryo-shaded-4.0.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-client-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-admissionregistration-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-apiextensions-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-apps-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-autoscaling-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-batch-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-certificates-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-common-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-coordination-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-core-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-discovery-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-events-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-extensions-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-flowcontrol-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-metrics-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-networking-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-node-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-policy-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-rbac-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-scheduling-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/kubernetes-model-storageclass-5.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/lapack-2.2.1.jar
    spark-3.3.3-bin-hadoop3/jars/leveldbjni-all-1.8.jar
    spark-3.3.3-bin-hadoop3/jars/libfb303-0.9.3.jar
    spark-3.3.3-bin-hadoop3/jars/libthrift-0.12.0.jar
    spark-3.3.3-bin-hadoop3/jars/log4j-1.2-api-2.17.2.jar
    spark-3.3.3-bin-hadoop3/jars/log4j-api-2.17.2.jar
    spark-3.3.3-bin-hadoop3/jars/log4j-core-2.17.2.jar
    spark-3.3.3-bin-hadoop3/jars/log4j-slf4j-impl-2.17.2.jar
    spark-3.3.3-bin-hadoop3/jars/logging-interceptor-3.12.12.jar
    spark-3.3.3-bin-hadoop3/jars/lz4-java-1.8.0.jar
    spark-3.3.3-bin-hadoop3/jars/mesos-1.4.3-shaded-protobuf.jar
    spark-3.3.3-bin-hadoop3/jars/metrics-core-4.2.7.jar
    spark-3.3.3-bin-hadoop3/jars/metrics-graphite-4.2.7.jar
    spark-3.3.3-bin-hadoop3/jars/metrics-jmx-4.2.7.jar
    spark-3.3.3-bin-hadoop3/jars/metrics-json-4.2.7.jar
    spark-3.3.3-bin-hadoop3/jars/metrics-jvm-4.2.7.jar
    spark-3.3.3-bin-hadoop3/jars/minlog-1.3.0.jar
    spark-3.3.3-bin-hadoop3/jars/netty-all-4.1.74.Final.jar
    spark-3.3.3-bin-hadoop3/jars/netty-buffer-4.1.74.Final.jar
    spark-3.3.3-bin-hadoop3/jars/netty-codec-4.1.74.Final.jar
    spark-3.3.3-bin-hadoop3/jars/netty-common-4.1.74.Final.jar
    spark-3.3.3-bin-hadoop3/jars/netty-handler-4.1.74.Final.jar
    spark-3.3.3-bin-hadoop3/jars/netty-resolver-4.1.74.Final.jar
    spark-3.3.3-bin-hadoop3/jars/netty-tcnative-classes-2.0.48.Final.jar
    spark-3.3.3-bin-hadoop3/jars/netty-transport-4.1.74.Final.jar
    spark-3.3.3-bin-hadoop3/jars/netty-transport-classes-epoll-4.1.74.Final.jar
    spark-3.3.3-bin-hadoop3/jars/netty-transport-classes-kqueue-4.1.74.Final.jar
    spark-3.3.3-bin-hadoop3/jars/netty-transport-native-epoll-4.1.74.Final-linux-aarch_64.jar
    spark-3.3.3-bin-hadoop3/jars/netty-transport-native-epoll-4.1.74.Final-linux-x86_64.jar
    spark-3.3.3-bin-hadoop3/jars/netty-transport-native-kqueue-4.1.74.Final-osx-aarch_64.jar
    spark-3.3.3-bin-hadoop3/jars/netty-transport-native-kqueue-4.1.74.Final-osx-x86_64.jar
    spark-3.3.3-bin-hadoop3/jars/netty-transport-native-unix-common-4.1.74.Final.jar
    spark-3.3.3-bin-hadoop3/jars/objenesis-3.2.jar
    spark-3.3.3-bin-hadoop3/jars/okhttp-3.12.12.jar
    spark-3.3.3-bin-hadoop3/jars/okio-1.14.0.jar
    spark-3.3.3-bin-hadoop3/jars/opencsv-2.3.jar
    spark-3.3.3-bin-hadoop3/jars/orc-core-1.7.8.jar
    spark-3.3.3-bin-hadoop3/jars/orc-mapreduce-1.7.8.jar
    spark-3.3.3-bin-hadoop3/jars/orc-shims-1.7.8.jar
    spark-3.3.3-bin-hadoop3/jars/oro-2.0.8.jar
    spark-3.3.3-bin-hadoop3/jars/osgi-resource-locator-1.0.3.jar
    spark-3.3.3-bin-hadoop3/jars/paranamer-2.8.jar
    spark-3.3.3-bin-hadoop3/jars/parquet-column-1.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/parquet-common-1.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/parquet-encoding-1.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/parquet-format-structures-1.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/parquet-hadoop-1.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/parquet-jackson-1.12.2.jar
    spark-3.3.3-bin-hadoop3/jars/pickle-1.2.jar
    spark-3.3.3-bin-hadoop3/jars/protobuf-java-2.5.0.jar
    spark-3.3.3-bin-hadoop3/jars/py4j-0.10.9.5.jar
    spark-3.3.3-bin-hadoop3/jars/rocksdbjni-6.20.3.jar
    spark-3.3.3-bin-hadoop3/jars/scala-collection-compat_2.12-2.1.1.jar
    spark-3.3.3-bin-hadoop3/jars/scala-compiler-2.12.15.jar
    spark-3.3.3-bin-hadoop3/jars/scala-library-2.12.15.jar
    spark-3.3.3-bin-hadoop3/jars/scala-parser-combinators_2.12-1.1.2.jar
    spark-3.3.3-bin-hadoop3/jars/scala-reflect-2.12.15.jar
    spark-3.3.3-bin-hadoop3/jars/scala-xml_2.12-1.2.0.jar
    spark-3.3.3-bin-hadoop3/jars/shapeless_2.12-2.3.7.jar
    spark-3.3.3-bin-hadoop3/jars/shims-0.9.25.jar
    spark-3.3.3-bin-hadoop3/jars/slf4j-api-1.7.32.jar
    spark-3.3.3-bin-hadoop3/jars/snakeyaml-1.31.jar
    spark-3.3.3-bin-hadoop3/jars/snappy-java-1.1.8.4.jar
    spark-3.3.3-bin-hadoop3/jars/spark-catalyst_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-core_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-graphx_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-hive-thriftserver_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-hive_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-kubernetes_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-kvstore_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-launcher_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-mesos_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-mllib-local_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-mllib_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-network-common_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-network-shuffle_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-repl_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-sketch_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-sql_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-streaming_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-tags_2.12-3.3.3-tests.jar
    spark-3.3.3-bin-hadoop3/jars/spark-tags_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-unsafe_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spark-yarn_2.12-3.3.3.jar
    spark-3.3.3-bin-hadoop3/jars/spire-macros_2.12-0.17.0.jar
    spark-3.3.3-bin-hadoop3/jars/spire-platform_2.12-0.17.0.jar
    spark-3.3.3-bin-hadoop3/jars/spire-util_2.12-0.17.0.jar
    spark-3.3.3-bin-hadoop3/jars/spire_2.12-0.17.0.jar
    spark-3.3.3-bin-hadoop3/jars/stax-api-1.0.1.jar
    spark-3.3.3-bin-hadoop3/jars/stream-2.9.6.jar
    spark-3.3.3-bin-hadoop3/jars/super-csv-2.2.0.jar
    spark-3.3.3-bin-hadoop3/jars/threeten-extra-1.5.0.jar
    spark-3.3.3-bin-hadoop3/jars/tink-1.6.1.jar
    spark-3.3.3-bin-hadoop3/jars/transaction-api-1.1.jar
    spark-3.3.3-bin-hadoop3/jars/univocity-parsers-2.9.1.jar
    spark-3.3.3-bin-hadoop3/jars/velocity-1.5.jar
    spark-3.3.3-bin-hadoop3/jars/xbean-asm9-shaded-4.20.jar
    spark-3.3.3-bin-hadoop3/jars/xz-1.9.jar
    spark-3.3.3-bin-hadoop3/jars/zjsonpatch-0.3.0.jar
    spark-3.3.3-bin-hadoop3/jars/zookeeper-3.6.2.jar
    spark-3.3.3-bin-hadoop3/jars/zookeeper-jute-3.6.2.jar
    spark-3.3.3-bin-hadoop3/jars/zstd-jni-1.5.2-1.jar
    spark-3.3.3-bin-hadoop3/kubernetes/
    spark-3.3.3-bin-hadoop3/kubernetes/dockerfiles/
    spark-3.3.3-bin-hadoop3/kubernetes/dockerfiles/spark/
    spark-3.3.3-bin-hadoop3/kubernetes/dockerfiles/spark/Dockerfile
    spark-3.3.3-bin-hadoop3/kubernetes/dockerfiles/spark/Dockerfile.java17
    spark-3.3.3-bin-hadoop3/kubernetes/dockerfiles/spark/bindings/
    spark-3.3.3-bin-hadoop3/kubernetes/dockerfiles/spark/bindings/R/
    spark-3.3.3-bin-hadoop3/kubernetes/dockerfiles/spark/bindings/R/Dockerfile
    spark-3.3.3-bin-hadoop3/kubernetes/dockerfiles/spark/bindings/python/
    spark-3.3.3-bin-hadoop3/kubernetes/dockerfiles/spark/bindings/python/Dockerfile
    spark-3.3.3-bin-hadoop3/kubernetes/dockerfiles/spark/decom.sh
    spark-3.3.3-bin-hadoop3/kubernetes/dockerfiles/spark/entrypoint.sh
    spark-3.3.3-bin-hadoop3/kubernetes/tests/
    spark-3.3.3-bin-hadoop3/kubernetes/tests/autoscale.py
    spark-3.3.3-bin-hadoop3/kubernetes/tests/decommissioning.py
    spark-3.3.3-bin-hadoop3/kubernetes/tests/decommissioning_cleanup.py
    spark-3.3.3-bin-hadoop3/kubernetes/tests/py_container_checks.py
    spark-3.3.3-bin-hadoop3/kubernetes/tests/pyfiles.py
    spark-3.3.3-bin-hadoop3/kubernetes/tests/python_executable_check.py
    spark-3.3.3-bin-hadoop3/kubernetes/tests/worker_memory_check.py
    spark-3.3.3-bin-hadoop3/licenses/
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-AnchorJS.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-CC0.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-JLargeArrays.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-JTransforms.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-antlr.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-arpack.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-automaton.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-blas.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-bootstrap.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-cloudpickle.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-d3.min.js.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-dagre-d3.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-datatables.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-dnsjava.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-f2j.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-graphlib-dot.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-istack-commons-runtime.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-jakarta-annotation-api
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-jakarta-ws-rs-api
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-jakarta.activation-api.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-jakarta.xml.bind-api.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-janino.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-javassist.html
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-javax-transaction-transaction-api.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-javolution.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-jaxb-runtime.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-jline.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-jodd.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-join.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-jquery.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-json-formatter.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-jsp-api.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-kryo.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-leveldbjni.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-machinist.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-matchMedia-polyfill.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-minlog.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-modernizr.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-mustache.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-netlib.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-paranamer.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-pmml-model.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-protobuf.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-py4j.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-pyrolite.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-re2j.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-reflectasm.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-respond.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-sbt-launch-lib.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-scala.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-scopt.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-slf4j.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-sorttable.js.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-spire.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-vis-timeline.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-xmlenc.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-zstd-jni.txt
    spark-3.3.3-bin-hadoop3/licenses/LICENSE-zstd.txt
    spark-3.3.3-bin-hadoop3/python/
    spark-3.3.3-bin-hadoop3/python/.coveragerc
    spark-3.3.3-bin-hadoop3/python/.gitignore
    spark-3.3.3-bin-hadoop3/python/MANIFEST.in
    spark-3.3.3-bin-hadoop3/python/README.md
    spark-3.3.3-bin-hadoop3/python/dist/
    spark-3.3.3-bin-hadoop3/python/docs/
    spark-3.3.3-bin-hadoop3/python/docs/Makefile
    spark-3.3.3-bin-hadoop3/python/docs/make.bat
    spark-3.3.3-bin-hadoop3/python/docs/make2.bat
    spark-3.3.3-bin-hadoop3/python/docs/source/
    spark-3.3.3-bin-hadoop3/python/docs/source/_static/
    spark-3.3.3-bin-hadoop3/python/docs/source/_static/copybutton.js
    spark-3.3.3-bin-hadoop3/python/docs/source/_static/css/
    spark-3.3.3-bin-hadoop3/python/docs/source/_static/css/pyspark.css
    spark-3.3.3-bin-hadoop3/python/docs/source/_templates/
    spark-3.3.3-bin-hadoop3/python/docs/source/_templates/autosummary/
    spark-3.3.3-bin-hadoop3/python/docs/source/_templates/autosummary/class.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/_templates/autosummary/class_with_docs.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/conf.py
    spark-3.3.3-bin-hadoop3/python/docs/source/development/
    spark-3.3.3-bin-hadoop3/python/docs/source/development/contributing.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/development/debugging.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/development/index.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/development/setting_ide.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/development/testing.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/getting_started/
    spark-3.3.3-bin-hadoop3/python/docs/source/getting_started/index.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/getting_started/install.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/getting_started/quickstart_df.ipynb
    spark-3.3.3-bin-hadoop3/python/docs/source/getting_started/quickstart_ps.ipynb
    spark-3.3.3-bin-hadoop3/python/docs/source/index.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/migration_guide/
    spark-3.3.3-bin-hadoop3/python/docs/source/migration_guide/index.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/migration_guide/koalas_to_pyspark.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/migration_guide/pyspark_1.0_1.2_to_1.3.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/migration_guide/pyspark_1.4_to_1.5.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/migration_guide/pyspark_2.2_to_2.3.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/migration_guide/pyspark_2.3.0_to_2.3.1_above.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/migration_guide/pyspark_2.3_to_2.4.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/migration_guide/pyspark_2.4_to_3.0.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/migration_guide/pyspark_3.1_to_3.2.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/migration_guide/pyspark_3.2_to_3.3.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/index.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.ml.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.mllib.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.pandas/
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.pandas/extensions.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.pandas/frame.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.pandas/general_functions.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.pandas/groupby.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.pandas/index.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.pandas/indexing.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.pandas/io.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.pandas/ml.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.pandas/series.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.pandas/window.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.resource.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.sql/
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.sql/avro.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.sql/catalog.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.sql/column.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.sql/configuration.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.sql/core_classes.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.sql/data_types.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.sql/dataframe.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.sql/functions.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.sql/grouping.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.sql/index.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.sql/io.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.sql/observation.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.sql/row.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.sql/spark_session.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.sql/window.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.ss/
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.ss/core_classes.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.ss/index.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.ss/io.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.ss/query_management.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/reference/pyspark.streaming.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/arrow_pandas.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/index.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/pandas_on_spark/
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/pandas_on_spark/best_practices.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/pandas_on_spark/faq.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/pandas_on_spark/from_to_dbms.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/pandas_on_spark/index.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/pandas_on_spark/options.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/pandas_on_spark/pandas_pyspark.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/pandas_on_spark/supported_pandas_api.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/pandas_on_spark/transform_apply.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/pandas_on_spark/typehints.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/pandas_on_spark/types.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/python_packaging.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/sql/
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/sql/arrow_pandas.rst
    spark-3.3.3-bin-hadoop3/python/docs/source/user_guide/sql/index.rst
    spark-3.3.3-bin-hadoop3/python/lib/
    spark-3.3.3-bin-hadoop3/python/lib/PY4J_LICENSE.txt
    spark-3.3.3-bin-hadoop3/python/lib/py4j-0.10.9.5-src.zip
    spark-3.3.3-bin-hadoop3/python/lib/pyspark.zip
    spark-3.3.3-bin-hadoop3/python/mypy.ini
    spark-3.3.3-bin-hadoop3/python/pyspark/
    spark-3.3.3-bin-hadoop3/python/pyspark/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/__pycache__/
    spark-3.3.3-bin-hadoop3/python/pyspark/__pycache__/install.cpython-38.pyc
    spark-3.3.3-bin-hadoop3/python/pyspark/_globals.py
    spark-3.3.3-bin-hadoop3/python/pyspark/_typing.pyi
    spark-3.3.3-bin-hadoop3/python/pyspark/accumulators.py
    spark-3.3.3-bin-hadoop3/python/pyspark/broadcast.py
    spark-3.3.3-bin-hadoop3/python/pyspark/cloudpickle/
    spark-3.3.3-bin-hadoop3/python/pyspark/cloudpickle/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/cloudpickle/cloudpickle.py
    spark-3.3.3-bin-hadoop3/python/pyspark/cloudpickle/cloudpickle_fast.py
    spark-3.3.3-bin-hadoop3/python/pyspark/cloudpickle/compat.py
    spark-3.3.3-bin-hadoop3/python/pyspark/conf.py
    spark-3.3.3-bin-hadoop3/python/pyspark/context.py
    spark-3.3.3-bin-hadoop3/python/pyspark/daemon.py
    spark-3.3.3-bin-hadoop3/python/pyspark/files.py
    spark-3.3.3-bin-hadoop3/python/pyspark/find_spark_home.py
    spark-3.3.3-bin-hadoop3/python/pyspark/install.py
    spark-3.3.3-bin-hadoop3/python/pyspark/instrumentation_utils.py
    spark-3.3.3-bin-hadoop3/python/pyspark/java_gateway.py
    spark-3.3.3-bin-hadoop3/python/pyspark/join.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/_typing.pyi
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/base.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/classification.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/clustering.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/common.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/evaluation.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/feature.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/fpm.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/functions.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/image.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/linalg/
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/linalg/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/param/
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/param/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/param/_shared_params_code_gen.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/param/shared.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/pipeline.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/recommendation.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/regression.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/stat.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/test_algorithms.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/test_base.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/test_evaluation.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/test_feature.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/test_image.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/test_linalg.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/test_param.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/test_persistence.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/test_pipeline.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/test_stat.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/test_training_summary.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/test_tuning.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/test_util.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/test_wrapper.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/typing/
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/typing/test_classification.yml
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/typing/test_clustering.yaml
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/typing/test_evaluation.yml
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/typing/test_feature.yml
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/typing/test_param.yml
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/typing/test_readable.yml
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tests/typing/test_regression.yml
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tree.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/tuning.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/util.py
    spark-3.3.3-bin-hadoop3/python/pyspark/ml/wrapper.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/_typing.pyi
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/classification.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/clustering.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/common.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/evaluation.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/feature.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/fpm.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/linalg/
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/linalg/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/linalg/distributed.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/random.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/random.pyi
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/recommendation.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/recommendation.pyi
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/regression.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/stat/
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/stat/KernelDensity.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/stat/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/stat/_statistics.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/stat/distribution.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/stat/test.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/tests/
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/tests/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/tests/test_algorithms.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/tests/test_feature.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/tests/test_linalg.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/tests/test_stat.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/tests/test_streaming_algorithms.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/tests/test_util.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/tree.py
    spark-3.3.3-bin-hadoop3/python/pyspark/mllib/util.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/_typing.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/accessors.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/base.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/categorical.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/config.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/data_type_ops/
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/data_type_ops/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/data_type_ops/base.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/data_type_ops/binary_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/data_type_ops/boolean_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/data_type_ops/categorical_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/data_type_ops/complex_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/data_type_ops/date_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/data_type_ops/datetime_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/data_type_ops/null_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/data_type_ops/num_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/data_type_ops/string_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/data_type_ops/timedelta_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/data_type_ops/udt_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/datetimes.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/exceptions.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/extensions.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/frame.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/generic.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/groupby.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/indexes/
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/indexes/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/indexes/base.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/indexes/category.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/indexes/datetimes.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/indexes/multi.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/indexes/numeric.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/indexes/timedelta.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/indexing.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/internal.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/missing/
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/missing/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/missing/common.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/missing/frame.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/missing/general_functions.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/missing/groupby.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/missing/indexes.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/missing/series.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/missing/window.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/ml.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/mlflow.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/namespace.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/numpy_compat.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/plot/
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/plot/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/plot/core.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/plot/matplotlib.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/plot/plotly.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/series.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/spark/
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/spark/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/spark/accessors.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/spark/functions.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/spark/utils.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/sql_formatter.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/sql_processor.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/strings.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/data_type_ops/
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/data_type_ops/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/data_type_ops/test_base.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/data_type_ops/test_binary_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/data_type_ops/test_boolean_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/data_type_ops/test_categorical_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/data_type_ops/test_complex_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/data_type_ops/test_date_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/data_type_ops/test_datetime_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/data_type_ops/test_null_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/data_type_ops/test_num_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/data_type_ops/test_string_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/data_type_ops/test_timedelta_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/data_type_ops/test_udt_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/data_type_ops/testing_utils.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/indexes/
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/indexes/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/indexes/test_base.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/indexes/test_category.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/indexes/test_datetime.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/indexes/test_timedelta.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/plot/
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/plot/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/plot/test_frame_plot.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/plot/test_frame_plot_matplotlib.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/plot/test_frame_plot_plotly.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/plot/test_series_plot.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/plot/test_series_plot_matplotlib.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/plot/test_series_plot_plotly.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_categorical.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_config.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_csv.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_dataframe.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_dataframe_conversion.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_dataframe_spark_io.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_default_index.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_expanding.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_extension.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_frame_spark.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_groupby.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_indexing.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_indexops_spark.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_internal.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_namespace.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_numpy_compat.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_ops_on_diff_frames.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_ops_on_diff_frames_groupby.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_ops_on_diff_frames_groupby_expanding.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_ops_on_diff_frames_groupby_rolling.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_repr.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_reshape.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_rolling.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_series.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_series_conversion.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_series_datetime.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_series_string.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_spark_functions.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_sql.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_stats.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_typedef.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_utils.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/tests/test_window.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/typedef/
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/typedef/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/typedef/typehints.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/usage_logging/
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/usage_logging/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/usage_logging/usage_logger.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/utils.py
    spark-3.3.3-bin-hadoop3/python/pyspark/pandas/window.py
    spark-3.3.3-bin-hadoop3/python/pyspark/profiler.py
    spark-3.3.3-bin-hadoop3/python/pyspark/py.typed
    spark-3.3.3-bin-hadoop3/python/pyspark/python/
    spark-3.3.3-bin-hadoop3/python/pyspark/python/pyspark/
    spark-3.3.3-bin-hadoop3/python/pyspark/python/pyspark/shell.py
    spark-3.3.3-bin-hadoop3/python/pyspark/rdd.py
    spark-3.3.3-bin-hadoop3/python/pyspark/rddsampler.py
    spark-3.3.3-bin-hadoop3/python/pyspark/resource/
    spark-3.3.3-bin-hadoop3/python/pyspark/resource/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/resource/information.py
    spark-3.3.3-bin-hadoop3/python/pyspark/resource/profile.py
    spark-3.3.3-bin-hadoop3/python/pyspark/resource/requests.py
    spark-3.3.3-bin-hadoop3/python/pyspark/resource/tests/
    spark-3.3.3-bin-hadoop3/python/pyspark/resource/tests/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/resource/tests/test_resources.py
    spark-3.3.3-bin-hadoop3/python/pyspark/resultiterable.py
    spark-3.3.3-bin-hadoop3/python/pyspark/serializers.py
    spark-3.3.3-bin-hadoop3/python/pyspark/shell.py
    spark-3.3.3-bin-hadoop3/python/pyspark/shuffle.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/_typing.pyi
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/avro/
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/avro/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/avro/functions.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/catalog.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/column.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/conf.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/context.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/dataframe.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/functions.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/group.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/observation.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/pandas/
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/pandas/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/pandas/_typing/
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/pandas/_typing/__init__.pyi
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/pandas/_typing/protocols/
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/pandas/_typing/protocols/__init__.pyi
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/pandas/_typing/protocols/frame.pyi
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/pandas/_typing/protocols/series.pyi
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/pandas/conversion.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/pandas/functions.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/pandas/functions.pyi
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/pandas/group_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/pandas/map_ops.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/pandas/serializers.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/pandas/typehints.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/pandas/types.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/pandas/utils.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/readwriter.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/session.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/sql_formatter.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/streaming.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_arrow.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_arrow_map.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_catalog.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_column.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_conf.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_context.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_dataframe.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_datasources.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_functions.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_group.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_pandas_cogrouped_map.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_pandas_grouped_map.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_pandas_map.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_pandas_udf.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_pandas_udf_grouped_agg.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_pandas_udf_scalar.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_pandas_udf_typehints.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_pandas_udf_typehints_with_future_annotations.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_pandas_udf_window.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_readwriter.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_serde.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_session.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_streaming.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_types.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_udf.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_udf_profiler.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/test_utils.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/typing/
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/typing/test_column.yml
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/typing/test_dataframe.yml
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/typing/test_functions.yml
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/typing/test_readwriter.yml
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/typing/test_session.yml
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/tests/typing/test_udf.yml
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/types.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/udf.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/utils.py
    spark-3.3.3-bin-hadoop3/python/pyspark/sql/window.py
    spark-3.3.3-bin-hadoop3/python/pyspark/statcounter.py
    spark-3.3.3-bin-hadoop3/python/pyspark/status.py
    spark-3.3.3-bin-hadoop3/python/pyspark/storagelevel.py
    spark-3.3.3-bin-hadoop3/python/pyspark/streaming/
    spark-3.3.3-bin-hadoop3/python/pyspark/streaming/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/streaming/context.py
    spark-3.3.3-bin-hadoop3/python/pyspark/streaming/dstream.py
    spark-3.3.3-bin-hadoop3/python/pyspark/streaming/kinesis.py
    spark-3.3.3-bin-hadoop3/python/pyspark/streaming/listener.py
    spark-3.3.3-bin-hadoop3/python/pyspark/streaming/tests/
    spark-3.3.3-bin-hadoop3/python/pyspark/streaming/tests/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/streaming/tests/test_context.py
    spark-3.3.3-bin-hadoop3/python/pyspark/streaming/tests/test_dstream.py
    spark-3.3.3-bin-hadoop3/python/pyspark/streaming/tests/test_kinesis.py
    spark-3.3.3-bin-hadoop3/python/pyspark/streaming/tests/test_listener.py
    spark-3.3.3-bin-hadoop3/python/pyspark/streaming/util.py
    spark-3.3.3-bin-hadoop3/python/pyspark/taskcontext.py
    spark-3.3.3-bin-hadoop3/python/pyspark/testing/
    spark-3.3.3-bin-hadoop3/python/pyspark/testing/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/testing/mllibutils.py
    spark-3.3.3-bin-hadoop3/python/pyspark/testing/mlutils.py
    spark-3.3.3-bin-hadoop3/python/pyspark/testing/pandasutils.py
    spark-3.3.3-bin-hadoop3/python/pyspark/testing/sqlutils.py
    spark-3.3.3-bin-hadoop3/python/pyspark/testing/streamingutils.py
    spark-3.3.3-bin-hadoop3/python/pyspark/testing/utils.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/__init__.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_appsubmit.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_broadcast.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_conf.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_context.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_daemon.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_install_spark.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_join.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_pin_thread.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_profiler.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_rdd.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_rddbarrier.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_readwrite.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_serializers.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_shuffle.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_statcounter.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_taskcontext.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_util.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/test_worker.py
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/typing/
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/typing/test_context.yml
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/typing/test_core.yml
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/typing/test_rdd.yml
    spark-3.3.3-bin-hadoop3/python/pyspark/tests/typing/test_resultiterable.yml
    spark-3.3.3-bin-hadoop3/python/pyspark/traceback_utils.py
    spark-3.3.3-bin-hadoop3/python/pyspark/util.py
    spark-3.3.3-bin-hadoop3/python/pyspark/version.py
    spark-3.3.3-bin-hadoop3/python/pyspark/worker.py
    spark-3.3.3-bin-hadoop3/python/pyspark.egg-info/
    spark-3.3.3-bin-hadoop3/python/pyspark.egg-info/PKG-INFO
    spark-3.3.3-bin-hadoop3/python/pyspark.egg-info/SOURCES.txt
    spark-3.3.3-bin-hadoop3/python/pyspark.egg-info/dependency_links.txt
    spark-3.3.3-bin-hadoop3/python/pyspark.egg-info/requires.txt
    spark-3.3.3-bin-hadoop3/python/pyspark.egg-info/top_level.txt
    spark-3.3.3-bin-hadoop3/python/run-tests
    spark-3.3.3-bin-hadoop3/python/run-tests-with-coverage
    spark-3.3.3-bin-hadoop3/python/run-tests.py
    spark-3.3.3-bin-hadoop3/python/setup.cfg
    spark-3.3.3-bin-hadoop3/python/setup.py
    spark-3.3.3-bin-hadoop3/python/test_coverage/
    spark-3.3.3-bin-hadoop3/python/test_coverage/conf/
    spark-3.3.3-bin-hadoop3/python/test_coverage/conf/spark-defaults.conf
    spark-3.3.3-bin-hadoop3/python/test_coverage/coverage_daemon.py
    spark-3.3.3-bin-hadoop3/python/test_coverage/sitecustomize.py
    spark-3.3.3-bin-hadoop3/python/test_support/
    spark-3.3.3-bin-hadoop3/python/test_support/SimpleHTTPServer.py
    spark-3.3.3-bin-hadoop3/python/test_support/hello/
    spark-3.3.3-bin-hadoop3/python/test_support/hello/hello.txt
    spark-3.3.3-bin-hadoop3/python/test_support/hello/sub_hello/
    spark-3.3.3-bin-hadoop3/python/test_support/hello/sub_hello/sub_hello.txt
    spark-3.3.3-bin-hadoop3/python/test_support/sql/
    spark-3.3.3-bin-hadoop3/python/test_support/sql/ages.csv
    spark-3.3.3-bin-hadoop3/python/test_support/sql/ages_newlines.csv
    spark-3.3.3-bin-hadoop3/python/test_support/sql/orc_partitioned/
    spark-3.3.3-bin-hadoop3/python/test_support/sql/orc_partitioned/_SUCCESS
    spark-3.3.3-bin-hadoop3/python/test_support/sql/orc_partitioned/b=0/
    spark-3.3.3-bin-hadoop3/python/test_support/sql/orc_partitioned/b=0/c=0/
    spark-3.3.3-bin-hadoop3/python/test_support/sql/orc_partitioned/b=0/c=0/.part-r-00000-829af031-b970-49d6-ad39-30460a0be2c8.orc.crc
    spark-3.3.3-bin-hadoop3/python/test_support/sql/orc_partitioned/b=0/c=0/part-r-00000-829af031-b970-49d6-ad39-30460a0be2c8.orc
    spark-3.3.3-bin-hadoop3/python/test_support/sql/orc_partitioned/b=1/
    spark-3.3.3-bin-hadoop3/python/test_support/sql/orc_partitioned/b=1/c=1/
    spark-3.3.3-bin-hadoop3/python/test_support/sql/orc_partitioned/b=1/c=1/.part-r-00000-829af031-b970-49d6-ad39-30460a0be2c8.orc.crc
    spark-3.3.3-bin-hadoop3/python/test_support/sql/orc_partitioned/b=1/c=1/part-r-00000-829af031-b970-49d6-ad39-30460a0be2c8.orc
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/_SUCCESS
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/_common_metadata
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/_metadata
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2014/
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2014/month=9/
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2014/month=9/day=1/
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2014/month=9/day=1/.part-r-00008.gz.parquet.crc
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2014/month=9/day=1/part-r-00008.gz.parquet
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2015/
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2015/month=10/
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2015/month=10/day=25/
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2015/month=10/day=25/.part-r-00002.gz.parquet.crc
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2015/month=10/day=25/.part-r-00004.gz.parquet.crc
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2015/month=10/day=25/part-r-00002.gz.parquet
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2015/month=10/day=25/part-r-00004.gz.parquet
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2015/month=10/day=26/
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2015/month=10/day=26/.part-r-00005.gz.parquet.crc
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2015/month=10/day=26/part-r-00005.gz.parquet
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2015/month=9/
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2015/month=9/day=1/
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2015/month=9/day=1/.part-r-00007.gz.parquet.crc
    spark-3.3.3-bin-hadoop3/python/test_support/sql/parquet_partitioned/year=2015/month=9/day=1/part-r-00007.gz.parquet
    spark-3.3.3-bin-hadoop3/python/test_support/sql/people.json
    spark-3.3.3-bin-hadoop3/python/test_support/sql/people1.json
    spark-3.3.3-bin-hadoop3/python/test_support/sql/people_array.json
    spark-3.3.3-bin-hadoop3/python/test_support/sql/people_array_utf16le.json
    spark-3.3.3-bin-hadoop3/python/test_support/sql/streaming/
    spark-3.3.3-bin-hadoop3/python/test_support/sql/streaming/text-test.txt
    spark-3.3.3-bin-hadoop3/python/test_support/sql/text-test.txt
    spark-3.3.3-bin-hadoop3/python/test_support/userlib-0.1.zip
    spark-3.3.3-bin-hadoop3/python/test_support/userlibrary.py
    spark-3.3.3-bin-hadoop3/sbin/
    spark-3.3.3-bin-hadoop3/sbin/decommission-slave.sh
    spark-3.3.3-bin-hadoop3/sbin/decommission-worker.sh
    spark-3.3.3-bin-hadoop3/sbin/slaves.sh
    spark-3.3.3-bin-hadoop3/sbin/spark-config.sh
    spark-3.3.3-bin-hadoop3/sbin/spark-daemon.sh
    spark-3.3.3-bin-hadoop3/sbin/spark-daemons.sh
    spark-3.3.3-bin-hadoop3/sbin/start-all.sh
    spark-3.3.3-bin-hadoop3/sbin/start-history-server.sh
    spark-3.3.3-bin-hadoop3/sbin/start-master.sh
    spark-3.3.3-bin-hadoop3/sbin/start-mesos-dispatcher.sh
    spark-3.3.3-bin-hadoop3/sbin/start-mesos-shuffle-service.sh
    spark-3.3.3-bin-hadoop3/sbin/start-slave.sh
    spark-3.3.3-bin-hadoop3/sbin/start-slaves.sh
    spark-3.3.3-bin-hadoop3/sbin/start-thriftserver.sh
    spark-3.3.3-bin-hadoop3/sbin/start-worker.sh
    spark-3.3.3-bin-hadoop3/sbin/start-workers.sh
    spark-3.3.3-bin-hadoop3/sbin/stop-all.sh
    spark-3.3.3-bin-hadoop3/sbin/stop-history-server.sh
    spark-3.3.3-bin-hadoop3/sbin/stop-master.sh
    spark-3.3.3-bin-hadoop3/sbin/stop-mesos-dispatcher.sh
    spark-3.3.3-bin-hadoop3/sbin/stop-mesos-shuffle-service.sh
    spark-3.3.3-bin-hadoop3/sbin/stop-slave.sh
    spark-3.3.3-bin-hadoop3/sbin/stop-slaves.sh
    spark-3.3.3-bin-hadoop3/sbin/stop-thriftserver.sh
    spark-3.3.3-bin-hadoop3/sbin/stop-worker.sh
    spark-3.3.3-bin-hadoop3/sbin/stop-workers.sh
    spark-3.3.3-bin-hadoop3/sbin/workers.sh
    spark-3.3.3-bin-hadoop3/yarn/
    spark-3.3.3-bin-hadoop3/yarn/spark-3.3.3-yarn-shuffle.jar



```python
!pip install findspark
```

    Requirement already satisfied: findspark in /usr/local/lib/python3.10/dist-packages (2.0.1)



```python
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.3.3-bin-hadoop3"
```


```python
import findspark
findspark.init()
```


```python
findspark.find()
```




    '/content/spark-3.3.3-bin-hadoop3'




```python
from pyspark.sql import SparkSession

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()
```


```python
sc = spark.sparkContext
```


```python
test = sc.parallelize([1, 2, 3, 4, 5])
test.map(lambda x: (x, x**2)).collect()
```




    [(1, 1), (2, 4), (3, 9), (4, 16), (5, 25)]




```python
from google.colab import drive
drive.mount('/content/drive')

```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
#List of file names
file_names = ['the_weeknd.csv', 'taylor_swift.csv', 'sza.csv', 'rihanna.csv', 'justin_bieber.csv', 'ed_sheeran.csv', 'drake.csv', 'doja_cat.csv', 'billie_eilish.csv', 'bad_bunny.csv']

# Initialize an empty DataFrame to union all your DataFrames
combined_df = None

# Loop over the file names, read each into a DataFrame, and combine them
for file_name in file_names:
    file_path = f'/content/drive/My Drive/raw-csvs-top-10/{file_name}'
    # Read the CSV file into a DataFrame
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    # Union the DataFrames
    if combined_df is None:
        combined_df = df
    else:
        combined_df = combined_df.union(df)

# Now `combined_df` contains all the data from the 10 CSV files
combined_df.show(5)  # Show the first 5 rows
```

    +--------------------+--------------------+--------------------+--------------------+--------------+-------------------+-------------+----------+--------------------+-------------------+--------------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-------+--------------+
    |          Spotify ID|          Artist IDs|          Track Name|          Album Name|Artist Name(s)|       Release Date|Duration (ms)|Popularity|            Added By|           Added At|              Genres|Danceability|Energy|Key|Loudness|Mode|Speechiness|Acousticness|Instrumentalness|Liveness|Valence|  Tempo|Time Signature|
    +--------------------+--------------------+--------------------+--------------------+--------------+-------------------+-------------+----------+--------------------+-------------------+--------------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-------+--------------+
    |2ye9iWj5V4g6k6HFe...|1Xyo4u8uXC1ZmMpat...|High For This - O...|House Of Balloons...|    The Weeknd|2011-03-21 00:00:00|       249102|        68|spotify:user:1213...|2023-10-09 20:53:51|canadian contempo...|       0.466| 0.403|  4| -10.441|   0|     0.0502|        0.59|         9.51E-5|     0.1| 0.0967| 75.033|             4|
    |4Yw8TyBm9J8cpy2gu...|1Xyo4u8uXC1ZmMpat...|What You Need - O...|House Of Balloons...|    The Weeknd|2011-03-21 00:00:00|       206421|        72|spotify:user:1213...|2023-10-09 20:53:51|canadian contempo...|       0.638| 0.335|  7| -10.522|   0|     0.0854|       0.505|          0.0139|   0.116| 0.0608|133.853|             4|
    |2r7BPog74oaTG5shN...|1Xyo4u8uXC1ZmMpat...|House Of Balloons...|House Of Balloons...|    The Weeknd|2011-03-21 00:00:00|       407315|        74|spotify:user:1213...|2023-10-09 20:53:51|canadian contempo...|       0.662| 0.511|  0|  -8.419|   1|     0.0342|     0.00237|          0.0036|   0.275|  0.228| 88.989|             4|
    |4jBfUB4kQJCWOrjGL...|1Xyo4u8uXC1ZmMpat...|The Morning - Ori...|House Of Balloons...|    The Weeknd|2011-03-21 00:00:00|       314109|        74|spotify:user:1213...|2023-10-09 20:53:51|canadian contempo...|       0.682|  0.51|  6|  -9.987|   0|     0.0441|       0.143|         4.46E-6|  0.0797|  0.191|120.097|             4|
    |00aqkszH1FdUiJJWv...|1Xyo4u8uXC1ZmMpat...|Wicked Games - Or...|House Of Balloons...|    The Weeknd|2011-03-21 00:00:00|       325305|        72|spotify:user:1213...|2023-10-09 20:53:51|canadian contempo...|       0.606|  0.57|  9|  -6.684|   1|      0.032|      0.0217|         8.47E-6|   0.301|  0.258|114.033|             4|
    +--------------------+--------------------+--------------------+--------------------+--------------+-------------------+-------------+----------+--------------------+-------------------+--------------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-------+--------------+
    only showing top 5 rows
    



```python
from pyspark.sql.functions import col, split

row_count_before_clean = combined_df.count()
print(f"The number of rows before data cleaning is: {row_count_before_clean}")

# List of top 10 artist names:
top_10_artist_names = ["The Weeknd", "Taylor Swift", "SZA", "Rihanna", "Justin Bieber", "Ed Sheeran", "Drake", "Doja Cat", "Billie Eilish", "Bad Bunny"]

# 1. Filter entries where the first artist listed in "Artist Name(s)" is one of the top 10 artists
filtered_df = combined_df.withColumn("FirstArtist", split(col("Artist Name(s)"), ",").getItem(0)) \
                         .filter(col("FirstArtist").isin(top_10_artist_names))

# 2. Remove duplicates based on 'Track Name'
no_duplicates_df = filtered_df.dropDuplicates(["Track Name"])

# 3. Remove live versions of songs
# Filter out rows where 'Track Name' contains "- Live" or "Live Version"
cleaned_df = no_duplicates_df.filter(~col("Track Name").like("%- Live%") &
                                     ~col("Track Name").like("%Live Version%"))

row_count_clean = cleaned_df.count()
print(f"Removed duplicates, live versions, and tracks where the top 10 artist is NOT a primary artist.")
print(f"The number of rows after data cleaning is: {row_count_clean}")

```

    The number of rows before data cleaning is: 3084
    Removed duplicates, live versions, and tracks where the top 10 artist is NOT a primary artist.
    The number of rows after data cleaning is: 1906



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

