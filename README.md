# Final-Project-Divvy
Github repository for final project of MACS 30123 Large Scale Computing (Fall, 2021).

Author: David Xu, William Zhu

Explanation of notebook:

[clean_divvy.ipynb](https://github.com/lsc4ss-a21/final-project-divvy/blob/main/clean_divvy.ipynb) cleans the divvy bike trip data and export cleaned data as parquet to s3 bucket, with Dask and Geopandas.

[dask_visualizations.ipynb](https://github.com/lsc4ss-a21/final-project-divvy/blob/main/dask_visualizations.ipynb) creates maps of the starting and ending positions of all electric Divvy bike trips in our sample, with dask and holoviews.

[dask_data_exploration.ipynb](https://github.com/lsc4ss-a21/final-project-divvy/blob/main/dask_data_exploration.ipynb) explores the relationship between potential features and our dependent variable in preparation for machine learning model, with dask.

[pyspark_divvy_ML.ipynb](https://github.com/lsc4ss-a21/final-project-divvy/blob/main/pyspark_divvy_ML.ipynb) documents our exercise of predicting whether a bike trip ends at a different zip code from starting position with logistic regression, and random forest models. The modeling exercise was done in PySpark.

David Xu was responsible for data cleaning, and contributed equally to data exploration and running machine learning models. William Zhu was responsible for data visualizations and contributed equally to data exploration and running machine learning models.

## Introduction

In July 2020, Divvy bike, Chicago’s public bike share system managed by Lyft, introduced Electric bikes (e-bikes). These e-bikes enable riders to ride much faster than normal bikes and make long distance bike rides more convenient. As residents of the East Hyde Park neighborhood, both David and William are big fans of the divvy e-bikes and frequently plan e-bike trips to the University campus or Downtown. The trouble is that, oftentimes, we are unable to find available e-bikes nearby using our Lyft mobile apps. It gets us thinking, can Lyft ensure that there are always sufficient e-bikes in every zip code? The first step to answering this question is to determine how well the destinations of e-bike trips can be forecasted in advance, so that Divvy service vans can plan ahead for e-bike redistribution. More specifically, this project is interested in using every Divvy e-bike trip’s starting information (coordinates, timestamp, ridership type, etc) to predict whether the trip ends in the same zip code as the starting location. 

Under the larger context, dockless bikes (such as Divvy E-bike) have important implications on urban mobility. According to a report from World Resources Institute [1], dockless bikes contributed to increasing urban mobility by enhancing connectivity to transit and reducing motorized trips. In the period of Covid19, bikes also offered a safer option of traveling. Dockless bikes have the potential to become one of the important solutions to urban mobility problems. Considering that In the city of Chicago particularly, the south and west side have fewer docks than the north [2], electric bikes could potentially be an equitable solution for urban mobility that the original docked bikes could not offer. In the literature, dynamic bike relocation and bike station allocations are also important questions to consider in the bike-sharing service planning problems. [3] Our project, though simple and small scale, takes a preliminary attempt at addressing the problem of bike reallocation with scalable coding which could potentially be used to address issues at larger scale. 

## Data Cleaning, Feature Engineering, and Visualization

In this project, we analyzed divvy bike trip records from July 2020 to October 2021. The
dataset is collected by Lyft and publicly accessible from here:
https://www.divvybikes.com/system-data. Trips taken by service staff or lasted below 60
seconds were removed by the data provider. The monthly datasets are stored in csv format. Each row represents a bike trip, with features including the starting and ending coordinates, timestamps, bike type, rider type, etc. We stored these datasets in AWS S3 and kept 2.1M e-bike trip records for further analysis. 

Based on the starting and ending coordinates, we identified the start and end zip code of each trip using geopandas Python package. The binary outcome variable, “cross zip code trip” equals True (1) if each trip’s start zip code is different from the end zip code and False (0) otherwise. Among the 2.1M e-bike trips, 1.5M (70.6%) are cross zip code trips. In the ML modeling session, we down-sampled the cross zip code trips by about 42% to ensure a balanced dataset.

Seven feature variables are prepared for ML modeling: Based on the start time stamp, we engineered features including Month of the Year, Day of the Week, Hour of the Day. Features including starting zip code, starting station, and rider type (casual or member) are provided in the dataset. Based on the start and end time stamp, we generated trip duration as an additional feature. Table 1 lists the variables included in the ML models and their data types.

Table 1: Outcome Variable and Features
| Variable                                         | Data Type   |
|--------------------------------------------------|-------------|
| Whether the bike trip crosses zip code (Outcome) | Binary      |
| Month                                            | Categorical |
| Day of the Week                                  | Categorical |
| Hour of the Day                                  | Categorical |
| Starting Zip Code                                | Categorical |
| Starting Station                                 | Categorical |
| Rider type: Member or Casual                     | Binary      |
| Trip Duration                                    | Numerical   |


Figure 1 shows the plots of 2.1M starting and ending e-bike coordinates using datashader. We can see a sharp contrast between the starting and ending coordinates. The starting e-bike coordinates are concentrated in certain spots, which are mostly likely to be the bike stations. Meanwhile, the end coordinates are scattered throughout major roads in Chicago. It means that e-bike trips are far more likely to end than to start on the streets (not docked to bike stations). It suggests that the Divvy bike service team plays an important role in docking e-bikes to bike stations. 

Figure 1: Mapping Start Coordinates vs End Coordinates of Divvy e-bikes
![plot_combined](https://github.com/william-wei-zhu/large_scale_divvy/blob/main/chart/plot_combined.png)


- note: necessary shape files for geopandas in data cleaning are included in the shape file folder.

## Model
Before creating a machine learning model pipeline, we first explored the relationship between seven features and the dependent variable of whether the bike trip ended at a different zip code than starting position. Our seven features included 3 time variables (month, weekday, hour), 2 location variables (starting zip code, starting station (for those trips that started from a station)), one variable that denotes the membership status of the rider, and one variable which describes the length of the bike trip. Here, I listed some of the visualizations (rest are included in chart folder): 
![features](https://github.com/william-wei-zhu/large_scale_divvy/blob/main/chart/feature.png)
(Note: while we did not create a visualization for trip duration, on average the bike trip that ends within the same code has a duration of 10.97 min, and bike trip that ends at different zip code has a duration of 15.87 min, as shown in 
[pyspark_divvy_ML.ipynb](https://github.com/william-wei-zhu/large_scale_divvy/blob/main/pyspark_divvy_ML.ipynb))

As shown in the six charts, month, weekday, and membership status do not seem to be good indicators by themselves. There are some variations in the hours, but the variations are relatively small in magnitude. Both location variables seem to be more helpful indicators, as both distributions demonstrated that there exists consideration variations across starting locations: some locations will have cross-zip-code trips, while others have less. Lastly, while not directly reflected in visualization, the average duration gap of 5 minute gap between two types of rides do seem to be significant enough to provide some prediction power. 

In this project, we adopted two machine learning models for prediction. We first started with the logistic regression model. While taking a rather naive linear assumption, the simplicity of logistic regression model (and ease of interpretation) offers a good baseline. To prevent the issue of contaminating training/testing data, we created machine learning pipelines in [pyspark_divvy_ML.ipynb](https://github.com/william-wei-zhu/large_scale_divvy/blob/main/pyspark_divvy_ML.ipynb). To ensure that the evaluation metrics of the prediction are not random, and to optimize prediction results, we incorporate both cross-validation and hyperparameter tuning in our pipeline. The best model from the pipeline has an accuracy of 0.632 on the test data. The AUC is 0.687. The resulting confusion matrix of logistic regression is shown below. (0 denotes cross-zip-code trips, 1 denotes within-zip-code trips)

![confusion_matrix_lr](https://github.com/william-wei-zhu/large_scale_divvy/blob/main/chart/cmat_lr.PNG)

The logistic regression yields a reasonable prediction. It is not very satisfactory as both the AUC and accuracy are not very high. The confusion matrix also shows considerable portions of false positive and false negative predictions. An important limitation with logistic regression lies on its linear assumption. Considering the relationships between features and dependable variables in the previous chart, it is likely that the relationships are nonlinear. Trees which partition observations in the feature space could perform better. Therefore, we adopt a random forest model as our second model. Similar to the previous attempt, we constructed a pipeline and incorporated cross-validation and hyperparameter tuning in our pipeline. The best model from the pipeline has an accuracy of 0.690 on the test data. The AUC is 0.765. The resulting confusion matrix of a random forest is shown below. (0 denotes cross-zip-code trips, 1 denotes within-zip-code trips)

![confusion_matrix_rf](https://github.com/william-wei-zhu/large_scale_divvy/blob/main/chart/cmat_rf.PNG)

Random forest models certainly performed better than logistic regression, as we anticipated. In fact, it does offer a significantly better AUC score, and it performs significantly better at distinguishing true positive(negative) from false positive(negative). The roc curve below also reflected the same conclusion. 

![roc](https://github.com/william-wei-zhu/large_scale_divvy/blob/main/chart/roc.PNG)

## Large Scale Computation Consideration
This section discusses our considerations of scalability of our code, and it could be helpful in addressing large-scale computational problems. We believe our code performs well in the following points:

- We converted our data into parquet format, which works well in Hadoop clusters, and is optimized for query performance and minimizing I/O. In a cloud computing server, given its higher efficiency, parquet also helps save cost, which is an important consideration for large scale computation.
- Utilizing PySpark and Dask could help save memory as the data are only loaded when actions are called. This also helps in both saving cost and increasing efficiency for large scale computation.
- Running machine learning models on PySpark could especially increase the efficiency in the stage of hyperparameter tuning. Grid searching is an intensive task, and running it parallelly generates significant speedup as opposed to running it serially. It also lowers the memory requirement. 

We also recognize the limitations in our code:
- While the assign function in dask dataframe did the job, it is still an embarrassing parallel operation, and could certainly be improved.
- In data-cleaning, due to the incompatibility of geopandas with dask, we first converted the data into dataframe. While relevant packages are being developed (dask_geopandas), they are not yet fully functional. Scaling the data cleaning task could be more challenging given this limitation. 

## Conclusion
In this project, we were able to utilize parallel computing techniques and came up with a prediction model (random forest) that could reasonably predict the end destination of a Divvy electric bike trip given only related bike trip data. Our coding structure, while has room for improvement, could potentially be used for larger scale analysis (e.g. all bikes in Divvy, or even all public bike sharing systems in the US, including those in NYC and Boston, for example.) To further improve on the prediction, we could also merge with other dataset (e.g. granular home value data).

## Reference

[1] Hui Jang, Su Song, Lu Lu. "Dockless Bike Sharing Can Create Healthy, Resilient Urban Mobility". Nov 17, 2020. https://www.wri.org/insights/dockless-bike-sharing-can-create-healthy-resilient-urban-mobility

[2] Greenfield, John. "E-Divvy bikes debut Wednesday, equitable-but-complex pricing system announced". Jul 27, 2020. https://chi.streetsblog.org/2020/07/27/e-divvy-bikes-debut-wednesday-equitable-but-complex-pricing-system-announced/

[3] Shui, C.S, Szeto, W.Y. (2020). A review of bicycle-sharing service planning problems. Transportation Research Part C: Emerging Technologies. Vol 117. Doi: https://doi.org/10.1016/j.trc.2020.102648

Dask Dataframe reference: https://docs.dask.org/en/stable/dataframe.html

PySpark logistic regression reference: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegression.html

PySpark random forest classifier reference: https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.ml.classification.RandomForestClassifier.html

Geopandas reference: https://geopandas.org/en/stable/docs/reference.html

Dask Geopandas reference: https://github.com/geopandas/dask-geopandas

Advantages of Parquet: https://dzone.com/articles/how-to-be-a-hero-with-powerful-parquet-google-and
