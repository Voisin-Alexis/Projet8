import pandas as pd
import numpy as np
from PIL import Image
import PIL
import io
from io import StringIO 
import pyarrow

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
from pyspark import SparkContext, SparkConf
from pyspark.rdd import RDD

from pyspark.sql import SparkSession
#from pyspark.sql.types import *

from pyspark.sql.functions import udf, pandas_udf, PandasUDFType
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import urllib.request

#!pip install findspark
import findspark

import pyspark
pyspark.__version__

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.models import Model
from keras.layers import Dense
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import optimizers
from keras.layers import Dense, GlobalMaxPooling2D, Flatten

import sklearn.decomposition
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector

import json
import boto.s3, boto.s3.key
#!pip install boto3
import boto3
import time

start_time = time.time()
pyspark.__version__

findspark.init()

spark = SparkSession.builder.master("local[*]").appName('Projet8_AWS').getOrCreate()

#spark

cred = boto3.Session().get_credentials()
ACCESS_KEY = cred.access_key
SECRET_KEY = cred.secret_key

AWS_KEY = ACCESS_KEY
SEC_KEY = SECRET_KEY

spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "512")


img_dir = 's3://imageprojet8oc/ImageAWS/'
s3 = boto3.resource('s3')
bucket = s3.Bucket("imageprojet8oc")

print('dataset_path =', img_dir)  

categorie = []

for objet in bucket.objects.filter(Prefix="ImageAWS"):
    if objet.key.split('/')[1] not in categorie:
        categorie.append(str(objet.key.split('/')[1]))
    else:
        continue
        
print('Nombre de catégorie: ', len(categorie), '\n')
print('Catégories à notre disposition: ')
for i in range(len(categorie)):
    print('  - ', categorie[i])

cred = boto3.Session().get_credentials()
ACCESS_KEY = cred.access_key
SECRET_KEY = cred.secret_key

AWS_KEY = ACCESS_KEY
SEC_KEY = SECRET_KEY

session = boto3.session.Session(aws_access_key_id = AWS_KEY,
                                aws_secret_access_key = SEC_KEY)
s3_client = session.client(service_name = 's3', region_name = 'eu-west-3')

response = s3_client.generate_presigned_url('get_object',
                                            Params = {'Bucket': 'imageprojet8oc',
                                                      'Key': "ImageAWS/Kaki/12_100.jpg"}
                                           )
resp = urllib.request.urlopen(response)
img_show = Image.open(io.BytesIO(resp.read()))

#img_show

img_dirS3 = 's3a://imageprojet8oc/ImageAWS/*'

sc = spark.sparkContext
sc.setSystemProperty('com.amazonaws.services.s3.enableV4', 'true')
sc._jsc.hadoopConfiguration().set('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:2.7.4')
#sc._jsc.hadoopConfiguration().set('spark.jars.packages', 'com.amazonaws:aws-java-sdk:1.12.117')
#sc._jsc.hadoopConfiguration().set('spark.jars.packages', 'com.amazonaws:aws-java-sdk-s3-1.12.120')
sc._jsc.hadoopConfiguration().set('spark.jars.packages', 'com.amazonaws:aws-java-sdk-core-1.12.134')
sc._jsc.hadoopConfiguration().set('spark.jars.packages', 'com.amazonaws:aws-java-sdk-dynamodb-1.12.134')
sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.eu-west-3.amazonaws.com")
sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", AWS_KEY)
sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", SEC_KEY)

imagesALL_df = spark.read.format("binaryFile") \
                    .option("pathGlobFilter", "*.jpg") \
                    .option("recursiveFileLookup", "true") \
                    .load(img_dirS3)

imagesALL_df.show(5)
imagesALL_df.select('path').show(5, False)

from pyspark.sql.functions import split
labelAll0 = imagesALL_df.withColumn("Label0", split(imagesALL_df['path'], "//").getItem(1))
labelAll0.select('Label0').show(1, False)

imagesALL_dfLabel = labelAll0.withColumn("Label", split(labelAll0['Label0'], "/").getItem(2))
imagesALL_dfLabel = imagesALL_dfLabel.select('path', 'content', 'Label')
imagesALL_dfLabel.show(1)

imagesALL_dfLabel.groupBy("Label").count().show()

label_indexer = StringIndexer(inputCol="Label", outputCol="Label_index")
label_indexer_transformer = label_indexer.fit(imagesALL_dfLabel)
imagesALL_dfLabel = label_indexer_transformer.transform(imagesALL_dfLabel)

imagesALL_dfLabel.show(1)

def preprocess(content):
    image = PIL.Image.open(io.BytesIO(content))
    imageResize = image.resize([224, 224])
    imageArray = img_to_array(imageResize)
    preprocessingImage = preprocess_input(imageArray)
    return preprocessingImage


def model_MobileNetV2():
    model = MobileNetV2(include_top = False, input_shape=(224, 224, 3), weights = "imagenet", pooling = 'max')
    return model

def featurize_series(model, content_series):
    input = np.stack(content_series.map(preprocess))
    preds = model.predict(input)
    output = [p.flatten() for p in preds]
    return pd.Series(output)

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    model = model_MobileNetV2()
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)

features_df = imagesALL_dfLabel.select("path", 'Label', 'Label_index', featurize_udf("content"))

features_df.persist()

features_df.show(1)

features_df = features_df.withColumnRenamed("featurize_udf(content)", "Features")
features_df.show(1)

from pyspark.ml.feature import PCA

conversionVectorUDT = udf(lambda feature: Vectors.dense(feature), VectorUDT())
features_df = features_df.withColumn("VectorUdt", conversionVectorUDT('Features'))
acp_train, acp_test = features_df.randomSplit([0.7, 0.3], seed = 7)

pca = PCA(k = 2, inputCol = 'VectorUdt', outputCol = 'X_acp')
model = pca.fit(acp_train)
acp_df = model.transform(acp_train)
acp_df.show(1)

lr = LogisticRegression(maxIter = 20, regParam = 0.05, elasticNetParam = 0.3, labelCol = "Label_index", featuresCol = 'X_acp')
p_model = lr.fit(acp_df)

acp_Test = model.transform(acp_test)
predictions = p_model.transform(acp_Test)

predictions.groupBy("Label").count().show()

evaluatorF1 = MulticlassClassificationEvaluator(labelCol = "Label_index", predictionCol = "prediction", metricName = "f1")

print("F1 = ", evaluatorF1.evaluate(predictions))

evaluatorAccuracy = MulticlassClassificationEvaluator(labelCol = "Label_index", predictionCol = "prediction", metricName = "accuracy")

print("Accuracy = ", evaluatorAccuracy.evaluate(predictions))

resultat_dir = 's3a://imageprojet8oc/resultat/'

resultattoPandas = predictions.toPandas()

csv_buffer = StringIO()
resultattoPandas.to_csv(csv_buffer)
s3.Object('imageprojet8oc', 'resultat/resultat.csv').put(Body=csv_buffer.getvalue())

interval0 = time.time() - start_time
print('Total time in seconds 0:', interval0)

print('VALIDATION VALIDATION')

img_dir = 's3://imageprojet8octest/ImageAWSTest/'
s3 = boto3.resource('s3')
bucket = s3.Bucket("imageprojet8oc")
#key = boto.s3.key.Key(bucket, "words.txt")

print('dataset_path =', img_dir)  

img_dirS3Test = 's3a://imageprojet8octest/ImageAWSTest/*'

sc = spark.sparkContext
sc.setSystemProperty('com.amazonaws.services.s3.enableV4', 'true')
sc._jsc.hadoopConfiguration().set('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:2.7.4')
#sc._jsc.hadoopConfiguration().set('spark.jars.packages', 'com.amazonaws:aws-java-sdk:1.12.117')
#sc._jsc.hadoopConfiguration().set('spark.jars.packages', 'com.amazonaws:aws-java-sdk-s3-1.12.120')
sc._jsc.hadoopConfiguration().set('spark.jars.packages', 'com.amazonaws:aws-java-sdk-core-1.12.134')
sc._jsc.hadoopConfiguration().set('spark.jars.packages', 'com.amazonaws:aws-java-sdk-dynamodb-1.12.134')
sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.eu-west-3.amazonaws.com")
sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", AWS_KEY)
sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", SEC_KEY)

imagesALL_dfTest = spark.read.format("binaryFile") \
                    .option("pathGlobFilter", "*.jpg") \
                    .option("recursiveFileLookup", "true") \
                    .load(img_dirS3Test)

labelAll0Test = imagesALL_dfTest.withColumn("Label0", split(imagesALL_dfTest['path'], "//").getItem(1))
imagesALL_dfLabelTest = labelAll0Test.withColumn("Label", split(labelAll0Test['Label0'], "/").getItem(2))
imagesALL_dfLabelTest = imagesALL_dfLabelTest.select('path', 'content', 'Label')

label_indexerTest = StringIndexer(inputCol="Label", outputCol="Label_index")
label_indexer_transformerTest = label_indexerTest.fit(imagesALL_dfLabelTest)
imageTest_dfLabelTest = label_indexer_transformerTest.transform(imagesALL_dfLabelTest)

def preprocessTest(content):
    image = PIL.Image.open(io.BytesIO(content))
    imageResize = image.resize([224, 224])
    imageArray = img_to_array(imageResize)
    preprocessingImage = preprocess_input(imageArray)
    return preprocessingImage

def model_MobileNetV2():
    model = MobileNetV2(include_top = False, input_shape=(224, 224, 3), weights = "imagenet", pooling = 'max')
    return model

def featurize_seriesX(model, content_series):
    input = np.stack(content_series.map(preprocessTest))
    preds = model.predict(input)
    output = [p.flatten() for p in preds]
    return pd.Series(output)

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udfX(content_series_iter):
    model = model_MobileNetV2()
    for content_series in content_series_iter:
        yield featurize_seriesX(model, content_series)

features_dfTest = imageTest_dfLabelTest.select("path", 'Label', 'Label_index', featurize_udfX("content"))

features_dfTest.persist()

features_dfTest = features_dfTest.withColumnRenamed("featurize_udfX(content)", "Features")

features_dfTest = features_dfTest.withColumn("VectorUdt", conversionVectorUDT('Features'))

acp_dfTest = model.transform(features_dfTest)

predictions = p_model.transform(acp_dfTest)

interval = time.time() - start_time
print('Total time in seconds:', interval)




