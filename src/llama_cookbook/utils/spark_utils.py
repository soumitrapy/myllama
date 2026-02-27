import pyspark

def get_spark_info(df: pyspark.sql.DataFrame): 
    print(f'number of partitions: {df.rdd.getNumPartitions()},\n\
    number of rows in each partition:{df.rdd.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()}',
          )