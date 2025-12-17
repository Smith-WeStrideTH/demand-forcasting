from __future__ import print_function
from __future__ import unicode_literals

import time
import sys
import os
import shutil
import csv
import subprocess

# ติดตั้ง library ที่จำเป็นภายใน Processing container
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "pydeequ==0.1.5"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas==1.1.4"])

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, DoubleType
from pyspark.sql.functions import *

from pydeequ.analyzers import *
from pydeequ.checks import *
from pydeequ.verification import *
from pydeequ.suggestions import *


def main():
    # -----------------------------
    # 1) อ่าน arguments จาก sys.argv
    # -----------------------------
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))

    # Retrieve the args and replace 's3://' with 's3a://' (used by Spark)
    s3_input_data = args["s3_input_data"].replace("s3://", "s3a://")
    s3_output_analyze_data = args["s3_output_analyze_data"].replace("s3://", "s3a://")

    print("Input  (Spark):", s3_input_data)
    print("Output (Spark):", s3_output_analyze_data)

    # -----------------------------
    # 2) สร้าง SparkSession
    # -----------------------------
    spark = (
        SparkSession.builder
        .appName("RetailDemand-DataQuality-Deequ")
        .getOrCreate()
    )

    # -----------------------------
    # 3) กำหนด schema ให้ตรงกับ retail-demand-forecasting.csv
    # -----------------------------
    schema = StructType(
        [
            StructField("record_id",    IntegerType(), True),
            StructField("date",         StringType(),  True),
            StructField("store_id",     IntegerType(), True),
            StructField("day_of_week",  StringType(),  True),
            StructField("is_weekend",   IntegerType(), True),
            StructField("is_holiday",   IntegerType(), True),
            StructField("holiday_name", StringType(),  True),
            StructField("max_temp_c",   DoubleType(),  True),
            StructField("rainfall_mm",  DoubleType(),  True),
            StructField("is_hot_day",   IntegerType(), True),
            StructField("is_rainy_day", IntegerType(), True),
            StructField("base_price",   DoubleType(),  True),
            StructField("discount_pct", DoubleType(),  True),
            StructField("is_promo",     IntegerType(), True),
            StructField("promo_type",   StringType(),  True),
            StructField("final_price",  DoubleType(),  True),
            StructField("units_sold",   IntegerType(), True),
        ]
    )

    # -----------------------------
    # 4) อ่านไฟล์ CSV จาก S3
    #    (ของเราเป็น comma-separated ไม่ใช่ \t)
    # -----------------------------
    dataset = (
        spark.read
        .csv(
            s3_input_data,  
            header=True,
            schema=schema,
            sep=",",
            quote='"'
        )
    )

    print("===== Dataset schema =====")
    dataset.printSchema()
    print("===== Example rows =====")
    dataset.show(5, truncate=False)

    # ==========================================================
    # 5) ANALYZERS – คำนวณ statistics / metrics ของ dataset
    # ==========================================================
    analysisResult = (
        AnalysisRunner(spark)
        .onData(dataset)
        # Size – จำนวน rows ทั้งหมด
        .addAnalyzer(Size())
        # Completeness – มี null หรือไม่
        .addAnalyzer(Completeness("record_id"))
        .addAnalyzer(Completeness("date"))
        .addAnalyzer(Completeness("store_id"))
        .addAnalyzer(Completeness("units_sold"))
        .addAnalyzer(Completeness("base_price"))
        # ApproxCountDistinct – จำนวน category โดยประมาณ
        .addAnalyzer(ApproxCountDistinct("store_id"))
        .addAnalyzer(ApproxCountDistinct("day_of_week"))
        # Mean – ค่าเฉลี่ยของ numeric columns
        .addAnalyzer(Mean("units_sold"))
        .addAnalyzer(Mean("base_price"))
        .addAnalyzer(Mean("discount_pct"))
        .addAnalyzer(Mean("max_temp_c"))
        .addAnalyzer(Mean("rainfall_mm"))
        # Compliance – กฎเงื่อนไขที่ควรเป็นจริง
        .addAnalyzer(Compliance(
            "non_negative_units_sold",
            "units_sold >= 0"
        ))
        .addAnalyzer(Compliance(
            "valid_discount_range",
            "discount_pct >= 0 AND discount_pct <= 1"
        ))
        .addAnalyzer(Compliance(
            "positive_base_price",
            "base_price > 0"
        ))
        # Correlation – ความสัมพันธ์ระหว่างตัวเลข
        .addAnalyzer(Correlation("units_sold", "base_price"))
        .addAnalyzer(Correlation("units_sold", "max_temp_c"))
        .run()
    )

    metrics = AnalyzerContext.successMetricsAsDataFrame(spark, analysisResult)
    print("===== Analyzer metrics =====")
    metrics.show(truncate=False)

    # เขียนผล analyzer metrics ออกเป็น TSV (sep = '\t')
    metrics.repartition(1).write.format("csv") \
        .mode("overwrite") \
        .option("header", True) \
        .option("sep", "\t") \
        .save("{}/dataset-metrics".format(s3_output_analyze_data))

    # ==========================================================
    # 6) CHECKS – เช็ค data quality rule ที่เรากำหนดเอง
    # ==========================================================
    verificationResult = (
        VerificationSuite(spark)
        .onData(dataset)
        .addCheck(
            Check(spark, CheckLevel.Error, "Retail Demand Data Quality Check")
            # จำนวนแถวอย่างน้อย 500 (ถ้าในอนาคตมีมากกว่า ก็ยังผ่าน)
            .hasSize(lambda x: x >= 500)
            # ค่าต่ำสุดของ units_sold ต้อง >= 0
            .hasMin("units_sold", lambda v: v >= 0)
            # ส่วนลดห้ามเกิน 1 (100%)
            .hasMax("discount_pct", lambda v: v <= 1.0)
            # ห้าม null คอลัมน์สำคัญ
            .isComplete("record_id")
            .isComplete("date")
            .isComplete("store_id")
            .isComplete("units_sold")
            # record_id ต้อง unique
            .isUnique("record_id")
            # day_of_week ต้องอยู่ในเซ็ตนี้เท่านั้น
            .isContainedIn(
                "day_of_week",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )
            # flag ต่าง ๆ ต้องเป็น 0 หรือ 1
            .isContainedIn("is_weekend", [0, 1])
            .isContainedIn("is_holiday", [0, 1])
            .isContainedIn("is_hot_day", [0, 1])
            .isContainedIn("is_rainy_day", [0, 1])
            .isContainedIn("is_promo", [0, 1])
        )
        .run()
    )

    print("===== Verification Run Status =====")
    print("Status:", verificationResult.status)

    # ตารางสรุปว่าแต่ละ check ผ่าน/ไม่ผ่าน
    resultsDataFrame = VerificationResult.checkResultsAsDataFrame(spark, verificationResult)
    print("===== Check Results =====")
    resultsDataFrame.show(truncate=False)
    resultsDataFrame.repartition(1).write.format("csv") \
        .mode("overwrite") \
        .option("header", True) \
        .option("sep", "\t") \
        .save("{}/constraint-checks".format(s3_output_analyze_data))

    # success metrics (ค่าที่ใช้ในการ evaluate checks)
    verificationSuccessMetricsDataFrame = VerificationResult.successMetricsAsDataFrame(
        spark, verificationResult
    )
    print("===== Success Metrics =====")
    verificationSuccessMetricsDataFrame.show(truncate=False)
    verificationSuccessMetricsDataFrame.repartition(1).write.format("csv") \
        .mode("overwrite") \
        .option("header", True) \
        .option("sep", "\t") \
        .save("{}/success-metrics".format(s3_output_analyze_data))

    # ==========================================================
    # 7) Suggestion – ให้ Deequ ช่วย generate constraint ใหม่
    # ==========================================================
    suggestionsResult = (
        ConstraintSuggestionRunner(spark)
        .onData(dataset)
        .addConstraintRule(DEFAULT())
        .run()
    )

    suggestions = suggestionsResult["constraint_suggestions"]
    parallelizedSuggestions = spark.sparkContext.parallelize(suggestions)
    suggestionsResultsDataFrame = spark.createDataFrame(parallelizedSuggestions)

    print("===== Constraint Suggestions =====")
    suggestionsResultsDataFrame.show(truncate=False)
    suggestionsResultsDataFrame.repartition(1).write.format("csv") \
        .mode("overwrite") \
        .option("header", True) \
        .option("sep", "\t") \
        .save("{}/constraint-suggestions".format(s3_output_analyze_data))

    # ปิด SparkSession
    spark.stop()


if __name__ == "__main__":
    main()
