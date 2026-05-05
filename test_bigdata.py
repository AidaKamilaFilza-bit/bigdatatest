# app.py

import os
import io
import tempfile
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


# -------------------------------------------------
# Page Setup
# -------------------------------------------------

st.set_page_config(
    page_title="Hacker Attack Clustering System",
    layout="wide"
)

st.title("Hacker Attack K-Means Clustering System")
st.write(
    "This web-based system uses Streamlit and PySpark to group hacker attack "
    "sessions using K-Means clustering."
)


# -------------------------------------------------
# Spark Session
# -------------------------------------------------

@st.cache_resource
def get_spark():
    spark = (
        SparkSession.builder
        .appName("HackerAttackClustering")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    return spark


spark = get_spark()


# -------------------------------------------------
# Sidebar Settings
# -------------------------------------------------

st.sidebar.header("Clustering Settings")

k_value = st.sidebar.slider(
    "Select Number of Clusters (k)",
    min_value=2,
    max_value=3,
    value=2,
    step=1
)

seed_value = st.sidebar.number_input(
    "Random Seed",
    min_value=1,
    value=42,
    step=1
)

preview_rows = st.sidebar.slider(
    "Preview Rows",
    min_value=5,
    max_value=50,
    value=10,
    step=5
)


# -------------------------------------------------
# Upload CSV
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload hack_data.csv",
    type=["csv"]
)


if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    try:
        # -------------------------------------------------
        # Step 2: Load Data Using PySpark
        # -------------------------------------------------

        df = spark.read.csv(
            temp_path,
            header=True,
            inferSchema=True
        )

        st.subheader("1. Dataset Preview")
        st.dataframe(df.limit(preview_rows).toPandas(), use_container_width=True)

        # -------------------------------------------------
        # Step 3: Basic Dataset Info
        # -------------------------------------------------

        row_count = df.count()
        col_count = len(df.columns)

        col1, col2 = st.columns(2)
        col1.metric("Number of Rows", row_count)
        col2.metric("Number of Columns", col_count)

        st.subheader("2. Dataset Columns")
        st.write(df.columns)

        st.subheader("3. Dataset Schema")
        schema_info = []

        for field in df.schema.fields:
            schema_info.append({
                "Column": field.name,
                "Data Type": str(field.dataType),
                "Nullable": field.nullable
            })

        schema_df = pd.DataFrame(schema_info)
        st.dataframe(schema_df, use_container_width=True)

        # -------------------------------------------------
        # Step 4: Prepare Data
        # -------------------------------------------------

        st.subheader("4. Data Preparation")

        numeric_cols = [
            "Session_Connection_Time",
            "Bytes Transferred",
            "Kali_Trace_Used",
            "Servers_Corrupted",
            "Pages_Corrupted",
            "WPM_Typing_Speed"
        ]

        st.write("Selected numeric columns for clustering:")
        st.write(numeric_cols)

        st.info(
            "The Location column is excluded because it is categorical and may not be reliable "
            "since attackers can use VPNs."
        )

        # Cast selected columns into double
        working_df = df

        for col_name in numeric_cols:
            working_df = working_df.withColumn(
                col_name,
                F.col(col_name).cast("double")
            )

        df_clean = working_df.select(numeric_cols).dropna()

        st.write("Rows after removing missing values:", df_clean.count())

        st.subheader("Missing Value Check")

        missing_exprs = [
            F.count(F.when(F.col(c).isNull(), c)).alias(c)
            for c in numeric_cols
        ]

        missing_df = df_clean.select(missing_exprs).toPandas().T.reset_index()
        missing_df.columns = ["Column", "Missing Count"]
        st.dataframe(missing_df, use_container_width=True)

        # -------------------------------------------------
        # Step 5: Create Feature Vector
        # -------------------------------------------------

        st.subheader("5. Feature Vector Creation")

        assembler = VectorAssembler(
            inputCols=numeric_cols,
            outputCol="features_raw",
            handleInvalid="skip"
        )

        assembled_df = assembler.transform(df_clean)

        st.write("The selected numeric columns were combined into one feature vector.")

        with st.expander("View Feature Vector Sample"):
            feature_sample = assembled_df.select(
                F.col("features_raw").cast("string").alias("features_raw")
            ).limit(10).toPandas()

            st.dataframe(feature_sample, use_container_width=True)

        # -------------------------------------------------
        # Step 6: Scale Features
        # -------------------------------------------------

        st.subheader("6. Feature Scaling")

        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )

        scaler_model = scaler.fit(assembled_df)
        scaled_df = scaler_model.transform(assembled_df)

        st.write(
            "StandardScaler was applied because the selected features have different value ranges."
        )

        with st.expander("View Scaled Feature Sample"):
            scaled_sample = scaled_df.select(
                F.col("features").cast("string").alias("scaled_features")
            ).limit(10).toPandas()

            st.dataframe(scaled_sample, use_container_width=True)

        # -------------------------------------------------
        # Step 7: Run K-Means Clustering
        # -------------------------------------------------

        st.subheader("7. K-Means Clustering")

        kmeans = KMeans(
            featuresCol="features",
            predictionCol="cluster",
            k=int(k_value),
            seed=int(seed_value)
        )

        model = kmeans.fit(scaled_df)
        clustered_df = model.transform(scaled_df)

        st.write(f"K-Means clustering was applied using k = {k_value}.")

        # -------------------------------------------------
        # Clustered Output
        # -------------------------------------------------

        st.subheader("8. Clustered Output")

        clustered_pd = clustered_df.select(
            *numeric_cols,
            "cluster"
        ).toPandas()

        st.dataframe(clustered_pd.head(50), use_container_width=True)

        # -------------------------------------------------
        # Cluster Distribution
        # -------------------------------------------------

        st.subheader("9. Cluster Distribution")

        cluster_counts = (
            clustered_df
            .groupBy("cluster")
            .count()
            .orderBy("cluster")
            .toPandas()
        )

        st.dataframe(cluster_counts, use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            cluster_counts["cluster"].astype(str),
            cluster_counts["count"]
        )
        ax.set_title("Number of Attack Sessions per Cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Sessions")
        st.pyplot(fig)

        # -------------------------------------------------
        # Cluster Summary
        # -------------------------------------------------

        st.subheader("10. Cluster Summary")

        agg_exprs = []

        for c in numeric_cols:
            agg_exprs.append(F.avg(c).alias(f"{c}_mean"))
            agg_exprs.append(F.stddev(c).alias(f"{c}_std"))

        cluster_summary = (
            clustered_df
            .groupBy("cluster")
            .agg(*agg_exprs)
            .orderBy("cluster")
            .toPandas()
        )

        st.dataframe(cluster_summary, use_container_width=True)

        # -------------------------------------------------
        # Cluster Centers
        # -------------------------------------------------

        st.subheader("11. Cluster Centers")

        centers = model.clusterCenters()

        centers_df = pd.DataFrame(
            centers,
            columns=numeric_cols
        )

        centers_df.index = [f"Cluster {i}" for i in range(len(centers_df))]

        st.dataframe(centers_df, use_container_width=True)

        # -------------------------------------------------
        # Evaluation
        # -------------------------------------------------

        st.subheader("12. Clustering Evaluation")

        evaluator = ClusteringEvaluator(
            featuresCol="features",
            predictionCol="cluster",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean"
        )

        silhouette = evaluator.evaluate(clustered_df)

        st.metric("Silhouette Score", f"{silhouette:.4f}")

        st.write(
            "A higher silhouette score means the clusters are more separated and better formed."
        )

        # -------------------------------------------------
        # Interpretation
        # -------------------------------------------------

        st.subheader("13. Interpretation")

        if k_value == 2:
            st.write(
                "For k = 2, the attack sessions are grouped into two hacker behaviour patterns. "
                "If the cluster sizes are balanced and the feature averages are clearly different, "
                "this suggests that two main hackers are strongly represented in the dataset."
            )
        elif k_value == 3:
            st.write(
                "For k = 3, the attack sessions are grouped into three clusters. "
                "This can be used to test whether a third hacker may be involved. "
                "If one cluster is very small or has similar behaviour to another cluster, "
                "then the third hacker involvement may be uncertain."
            )

        # -------------------------------------------------
        # Download Clustered Output
        # -------------------------------------------------

        st.subheader("14. Download Clustered Result")

        output_df = clustered_df.select(
            *numeric_cols,
            "cluster"
        ).toPandas()

        csv_buffer = io.StringIO()
        output_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="Download Clustered Output CSV",
            data=csv_buffer.getvalue(),
            file_name="clustered_hacker_output.csv",
            mime="text/csv"
        )

    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass

else:
    st.info("Please upload the hack_data.csv file to begin clustering analysis.")