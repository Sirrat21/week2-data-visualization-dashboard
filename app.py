import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(
    page_title="Netflix Data Visualization Dashboard",
    layout="centered",
)

st.title("Netflix Data Visualization Dashboard")

st.markdown("""
### Project Introduction

This dashboard analyzes the **Netflix Movies & TV Shows Dataset** using Python, Pandas, Plotly, Matplotlib, Seaborn, and Streamlit.

**Objectives**
- Perform Exploratory Data Analysis (EDA)
- Create interactive visualizations
- Display KPIs
- Generate meaningful insights
""")

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")

    df.drop_duplicates(inplace=True)

    df["director"] = df["director"].fillna("Unknown")
    df["cast"] = df["cast"].fillna("Unknown")
    df["country"] = df["country"].fillna("Unknown")
    df["rating"] = df["rating"].fillna("Unknown")
    df["duration"] = df["duration"].fillna("Unknown")

    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

    return df

df = load_data()

st.sidebar.title("Dashboard Menu")
st.sidebar.write("Week 2 Internship Project")
st.sidebar.write("Interactive Data Visualization Dashboard")

st.sidebar.subheader("Filter by Type")

selected_type = st.sidebar.selectbox(
    "Choose Content Type",
    ["All"] + list(df["type"].unique())
)

if selected_type == "All":
    filtered_df = df
else:
    filtered_df = df[df["type"] == selected_type]

selected_country = st.sidebar.selectbox(
    "Choose Country",
    ["All"] + list(df["country"].unique())
)

if selected_country == "All":
    filtered_df = df
else:
    filtered_df = df[df["country"] == selected_country]

st.header("Dataset Overview")

col1, col2 = st.columns(2)

col1.metric("Rows", filtered_df.shape[0])
col2.metric("Columns", filtered_df.shape[1])

st.subheader("Dataset Preview")
st.dataframe(filtered_df.head())

st.header("Key Performance Indicators")

k1, k2, k3, k4 = st.columns(4)

movies = filtered_df[filtered_df["type"] == "Movie"].shape[0]
tvshows = filtered_df[filtered_df["type"] == "TV Show"].shape[0]
countries = filtered_df["country"].nunique()

k1.metric("Total Records", filtered_df.shape[0])
k2.metric("Movies", movies)
k3.metric("TV Shows", tvshows)
k4.metric("Countries", countries)

st.header("Movies vs TV Shows")

type_count = filtered_df["type"].value_counts().reset_index()
type_count.columns = ["Type", "Count"]

fig = px.bar(
    type_count,
    x="Type",
    y="Count",
    color="Type",
    text="Count",
    title="Movies vs TV Shows"
)

st.plotly_chart(fig, use_container_width=True)
st.header("Rating Distribution")

fig = px.pie(
    filtered_df,
    names="rating",
    title="Content Rating Distribution"
)

st.plotly_chart(fig, use_container_width=True)

st.header("Content Added Over Years")

filtered_df["Year Added"] = filtered_df["date_added"].dt.year

year_data = (
    filtered_df["Year Added"]
    .value_counts()
    .sort_index()
    .reset_index()
)

year_data.columns = ["Year", "Titles Added"]

fig = px.line(
    year_data,
    x="Year",
    y="Titles Added",
    markers=True,
    title="Content Added Over the Years"
)

st.plotly_chart(fig, use_container_width=True)

st.header("Release Year Distribution")

fig = px.histogram(
    filtered_df,
    x="release_year",
    nbins=30,
    title="Release Year Distribution"
)

st.plotly_chart(fig, use_container_width=True)

st.header("Release Year by Type")

fig = px.box(
    filtered_df,
    x="type",
    y="release_year",
    color="type",
    title="Release Year by Type"
)

st.plotly_chart(fig, use_container_width=True)

st.header("Top 10 Countries")

top_country = (
    filtered_df["country"]
    .value_counts()
    .head(10)
    .reset_index()
)

top_country.columns = ["Country", "Count"]

fig = px.bar(
    top_country,
    x="Country",
    y="Count",
    color="Country",
    text="Count",
    title="Top 10 Countries"
)

st.plotly_chart(fig, use_container_width=True)

st.header("Release Year vs Type")

fig = px.scatter(
    filtered_df,
    x="release_year",
    y="type",
    color="rating",
    title="Release Year vs Content Type"
)

st.plotly_chart(fig, use_container_width=True)

st.header("Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(include=np.number)

if len(numeric_df.columns) > 1:

    fig, ax = plt.subplots(figsize=(8,5))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

else:
    st.info("Not enough numeric columns for a heatmap.")

st.header("Download Filtered Dataset")
st.download_button(
    label="Download Filtered Dataset",
    data=filtered_df.to_csv(index=False),
    file_name="DATASET.csv",
    mime="text/csv"
)