import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Configure the Streamlit page layout and styling.
st.set_page_config(layout="centered")
sns.set_theme(style="whitegrid")

# Dictionary mapping descriptive event names to their corresponding parquet files.
# This allows the user to select an event by name and load the associated data.
event_files = {
    "Event 1: Abrupt Increase of BSW": "combined_data_event1.parquet",
    "Event 2: Spurious Closure of DHSV": "combined_data_event2.parquet",
    "Event 5: Severe Slugging": "combined_data_event5.parquet",
    "Event 6: Flow Instability": "combined_data_event6.parquet",
    "Event 7: Rapid Productivity Loss": "combined_data_event7.parquet",
    "Event 8: Quick Restriction in PCK": "combined_data_event8.parquet",
    "Event 9: Hydrate in Production Line": "combined_data_event9.parquet"
}

# Sidebar UI elements to select the event and the page type (Main Dashboard or Detailed EDA).
selected_event = st.sidebar.selectbox("Choose an Event", list(event_files.keys()))
page = st.sidebar.selectbox("Choose a page", ["Main Dashboard", "Detailed EDA"])

# Load the data for the currently selected event file.
df_event = pd.read_parquet(event_files[selected_event])
df_event = df_event.dropna(axis=1, how='all')  # Drop columns that are entirely empty.

# Display the title and a brief message at the top of the page.
st.title(f"Exploratory Data Analysis (EDA) for {selected_event}")
st.write("Loading and preparing the data...")

# Specify columns to exclude from analysis (like identifiers or labels) and
# define a threshold for considering columns "high-quality" (≥70% filled).
exclude_columns = ['state', 'label', 'source', 'well_id', 'filename', 'class']
threshold = 0.5

# If the user selected "Main Dashboard", display high-level KPIs and basic distributions.
if page == "Main Dashboard":
    st.header("Main Dashboard")

    # Compute key performance indicators (KPIs) such as total rows, unique filenames, and filled percentages.
    st.write("### Key Performance Indicators (KPIs)")
    total_rows = df_event.shape[0]
    unique_filenames = df_event['filename'].nunique()
    filled_percentage = df_event.notna().mean() * 100

    # Display the computed KPIs as metrics.
    st.metric(label="Total Rows", value=total_rows)
    st.metric(label="Unique Filenames", value=unique_filenames)
    st.metric(label="Average Filled Percentage", value=f"{filled_percentage.mean():.2f}%")
    
    # Show a simple bar chart of the distribution of 'class' values in the dataset.
    st.write("#### Class Distribution")
    class_counts = df_event['class'].value_counts()
    st.bar_chart(class_counts)

    # Show a bar chart of the filled percentage per column to quickly assess data completeness.
    st.write("#### Filled Percentage per Column")
    st.bar_chart(filled_percentage)

    # Display a summary of the numeric columns for the entire event.
    st.write("#### Summary Statistics for Numeric Columns")
    st.dataframe(df_event.describe())

    # Plot a heatmap to show correlations between numeric columns for the entire event.
    st.write("#### Correlation Heatmap")
    numeric_columns = df_event.select_dtypes(include=np.number).columns
    if len(numeric_columns) > 1:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df_event[numeric_columns].corr(), annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", ax=ax)
        st.pyplot(fig)

# If the user selected "Detailed EDA", provide more in-depth analysis and visualizations at the file level.
elif page == "Detailed EDA":
    st.header("Detailed EDA")

    # Allow the user to pick a specific filename within the event data for more granular analysis.
    filenames = df_event['filename'].unique()
    selected_filename = st.selectbox("Select a filename for detailed analysis", filenames)

    # Filter the main dataset to include only rows for the chosen filename.
    subset = df_event[df_event['filename'] == selected_filename]
    st.write(f"**Data shape for {selected_filename}**: {subset.shape}")

    # Calculate the duration for each class in seconds and minutes
    class_durations = subset['class'].value_counts().sort_index()

    # Display the duration of each class in seconds and minutes
    st.write("### Duration of Each Class")
    for cls, duration in class_durations.items():
        duration_seconds = duration
        duration_minutes = duration / 60
        st.write(f"Class {cls}: {duration_seconds} seconds ({duration_minutes:.2f} minutes)")

    # Display a summary of the numeric columns for the entire event.
    st.write("#### Summary Statistics for Numeric Columns")
    st.dataframe(subset.describe())

    # Clean up the 'class' column by forward and backward filling missing values.
    # This ensures each row has a 'class' value.
    subset['class'] = subset['class'].fillna(method='bfill').fillna(method='ffill').fillna(0)

    # Determine which columns are "high-quality" based on the threshold for completeness,
    # excluding certain known non-informative or label-like columns.
    filled_percentage = subset.notna().mean()
    valid_columns = filled_percentage[filled_percentage >= threshold].index.difference(exclude_columns).tolist()
    st.write(f"High-quality variables (≥50% filled): {valid_columns}")

    # Plot a heatmap to show correlations between high-quality numeric columns
    if valid_columns:
        st.write("### Correlation Heatmap for High-Quality Variables")

        # Calculate the correlation matrix for the selected high-quality variables
        correlation_matrix = subset[valid_columns].select_dtypes(include=np.number).corr()

        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap of High-Quality Variables")
        st.pyplot(fig)
    else:
        st.write("No high-quality variables available for correlation analysis.")


    # Plot the distribution of 'class' labels for the chosen filename.
    # This provides a sense of how many rows belong to each class.
    st.write(f"### Distribution of Class Labels for {selected_filename}")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.countplot(x='class', data=subset, palette='tab10', ax=ax)
    ax.set_title(f'Distribution of Class Labels for {selected_filename}')
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # For each high-quality variable, plot a time series showing how its values evolve over time,
    # with coloring by 'class' to see how class transitions might relate to changes in the variable.
    st.write(f"### Time Series for High-Quality Variables in {selected_filename}")
    for var in valid_columns:
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.lineplot(data=subset, x=subset.index, y=var, hue='class', palette='tab20', ax=ax)
        ax.set_title(f"Time Series of {var}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel(var)
        st.pyplot(fig)


    # Plot rolling statistics for each high-quality variable, such as mean and standard deviation,
    # to provide a sense of how the variable changes over time.
    # Select rolling window size using a slider
    window_size = st.slider("Select Rolling Window Size", 5, 240, 60)

    st.write("### Rolling Statistics for High-Quality Variables")

    for var in valid_columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by 'filename' to calculate rolling statistics within each file
        moving_avg = subset.groupby('filename')[var].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
        moving_diff = subset.groupby('filename')[var].transform(lambda x: x.diff(periods=window_size).fillna(0))
        
        # Plot original data
        ax.plot(subset.index, subset[var], label='Original Data', color='blue', alpha=0.5)
        
        # Plot moving average
        ax.plot(subset.index, moving_avg, label='Moving Average', color='green')
        
        # Plot moving difference
        ax.plot(subset.index, moving_diff, label='Moving Difference', color='red')
        
        # Add title and legend
        ax.set_title(f"Moving Average and Moving Difference for {var}")
        ax.set_xlabel("Index")
        ax.set_ylabel(var)
        ax.legend()
        
        # Display the plot in Streamlit
        st.pyplot(fig)


    # Provide a dropdown for the user to choose different types of additional graphs to explore:
    # "Histogram by Class" to look at distributions per class, or 
    # "Custom Time Series (All Files)" to compare across all files.
    graph_type = st.selectbox("Select a graph type", ["Histogram by Class", "Custom Time Series (All Files)"])

    # If the user chooses histograms by class, show distributions of each high-quality variable, separated by class.
    if graph_type == "Histogram by Class":
        st.write(f"### Distributions by Class for High-Quality Variables in {selected_filename}")
        for var in valid_columns:
            fig, ax = plt.subplots(figsize=(6, 3))
            try:
                # Attempt to plot a kernel density estimate (KDE). If it fails due to low variance, fallback to no KDE.
                sns.histplot(data=subset, x=var, hue='class', kde=True, bins=20, palette='tab20', ax=ax)
            except np.linalg.LinAlgError:
                sns.histplot(data=subset, x=var, hue='class', kde=False, bins=20, palette='tab20', ax=ax)
                st.warning(f"KDE could not be applied to '{var}' due to low variance.")
            ax.set_title(f"Distribution of {var}")
            ax.set_xlabel(var)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    # If the user chooses the custom time series option, plot all files together to see how a variable behaves across
    # different files, and mark transitions from class 0 to transient classes.
    elif graph_type == "Custom Time Series (All Files)":
        st.write("### Custom Time Series (All Files)")
        
        # Define line styles and colors for visual differentiation among multiple files.
        line_styles = ['-', '--', '-.', ':']
        max_unique_files = 40
        colors = cm.get_cmap('tab20', max_unique_files)
        filename_styles = {
            filename: (mcolors.to_hex(colors(i)), line_styles[i % len(line_styles)])
            for i, filename in enumerate(filenames)
        }

        # For each high-quality variable, plot time series data from all files,
        # splitting the timeline into a "class 0" portion and a "transient class" portion.
        for column in valid_columns:
            fig, ax = plt.subplots(figsize=(18, 10))
            
            # Group the data by filename to plot each file separately with a unique style.
            for filename, group_data in df_event.groupby('filename'):
                if column in group_data.columns:
                    class_0_data = group_data[group_data['class'] == 0][column]
                    transient_class_data = group_data[group_data['class'] > 0][column]
                    
                    x_class_0 = np.arange(len(class_0_data))
                    x_transient = np.arange(len(class_0_data), len(class_0_data) + len(transient_class_data))
                    
                    color, line_style = filename_styles[filename]
                    
                    # Plot the class 0 segment and the transient segment of the timeline with distinct sections.
                    ax.plot(x_class_0, class_0_data, label=f'{filename}', linestyle=line_style, color=color, alpha=0.7)
                    ax.plot(x_transient, transient_class_data, linestyle=line_style, color=color, alpha=0.7)

            # Draw a vertical line to indicate where data transitions from class 0 to transient classes.
            ax.axvline(x=len(class_0_data), color='black', linestyle='--', label='Transition 0 → Transient')
            
            ax.set_title(f'Time Series for {column} (Segmented by Class for All Files)', fontsize=16)
            ax.set_xlabel('Segmented by Class (0, Transient)', fontsize=14)
            ax.set_ylabel(column, fontsize=14)
            ax.legend(title='Filename', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)
            
            plt.tight_layout(pad=3)
            st.pyplot(fig)

st.write("### EDA Complete")
