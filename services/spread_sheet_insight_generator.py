import pandas as pd
import yaml
from dotenv import load_dotenv
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
from langchain_ollama import OllamaLLM 

# Load environment variables
load_dotenv()

# Function to load configuration from YAML file
def load_config(config_path="config\config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load configuration
config = load_config()

# Function to select the LLM based on the config.yaml file
def initialize_llm(config):
    if config['llm_provider'] == "groq":
        print(f"Using Groq LLM Model: {config['llm_model']}")
        return ChatGroq(model=config['llm_model'])
    elif config['llm_provider'] == "ollama":
        print(f"Using Ollama LLM Model: {config['llm_model']}")
        return OllamaLLM(model=config['llm_model']) 
    else:
        raise ValueError("Unsupported LLM provider specified in config.yaml.")

# Initialize the LLM based on configuration
llm = initialize_llm(config)

# Load the dataset
try:
    df = pd.read_csv("samples\Train.csv")  # Specify correct path to the CSV file
except FileNotFoundError:
    raise Exception("CSV file not found. Please check the path to the dataset.")

# Function to generate key insights from the dataset
def generate_insights(dataframe):
    # Initialize SmartDataframe with the selected LLM
    sdf = SmartDataframe(dataframe, config={"llm": llm})

    print("===== Data Insights Report =====\n")

    # 1. Basic Information: Displays general info about the dataset
    print("1. Basic Information:\n")
    print(dataframe.info(), "\n")

    # 2. Missing Values: Summarizes the number of missing values in each column
    print("2. Missing Values Summary:\n")
    print(dataframe.isnull().sum(), "\n")

    # 3. Descriptive Statistics: Generates summary statistics for numeric columns
    print("3. Descriptive Statistics (Numeric Columns):\n")
    print(dataframe.describe(), "\n")

    # 4. Correlation Matrix (Plotting using LLM)
    print("4. Correlation Matrix:\n")
    correlation_matrix_prompt = "Create a plot of the correlation matrix."
    correlation_matrix = sdf.chat(correlation_matrix_prompt)
    print(correlation_matrix)

    # 5. Outlier Detection: Detects any outliers in the dataset using LLM
    print("\n5. Outlier Detection:\n")
    outlier_prompt = "Identify any outliers in the dataset."
    outliers = sdf.chat(outlier_prompt)
    print(outliers)

    # 6. Value Counts for Categorical Columns: Displays the distribution of values for each categorical column
    print("\n6. Categorical Columns Distribution:\n")
    categorical_cols = dataframe.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            print(f"Value counts for {col}:\n", dataframe[col].value_counts(), "\n")

            # Create a plot of the distribution of values for the categorical column
            plot_prompt = f"Create a plot showing the distribution of {col}."
            plot_output = sdf.chat(plot_prompt)
            print(plot_output)
    else:
        print("No categorical columns found.\n")

    # 7. Time-Series Trend Analysis: If 'date' column exists, plots trends over time
    if 'date' in dataframe.columns:
        print("\n7. Time-Series Trend Analysis:\n")
        trend_prompt = "Plot the trends in the data over time."
        trends = sdf.chat(trend_prompt)
        print(trends)
    else:
        print("\nNo date column detected for trend analysis.\n")

# Call the function to generate insights
generate_insights(df)
