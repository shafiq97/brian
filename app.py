import pandas as pd
import numpy as np

if __name__ == "__main__":
    # read the specified worksheets from the Excel file
    file_path = "whs2022_annex2.xlsx"
    sheets = ["Annex 2-1", "Annex 2-2", "Annex 2-3", "Annex 2-4"]

    # 1
    try:
        with pd.ExcelFile(file_path) as xls:
            dfs = [pd.read_excel(xls, sheet_name=sheet) for sheet in sheets]
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")

    # store the dataframes in a dictionary
    data = {
        "annex1": dfs[0] if len(dfs) > 0 else None,
        "annex2": dfs[1] if len(dfs) > 1 else None,
        "annex3": dfs[2] if len(dfs) > 2 else None,
        "annex4": dfs[3] if len(dfs) > 3 else None
    }

    # 2
    # read the footnotes worksheet
    footnotes_df = pd.read_excel("whs2022_annex2.xlsx", sheet_name="Footnotes")

    # create a Series with footnote markers as index
    footnotes = pd.Series(
        data=footnotes_df.iloc[:, 1].values, index=footnotes_df.iloc[:, 0].values)

    # print the resulting Series
    # print(footnotes)

    # 3
    # read the Annex 2-1 worksheet

    # Load the dataset

    # Read the Excel file
    file_name = "whs2022_annex2.xlsx"
    sheet_name = "Annex 2-1"
    df = pd.read_excel(file_name, sheet_name=sheet_name, header=None)

    # Forward fill missing values in the first and fourth rows
    df.iloc[[0, 3]] = df.iloc[[0, 3]].fillna(method='ffill', axis=1)

    # Concatenate gender information to the values in the first row
    gender_info = df.iloc[2, 1:4].values
    new_columns = []

    for col in df.iloc[0]:
        if col in gender_info:
            prev_col = new_columns[-1]
            new_columns.append(f"{prev_col} {col}")
        else:
            new_columns.append(col)

    df.iloc[0] = new_columns

    # Save the modified dataset to a new Excel file
    output_file = "modified_excel_file.xlsx"
    df.to_excel(output_file, sheet_name=sheet_name, index=False, header=False)

    # print(annex_df)

    # 4
    def process_df(df):
        # Delete the last column
        df = df.drop(df.columns[-1], axis=1)

        # Delete the second and third row
        df = df.drop(df.index[[1, 2]], axis=0)

        # Delete all blank rows and all blank columns
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # Delete columns with footnote marker symbols
        for col in df.columns:
            if df[col].isin(footnotes).any():
                df = df.drop(col, axis=1)

        # Set the first two rows as the two-level header
        df.columns = pd.MultiIndex.from_tuples(
            zip(df.iloc[0], df.iloc[1]), names=["Statistic", "Year"])
        df = df.drop(df.index[0:2])

        # Set the first column as the index
        df = df.set_index(df.columns[0])
        df.index.name = 'Member State'

        return df

    # 5
    def replace_outer_headers(df, header_series):
        outer_headers = df.columns.get_level_values(0).unique()
        new_outer_headers = {
            outer_headers[i]: header_series[i] for i in range(len(outer_headers))}

        df.columns = df.columns.set_levels(
            df.columns.levels[0].map(lambda x: new_outer_headers.get(x, x)),
            level=0
        )
        return df

    # 6
    def process_missing_values_and_data_types(df):
        # Print the total number of missing values
        print(f"Total missing values: {df.isna().sum().sum()}")

        # Display the rows with missing values
        print("Rows with missing values:")
        print(df[df.isna().any(axis=1)])

        # Remove the row with missing values
        df = df.dropna(how='any', axis=0)

        # Print the data type of each column
        print("Data types of columns:")
        print(df.dtypes)

        # Replace "<0.1", "<1", and "-" entries
        df = df.replace(["<0.1", "<1", "-"], [0, 0, np.nan])

        # Convert all columns to numeric data type
        df = df.apply(pd.to_numeric, errors='coerce')

        # Verify that the conversions are successful
        print("Data types after conversion:")
        print(df.dtypes)

        # Print the number of missing values of each column as a percentage of the total length of the column
        print("Percentage of missing values for each column:")
        print((df.isna().sum() / len(df)) * 100)

        return df

    a1 = process_df(data["annex1"])
    a2 = process_df(data["annex2"])
    a3 = process_df(data["annex3"])
    a4 = process_df(data["annex4"])

    ha1 = pd.Series(a1.columns.get_level_values(0).unique())
    ha2 = pd.Series(a2.columns.get_level_values(0).unique())
    ha3 = pd.Series(a3.columns.get_level_values(0).unique())
    ha4 = pd.Series(a4.columns.get_level_values(0).unique())

    a1 = replace_outer_headers(a1, ha1)
    a2 = replace_outer_headers(a2, ha2)
    a3 = replace_outer_headers(a3, ha3)
    a4 = replace_outer_headers(a4, ha4)

    # Process each DataFrame
    a1 = process_missing_values_and_data_types(a1)
    print("\n---\n")
    a2 = process_missing_values_and_data_types(a2)
    print("\n---\n")
    a3 = process_missing_values_and_data_types(a3)
    print("\n---\n")
    a4 = process_missing_values_and_data_types(a4)

    # Save the processed DataFrame a1 to an Excel file
    output_file = "processed_a1111.xlsx"
    a1.to_excel(output_file, sheet_name='Annex 2-1 Processed')

    # 7
    # Extract relevant columns from DataFrame a1
    child_mortality = a1.loc[:, [('Under-five mortality ratee (per 1000 live births)', 2020),
                                 ('Neonatal mortality ratee (per 1000 live births)', 2020)]]

    # Replace "<1" and "<0.1" with 0, and replace "-" with NaN
    child_mortality = child_mortality.replace(
        {'<1': 0, '<0.1': 0, '-': np.nan})

    # Convert columns to numeric data type
    child_mortality = child_mortality.apply(pd.to_numeric)

    # Calculate mortality rates as percentages
    child_mortality = child_mortality * 0.1

    # Sort by under-five mortality rate, then by neonatal mortality rate
    child_mortality = child_mortality.sort_values(by=[('Under-five mortality ratee (per 1000 live births)', 2020),
                                                      ('Neonatal mortality ratee (per 1000 live births)', 2020)],
                                                  ascending=True)

    # Show top 20 countries
    child_mortality = child_mortality.head(20)

    # Rename columns
    child_mortality.columns = [
        'Under-five mortality rate (%)', 'Neonatal mortality rate (%)']

    # Display table
    print(child_mortality)
