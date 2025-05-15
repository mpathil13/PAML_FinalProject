import streamlit as st                  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from itertools import combinations
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from helper_functions import fetch_dataset

alphafold_features = ['ranking_score', 'ptm', 'iptm', 'chainA_iptm',
       'chainC_iptm', 'chainD_iptm', 'chainE_iptm', 'A1_pLDDT', 'A2_pLDDT',
       'A3_pLDDT', 'A4_pLDDT', 'A5_pLDDT', 'A6_pLDDT', 'A7_pLDDT', 'A8_pLDDT',
       'A9_pLDDT', 'A_average_pLDDT', 'C_average_pLDDT', 'D_average_pLDDT',
       'E_average_pLDDT']

rmsd_features = ['rmsd_complex_Ca',
       'rmsd_HLA_groove_all_atom', 'rmsd_TCR_Ca', 'rmsd_peptide_all_atom',
       'rmsd_peptide_backbone', 'rmsd_peptide_position_1',
       'rmsd_peptide_position_2', 'rmsd_peptide_position_3',
       'rmsd_peptide_position_4', 'rmsd_peptide_position_5',
       'rmsd_peptide_position_6', 'rmsd_peptide_position_7',
       'rmsd_peptide_position_8', 'rmsd_peptide_position_9']

target_columns = ['logA', 'obs_logC', 'affinity',  'presenter_category']

#############################################

st.markdown('# Visualize')

#############################################

def display_features(df):
    """
    This function displays feature names and descriptions from dataset columns.
    
    Inputs:
    - df (pandas.DataFrame): The input DataFrame to with features to be displayed.
    Outputs: None
    """
    st.markdown("**AlphaFold Confidence Metrics**")

    columns = st.columns(3) # 3 columns
    for i, col in enumerate(alphafold_features):
        col_index = i % 3
        with columns[col_index]:
            st.markdown(f"{col}")

    columns = st.columns(1)
    st.markdown("**RMSD Metrics**")

    columns = st.columns(3) # 3 columns
    for i, col in enumerate(rmsd_features):
        col_index = i % 3
        with columns[col_index]:
            st.markdown(f"{col}")

    columns = st.columns(1)
    st.markdown(f"**Target Labels**")

    columns = st.columns(2) # 2 columns
    for i, col in enumerate(target_columns):
        col_index = i % 2
        with columns[col_index]:
            st.markdown(f"{col}")

def categorize_substitution(x):
    nonpolar_aliphatic = ["G", "A", "V", "P", "L", "M", "I"]
    polar_uncharged = ["S", "T", "C", "N", "Q"]
    aromatic = ["F", "Y", "W"]
    positive = ["K", "R", "H"]
    negative = ["D", "E"] 
    if x in nonpolar_aliphatic:
        return "Nonpolar, Aliphatic"
    if x in polar_uncharged:
        return "Polar, Uncharged"
    if x in aromatic:
        return "Aromatic"
    if x in positive:
        return "Positive"
    if x in negative:
        return "Negative"

def remove_features(df,removed_features):
    """
    Remove the features in removed_features (list) from the input pandas dataframe df. 

    Input
    - df is dataset in pandas dataframe
    Output: 
    - df: pandas dataframe with features removed
    """
    X = df.drop(removed_features, axis=1)
    st.session_state['drop_features_df'] = df
    return X

def compute_correlation(df, features):
    """
    This function computes pair-wise correlation coefficents of X and render summary strings

    Input
        - df: pandas dataframe 
        - features: a list of feature name (string), e.g. ['age','height']
    Output
        - correlation: correlation coefficients between one or more features
        - summary statements: a list of summary strings where each of it is in the format: 
            '- Features X and Y are {strongly/weakly} {positively/negatively} correlated: {correlation value}'
    """
    correlation = None
    cor_summary_statements = []
    
    correlation = df[features].corr()
    feature_pairs = combinations(features, 2)
    for f1, f2 in feature_pairs:
        corr = correlation[f1][f2]
        summary = '- Features %s and %s are %s %s correlated: %.2f' % (  
f1, f2, 'strongly' if corr > 0.5 else 'weakly', 'positively' if corr > 0 else 'negatively', corr) 
        st.markdown(summary)
        cor_summary_statements.append(summary)
    
    st.session_state['correlation_df'] = df
    return correlation, cor_summary_statements

###################### FETCH DATASET #######################

df=None
df = fetch_dataset()

st.markdown("We loaded our pMHC-TCR dataset. Here's a key to help you follow along: \
    \n :whale: - headings for steps in our approach \
    \n :octopus: - user-input opportunities for data visualizations \
    \n :penguin: - data processing steps handled behind the scenes")

######################### MAIN BODY #########################

######################### EXPLORE DATASET #########################

if df is not None:
    st.markdown('#### :whale: KEY FEATURES')

    # Display feature names and descriptions
    display_features(df)
    
    # Display dataframe as table
    st.dataframe(df)

    ###################### VISUALIZE DATASET #######################
    st.markdown('#### :whale: DATA VISUALIZATION')
    st.markdown(':octopus: Try it yourself! Use the sidebar to select a numerical feature for histogram visualization.')
    numeric_columns = list(df.select_dtypes(['float','int']).columns)

    # Plot histograms
    st.sidebar.header('Select data for histogram plots')
    try:
        x_value = st.sidebar.selectbox('Feature to plot', options=numeric_columns)
        fig, ax = plt.subplots()
        df[x_value].hist(ax=ax)
        plt.title(f'{x_value}', fontweight="bold")
        st.pyplot(fig)
    except Exception as e:
        print(e)

    st.markdown(':octopus: Generate super useful plots to assess pLDDT and RMSD data distribution \
                based on MHC presentation and peptide position.')
    generate_plots_button = st.button('Generate plots')

    if generate_plots_button:  
        st.markdown('To validate our approach using MHCflurry-predicted pMHC binding affinity, view predicted binding' \
        ' affinity vs. position of amino acid mutation for wild-type peptide NLVPMVATV. Biochemical properties ' \
        ' are indicated by the datapoint color, per the legend.')
        try:
            AA_cat_order = ["Positive", "Negative", "Aromatic", "Polar, Uncharged", "Nonpolar, Aliphatic"]
            AA_category_colors = {
                "Nonpolar, Aliphatic": "#ffbe0b",
                "Polar, Uncharged": "#fb5607",
                "Aromatic": "#ff006e",
                "Positive": "#8338ec",
                "Negative": "#3a86ff",
            }
            TCR_peptide = df[['peptide','affinity']]
            TCR_peptide["position"] = TCR_peptide['peptide'].apply(lambda x: x[0:2])
            TCR_peptide["substitution"] = TCR_peptide['peptide'].apply(lambda x: x[2])
            TCR_peptide["category"] = TCR_peptide['substitution'].apply(categorize_substitution)
            TCR_peptide = TCR_peptide.drop_duplicates()
            positions_numbers = ["N1", "L2", "V3", "P4", "M5", "V6", "A7", "T8", "V9"]
            TCR_peptide['log_Kd'] = np.log10(TCR_peptide['affinity'])

            fig, ax = plt.subplots()
            sns.boxplot(x='position', y='log_Kd', data=TCR_peptide, order=positions_numbers, color="whitesmoke", ax=ax)
            ax = sns.stripplot(x='position', y='log_Kd', hue='category', order=positions_numbers, jitter=True, data=TCR_peptide, hue_order=AA_cat_order, palette=AA_category_colors, dodge=True)
            plt.legend(title='Residue Type')
            ax.axhline(y=1.7, color='black', linestyle='--')
            ax.axhline(y=2.7, color='black', linestyle='--')
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.8))
            ax.set_xlabel('Mutant Residue Position')
            ax.set_ylabel('log(Kd)')
            ax.set_title('Predicted Binding Affinity vs. Mutant Position', fontweight="bold")
            st.pyplot(fig)
        except Exception as e:
            print(e)

        st.markdown('Visualize the distributions of pLDDT (AlphaFold local confidence metric) and RMSD. We categorize by ' \
        'pMHC presentation strength and peptide position.')
        try:
            positions_numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
            rmsd_positions = ['rmsd_peptide_position_1',
        'rmsd_peptide_position_2', 'rmsd_peptide_position_3',
        'rmsd_peptide_position_4', 'rmsd_peptide_position_5',
        'rmsd_peptide_position_6', 'rmsd_peptide_position_7',
        'rmsd_peptide_position_8', 'rmsd_peptide_position_9']
            peptide_positions_dict = dict(zip(rmsd_positions, positions_numbers))

            df_temp = df.copy()
            df_temp.rename(columns=peptide_positions_dict, inplace=True)

            df_nonbinder = df_temp[df_temp['presenter_category'] == 'nonbinder'].set_index('peptide')
            df_strong = df_temp[df_temp['presenter_category'] == 'strong_intermediate'].set_index('peptide')
            unique_nonbinder, unique_strong = len(df_nonbinder['ID'].unique()), len(df_strong['ID'].unique())
            xlabel, ylabel = 'Epitope Position', 'RMSD (\u00C5)'

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
            sns.boxplot(data=df_nonbinder[positions_numbers], color="#e9ecef", ax=axes[1])
            axes[1].set_xlabel(xlabel, fontsize=16)
            axes[1].set_ylabel(ylabel, labelpad=0.0005, fontsize=16)
            axes[1].set_title(f'Non-Binders ({unique_nonbinder})', fontsize=16)
            sns.boxplot(data=df_strong[positions_numbers], color="#495057", ax=axes[0])
            axes[0].set_xlabel(xlabel, fontsize=16)
            axes[0].set_ylabel(ylabel, labelpad=0.0005, fontsize=16)
            axes[0].set_ylim(-0.5, 5.7)
            axes[0].set_title(f'Strong/Intermediate Binders ({unique_strong})', fontsize=16)
            axes[0].axhline(y=2, color='black', linestyle='--')
            axes[1].axhline(y=2, color='black', linestyle='--')
            plt.suptitle(f"Per-Residue RMSD Distribution for All Mutant Samples", fontsize=18, y =1.05, fontweight="bold")
            st.pyplot(fig)

            AF3_positions = ['A1_pLDDT', 'A2_pLDDT',
        'A3_pLDDT', 'A4_pLDDT', 'A5_pLDDT', 'A6_pLDDT', 'A7_pLDDT', 'A8_pLDDT',
        'A9_pLDDT']
            AF3_peptide_positions_dict = dict(zip(AF3_positions, positions_numbers))

            AF3_df = df.copy()
            AF3_df.rename(columns=AF3_peptide_positions_dict, inplace=True)

            AF3_df_nonbinder = AF3_df[AF3_df['presenter_category'] == 'nonbinder'].set_index('peptide')
            AF3_df_strong = AF3_df[AF3_df['presenter_category'] == 'strong_intermediate'].set_index('peptide')
            unique_nonbinder, unique_strong = len(AF3_df_nonbinder['ID'].unique()), len(AF3_df_strong['ID'].unique())
            xlabel, ylabel = "Epitope Position", "pLDDT"

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
            sns.violinplot(data=AF3_df_nonbinder[positions_numbers],  color="#e9ecef", ax=axes[1])
            sns.violinplot(data=AF3_df_strong[positions_numbers], color="#495057", ax=axes[0])

            axes[1].set_xlabel(xlabel, fontsize=16)
            axes[1].set_ylabel(ylabel, labelpad=0.0005, fontsize=16)
            axes[1].set_title(f'Non-Binders ({unique_nonbinder})', fontsize=16)
            axes[0].set_xlabel(xlabel, fontsize=16)
            axes[0].set_ylabel(ylabel, labelpad=0.0005, fontsize=16)
            axes[0].set_title(f'Strong/Intermediate Binders ({unique_strong})', fontsize=16)
            axes[0].axhline(y=70, color='black', linestyle='--')
            axes[1].axhline(y=70, color='black', linestyle='--')
            plt.setp(axes, ylim=(35,105))
            plt.suptitle(f"Per-Residue pLDDT Distribution for All Mutant Samples", fontsize=18, y =1.05, fontweight="bold")
            st.pyplot(fig)
        except Exception as e:
            print(e)

    st.markdown("Correlation time! The following heatmap summarizes correlations between numerical features.")
    try:
        # Evaluate correlation between numerical features
        features= ['ptm', 'iptm', 'chainA_iptm', 'chainC_iptm', 'chainD_iptm',
            'chainE_iptm', 'A1_pLDDT', 'A2_pLDDT', 'A3_pLDDT', 'A4_pLDDT',
            'A5_pLDDT', 'A6_pLDDT', 'A7_pLDDT', 'A8_pLDDT', 'A9_pLDDT',
            'A_average_pLDDT', 'C_average_pLDDT', 'D_average_pLDDT',
            'E_average_pLDDT', 'rmsd_complex_Ca', 'rmsd_TCR_Ca',
            'rmsd_peptide_all_atom', 'rmsd_peptide_backbone',
            'rmsd_peptide_position_1', 'rmsd_peptide_position_2',
            'rmsd_peptide_position_3', 'rmsd_peptide_position_4',
            'rmsd_peptide_position_5', 'rmsd_peptide_position_6',
            'rmsd_peptide_position_7', 'rmsd_peptide_position_8',
            'rmsd_peptide_position_9']
        correlation = df[features].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        plt.title('Confidence and RMSD Feature Correlation Heatmap', fontweight="bold")
        st.pyplot(fig)
    except Exception as e:
        print(e)

    st.markdown(":octopus: Try it yourself! Select features to visualize their correlation.")
    
    # Collect features for correlation analysis using multiselect
    numeric_columns = list(df.select_dtypes(['float','int']).columns)

    select_features_for_correlation = st.multiselect(
        'Select features for visualizing the correlation analysis (up to 4 recommended)',
        numeric_columns,
    )
    
    generate_correlation_button = st.button('Generate correlation analysis')

    if generate_correlation_button:  
        # Compute correlation between selected features
        correlation, correlation_summary = compute_correlation(
            df, select_features_for_correlation)
        st.write(correlation)
        # Display correlation of all feature pairs
        if select_features_for_correlation:
            try:
                fig = scatter_matrix(
                    df[select_features_for_correlation], figsize=(12, 8))
                st.pyplot(fig[0][0].get_figure())
            except Exception as e:
                print(e)

    ###################### PREPROCESS DATASET #######################
    st.markdown("#### :whale: DATA PREPROCESSING")

    st.markdown(':octopus: Preprocess the dataset before continuing to `Classify`')
    if st.button('Preprocess Data'):
        try:
            st.markdown(':penguin: Removed missing data.')
            ### View data with missing values or invalid inputs
            top_n=3 # Used for top categories with missing data
            missing_column_counts = df[df.columns[df.isnull().any()]].isnull().sum()
            max_idxs = np.argsort(missing_column_counts.to_numpy())[::-1][:top_n]

            # Compute missing statistics
            num_categories = df.isna().any(axis=0).sum()
            average_per_category = df.isna().sum().sum()/len(df.columns)
            total_missing_values = df.isna().sum().sum()
            top_missing_categories = df.columns[max_idxs[:top_n]].to_numpy()
                
            st.markdown('Number of categories with missing values: {0:.2f}'.format(num_categories))
            st.markdown('Average number of missing values per category: {0:.2f}'.format(average_per_category))
            st.markdown('Total number of missing values: {0:.2f}'.format(total_missing_values))
            st.markdown('Top {} categories with most missing values: {}'.format(top_n, top_missing_categories))
            st.markdown(df.columns[df.isnull().any()].tolist())

            # Drop rows with missing values
            df = df.dropna() 
            st.markdown(f'Dataframe shape after dropping rows with missing values: {df.shape}')

            st.markdown(':penguin: Removed irrelevant or highly correlated features and features with' \
            ' experimental values used to generate target binary labels.')
            df = df.drop_duplicates(subset=['epitope', 'rmsd_complex_Ca'], keep='last')
            highly_correlated_features = ['iptm', 'chainA_iptm', 'chainC_iptm', 'chainD_iptm', 'chainE_iptm', \
                            'A1_pLDDT', 'C_average_pLDDT', 'D_average_pLDDT', 'rmsd_peptide_position_1']
            df = remove_features(df, highly_correlated_features)
            experimental_features = ['logA', 'obs_logC', 'affinity']
            df = remove_features(df, experimental_features)
            st.markdown(f'Highly correlated features dropped: {highly_correlated_features}')
            st.markdown(f'Experimental features dropped: {experimental_features}')

            st.markdown(":penguin: Created new features for average RMSD and average pLDDT across " \
            "peptide recognition region.")

            # Average RMSD of peptide recognition region (positions 3-8)
            rmsd_recognition = ['rmsd_peptide_position_3', 'rmsd_peptide_position_4',
                'rmsd_peptide_position_5', 'rmsd_peptide_position_6',
                'rmsd_peptide_position_7', 'rmsd_peptide_position_8']
            df['rmsd_peptide_recognition'] = df[rmsd_recognition].mean(axis=1)
            # Average pLDDT of peptide recognition region (positions 3-8)
            pLDDT_recognition = ['A3_pLDDT', 'A4_pLDDT',
                'A5_pLDDT', 'A6_pLDDT', 'A7_pLDDT', 'A8_pLDDT']
            df['pLDDT_recognition'] = df[pLDDT_recognition].mean(axis=1)
            # Drop averaged features
            df = remove_features(df, rmsd_recognition)
            df = remove_features(df, pLDDT_recognition)
            st.markdown("`rmsd_peptide_recognition` and `pLDDT_recognition` created.")
        
            # Separate dataframe into numerical and categorical features
            df_numerical = df.select_dtypes(include=['number'])
            df_categorical = df.select_dtypes(include=['object'])

            st.markdown(":penguin: Standardized numerical features.")
            # Standardize numerical features
            df_numerical_std = pd.DataFrame()
            for feature in df_numerical:
                df_numerical_std[feature+'_std'] = (df_numerical[feature] - df_numerical[feature].mean()) / df_numerical[feature].std()
            
            st.markdown(":penguin: Encoded MHC presentation, our categorical target.")
            target_feature = 'presenter_category'
            enc = OrdinalEncoder()
            df_categorical_encoded = pd.DataFrame()
            df_categorical_encoded[[target_feature+'_target']] = enc.fit_transform(df_categorical[[target_feature]])

            df = pd.concat([df_numerical_std.reset_index(drop=True),df_categorical_encoded.reset_index(drop=True)], axis=1)
        
            # Show updated dataset
            st.markdown("#### :whale: FINAL PREPROCESSED DATA")
            st.markdown("Our preprocessed dataset (with descriptive stats!)")
            st.write(df)
            st.write(df.describe())

        except ValueError as err:
                st.write({str(err)})

    st.session_state['data'] = df # save dataframe
    

 
