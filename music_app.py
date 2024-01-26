import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  GridSearchCV, cross_val_score

# Configuring the page
st.set_page_config(layout="wide")

custom_title_style = """
<style>
    @import url('https://fonts.googleapis.com/css?family=Audiowide&display=swap');
    .custom-title {
        font-family: 'Audiowide', sans serif;
        font-size: 75px;
    }
</style>
"""
st.markdown(custom_title_style, unsafe_allow_html=True)

# Sidebar menu
selected = option_menu(
    menu_title=None,
    options=["Visualization", "Models", "Improve"],
    icons=["search", "book", "gear"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "gray"},
    }
)

col1, col2 = st.columns(2)
with col1:
    st.image('la-musique.png')
with col2:
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("<h1 class='custom-title'>Can music improve health?</h1>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_excel('clean_v1_mxmh_survey_results (1).xlsx')

# Function to load encoded data
@st.cache_data
def load_data2():
    return pd.read_excel('encoded_df (1).xlsx')

# Function to calculate correlation matrix
@st.cache_data
def get_correlation_matrix(dt):
    dt.drop(columns=['Timestamp'], inplace=True)
    return dt.corr()

@st.cache_data
def feature_selection(dt):
    selected_columns = ['Age', 'Hours per day', 'Anxiety', 'Depression', 'Insomnia', 'OCD', 'Music effects'] + [col for col in dt if 'Fav genre_'in col]
    dt_for_manifold = dt[selected_columns]
    return dt_for_manifold
@st.cache_data
def data_standardization(dt):
    scaler = StandardScaler()
    normalized = scaler.fit_transform(dt)
    return normalized
@st.cache_data
def pca_fonction(dt):
    pca = PCA(n_components=2)
    result = pca.fit_transform(dt)
    return result
@st.cache_data
def tsne_fonction(dt):
    tsne = TSNE(n_components=2, random_state=42)
    result = tsne.fit_transform(dt)
    return result
@st.cache_data
def lle_fonction(dt):
    lle = LocallyLinearEmbedding(n_components=2, random_state=42)
    result = lle.fit_transform(dt)
    return result


@st.cache_data
def visualization_method(dt, pca_result, tsne_result, lle_result):
    method_results = [pca_result, tsne_result, lle_result]
    method_names = ['PCA', 't-SNE', 'LLE']
    health_measures = ['Anxiety', 'Depression', 'Insomnia', 'OCD', 'Music effects']
    normalized_measures = {}
    color_maps = ['plasma', 'inferno', 'magma', 'viridis', 'cividis']

    for measure in health_measures:
        normalized_measures[measure] = (dt[measure] - dt[measure].min()) / (dt[measure].max() - dt[measure].min())

    for measure, cmap in zip(health_measures, color_maps):
        fig, axes = plt.subplots(1, len(method_results), figsize=(24, 6))
        colors = plt.cm.get_cmap(cmap)(normalized_measures[measure])

        for ax, result, method_name in zip(axes, method_results, method_names):
            sc = ax.scatter(result[:, 0], result[:, 1], alpha=0.7, c=colors)
            ax.set_title(f'{method_name} Colored by {measure}')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            fig.colorbar(ScalarMappable(Normalize(0, 1), cmap=cmap), ax=ax, label=f'{measure} Level')

        plt.tight_layout()
        st.pyplot(fig)

@st.cache_data
def visualization3D(dt):
    pca = PCA(n_components=3)
    pca_result_3d = pca.fit_transform(normalized_data)
    tsne = TSNE(n_components=3, random_state=42)
    tsne_result_3d = tsne.fit_transform(normalized_data)
    lle = LocallyLinearEmbedding(n_components=3, random_state=42)
    lle_result_3d = lle.fit_transform(normalized_data)


    method_results = [pca_result_3d, tsne_result_3d, lle_result_3d]
    method_names = ['PCA', 't-SNE', 'LLE']

    health_measures = ['Anxiety', 'Depression', 'Insomnia', 'OCD','Music effects']
    normalized_measures = {}
    color_maps = ['plasma', 'inferno', 'magma', 'viridis', 'cividis']


    for measure in health_measures:
        normalized_measures[measure] = (dt[measure] - dt[measure].min()) / (dt[measure].max() - dt[measure].min())

    for measure, cmap in zip(health_measures, color_maps):
        colors = plt.cm.get_cmap(cmap)(normalized_measures[measure])

        for result, method_name in zip(method_results, method_names):
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(result[:, 0], result[:, 1], result[:, 2], alpha=0.7, c=colors)
            ax.set_title(f'3D {method_name} Colored by {measure}')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            fig.colorbar(ScalarMappable(Normalize(0, 1), cmap=cmap), ax=ax, label=f'{measure} Level')
            plt.tight_layout()
            st.pyplot(fig)

def effect(dt_for_manifold):
    music_effects_counts = dt_for_manifold['Music effects'].value_counts(normalize=True)

    music_effects_mean = dt_for_manifold['Music effects'].mean()
    music_effects_median = dt_for_manifold['Music effects'].median()

    music_effects_analysis = {
        'Counts': music_effects_counts.to_dict(),
        'Mean': music_effects_mean,
        'Median': music_effects_median
    }
    return music_effects_analysis



if selected == "Visualization":
    st.title(f"Step 1: {selected}")

    data = load_data()
    
    st.subheader("Display data:")
    if st.checkbox('Show raw data'):
        st.write(data.head())

    st.markdown("Encoding categorical variables is crucial for converting textual data into a numerical format, which is necessary for most statistical analyses and machine learning algorithms. It also enables relationships between categorical features and other numerical variables to be revealed and exploited.")

    st.markdown('In this case, two encoding methods were used:')

    st.markdown('One-Hot encoding for categorical variables (with no inherent order), transforming each category into a new binary column.')
    st.markdown('Ordinal encoding for ordinal variables (with an order or a limited number of categories), assigning a unique number to each category according to their order.')
    st.markdown(" Encoding categorical variables is crucial for converting textual data into a numerical format, which is necessary for most statistical analyses and machine learning algorithms. It also enables relationships between categorical features and other numerical variables to be revealed and exploited.")
    st.markdown('Encoding techniques can be found on the ***notebook***')
    data2 = load_data2()
    if st.checkbox('Show encoded data'):
        st.write(data2.head())

    st.subheader("1. Analyse the variables, by computing the correlations in order to detect the most important and explain your analysis")
    correlation_matrix = get_correlation_matrix(data2)
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix of Encoded Data")
    st.pyplot(fig)

    st.markdown('Some key observations:')

    st.markdown('There is a strong correlation between Frequency [Hip hop] and Frequency [Rap] (correlation of 0.785). This suggests a potential redundancy between these two variables, which makes sense as Hip hop and Rap genres are often linked.')

    st.markdown('Age with Anxiety (-0.202): A moderate negative correlation, suggesting that anxiety may decrease with age.')

    st.markdown('Frequency [Pop] with Anxiety (0.118): A slightly positive correlation, indicating a weak relationship between listening to pop music and anxiety.')

    st.markdown('Frequency [Video game music] with Anxiety (0.103): A slightly positive correlation, suggesting a weak relationship between listening to video game music and anxiety.')

    st.markdown('Anxiety and Depression are strongly correlated with each other (0.521), in line with trends observed in psychology.')

    st.markdown('These results provide useful pointers for the next steps, including feature selection and the dimensionality reduction methods to be applied. They also help us to understand how the different variables relate to each other and to mental health indicators.')
 

    st.subheader("2.Apply different manifold learning (projections) approaches on the dataset by visualising the results in 2D or 3D. If possible, compute the errors.")

    st.markdown('Thanks to the next step, we noticed what needed to be used in step, and kept it:')
    st.code('''selected_columns = ['Age', 'Hours per day', 'Anxiety', 'Depression', 'Insomnia', 'OCD', 'Music effects'] + [col for col in encoded_df if 'Fav genre_' in col]
    data_for_manifold = encoded_df[selected_columns]''')
    
    data_for_manifold = feature_selection(data2)
    normalized_data = data_standardization(data_for_manifold)
    pca_result = pca_fonction(normalized_data)
    tsne_result = tsne_fonction(normalized_data)
    lle_result = lle_fonction(normalized_data)

    st.markdown("Applying different manifold learning techniques then visualize :")

    vis_type = st.selectbox("***Choose a Visualization Type***", ["2D Visualizations", "3D Visualizations"])

    if vis_type == "2D Visualizations":
        # Code for 2D visualizations
        st.markdown("***2D Visualizations***")
    
        visualization_method(data2, pca_result, tsne_result, lle_result)

        st.markdown("Correlation between musical genres and mental health: There appear to be distinct patterns, especially in the t-SNE and LLE visualizations, indicating that certain structures in the data are indicative of correlations between musical tastes and mental health states.")
        st.markdown("Anxiety, Depression, Insomnia, OCD: The different states appear to manifest themselves differently among respondents, as shown by variations in the distribution of points in the component space. For example, the points in the insomnia graphs appear more clustered, perhaps suggesting a stronger correlation between musical genres and insomnia.")

        st.markdown("Effects of Music: The visualizations indicate that for some individuals, music has a marked beneficial effect on their condition (yellow dots), while for others there is little or no effect (blue dots). This may indicate that music affects people differently, which is important for the application of music therapy.")

        st.markdown("Conclusion: These visualized data suggest a relationship between individuals' musical preferences and their self-reported mental health. Different data visualization techniques reveal varying degrees of this correlation, with some conditions such as insomnia showing clearer groupings than others. The results could inform a more targeted application of music therapy, taking into account individuals' musical preferences to treat specific mental health conditions. It is also clear that music does not affect everyone in the same way, underlining the importance of a personalized approach in the application of music therapy.")
        
    if vis_type == "3D Visualizations":
        
        st.markdwn("***3D Visualizations***")
        visualization3D(data2)

        st.markdown('The increase in explained variance to around 22% indicates that the first three PCA principal components capture slightly more of the total data variance than the first two components. This remains relatively low, meaning that much of the information contained in the original data is not represented in these three components.')

        st.markdown('A reconstruction error of almost zero (very close to zero) suggests that the LLE was able to preserve the local structure of the data very precisely during the reduction to three dimensions.')

    st.subheader('Small study on the effect of music:')
    result = effect(data_for_manifold)
    st.write(result)

    st.markdown('Distribution of values:')
    st.markdown('74.54% of participants reported a score of 0, indicating that they perceive an improvement due to the music.')

    st.markdown('23.07% of participants reported a score of 1, indicating that they perceive no effect from the music.')

    st.markdown('2.39% of participants reported a score of 2, which could indicate a worsening of their condition due to the music or some other non-beneficial effect.')

    st.markdown('Central tendency: The mean score is 0.28, which is closer to 0 (improvement) than to 1 (no effect), indicating a general trend towards improvement. The median score is 0, confirming that half the participants reported an improvement.')

    st.markdown('These results suggest that a significant majority of study participants perceive a positive effect of music on their well-being. With the mean and median closer to 0, this reinforces the idea that music has a good effect overall, according to the data collected.')


@st.cache_data
def split_and_normalize_data(data,  dt_for_manifold, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        data,  dt_for_manifold['Music effects'], 
        test_size=test_size, 
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


@st.cache_data
def apply_manifold_learning(X_train, X_test):
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, random_state=42, learning_rate='auto', init='random')
    lle = LocallyLinearEmbedding(n_components=2, random_state=42)

    # Training transformations
    pca_train = pca.fit_transform(X_train)
    tsne_train = tsne.fit_transform(X_train)
    lle_train = lle.fit_transform(X_train)

    # Testing transformations
    pca_test = pca.transform(X_test)
    tsne_test = tsne.fit_transform(X_test) # Note: TSNE does not support transform on test data
    lle_test = lle.transform(X_test)

    return pca_train, pca_test, tsne_train, tsne_test, lle_train, lle_test


@st.cache_data
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=42)
    accuracies = {}

    for name, train_data, test_data in [
        ('PCA', pca_train, pca_test), 
        ('t-SNE', tsne_train, tsne_test), 
        ('LLE', lle_train, lle_test)
    ]:
        clf.fit(train_data, y_train)
        y_pred = clf.predict(test_data)
        accuracies[name] = accuracy_score(y_test, y_pred)

    return accuracies

@st.cache_data
def visu_pca(pca_result_test_2D, y_test):
  
    fig, ax = plt.subplots(figsize=(7, 4))
    
    colors = plt.cm.get_cmap('plasma')(y_test)

    sc = ax.scatter(pca_result_test_2D[:, 0], pca_result_test_2D[:, 1], alpha=0.7, c=colors)
    ax.set_title('PCA Results Colored by Music Effects')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Music Effects Level')


    plt.tight_layout()
    st.pyplot(fig)
@st.cache_data
def apply_3d_manifold_learning(X_train, X_test):
    pca = PCA(n_components=3)
    tsne = TSNE(n_components=3, random_state=42, learning_rate='auto', init='random')
    lle = LocallyLinearEmbedding(n_components=3, random_state=42)

    pca_train_3D = pca.fit_transform(X_train)
    tsne_train_3D = tsne.fit_transform(X_train)
    lle_train_3D = lle.fit_transform(X_train)

    pca_test_3D = pca.transform(X_test)
    tsne_test_3D = tsne.fit_transform(X_test) 
    lle_test_3D = lle.transform(X_test)

    return pca_train_3D, pca_test_3D, tsne_train_3D, tsne_test_3D, lle_train_3D, lle_test_3D
@st.cache_data
def train_and_evaluate_model_3d(pca_train_3D, pca_test_3D, tsne_train_3D, tsne_test_3D, lle_train_3D, lle_test_3D, y_train, y_test):
    clf = RandomForestClassifier(random_state=42)
    accuracies = {}

    for name, train_data, test_data in [
        ('PCA', pca_train_3D, pca_test_3D), 
        ('t-SNE', tsne_train_3D, tsne_test_3D), 
        ('LLE', lle_train_3D, lle_test_3D)
    ]:
        clf.fit(train_data, y_train)
        y_pred = clf.predict(test_data)
        accuracies[name] = accuracy_score(y_test, y_pred)

    return accuracies
@st.cache_data
def visu3D_pca(pca_result_test_3D, y_test):

    colors = plt.cm.get_cmap('plasma')(y_test)

    fig, ax = plt.subplots(figsize=(7, 4))
    sc = plt.scatter(pca_result_test_3D[:, 0], pca_result_test_3D[:, 1], alpha=0.7, c=colors)
    plt.title('PCA Results Colored by Music Effects')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')


    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Music Effects Level')


    plt.tight_layout()
    st.pyplot(fig)
@st.cache_data
def prepare_data(dt):

    X = dt.drop(['Anxiety', 'Depression', 'Insomnia', 'OCD'], axis=1)
    Y = dt[['Anxiety', 'Depression', 'Insomnia', 'OCD']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@st.cache_data
def apply_manifold_learning_mental(X_train, X_test, n_components=2):
    pca = PCA(n_components=n_components)
    lle = LocallyLinearEmbedding(n_components=n_components, random_state=42)
    tsne = TSNE(n_components=n_components, random_state=42)

    X_train_pca = pca.fit_transform(X_train)
    X_train_lle = lle.fit_transform(X_train)
    X_train_tsne = tsne.fit_transform(X_train)

    X_test_pca = pca.transform(X_test)
    X_test_lle = lle.transform(X_test)
    X_test_tsne = tsne.fit_transform(X_test)

    return X_train_pca, X_test_pca, X_train_lle, X_test_lle, X_train_tsne, X_test_tsne

@st.cache_data
def train_and_evaluat_mental(X_train_pca, X_test_pca, X_train_lle, X_test_lle, X_train_tsne, X_test_tsne, y_train, y_test):
    model_pca = MultiOutputRegressor(Ridge()).fit(X_train_pca, y_train)
    model_lle = MultiOutputRegressor(Ridge()).fit(X_train_lle, y_train)
    model_tsne = MultiOutputRegressor(Ridge()).fit(X_train_tsne, y_train)

    mse_results = {}

    for name, model, X_test in [('PCA', model_pca, X_test_pca), ('LLE', model_lle, X_test_lle), ('t-SNE', model_tsne, X_test_tsne)]:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
        mse_results[name] = mse

    return mse_results

@st.cache_data
def apply_3d_manifold_learning_mental(X_train, X_test, random_state=42):
    pca = PCA(n_components=3)
    lle = LocallyLinearEmbedding(n_components=3, random_state=random_state)
    tsne = TSNE(n_components=3, random_state=random_state)

    X_train_pca_3D = pca.fit_transform(X_train)
    X_test_pca_3D = pca.transform(X_test)
    X_train_lle_3D = lle.fit_transform(X_train)
    X_test_lle_3D = lle.transform(X_test)
    X_train_tsne_3D = tsne.fit_transform(X_train)
    X_test_tsne_3D = tsne.fit_transform(X_test) 

    return X_train_pca_3D, X_test_pca_3D, X_train_lle_3D, X_test_lle_3D, X_train_tsne_3D, X_test_tsne_3D

@st.cache_data
def train_and_evaluate_3d_mental(X_train_pca, X_test_pca, X_train_lle, X_test_lle, X_train_tsne, X_test_tsne, y_train, y_test):
    model_pca = MultiOutputRegressor(Ridge()).fit(X_train_pca, y_train)
    model_lle = MultiOutputRegressor(Ridge()).fit(X_train_lle, y_train)
    model_tsne = MultiOutputRegressor(Ridge()).fit(X_train_tsne, y_train)

    mse_results = {}

    for name, model, X_test in [('PCA', model_pca, X_test_pca), ('LLE', model_lle, X_test_lle), ('t-SNE', model_tsne, X_test_tsne)]:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
        mse_results[name] = mse

    return mse_results

if selected == "Models":
    st.title(f"Step 2: {selected}")
    st.subheader('3-4. Apply an external validation process (classification and/or regression) to choose the best method. / Explain the model and why it works best for this problem/dataset. Use Python with matplotlib and/or KNIME to plot the dataset and the knowledge extracted from it. Explain the knowledge you extracted.')
    
    data = load_data()
    data2 = load_data2()
    
    data_for_manifold = feature_selection(data2)
    normalized_data = data_standardization(data_for_manifold)

    study_type = st.selectbox("***Choose a study case***", ["Predicts the effect of music on individuals", "Predicts mental health"])
    
    if study_type == "Predicts the effect of music on individuals":
        st.markdown("In this context, the classification model predicts the effect of music on individuals, in particular the variable 'Music effects'.")
        X_train, X_test, y_train, y_test = split_and_normalize_data(normalized_data, data_for_manifold)
        pca_train, pca_test, tsne_train, tsne_test, lle_train, lle_test = apply_manifold_learning(X_train, X_test)
        accuracies = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        st.write(accuracies)

        st.markdown('We obtain better results when using the PCA with the random forest. So we wanted to see the distribution of the data ')
        visu_pca(pca_test, y_test)

        st.markdown('***Dimensionality 3D reduction methods***')
        X_train, X_test, y_train, y_test = split_and_normalize_data(normalized_data,  data_for_manifold)
        pca_train_3D, pca_test_3D, tsne_train_3D, tsne_test_3D, lle_train_3D, lle_test_3D = apply_3d_manifold_learning(X_train, X_test)
        accuracies_3d = train_and_evaluate_model_3d(pca_train_3D, pca_test_3D, tsne_train_3D, tsne_test_3D, lle_train_3D, lle_test_3D, y_train, y_test)

        st.write(accuracies_3d)
        st.markdown('Here we see that the best results are obtained with the 3d dimensiality reduction using the PCA method, with a score of 74%. So, when classifying the effects of music on mental health, the model correctly predicted the effect 74% of the time in the test set.')

        visu3D_pca(pca_test_3D, y_test)

    if study_type == "Predicts mental health":
        st.markdown("In this context, the classification model predicts mental health.")
        st.markdown('Here we had to use a model with several outputs')
        
        st.markdown('***Dimensionality 2D reduction methods***')

        data_mental= data2.drop(columns=['Timestamp'])
        X_train, X_test, y_train, y_test = prepare_data(data_mental)
        X_train_pca, X_test_pca, X_train_lle, X_test_lle, X_train_tsne, X_test_tsne = apply_manifold_learning_mental(X_train, X_test)
        
        mse_results = train_and_evaluat_mental(X_train_pca, X_test_pca, X_train_lle, X_test_lle, X_train_tsne, X_test_tsne, y_train, y_test)

        st.write(mse_results)

        st.markdown("The PCA, LLE, T-SNE models performed comparably, but the PCA performed slightly better than the LLE and T-SNE in terms of MSE in this specific dataset.")

        st.markdown('***Dimensionality 3D reduction methods***')
        X_train_pca_3D, X_test_pca_3D, X_train_lle_3D, X_test_lle_3D, X_train_tsne_3D, X_test_tsne_3D = apply_3d_manifold_learning_mental(X_train, X_test)
        mse_results_3d = train_and_evaluate_3d_mental(X_train_pca_3D, X_test_pca_3D, X_train_lle_3D, X_test_lle_3D, X_train_tsne_3D, X_test_tsne_3D, y_train, y_test)

        st.write(mse_results_3d)

        st.markdown('The PCA and LLE models performed comparably, but the PCA performed slightly better than the LLE in terms of MSE in this specific dataset.')
        st.markdown('And the t-SNE gives very poor results')

        st.markdown('***So to conclude, the best model that gives the best result is with PCA dimensionality reduction***')
        st.markdown("In the end, we noticed that PCA gave us the best results with the random forest model. This made us a little confused at first, because in view of previous analyses, we noticed that PCA didn't necessarily give the best distribution/representation of points in clusters, while tsne and lle gave a better distribution, which implied (perhaps) better results with the latter two. So we tried to find the reasons for this in order to understand things in greater depth:")

        st.markdown("Good visual separation doesn't always guarantee good classification performance, and vice versa. PCA tends to preserve the overall variance of the data, which can help maximize the separation between classes for classification, even if this doesn't translate into clear separation in the visualization.")
        st.markdown("PCA is a linear technique that may work best when class separation is linearly achievable in feature space. Whereas t-SNE and LLE are non-linear techniques that seek to preserve local structures in the data. However, these techniques can sometimes exaggerate clusters or discontinuities in the data, which is not always ideal for classification.")
        st.markdown("Regressions are used to predict continuous values, while classification is used to predict discrete categories. The variance-maximizing principal components of PCA may be better suited to capturing trends in continuous data than t-SNE or LLE, which are often used to highlight data structures in a classification context.")
        st.markdown("Regressions can benefit from noise reduction and meaningful feature extraction, which are strengths of PCA. This may explain why PCA works better for MultiOutputRegressor in your case.")
        st.markdown("Regression models can sometimes be more sensitive to data complexity. PCA reduces this complexity by retaining only those components that explain the majority of the variance, which can simplify the problem for the regression model.")

@st.cache_data
def perform_pca_analysis(X_train, variance_threshold=0.95):
    pca_full = PCA()
    pca_full.fit(X_train)

    explained_variance_ratio_cumsum = np.cumsum(pca_full.explained_variance_ratio_)

    plt.figure(figsize=(10, 5))
    plt.plot(explained_variance_ratio_cumsum, marker='o', linestyle='--')
    plt.title('Variance Expliquée Cumulative par Composantes PCA')
    plt.xlabel('Nombre de Composantes')
    plt.ylabel('Variance Cumulée Expliquée')

    components_required = np.argmax(explained_variance_ratio_cumsum >= variance_threshold) + 1
    plt.axvline(components_required, color='r', linestyle='--', label=f'{components_required} composantes pour {variance_threshold*100}% variance')
    plt.legend()
    
    st.pyplot(plt)

    return components_required

@st.cache_data
def tune_and_train_model(X_train, y_train, n_components=20):
    pca = PCA(n_components=n_components)
    pca_result_train = pca.fit_transform(X_train)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(pca_result_train, y_train)

    return pca, grid_search.best_estimator_, grid_search.best_params_

@st.cache(hash_funcs={PCA: id})
def evaluate_model(pca, best_clf, X_test, y_test):
    pca_result_test = pca.transform(X_test)
    accuracy = cross_val_score(best_clf, pca_result_test, y_test, cv=5, scoring='accuracy').mean()
    return accuracy

if selected == "Improve":
    st.title(f"Step 3: {selected}")
    st.subheader('5.Analyse how the projection results can be improved and propose some solutions related to the dataset you analyse.')
    data2 = load_data2()

    data_for_manifold = feature_selection(data2)
    normalized_data = data_standardization(data_for_manifold)
    X_train, X_test, y_train, y_test = train_test_split(normalized_data, data_for_manifold['Music effects'], test_size=0.2, random_state=42)

    components_required = perform_pca_analysis(X_train)
    st.write(f'Nombre de composantes nécessaires pour atteindre 95% de la variance expliquée : {components_required}')

    pca, best_clf, best_params = tune_and_train_model(X_train, y_train, components_required)
    st.write(f'Best Random Forest parameters: {best_params}')

    accuracy = evaluate_model(pca, best_clf, X_test, y_test)
    st.write(f'Cross-validated accuracy: {accuracy}')

    st.markdown('In addition, GridSearchCV is used for an exhaustive search of the best hyperparameters for the Random Forest classifier, and cross_val_score for cross-validation to reliably assess model accuracy. We end up with a score of 0.90%.')
