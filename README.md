### **Movie Recommender System 🎬**

This project is a movie recommendation system built in Python that demonstrates two key machine learning approaches: **Content-Based Filtering** and **Collaborative Filtering** (User Recommendations). The app is hosted using Streamlit to provide an interactive user interface for exploring the models.

#### **Project Overview**

The goal of this project was to build a robust recommendation engine from scratch using the **MovieLens 100k dataset**. The project showcases a full data science workflow, from data cleaning and exploratory data analysis (EDA) to model building, evaluation, and deployment.

The two main models implemented are:

  * **Content-Based Filtering:** This model recommends movies based on the similarity of their features (genres, title) with vectors. It's great for new users or items, as it doesn't require prior interaction data.
  * **Collaborative Filtering:** This model recommends movies based on the preferences of similar users. It excels at discovering unexpected patterns in user behavior, leading to surprising recommendations.

-----

### **Key Features**

  * **Interactive UI:** A user-friendly interface built with Streamlit allows for real-time model interaction.
  * **Dual-Model Approach:** The app demonstrates and compares two distinct recommendation strategies.
  * **Model Evaluation:** Performance metrics (Root Mean Square Error, Mean Absolute Error) are used to evaluate the collaborative filtering model's prediction accuracy.

#### **Methodology**

1.  **Data Acquisition and Cleaning:** The MovieLens 100k dataset was loaded and preprocessed. Missing values and inconsistencies were filled, and the data was prepared for each model.
2.  **Exploratory Data Analysis (EDA):** Visualizations were created to understand the distribution of ratings, popular movies, and genre prevalence.
3.  **Model Building:**
      * **Content-Based:** A Term Frequency - Inverse Document Frequency vectorizer was used to convert movie content into numerical vectors using all the dimensions of the dataset. **Cosine similarity** was then calculated to find the most similar movies.
      * **Collaborative:** The **Singular Value Decomposition (SVD)** algorithm from the `surprise` library was trained on the user-item rating matrix in order to make training the models easier using more relative information.
4.  **Deployment:** The final application was containerized and hosted on the Streamlit Community Cloud for public access.

-----
<img width="3272" height="1836" alt="image" src="https://github.com/user-attachments/assets/1b6184a0-7c8e-4cc1-9279-a17e652af40c" />
----
Results and Inquiries:
<img width="708" height="468" alt="image" src="https://github.com/user-attachments/assets/67a0f2db-e60a-4aae-af99-bfc67f3b5b92" />
<img width="1019" height="545" alt="image" src="https://github.com/user-attachments/assets/a79a57ff-6ebe-41e9-8756-6b9fc6510493" />
<img width="1062" height="622" alt="image" src="https://github.com/user-attachments/assets/d6c2c446-1787-4c27-b0c7-57f920b55149" />


### **Future Work**

  * **Hybrid Model:** The next step would be to build a hybrid model that intelligently combines the strengths of both content-based and collaborative filtering to improve recommendation quality.
  * **User Profiles:** Implement a feature to allow users to create and save profiles, and use their full viewing history to generate more personalized recommendations.
  * **Scalability:** Adapt the codebase to handle much larger datasets ( using Spark or another distributed computing framework).


