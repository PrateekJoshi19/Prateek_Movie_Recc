# Import necessary libraries
import pandas as pd  # Library for data manipulation and analysis
import numpy as np  # Library for numerical operations


# Scikit-learn libraries for preprocessing, model building, and evaluation
from sklearn.preprocessing import StandardScaler  # Standardize features by removing the mean and scaling to unit variance
from sklearn.ensemble import RandomForestRegressor  # Random forest regressor for building the recommendation model
from sklearn.model_selection import train_test_split  # Function for splitting the dataset into training and testing sets
from sklearn.metrics import mean_squared_error, r2_score  # Functions to evaluate the performance of the model


import pickle  # Module for serializing and deserializing Python objects
import uuid  # Module for generating unique user IDs
import os  # Module for interacting with the operating system (e.g., file paths)
import ast  # Module for safely evaluating strings containing Python expressions


# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')  # This line will suppress warning messages to keep the output clean


# Load the movies dataset
movies_df = pd.read_csv('movies.csv')  # Read the CSV file containing movie data into a DataFrame


# First Layer: User Segmentation Layer
class UserSegmentLayer:
    def __init__(self):
        # Define questions to gather user preferences
        self.questions = [
            "How often do you watch movies? (1-5): ",
            "Rate your interest in action movies (1-5): ",
            "Rate your interest in comedy movies (1-5): ",
            "Rate your interest in drama movies (1-5): ",
            "Rate your interest in sci-fi movies (1-5): ",
            "Rate your interest in horror movies (1-5): ",
            "How important is the movie's budget to you? (1-5): ",
            "How much do you care about movie ratings? (1-5): ",
            "Do you prefer movies in English? (1-5): ",
            "How important is the movie's cast to you? (1-5): ",
            "Do you prefer recent movies or classics? (1-5, 1 for classics, 5 for recent): ",
            "How much do you care about movie duration? (1-5): ",
            "Rate your interest in foreign language films (1-5): ",
            "How important are special effects to you? (1-5): ",
            "Do you prefer watching movies in theaters or at home? (1-5, 1 for home, 5 for theaters): "
        ]
        self.scaler = StandardScaler()  # Initialize a scaler for future use
    

    # Function to get user answers to the questions
    def get_user_answers(self):
        answers = []  # Initialize an empty list to store user answers
        for question in self.questions:
            while True:
                try:
                    # Prompt the user for an answer to the current question
                    answer = int(input(f"{question} "))
                    # Check if the answer is within the valid range of 1 to 5
                    if 1 <= answer <= 5:
                        answers.append(answer)  # Append the valid answer to the list
                        break  # Exit the loop and move to the next question
                    else:
                        print("Please enter a number between 1 and 5.")  # Inform the user of the valid range
                except ValueError:
                    print("Please enter a valid number.")  # Inform the user that the input must be a number
        return answers  # Return the list of user answers
    

    # Function to segment user based on their answers
    def segment_user(self, answers):
        # Simple segmentation based on average score
        avg_score = sum(answers) / len(answers)
        if avg_score < 2:
            return 0  # Low preference user
        elif avg_score < 3:
            return 1  # Moderate preference user
        elif avg_score < 4:
            return 2  # High preference user
        else:
            return 3  # Very high preference user
    

    # Function to generate a unique user ID
    def generate_user_id(self):
         # Generate a new unique user ID using uuid4
         # uuid4() generates a random UUID (Universally Unique Identifier)
         # str() converts the UUID object to a string
        return str(uuid.uuid4())
    

    # Function to store user data
    def store_user_data(self, user_id, answers, segment):
        # Create a dictionary to store user data including user ID, answers to the questionnaire, and user segment
        user_data = {
            'user_id': user_id,  # Unique identifier for the user
            'answers': answers,  # User's answers to the questionnaire
            'segment': int(segment)  # User's segment based on their answers
        }
        # Create the 'user_data' directory if it doesn't already exist
        os.makedirs('user_data', exist_ok=True)
        # Save the user data to a file named with the user ID in the 'user_data' directory
        with open(f'user_data/{user_id}.pkl', 'wb') as f:
            pickle.dump(user_data, f)  # Use pickle to serialize the user data and write it to the file


    # Main function to process user, get their answers, segment them, and store their data
    def process_user(self):
        answers = self.get_user_answers()  # Get the user's answers to the questionnaire
        segment = self.segment_user(answers)  # Determine the user's segment based on their answers
        user_id = self.generate_user_id()  # Generate a unique user ID
        self.store_user_data(user_id, answers, segment)  # Store the user's data (ID, answers, segment)
        return user_id  # Return the generated user ID


# Second Layer: Recommendation Layer
class RecommendationLayer:
    def __init__(self, movies_df):
        self.movies_df = movies_df  # Load the movies DataFrame
        self.prepare_data()  # Prepare the data for training and recommendation
        self.model = RandomForestRegressor()  # Initialize the RandomForest model
        self.train_model()  # Train the model


    # Function to prepare data for training
    def prepare_data(self):
        def parse_genres(genre_str):
            try:
                genres = ast.literal_eval(genre_str)  # Safely evaluate the string to a Python object
                if isinstance(genres, list) and all(isinstance(g, dict) for g in genres):
                    return [g['name'] for g in genres]  # Extract genre names if in dict format
                elif isinstance(genres, list) and all(isinstance(g, str) for g in genres):
                    return genres  # Return list if already in str format
                elif isinstance(genres, str):
                    return [genres]  # Handle single genre as string
                else:
                    return []  # Return empty list if genres format is not recognized
            except:
                return []


        # Parse genres column and fill missing values with empty list
        self.movies_df['genres'] = self.movies_df['genres'].fillna('[]').apply(parse_genres)


        # Define a list of genres to create one-hot encoded features
        self.genre_list = ['Action', 'Comedy', 'Drama', 'Science Fiction', 'Horror']
        for genre in self.genre_list:
            # Create a binary column for each genre
            self.movies_df[f'genre_{genre}'] = self.movies_df['genres'].apply(lambda x: 1 if genre in x else 0)
        

        # Extract the release year from the release date
        self.movies_df['release_year'] = pd.to_datetime(self.movies_df['release_date'], errors='coerce').dt.year
        

        # Create a binary column to indicate if the movie is in English
        self.movies_df['is_english'] = (self.movies_df['original_language'] == 'en').astype(int)
        

        # List of numerical features to be normalized
        numerical_features = ['budget', 'popularity', 'release_year', 'runtime', 'vote_average', 'vote_count']
        for feature in numerical_features:
            # Normalize each numerical feature to the range [0, 1]
            self.movies_df[feature] = (self.movies_df[feature] - self.movies_df[feature].min()) / (self.movies_df[feature].max() - self.movies_df[feature].min())
        

        # Select features for the model
        self.features = [col for col in self.movies_df.columns if col.startswith('genre_')] + ['budget', 'popularity', 'release_year', 'runtime', 'vote_average', 'vote_count', 'is_english']
        

        # Prepare the feature matrix (X) and target vector (y)
        self.X = self.movies_df[self.features].fillna(0)  # Fill any missing values with 0
        self.y = self.movies_df['vote_average']  # Target is the average vote score


    # Function to train the model
    def train_model(self):
        # Splitting the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        self.model.fit(self.X_train, self.y_train) # Train the model on the training data
        

        # Calculate and print training accuracy
        y_train_pred = self.model.predict(self.X_train)  # Predict the target values for the training set
        train_mse = mean_squared_error(self.y_train, y_train_pred) # Calculate Mean Squared Error (MSE) for the training set predictions
        train_r2 = r2_score(self.y_train, y_train_pred)  # Calculate R^2 score for the training set predictions
        print()
        print(f"Training MSE: {train_mse:.4f}")  # Print the training Mean Squared Error
        print(f"Training R^2 score: {train_r2:.4f}")  # Print the training R^2 score
        

        # Calculate and print testing accuracy
        y_test_pred = self.model.predict(self.X_test)  # Predict the target values for the testing set
        test_mse = mean_squared_error(self.y_test, y_test_pred)  # Calculate Mean Squared Error (MSE) for the testing set predictions
        test_r2 = r2_score(self.y_test, y_test_pred)  # Calculate R^2 score for the testing set predictions
        print()
        print(f"Testing MSE: {test_mse:.4f}")  # Print the testing Mean Squared Error
        print(f"Testing R^2 score: {test_r2:.4f}")  # Print the testing R^2 score
        print()


    # Function to get user segment and answers based on user ID
    def get_user_segment(self, user_id):
        # Load user data from the saved file
        with open(f'user_data/{user_id}.pkl', 'rb') as f:
            user_data = pickle.load(f)  # Load the serialized user data from the file
        return user_data['segment'], user_data['answers']  # Return the user's segment and answers


    # Function to recommend movies based on user preferences
    def recommend_movies(self, user_id, top_n=10):
        # Retrieve user segment and answers
        _, user_answers = self.get_user_segment(user_id)
        

        # Construct user profile based on answers
        user_profile = {
            'watch_frequency': user_answers[0],       # How often the user watches movies
            'genre_preferences': user_answers[1:6],   # User's interest in different genres (Action, Comedy, Drama, Sci-Fi, Horror)
            'budget_importance': user_answers[6],     # Importance of the movie's budget to the user
            'rating_importance': user_answers[7],     # Importance of movie ratings to the user
            'english_preference': user_answers[8],    # Preference for movies in English
            'cast_importance': user_answers[9],       # Importance of the movie's cast to the user
            'recency_preference': user_answers[10],   # Preference for recent movies vs. classics
            'duration_importance': user_answers[11],  # Importance of movie duration to the user
            'foreign_preference': user_answers[12],   # Interest in foreign language films
            'effects_importance': user_answers[13],   # Importance of special effects to the user
            'theater_preference': user_answers[14]    # Preference for watching movies in theaters vs. at home
        }
        

        self.movies_df['user_score'] = 0  # Initialize user score column
        

        # Calculate user score based on genre preferences
        for i, genre in enumerate(self.genre_list):
            self.movies_df['user_score'] += self.movies_df[f'genre_{genre}'] * user_profile['genre_preferences'][i]
        

        # Calculate user score based on other preferences
        # Add budget preference
        self.movies_df['user_score'] += self.movies_df['budget'] * user_profile['budget_importance']
        # Add rating importance
        self.movies_df['user_score'] += self.movies_df['vote_average'] * user_profile['rating_importance']
        # Add preference for English language movies
        self.movies_df['user_score'] += self.movies_df['is_english'] * user_profile['english_preference']
        # Add importance of movie cast
        self.movies_df['user_score'] += self.movies_df['popularity'] * user_profile['cast_importance']
        # Add preference for recent movies
        self.movies_df['user_score'] += self.movies_df['release_year'] * user_profile['recency_preference']
        # Add importance of movie duration
        self.movies_df['user_score'] += self.movies_df['runtime'] * user_profile['duration_importance']
        # Add preference for foreign language films
        self.movies_df['user_score'] += (1 - self.movies_df['is_english']) * user_profile['foreign_preference']
        

        # Normalize the user score to the range [0, 1]
        self.movies_df['user_score'] = (self.movies_df['user_score'] - self.movies_df['user_score'].min()) / (self.movies_df['user_score'].max() - self.movies_df['user_score'].min())
        

        # Combine user score and movie rating to get the final score
        self.movies_df['final_score'] = 0.7 * self.movies_df['user_score'] + 0.3 * self.movies_df['vote_average']
        

        # Sort movies by final score and select top N recommendations
        recommended_movies = self.movies_df.sort_values('final_score', ascending=False).head(top_n)
        return recommended_movies[['title', 'final_score', 'genres', 'vote_average']]


# Main execution
if __name__ == "__main__":
    # Print the first few rows of the dataset to get an overview
    print()
    print("First few rows of the dataset:")
    print()
    print(movies_df.head())


    # Print the last few rows of the dataset to check for any anomalies
    print()
    print("Last rows of the dataset:")
    print()
    print(movies_df.tail())

    
    # Print the columns of the dataset to understand its structure
    print()
    print("\nColumns in the dataset:")
    print()
    print(movies_df.columns)


    # Instantiate user segmentation and recommendation layers
    user_segment_layer = UserSegmentLayer()  # Create an instance of the UserSegmentLayer class
    recommendation_layer = RecommendationLayer(movies_df)  # Create an instance of the RecommendationLayer class with the movies dataset


    while True:
        # Process user, get user ID and provide recommendations
        user_id = user_segment_layer.process_user()  # Process the user to get their answers, segment them, and generate a user ID
        print()
        print(f"User ID: {user_id}")  # Print the generated user ID


        # Get the top 10 movie recommendations for the user
        recommendations = recommendation_layer.recommend_movies(user_id)
        print()
        print("\nTop 10 Recommended Movies:")
        print()
        print(recommendations.to_string(index=True))  # Print the recommended movies without the index
        print()


        # Ask if another user wants to use the system
        add_another = input("\nDo you want to add another user? (yes/no): ").lower()
        if add_another != 'yes':  # Exit the loop if the answer is not 'yes'
            break


    # Print a thank you message after exiting the loop
    print()
    print("Thank you for using the movie recommendation system!")



'''
***END OF CODE***
'''


'''''''''
Code By:
(05613702022): Prateek Joshi
(01713702022): Aqsa Siddiqui
(03113702022): Riya Sharma
(03813702022): Yash Wadhwa 
'''''''''


'''
Submitted To: 
Vaseem Durrani
'''
 

'''
***ThankYou***
'''