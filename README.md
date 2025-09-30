
# Post Recommendation System
This repository contains the code for a post recommendation system that provides personalized content suggestions to users. The system uses a hybrid approach, combining content-based, collaborative filtering, and heuristic methods to generate the top 3 recommendations for each user.

# Approach
The recommendation system uses a multi-faceted approach to provide relevant recommendations:

1. Content-Based Filtering
Content Modeling: Post content (titles, body, and tags) is processed using TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical vectors. These vectors are then reduced in dimensionality and denoised using TruncatedSVD to create a dense embedding for each post.
User Profile Matching: User interests from their profiles are also transformed into a content vector using the same TF-IDF and SVD process. The similarity between a user's interest vector and a post's content embedding is calculated using cosine similarity.

2. Collaborative Filtering
An implicit user-item matrix is built from user engagement data. Different engagement types (e.g., likes, comments) are assigned different weights to reflect their importance.
SVD (Singular Value Decomposition) is applied to this matrix to discover latent factors for both users and posts. The dot product of a user's latent vector and a post's latent vector is used as the collaborative score.

3. Heuristics & Ensemble Scoring
Recency: Posts created more recently are given a boost using an exponential decay function.
Popularity: Posts with a higher total number of engagements are also given a boost.
Interest Overlap: A Jaccard similarity score is calculated between a user's specified interests and the tags on a post to measure direct topical overlap.
Final Score: All these components—content similarity, collaborative score, interest overlap, recency, and popularity—are combined into a single, final recommendation score using a linear combination with tunable weights (α, β, γ, δ, ε). Posts that a user has already engaged with are optionally deprioritized.

# Evaluation
The system's performance can be evaluated using standard information retrieval metrics.
Precision@3: The fraction of the top 3 recommended posts that the user actually engaged with.
nDCG@3 (normalized Discounted Cumulative Gain): A metric that measures the ranking quality, giving higher scores to relevant posts that appear higher in the recommendation list.
MAP (Mean Average Precision): The mean of the average precision scores for each user.
A time-aware validation split is used to simulate a real-world scenario, where the model is trained on past engagements and validated on future ones.

# Extensions
Several improvements can be made to this system:
Advanced Content Modeling: Replace TF-IDF with more powerful, pretrained sentence embeddings like SBERT or Universal Sentence Encoder for better semantic understanding.
Learning-to-Rank Models: Train a more sophisticated model like LightGBM Ranker or LambdaMART to learn the optimal combination of features (content, collaborative, and heuristic) instead of using a simple linear combination.
Robust Collaborative Filtering: Implement models like LightFM or Implicit ALS for more advanced collaborative filtering, especially for handling implicit feedback data.
Fairness and Diversity: Incorporate mechanisms to prevent filter bubbles and introduce serendipity by diversifying recommendations based on content type or topic.

# Setup and Usage
Prerequisites
* Python 3.11
* pandas
* numpy
* scikit-learn

# Data
The system requires three CSV files in a datasets folder:
* Users.csv
* Posts.csv
* Engagements.csv
