import numpy as np
from api.user_account_api import UserAccountAPI

# You need to add your Twitter API Credentials
CONSUMER_KEY: str = ""
CONSUMER_SECRET: str = ""
ACCESS_TOKEN: str = ""
ACCESS_TOKEN_SECRET: str = ""

# 1. Set up the API object
api: UserAccountAPI = UserAccountAPI(
    consumer_key=CONSUMER_KEY,
    consumer_secret=CONSUMER_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET)

# 2. Get Feature Input vectors using a Transformer model
print("get features")
# You retrieve the user account and generate the input vector
# either by screen name or by the user id.

# Let's analyse a Bot account

screen_name: str = "@elonmusk"
user_input_feature_vec1: np.ndarray = api.get_input_feature_vector_by_screen_name(
    screen_name=screen_name)
#
# user_id: str = "1183842404"  #
# user_input_feature_vec2: np.ndarray = api.get_input_feature_vector_by_id(
#     user_id=user_id)

# ---------------------------------------------------
# 3. Predict credibility

credibility1: float = api.get_user_account_credibility(
    input_user_embedding=user_input_feature_vec1)
print("credibility1: ", credibility1)
#
# credibility2: float = api.get_user_account_credibility(
#     input_user_embedding=user_input_feature_vec2)
# print(credibility2)

# ---------------------------------------------------
# 4. Get embedding
user_embedding1: np.ndarray = api.get_user_embedding(
    input_user_embedding=user_input_feature_vec1)
print(user_embedding1.shape)

# user_embedding2: np.ndarray = api.get_user_embedding(
#     input_user_embedding=user_input_feature_vec2)
# print(user_embedding2.shape)

# ---------------------------------------------------
# 5. Additional analysis: Similarity between a pair of embeddings
# similarity: float = api.embedding_cosine_similarity(
#     x=user_embedding1,
#     y=user_embedding2)
# print(similarity)