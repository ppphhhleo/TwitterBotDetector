import numpy as np
from tensorflow.keras.models import Model
from helper.settings import logger, model_directory, model_name, embedding_layer
from analysis.user_account_analysis import UserAccountAnalysis
from tweepy.models import User
from connectors.dnn_model_connector import ModelConnector


class UserAccountAPI:
    def __init__(self, consumer_secret: str, consumer_key: str,
                 access_token: str, access_token_secret: str):  # 连接Twitter的API
        self.user_account_analysis: UserAccountAnalysis = UserAccountAnalysis(
            consumer_secret=consumer_secret,
            consumer_key=consumer_key,
            access_token=access_token,
            access_token_secret=access_token_secret)
        self.model_connector: ModelConnector = ModelConnector()
        self.model_directory: str = model_directory
        self.model_name: str = model_name
        self.model: Model = object.__new__(Model)
        self.model_status: int = 0

    def set_up_model(self):
        if self.model_status == 0:
            self.model: Model = self.model_connector.load_trained_model(
                model_directory=self.model_directory,
                model_name=self.model_name)
            self.model_status: int = 1

    def get_input_feature_vector_by_screen_name(self, screen_name: str) -> np.ndarray:
        user_input_feature_vec: np.ndarray = np.array([])
        try:
            # 1. Preprocess
            screen_name: str = screen_name.replace("@", "")

            # 2. Retrieve data
            users_data: list = self.user_account_analysis.get_users_data_by_screen_name(
                screen_names=[screen_name])
            # print(users_data[0][0]) got the profiles
            logger.info("successfully get the screen name, begin to generate embedding")
            # 3. Generate embedding
            user: User = users_data[0][0]
            # user_features: dict = self.user_account_analysis.extract_features_from_user_account(user_data = user)
            # tw = self.user_account_analysis.my_get_last_k_tweets_from_user_account(screen_name=str(screen_name), k=10)
            logger.info("users_data[0][0]: {}, users_data[0][1]".format(users_data[0][0]))
            # usert = open("./userdata.txt", "w", encoding="utf-8"):

            user_input_feature_vec: np.ndarray = self.user_account_analysis.generate_user_feature_vector(
                user=user)
        except Exception as e:
            logger.error(e)
        return user_input_feature_vec

    def my_get_input_feature_vector_by_screen_name(self, screen_name: str) -> np.ndarray:
        user_input_feature_vec: np.ndarray = np.array([])

        # 1. Preprocess
        screen_name: str = screen_name.replace("@", "")
        # 2. Retrieve data
        users_data: list = self.user_account_analysis.get_users_data_by_screen_name(
            screen_names=[screen_name])
        # print(users_data[0][0]) got the profiles
        logger.info("successfully get the screen name, begin to generate embedding")
            # 3. Generate embedding
        user: User = users_data[0][0]
        user_features: dict = self.user_account_analysis.extract_features_from_user_accountA(
            user_data=user)
        user_input_feature_vec: np.ndarray = self.user_account_analysis.generate_user_feature_vector(
            user=user)
        return user_features, user_input_feature_vec

    def get_input_feature_vector_by_id(self, user_id: str) -> np.ndarray:
        user_input_feature_vec: np.ndarray = np.array([])
        try:
            # 1. Retrieve data
            users_data: list = self.user_account_analysis.get_users_data_by_id(
                user_ids=[user_id])

            # 2. Generate embedding
            user: User = users_data[0][0]
            user_input_feature_vec: np.ndarray = self.user_account_analysis.generate_user_feature_vector(
                user=user)
        except Exception as e:
            logger.error(e)
        return user_input_feature_vec

    def get_user_account_credibility(self, input_user_embedding: np.ndarray) -> float:
        credibility: float = 0.0
        try:
            # 1. Load model
            self.set_up_model()

            # 2. Obtain credibility
            logger.info("load model successfully, then begin to calculate")

            credibility: float = round(1 - float(
                self.model.predict(input_user_embedding)), 3)

        except Exception as e:
            logger.error(e)
        return credibility

    def get_user_embedding(self, input_user_embedding: np.ndarray) -> np.ndarray:
        user_embedding: np.ndarray = np.empty(shape=(1, ))
        try:
            # 1. Load model
            self.set_up_model()

            # 2. Obtain Low-dimensional embedding
            intermediate_layer_model: Model = Model(
                inputs=self.model.input,
                outputs=self.model.get_layer(name=embedding_layer).output)
            user_embedding: np.ndarray = intermediate_layer_model.predict(
                input_user_embedding)
        except Exception as e:
            logger.error(e)
        return user_embedding

    def embedding_cosine_similarity(self, x: np.array, y: np.array) -> float:
        similarity: float = 0.0
        try:
            similarity: float = self.user_account_analysis.embedding_cosine_similarity(
                x=x, y=y)
        except Exception as e:
            logger.error(e)
        return similarity