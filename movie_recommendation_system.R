############################################################
# HarvardX: PH125.8x - Data Science: Machine Learning
# Topic: Recommendation Systems - Model Fitting
#
# This script explores different models to predict movie 
# ratings using the MovieLens dataset, including:
#   - Baseline average model
#   - Movie effects model
#   - Movie + user effects model
#
# Key steps:
#   1. Data loading and exploration
#   2. Data partitioning into training and test sets
#   3. Model building and evaluation using RMSE
#
# References:
#   - Netflix Prize background and winning solutions
#   - https://bits.blogs.nytimes.com/2009/09/21/netflix-awards-1-million-prize-and-starts-a-new-contest/
#   - https://web.archive.org/web/20210804160538/https://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf
#
############################################################


# Load required libraries
library(dslabs)
library(tidyverse)

# Load the 'movielens' dataset
data("movielens")

# View the dataset as a tibble (100,004 × 7)
movielens %>% as_tibble()

# Summarize the number of distinct users and movies
movielens %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))
# Multiplying the number of unique users (n_users = 671) and the number unique movies (n_movies = 9066) gives a total more than 5 million. However, our movielens data table has 100004 rows. This implies that not every movie is rated by every user. Therefore it is obvious that if we generate a table with the unique users as rows and the movies as columns, there will be a lot many empty cells. 

# Select the top 5 movies based on number of ratings
keep <- movielens %>%
  dplyr::count(movieId) %>%
  top_n(5) %>%
  pull(movieId)

# Create a table: users 13 to 20 and their ratings for the top 5 movies
tab <- movielens %>%
  filter(userId %in% c(13:20)) %>% 
  filter(movieId %in% keep) %>% 
  dplyr::select(userId, title, rating) %>% 
  pivot_wider(names_from="title", values_from="rating")
tab %>% knitr::kable()

# Visualize a random sample of 100 users and 100 movies
users <- sample(unique(movielens$userId), 100)
rafalib::mypar()
movielens %>% filter(userId %in% users) %>% 
  dplyr::select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  pivot_wider(names_from = movieId, values_from = rating) %>% 
  (\(mat) mat[, sample(ncol(mat), 100)])() %>%  # Randomly select 100 movies
  as.matrix() %>% 
  t() %>%
  image(1:100, 1:100, ., xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

# Plot distributions of number of ratings per movie and per user
library(gridExtra)
p1 <- movielens %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

p2 <- movielens %>% 
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users")

grid.arrange(p1, p2, ncol = 2)

# -----------------------------------
# Building the Recommendation System
# -----------------------------------

# Split data into training and test sets
library(caret)
set.seed(755)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.2, list = FALSE)
train_set <- movielens[-test_index,]
test_set <- movielens[test_index,]

# Ensure all movies and users in test set are also in training set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Define RMSE (Root Mean Squared Error) function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# ------------------------
# A first simple model: 
# Predicting using the overall average rating
# ------------------------

# Calculate the mean rating (mu_hat)
mu_hat <- mean(train_set$rating)
mu_hat

# Calculate RMSE for naive model using mean rating
naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

# The following code confirms that any number other than the true rating for all movies and users would result into a higher RMSE
predictions <- rep(2.5, nrow(test_set))
RMSE(test_set$rating, predictions)

# Store RMSE result
rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)

# -------------------------------
# Modeling Movie Effects Only
# -------------------------------

# Calculate average deviation for each movie (movie effect)
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Visualize distribution of movie effects
movie_avgs %>% qplot(b_i, geom = "histogram", bins = 30, data = ., color = I("black"))

# Predict ratings: mean + movie effect
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

# Calculate RMSE for movie effect model
model_1_rmse <- RMSE(predicted_ratings, test_set$rating)

# Update RMSE results
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Movie Effect Model",
                                     RMSE = model_1_rmse))

# Show results
rmse_results %>% knitr::kable()

# -------------------------------
# Modeling Movie + User Effects
# -------------------------------

# Visualize user average ratings for users with at least 100 ratings
train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n() >= 100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

# Calculate user effect: average deviation after accounting for movie effect
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predict ratings: mean + movie effect + user effect
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Calculate RMSE for movie + user effect model
model_2_rmse <- RMSE(predicted_ratings, test_set$rating)

# Update RMSE results
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Movie + User Effects Model",
                                     RMSE = model_2_rmse))

# Show all RMSE comparison results
rmse_results %>% knitr::kable()
