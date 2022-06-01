
'''
In this project, I will analyse the interactions that users have with articles on the IBM Watson Studio platform
 and make recommendations about new articles they will like.
'''

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import project_tests as t
import pickle
import seaborn as sns
sns.set()

df = pd.read_csv('user-item-interactions.csv')
df_content = pd.read_csv('articles_community.csv')
del df['Unnamed: 0']
del df_content['Unnamed: 0']

# showing df.head()
df.head()

# showing df_content.head()
df_content.head()

# Part 1: Exploratory Data Analysis

# find df.shape
df.shape

# median
df.groupby('email')['article_id'].count().median()
# max
df.groupby('email')['article_id'].count().max()

# median and maximum number of user_article interactions
median_val = df.groupby('email')['article_id'].count().median() # 50% of individuals interact with 3 number of articles or fewer.
max_views_by_user = df.groupby('email')['article_id'].count().max() # The maximum number of user-article interactions by any 1 user is 364.

user_interacts = df.groupby('email')['article_id'].count()
user_interacts
# summary statistics
user_interacts.describe()

# plot graph
plt.figure(figsize=(8,6))
user_interacts.plot(kind='hist')
plt.title('Distribution of User-Article Interactions') 
plt.xlabel('Number of User-Article Interactions')

# Find and remove duplicate articles from df_content
# find duplicate articles
df_content.article_id.duplicated().sum()
ids = df_content['article_id']

# explore duplicate articles
df_content[ids.isin(ids[ids.duplicated()])]

# Remove article_id duplicates - only keep the first
df_content.drop_duplicates(subset = ['article_id'], keep = 'first', inplace = True)

'''
Find : a. The number of unique articles that have an interaction with a user.
b. The number of unique articles in the dataset (whether they have any interactions or not).
c. The number of unique users in the dataset. (excluding null values)
d. The number of user-article interactions in the dataset.
'''
# The number of unique articles that have at least one interaction
df.article_id.nunique()

# The number of unique articles on the IBM platform
df_content.article_id.nunique()

# The number of unique users
df.email.nunique()

# The number of user-article interactions
df.shape[0]

unique_articles = df.article_id.nunique() # The number of unique articles that have at least one interaction
total_articles =  df_content.article_id.nunique() # The number of unique articles on the IBM platform
unique_users = df.email.nunique() # The number of unique users
user_article_interactions = df.shape[0] # The number of user-article interactions

most_viewed_article_id = str(df.article_id.value_counts().index[0]) # The most viewed article in the dataset as a string with one value following the decimal 
max_views = df.article_id.value_counts().iloc[0] # The most viewed article in the dataset was viewed how many times?

# Run this cell to map the user email to a user_id column and remove the email column

def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

email_encoded = email_mapper()
del df['email']
df['user_id'] = email_encoded

# show header
df.head()


'''
Part II: Rank-Based Recommendations
We don't actually have ratings for whether a user liked an article or not. We only know that a user has interacted
with an article. In these cases, the popularity of an article can really only be based on how often an article was 
interacted with.
'''
# The function below returns the n top articles ordered with most interactions as the top.
def get_top_articles(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    '''
    top_articles = df['title'].value_counts().index.tolist()[:n]
    top_articles = [str(i) for i in top_articles]
    
    return top_articles # Return the top article titles from df (not df_content)

def get_top_article_ids(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    '''
    top_articles = df['article_id'].value_counts().index.tolist()[:n]
    top_articles = [str(i) for i in top_articles]

    return top_articles # Return the top article ids

print(get_top_articles(10))
print(get_top_article_ids(10))

# Test your function by returning the top 5, 10, and 20 articles
top_5 = get_top_articles(5)
top_10 = get_top_articles(10)
top_20 = get_top_articles(20)

'''
Part III: User-User Based Collaborative Filtering
The function below reformats the df dataframe to be shaped with users as the rows and articles as the columns.

Each user should only appear in each row once.
Each article should only show up in one column.
If a user has interacted with an article, then place a 1 where the user-row meets for that article-column. It does not matter how many times a user has interacted with the article, all entries where a user has interacted with an article should be a 1.
If a user has not interacted with an item, then place a zero where the user-row meets for that article-column.
'''
# create the user-article matrix with 1's and 0's

def create_user_item_matrix(df):
    '''
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    df_count = df.groupby(['user_id', 'article_id']).count().reset_index() # create a new df based on count
    user_item = df_count.pivot_table(values='title', index='user_id', columns='article_id') # pivot so users on rows and article on columns
    user_item.replace(np.nan, 0, inplace=True) # replace nulls with 0s
    user_item=user_item.applymap(lambda x: 1 if x > 0 else x) # entries should be a 1 or 0

    return user_item # return the user_item matrix 

user_item = create_user_item_matrix(df)

'''
The function below takes a user_id and provides an ordered list of the most similar users to that user 
(from most similar to least similar). The returned result does not contain the provided user_id, as we know 
that each user is similar to him/herself. Because the results for each user here are binary, it makes sense 
to compute similarity as the dot product of two users.
'''

def find_similar_users(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise    
    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first   
    Description:
    Computes the similarity of every pair of users based on the dot product
    Returns an ordered   
    '''
    # compute similarity of each user to the provided user
    dot_prod_users = user_item.dot(np.transpose(user_item))
    
    # sort by similarity
    sim_users = dot_prod_users[user_id].sort_values(ascending = False)
    
    # create list of just the ids
    most_similar_users = sim_users.index.tolist()
    
    # remove the own user's id
    most_similar_users.remove(user_id)
       
    return most_similar_users # return a list of the users in order from most to least similar

# Do a spot check of your function
print("The 10 most similar users to user 1 are: {}".format(find_similar_users(1)[:10]))

'''
Now that we have a function that provides the most similar users to each user, we want to use 
these users to find articles we can recommend. The functions below return the articles we would recommend to each user.

'''
def get_article_names(article_ids, df=df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''
    article_names = []
    for idx in article_ids:
        article_names.append(df[df['article_id']==float(idx)].max()['title'])
    
    return article_names # Return the article names associated with list of article ids


def get_user_articles(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    article_ids = user_item.loc[user_id][user_item.loc[user_id] == 1].index.astype('str')
    
    article_names = []

    for idx in article_ids:
        article_names.append(df[df['article_id']==float(idx)].max()['title']) # need to use df instead of df_content as it only has 1051 rows
    
    return article_ids, article_names # return the ids and names


def user_user_recs(user_id, m = 10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    Users who are the same closeness are chosen arbitrarily as the 'next' user
    
    For the user where the number of recommended articles starts below m 
    and ends exceeding m, the last items are chosen arbitrarily
    
    '''
    recs = np.array([]) # recommendations to be made
    
    user_articles_seen = get_user_articles(user_id)[0] #seen by our user
    closest_users = find_similar_users(user_id) # users closest to our user
    
    for others in closest_users:
        
        others_articles_seen = get_user_articles(others)[0] # articles seen by others like our user
        new_recs = np.setdiff1d(others_articles_seen, user_articles_seen, assume_unique=True) #find those not seen by user
        recs = np.unique(np.concatenate([new_recs, recs], axis = 0)) # concate arrays and only return unique values

        if len(recs) > m-1:
            break
            
    recs = recs[:m]
    recs.tolist()
    
    return recs # return your recommendations for this user_id    

# Check Results
get_article_names(user_user_recs(1, 10))

'''
Now we are going to improve the consistency of the user_user_recs function from above.
Instead of arbitrarily choosing when we obtain users who are all the same closeness to a given user - choose 
the users that have the most total article interactions before choosing those with fewer article interactions.
Instead of arbitrarily choosing articles from the user where the number of recommended articles starts below m 
and ends exceeding m, choose articles with the articles with the most total interactions before choosing those 
with fewer total interactions. This ranking should be what would be obtained from the top_articles function we wrote earlier.
'''
def get_top_sorted_users(user_id, df=df, user_item=user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook 
    user_item - (pandas dataframe) matrix of users by articles: 
            1's when a user has interacted with an article, 0 otherwise
              
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u
                    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
     
    '''
    # create neighbors dataframe with empty columns
    neighbors_df = pd.DataFrame(columns=['neighbor_id', 'similarity'])
    # set neighbor_id column equal to user_item index starting from 1
    neighbors_df['neighbor_id'] = user_item.index-1
    # make similarity column equal to most similar using dot product 
    dot_prod_users = user_item.dot(np.transpose(user_item))
    neighbors_df['similarity'] = dot_prod_users[user_id]
    # create new df based on number of interactions of users
    interacts_df = df.user_id.value_counts().rename_axis('neighbor_id').reset_index(name='num_interactions')
    # merge dataframes which creates number of interactions column from interacts_df
    neighbors_df = pd.merge(neighbors_df, interacts_df, on='neighbor_id', how='outer')
    # sortvalues on similarity and then number of interactions 
    neighbors_df = neighbors_df.sort_values(by=['similarity', 'num_interactions'], ascending = False)
    # reset index
    neighbors_df = neighbors_df.reset_index(drop=True)
    # drop row with the user_id as itself will be most similar
    neighbors_df = neighbors_df[neighbors_df.neighbor_id != user_id]
    
    return neighbors_df # Return the dataframe specified in the doc_string


def user_user_recs_part2(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    * Choose the users that have the most total article interactions 
    before choosing those with fewer article interactions.

    * Choose articles with the articles with the most total interactions 
    before choosing those with fewer total interactions. 
   
    '''
    recs = np.array([]) # recommendations to be made
    
    user_articles_ids_seen, user_articles_names_seen = get_user_articles(user_id, user_item) #article ids seen by our user
    closest_neighs = get_top_sorted_users(user_id, df, user_item).neighbor_id.tolist() # neighbour user ids closest to our user
    
    for neighs in closest_neighs:
        
        neigh_articles_ids_seen, neigh_articles_names_seen = get_user_articles(neighs, user_item) # articles seen by others like our user
        new_recs = np.setdiff1d(neigh_articles_ids_seen, user_articles_ids_seen, assume_unique=True) #find those not seen by user
        recs = np.unique(np.concatenate([new_recs, recs], axis = 0)) # concate arrays and only return unique values

        if len(recs) > m-1:
            break
            
    recs = recs[:m]
    recs = recs.tolist() # convert to a list
    
    rec_names = get_article_names(recs, df=df)
    
    return recs, rec_names

# Checking functions
rec_ids, rec_names = user_user_recs_part2(20, 10)
print("The top 10 recommendations for user 20 are the following article ids:")
print(rec_ids)
print()
print("The top 10 recommendations for user 20 are the following article names:")
print(rec_names)

### Tests with a dictionary of results
user1_most_sim = get_top_sorted_users(1).iloc[0].neighbor_id # Find the user that is most similar to user 1 
user131_10th_sim = get_top_sorted_users(131).iloc[9].neighbor_id # Find the 10th most similar user to user 131

'''
If given a new user, it would make sense to use Rank Based Recommendations and the get_top_articles function 
to make recommendations. We would just recommend the most popular articles since we do not have any information 
about the user or their interactions so cannot tell which other users they are most similar to. Once we have
more information about the user we could a blended approach of 3 types of recommendation techniques; Rank, 
Content, and Collaborative
'''
new_user = '0.0'

# What would your recommendations be for this new user '0.0'?  As a new user, they have no observed articles.
# Provide a list of the top 10 article ids you would give to 
new_user_recs = get_top_article_ids(10, df)

'''
Part IV: Matrix Factorization
In this part, we will use matrix factorization to make article recommendations to the users on the IBM Watson Studio platform.
We have already created a user_item matrix above in question 1 of Part III above. This first question here will just require that you run the cells to get things set up for the rest of Part IV of the notebook.
'''
user_item_matrix = user_item
# Perform SVD on the User-Item Matrix Here
u, s, vt = np.linalg.svd(user_item_matrix)

s.shape, u.shape, vt.shape

'''
How do we choose the number of latent features to use? Running the below cell, we can see that as the number of latent features increases, we obtain a lower error rate on making predictions for the 1 and 0 values in the user-item matrix. We get an idea of how the accuracy improves as we increase the number of latent features.
'''

num_latent_feats = np.arange(10, 710, 20)
sum_errs = []

for k in num_latent_feats:
    # restructure with k latent features
    s_new, u_new, vt_new = np.diag(s[:k]), u[:, :k], vt[:k, :]
    
    # take dot product
    user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new))
    
    # compute error for each prediction to actual value
    diffs = np.subtract(user_item_matrix, user_item_est)
    
    # total errors and keep track of them
    err = np.sum(np.sum(np.abs(diffs)))
    sum_errs.append(err)
    
plt.plot(num_latent_feats, 1 - np.array(sum_errs)/df.shape[0]);
plt.xlabel('Number of Latent Features');
plt.ylabel('Accuracy');
plt.title('Accuracy vs. Number of Latent Features')

'''
From the above, we can't really be sure how many features to use, because simply having a better way 
to predict the 1's and 0's of the matrix doesn't exactly give us an indication of if we are able to 
make good recommendations. Instead, we might split our dataset into a training and test set of data, as shown in the cell below.

Using the code from question 3 we can understand the impact on accuracy of the training and test 
sets of data with different numbers of latent features. Using the split below:

How many users can we make predictions for in the test set?
How many users are we not able to make predictions for because of the cold start problem?
How many articles can we make predictions for in the test set?
How many articles are we not able to make predictions for because of the cold start problem?
'''

df_train = df.head(40000)
df_test = df.tail(5993)

def create_test_and_train_user_item(df_train, df_test):
    '''
    INPUT:
    df_train - training dataframe
    df_test - test dataframe
    
    OUTPUT:
    user_item_train - a user-item matrix of the training dataframe 
                      (unique users for each row and unique articles for each column)
    user_item_test - a user-item matrix of the testing dataframe 
                    (unique users for each row and unique articles for each column)
    test_idx - all of the test user ids
    test_arts - all of the test article ids
    
    '''
    user_item_train = create_user_item_matrix(df_train)
    user_item_test = create_user_item_matrix(df_test)
    
    test_idx = user_item_test.index
    test_arts = user_item_test.columns
    
    return user_item_train, user_item_test, test_idx, test_arts

user_item_train, user_item_test, test_idx, test_arts = create_test_and_train_user_item(df_train, df_test)

train_idx = user_item_train.index # 4487 users in training set
train_idx 

test_idx.difference(train_idx) # of 682 users in test set, only 20 of them are in training set

train_arts = user_item_train.columns #714 movies in train set
train_arts

test_arts.difference(train_arts) # all articles in test set are in training set too

# fit SVD on the user_item_train matrix
u_train, s_train, vt_train = np.linalg.svd(user_item_train) # fit svd similar to above then use the cells below

num_latent_feats = np.arange(10,700+10,20)
sum_errs_train = []
sum_errs_test = []

#Decomposition
row_idx = user_item_train.index.isin(test_idx)
col_idx = user_item_train.columns.isin(test_arts)

u_test = u_train[row_idx, :]
vt_test = vt_train[:, col_idx]

# test users that we can predict for
users_can_predict = np.intersect1d(list(user_item_train.index),list(user_item_test.index))
    
for k in num_latent_feats:
    # restructure with k latent features
    s_train_new, u_train_new, vt_train_new = np.diag(s_train[:k]), u_train[:, :k], vt_train[:k, :]
    u_test_new, vt_test_new = u_test[:, :k], vt_test[:k, :]
    
    # take dot product
    user_item_train_preds = np.around(np.dot(np.dot(u_train_new, s_train_new), vt_train_new))
    user_item_test_preds = np.around(np.dot(np.dot(u_test_new, s_train_new), vt_test_new))
    
    # compute error for each prediction to actual value
    diffs_train = np.subtract(user_item_train, user_item_train_preds)
    diffs_test = np.subtract(user_item_test.loc[users_can_predict,:], user_item_test_preds)
    
    # total errors and keep track of them
    err_train = np.sum(np.sum(np.abs(diffs_train)))
    err_test = np.sum(np.sum(np.abs(diffs_test)))
    
    sum_errs_train.append(err_train)
    sum_errs_test.append(err_test)

# plotting the training and test accuracies
fig, ax1 = plt.subplots()

color = 'tab:orange'
ax1.set_xlabel('Number of Latent Features')
ax1.set_ylabel('Accuracy for Training', color=color)
ax1.plot(num_latent_feats, 1 - np.array(sum_errs_train)/df.shape[0], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_title('Accuracy vs. Number of Latent Features')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Accuracy for Test', color=color)  # we already handled the x-label with ax1
ax2.plot(num_latent_feats, 1 - np.array(sum_errs_test)/df.shape[0], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

'''
From above, we can see that the accuracy for the training data increases with an increase in the number of latent features, 
however the opposite is true for the accuracy of the test data. This is most likely due to overfitting of the data with the 
increase in latent features, therefore the number of latent features should be kept relatively low. It is important to note 
that using SVD here we can only actually make recommendations for the 20 users in both the training and test dataset, and we
have a very sparse matrix which is likely why the test data accuracy is so high at >99%. It would be interesting to look at
results if we had more users that appeared in both the test and training data. We can see that at approximately 80 features
there is a cross over point when the accuracy for test data begins to drop, therefore this would be a good number of latent
features to include, since beyond that our accuracy for training increases but testing decreases. To test how well our 
recommendation engine works in practice, we could conduct an A/B test for new users to help solve the cold start problem.
An example would be to recommend articles to one group using our recommendation engine and then to recommend just the most
popular articles to the other group of users. We would then compare the click through rates to effectively measure if our 
recommendation engine leads to an increase in clicks. If we saw a significant rise in clicks by using our recommendation 
engine then we could conclude this works well and should be deployed.

Conclusion
When using SVD we experience the cold start problem. That is, we can only make predictions for articles and users that exist 
in both the training and test sets. For users in the test dataset that are not in the training set, we cannot predict articles 
to recommend to that user. To improve upon our recommendation engine for new users, we could use a blended techniques of 
Knowledge based, collaborative filtering based, and content based recommendations. We could also conduct an A/B test to best 
understand which recommendation technique should be employed.
'''
