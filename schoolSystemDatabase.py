# --- PART 1: READING DATA ---

# 1.1
import numpy as np
import pandas as pd

def read_movies_data(f):

    cs = ["id", "title", "year", "genre"]

    data = []
    
    with open(f, 'r') as file:
        for line in file:

            a = line.split('|')

            id = a[0]
            title = a[1]
            year = a[2]
            genre = a[3]
            genre = genre[:-1]

            data.append([int(id), title, int(year), genre])

    

    df = pd.DataFrame(data, columns=cs)
    df.set_index('id', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    return df

# 1.2
def read_ratings_data(f):
    ratings = {}

    with open(f, 'r', newline='') as file:
        for line in file: 
            userid, movieid, rating = map(float, line.strip().split(','))

            if movieid in ratings:
                ratings[movieid].append(rating)
            else:
                ratings[movieid] = [rating]
    
    return ratings

# --- PART 2: PROCESSING DATA ---

# 2.1
def get_movie(movies_df, movie_id):
    pass

# 2.2
def create_genre_dict(movies_df):
    genredict = {}
    for index, row in movies_df.iterrows():
        movieid = index
        genres = row['genre'].split('|')

        for genre in genres:
            if genre in genredict:
                genredict[genre].append(movieid)
            else: 
                genredict[genre] = [movieid]
    
    return genredict

# 2.3
def calculate_average_rating(ratingsdict, moviesdataframe): 
    avgratings = {}

    for movieid, ratings in ratingsdict.items():
        avgrating = sum(ratings) / len(ratings)
        avgratings[movieid] = avgrating
    
    avgratingsseries = pd.Series(avgratings)
    return avgratingsseries

# --- PART 3: RECOMMENDATION ---

# 3.1
def get_popular_movies(avgratings, n=10):
    sortedratings = avgratings.sort_values(ascending=False)
    topmovies = sortedratings.head(n)
    return topmovies

# 3.2
def filter_movies(avg_ratings, thres_rating=3):
    filtered = {}
    for movieid, avgrating in avg_ratings.items():
        if avgrating >= thres_rating:
            filtered[movieid] = avgrating
    filteredseries = pd.Series(filtered)
    return filteredseries

# 3.3
def get_popular_in_genre(genre, genre_to_movies, avg_ratings, n=5):
    if genre in genre_to_movies: 
        moviesingenre = genre_to_movies[genre]
        genreratings = avg_ratings.loc[moviesingenre]
        sortedgenreratings = genreratings.sort_values(ascending=False)
        topgenremovies = sortedgenreratings.head(n)
        return topgenremovies
    else:
        return pd.Series()
# 3.4
def get_genre_rating(genre, genre_to_movies, avg_ratings):
    if genre in genre_to_movies:
        moviesingenre = genre_to_movies[genre]
        genreratings = avg_ratings[moviesingenre]
        genreavgrating = genreratings.mean()
        return genreavgrating
    else:
        return None
# 3.5
def get_movie_of_the_year(year, avg_ratings, movies_df):
    yearmovies = movies_df[movies_df['year'] == year]

    if not yearmovies.empty:
        highestratedid = avg_ratings.loc[yearmovies.index].idxmax()
        highestratedtitle = movies_df.loc[highestratedid]['title']
        return highestratedtitle
    else:
        return None

# --- PART 4: USER FOCUSED ---

# 4.1
def read_user_ratings(f):
    userratings = {}

    with open(f, 'r') as file:
        for line in file:
            userid, movieid, rating = line.strip().split(',')
            userid = int(userid)
            movieid = int(movieid)
            rating = float(rating)
            userrating = (movieid, rating)

            if userid in userratings:
                userratings[userid].append(userrating)
            else:
                userratings[userid] = [userrating]

    return userratings

# 4.2
def get_user_genre(user_id, user_to_movies, movies_df):
    if user_id not in user_to_movies:
        return None
    
    usermovieratings = user_to_movies[user_id]

    genreratings = {}
    for movieid, rating in usermovieratings:
        moviegenre = movies_df.loc[movieid]['genre']
        if moviegenre in genreratings:
            genreratings[moviegenre].append(rating)
        else:
            genreratings[moviegenre] = [rating]
    
    avggenreratings = {genre: sum(ratings) / len(ratings) for  genre, ratings in genreratings.items()}

    topgenre = max(avggenreratings, key=avggenreratings.get)

    return topgenre
# 4.3
def recommend_movies(user_id, user_to_movies, movies_df, avg_ratings):
    # Step 1: Determine the user's top genre
    top_user_genre = get_user_genre(user_id, user_to_movies, movies_df)
    if top_user_genre is None:
        return pd.Series([], name='avg_ratings')

    top_genre_movies = movies_df[movies_df['genre'] == top_user_genre]

    top_genre_movies = top_genre_movies.join(avg_ratings.rename('avg_ratings')).sort_values(by='avg_ratings', ascending=False)

    topthree = top_genre_movies.head(3)
    
    a = topthree['avg_ratings'].squeeze()
    return a

# --- PART 1: READING DATA ---
 
# 1.1
import numpy as np
import pandas as pd
 
def read_movies_data(f):
 
    cs = ["id", "title", "year", "genre"]
 
    data = []
    
    with open(f, 'r') as file:
        for line in file:
 
            a = line.split('|')
 
            id = a[0]
            title = a[1]
            year = a[2]
            genre = a[3]
            genre = genre[:-1]
 
            data.append([int(id), title, int(year), genre])
 
    
 
    df = pd.DataFrame(data, columns=cs)
    df.set_index('id', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    return df
 
# 1.2
def read_ratings_data(f):
    ratings = {}
 
    with open(f, 'r', newline='') as file:
        for line in file: 
            userid, movieid, rating = map(float, line.strip().split(','))
 
            if movieid in ratings:
                ratings[movieid].append(rating)
            else:
                ratings[movieid] = [rating]
    
    return ratings
 
# --- PART 2: PROCESSING DATA ---
 
# 2.1
def get_movie(movies_df, movie_id):
    pass
 
# 2.2
def create_genre_dict(movies_df):
    genredict = {}
    for index, row in movies_df.iterrows():
        movieid = index
        genres = row['genre'].split('|')
 
        for genre in genres:
            if genre in genredict:
                genredict[genre].append(movieid)
            else: 
                genredict[genre] = [movieid]
    
    return genredict
 
# 2.3
def calculate_average_rating(ratingsdict, moviesdataframe): 
    avgratings = {}
 
    for movieid, ratings in ratingsdict.items():
        avgrating = sum(ratings) / len(ratings)
        avgratings[movieid] = avgrating
    
    avgratingsseries = pd.Series(avgratings)
    return avgratingsseries
 
# --- PART 3: RECOMMENDATION ---
 
# 3.1
def get_popular_movies(avgratings, n=10):
    sortedratings = avgratings.sort_values(ascending=False)
    topmovies = sortedratings.head(n)
    return topmovies
 
# 3.2
def filter_movies(avg_ratings, thres_rating=3):
    filtered = {}
    for movieid, avgrating in avg_ratings.items():
        if avgrating >= thres_rating:
            filtered[movieid] = avgrating
    filteredseries = pd.Series(filtered)
    return filteredseries
 
# 3.3
def get_popular_in_genre(genre, genre_to_movies, avg_ratings, n=5):
    if genre in genre_to_movies: 
        moviesingenre = genre_to_movies[genre]
        genreratings = avg_ratings.loc[moviesingenre]
        sortedgenreratings = genreratings.sort_values(ascending=False)
        topgenremovies = sortedgenreratings.head(n)
        return topgenremovies
    else:
        return pd.Series()
# 3.4
def get_genre_rating(genre, genre_to_movies, avg_ratings):
    if genre in genre_to_movies:
        moviesingenre = genre_to_movies[genre]
        genreratings = avg_ratings[moviesingenre]
        genreavgrating = genreratings.mean()
        return genreavgrating
    else:
        return None
# 3.5
def get_movie_of_the_year(year, avg_ratings, movies_df):
    yearmovies = movies_df[movies_df['year'] == year]
 
    if not yearmovies.empty:
        highestratedid = avg_ratings.loc[yearmovies.index].idxmax()
        highestratedtitle = movies_df.loc[highestratedid]['title']
        return highestratedtitle
    else:
        return None
 
# --- PART 4: USER FOCUSED ---
 
# 4.1
def read_user_ratings(f):
    userratings = {}
 
    with open(f, 'r') as file:
        for line in file:
            userid, movieid, rating = line.strip().split(',')
            userid = int(userid)
            movieid = int(movieid)
            rating = float(rating)
            userrating = (movieid, rating)
 
            if userid in userratings:
                userratings[userid].append(userrating)
            else:
                userratings[userid] = [userrating]
 
    return userratings
 
# 4.2
def get_user_genre(user_id, user_to_movies, movies_df):
    if user_id not in user_to_movies:
        return None
    
    usermovieratings = user_to_movies[user_id]
 
    genreratings = {}
    for movieid, rating in usermovieratings:
        moviegenre = movies_df.loc[movieid]['genre']
        if moviegenre in genreratings:
            genreratings[moviegenre].append(rating)
        else:
            genreratings[moviegenre] = [rating]
    
    avggenreratings = {genre: sum(ratings) / len(ratings) for  genre, ratings in genreratings.items()}
 
    topgenre = max(avggenreratings, key=avggenreratings.get)
 
    return topgenre
# 4.3
def recommend_movies(user_id, user_to_movies, movies_df, avg_ratings):
    # Step 1: Determine the user's top genre
    top_user_genre = get_user_genre(user_id, user_to_movies, movies_df)
    if top_user_genre is None:
        return pd.Series([], name='avg_ratings')
 
    top_genre_movies = movies_df[movies_df['genre'] == top_user_genre]
 
    top_genre_movies = top_genre_movies.join(avg_ratings.rename('avg_ratings')).sort_values(by='avg_ratings', ascending=False)
 
    topthree = top_genre_movies.head(3)
    
    a = topthree['avg_ratings'].squeeze()
    return a
