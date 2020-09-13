import numpy as np
import pandas

#Class for Popularity based Recommender System model
class popularity_recommender():
    def __init__(self):
        self.train_data = None
        self.userID = None
        self.item_id = None
        self.popularity_recommendations = None
        
    #Create the popularity based recommender system model
    def create(self, train_data, userID, item_id):
        self.train_data = train_data
        self.userID = userID
        self.item_id = item_id

        #Get a count of userIDs for each unique name as recommendation score
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.userID: 'count'}).reset_index()
        train_data_grouped.rename(columns = {'userID': 'score'},inplace=True)
    
        #Sort the names based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])
    
        #Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        
        #Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)

    #Use the popularity based recommender system model to
    #make recommendations
    def recommend(self, userID):    
        user_recommendations = self.popularity_recommendations
        
        #Add userID column for which the recommendations are being generated
        user_recommendations['userID'] = userID
    
        #Bring userID column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        
        return user_recommendations
    

#Class for Item similarity based Recommender System model
class item_similarity_recommender():
    def __init__(self):
        self.train_data = None
        self.userID = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.names_dict = None
        self.rev_names_dict = None
        self.item_similarity_recommendations = None
        
    #Get unique items (names) corresponding to a given user
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.userID] == user]
        user_items = list(user_data[self.item_id].unique())
        
        return user_items
        
    #Get unique users for a given item (name)
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.userID].unique())
            
        return item_users
        
    #Get unique items (names) in the training data
    def get_all_items_train_data(self,name):
        all_items = list(self.train_data[self.item_id].unique())    
        return all_items
        
    #Construct cooccurence matrix
    def construct_cooccurence_matrix(self, user_names, all_names):
            
        ####################################
        #Get users for all names in user_names.
        ####################################
        user_names_users = []        
        for i in range(0, len(user_names)):
            user_names_users.append(self.get_item_users(user_names[i]))
            
        ###############################################
        #Initialize the item cooccurence matrix of size 
        #len(user_names) X len(names)
        ###############################################
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_names), len(all_names))), float)
           
        #############################################################
        #Calculate similarity between user names and all unique names
        #in the training data
        #############################################################
        for i in range(0,len(all_names)):
            #Calculate unique listeners (users) of name (item) i
            names_i_data = self.train_data[self.train_data[self.item_id] == all_names[i]]
            users_i = set(names_i_data[self.userID].unique())
            
            for j in range(0,len(user_names)):       
                    
                #Get unique listeners (users) of name (item) j
                users_j = user_names_users[j]
                    
                #Calculate intersection of listeners of names i and j
                users_intersection = users_i.intersection(users_j)
                
                #Calculate cooccurence_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:
                    #Calculate union of listeners of names i and j
                    users_union = users_i.union(users_j)
                    
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0
                    
        
        return cooccurence_matrix

    
    #Use the cooccurence matrix to make top recommendations
    def generate_top_recommendations(self, user, cooccurence_matrix, all_names, user_names):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))
        
        #Calculate a weighted average of the scores in cooccurence matrix for all user names.
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        #Sort the indices of user_sim_scores based upon their value
        #Also maintain the corresponding score
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
    
        #Create a dataframe from the following
        columns = ['userID', 'name', 'score', 'rank']
        #index = np.arange(1) # array of numbers for the number of samples
        df = pandas.DataFrame(columns=columns)
         
        #Fill the dataframe with top 10 item based recommendations
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_names[sort_index[i][1]] not in user_names and rank <= 10:
                df.loc[len(df)]=[user,all_names[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        #Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current user has no names for training the item similarity based recommendation model.")
            return -1
        else:
            return df
 
    #Create the item similarity based recommender system model
    def create(self, train_data, userID, item_id):
        self.train_data = train_data
        self.userID = userID
        self.item_id = item_id

    #Use the item similarity based recommender system model to
    #make recommendations
    def recommend(self, user):
        
        ########################################
        #A. Get all unique names for this user
        ########################################
        user_names = self.get_user_items(user)    
            
        print("No. of unique names for the user: %d" % len(user_names))
        
        ######################################################
        #B. Get all unique items (names) in the training data
        ######################################################
        all_names = self.get_all_items_train_data()
        
        print("no. of unique names in the training set: %d" % len(all_names))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(user_names) X len(names)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_names, all_names)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_names, user_names)
                
        return df_recommendations
    
    #Get similar items to given items
    def get_similar_items(self, item_list):
        
        user_names = item_list
        
        ######################################################
        #B. Get all unique items (names) in the training data
        ######################################################
        all_names = self.get_all_items_train_data()
        
        print("no. of unique names in the training set: %d" % len(all_names))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(user_names) X len(names)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_names, all_names)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_names, user_names)
         
        return df_recommendations