# SI 650 Final Project

# imports we will use
import streamlit as st
import altair as alt
import pandas as pd
import pyterrier as pt
import os
from pyterrier.measures import *
import re
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from nltk import ngrams, FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


debugging = 1


class Model:

    RANK_CUTOFF = 100
    SEED = 42
    QUERY_CUTOFF = 30
    def __init__(self, path_data, path_topic, path_qrel):
        self.init_data(path_data)
        self.create_index()
        self.get_topic_qrel(path_topic, path_qrel)
        self.init_pymodels()
        self.train_model()
        if debugging == 1:
          print("init finished")
    
    
    def init_data(self, path):
        foodresults = pd.read_csv(path, encoding="latin-1")
        data=[]
        foodresults["docno"] = range(0,len(foodresults))
        foodresults["recipe_id"] = range(0,len(foodresults))
        foodresults.head(5)
        for index,row in foodresults.iterrows():
            # docno = "d"+str(row["docno"])
            recipe_id = str(row["recipe_id"])
            recipe_name = " ".join(re.findall("[a-zA-Z]+", row["name"]))
            ingredients = " ".join(re.findall("[a-zA-Z]+", str(row["ingredients"])))
            recipe = " ".join(re.findall("[a-zA-Z]+", str(row["recipe"])))
            rate = str(float(row["rate"]))
            views = str(int(row["views"]))
            description = " ".join(re.findall("[a-zA-Z]+", row["description"]))
            keywords = " ".join(re.findall("[a-zA-Z]+", str(row["keywords"])))
            calories = row["calories"]
            fatContent = row["fatContent"]
            proteinContent = row["proteinContent"]
            url = row["url"]

            original_recipe_name = str(row['name'])
            original_ingredients = str(row["ingredients"])
            original_recipe = str(row["recipe"])
            original_description = str(row["description"])
            original_keywords = str(row["keywords"])

            data.append([recipe_id,recipe_name,ingredients,recipe,
                        rate,views,description,keywords,calories,fatContent,proteinContent, url, original_recipe_name, 
                        original_ingredients, original_recipe, original_description, original_keywords])
        self.df = pd.DataFrame(data,columns=["docno","recipe_name","ingredients","recipe",
                    "rate","views","description","keywords","calories","fatContent","proteinContent", "url", 
                    "original_recipe_name", "original_ingredients","original_recipe", "original_description",
                    "original_keywords"])
        if debugging == 1:
          print("init_data finished")

    
    def create_index(self):
    	#TODO: add if condition
        self.index_dir = "./docs_index"
        indexer = pt.DFIndexer(self.index_dir, overwrite=True)
        index_ref = indexer.index(self.df["recipe"], self.df["docno"], self.df["recipe_name"], self.df["keywords"], 
                            self.df["description"], self.df["ingredients"], self.df["rate"], self.df["views"], self.df["url"])
        # index_ref.toString()
        self.index = pt.IndexFactory.of(index_ref)
        if debugging == 1:
          print("create_index finished")

    def get_topic_qrel(self, path_topic, path_qrel):
        topics = pd.read_csv(path_topic)
        qrels = pd.read_csv(path_qrel)
        qrels = qrels.drop(columns=["iteration"])
        topics["qid"] = topics["qid"].astype(str)
        qrels["qid"] = qrels["qid"].astype(str)
        qrels["docno"] = qrels["docno"].astype(str)
        self.topics = topics
        self.qrels = qrels
        if debugging == 1:
          print("get_topic_qrel finished")

    def init_pymodels(self):
        self.bm25 = pt.BatchRetrieve(self.index, wmodel="BM25")
        self.qe = pt.rewrite.Bo1QueryExpansion(self.index)
        self.ltr_feats = (self.bm25 % self.RANK_CUTOFF) >> pt.text.get_text(self.index, ["recipe_name","keywords","description","ingredients","rate","views"]) >> (
            pt.transformer.IdentityTransformer()
            **
            (self.bm25 >> self.qe >> self.bm25)
            **
            (pt.text.scorer(body_attr="recipe_name", takes='docs', wmodel='BM25'))
            ** # score of title (not originally indexed)
            (pt.text.scorer(body_attr="keywords", takes='docs', wmodel='BM25'))
            ** 
            (pt.text.scorer(body_attr="description", takes='docs', wmodel='BM25'))
            ** 
            (pt.text.scorer(body_attr="ingredients", takes='docs', wmodel='BM25'))
            **
            pt.apply.doc_score(lambda row: float(row["rate"]))
            **
            pt.apply.doc_score(lambda row: int(row["views"]))
            # **
            # pt.BatchRetrieve(self.index, wmodel="CoordinateMatch")
        )
        self.fnames=["bm25", "qe", "bm25_recipe_name", "bm25_keywords", "bm25_description", "bm25_ingredients", "rate", "views"]
        lmart_l = lgb.LGBMRanker(
            task="train",
            silent=False,
            min_data_in_leaf=1,
            min_sum_hessian_in_leaf=1,
            max_bin=255,
            num_leaves=31,
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[10],
            ndcg_at=[10],
            eval_at=[10],
            learning_rate= .1,
            importance_type="gain",
            num_iterations=100,
            early_stopping_rounds=5
        )
        self.lmart_x_pipe = self.ltr_feats >> pt.ltr.apply_learned_model(lmart_l, form="ltr", fit_kwargs={'eval_at':[10]})
        if debugging == 1:
          print("init_pymodels finished")

    def train_model(self):
        tr_va_topics, test_topics = train_test_split(self.topics, test_size=0.15, random_state=self.SEED)
        train_topics, valid_topics =  train_test_split(tr_va_topics, test_size=0.17, random_state=self.SEED)
        self.lmart_x_pipe.fit(train_topics, self.qrels, valid_topics, self.qrels)
        if debugging == 1:
          print("train_model finished")

    
    def get_query_results(self, query):
        query_results = self.lmart_x_pipe(query).head(self.QUERY_CUTOFF)["docno"].to_list()
        results = []
        rank = 0
        for result in query_results:
            result_dict = {'docno': result, 
            'recipe_name': self.df[self.df['docno'] == result]['original_recipe_name'].to_list()[0],
            'ingredients': self.df[self.df['docno'] == result]['original_ingredients'].to_list()[0],
            'recipe': self.df[self.df['docno'] == result]['original_recipe'].to_list()[0],
            'rate': self.df[self.df['docno'] == result]['rate'].to_list()[0],
            'views': self.df[self.df['docno'] == result]['views'].to_list()[0],
            'description': self.df[self.df['docno'] == result]['original_description'].to_list()[0],
            'calories': self.df[self.df['docno'] == result]['calories'].to_list()[0],
            'fatContent': self.df[self.df['docno'] == result]['fatContent'].to_list()[0],
            'proteinContent': self.df[self.df['docno'] == result]['proteinContent'].to_list()[0],
            'url': self.df[self.df['docno'] == result]['url'].to_list()[0],
            'rank': rank}
            results.append(result_dict)
            rank += 1
        return results

    def filter_result(self, df):
    	# Add filter here
    	pass

    def expand_query(self, query, results):
        data = ''
        for i in range(len(results)):
          data+=results.iloc[i]["ingredients"]
          data+=" "

        stop_words = set(stopwords.words('english'))
        my_stop_list = ["lb", "teaspoons", "teaspoon", "optional", "tablespoon", "tablespoons", "tsp", "g", "kg", "lbs"]
        for stop_word in my_stop_list:
          stop_words.add(stop_word)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(data)
        new_tokens= [token for token in tokens if not token.isnumeric() 
                        and not token in stop_words]
        all_counts = FreqDist(ngrams(new_tokens, 2))
        expand_dict = dict()
        for key in all_counts.keys():
          if query in key:
            expand_dict[key] = all_counts[key]
        sort_expand_dict = sorted(expand_dict.items(), key=lambda x: x[1], reverse=True)
        expand_list = sort_expand_dict[0:3]
        drop_list = [query]
        for item in expand_list:
          drop_list_item = item[0][0]+" "+item[0][1]
          drop_list.append(drop_list_item)
        return drop_list
    
    def ingredients_sub(self, results):
        pass


def pyterrier_init():
    if not pt.started():
        pt.init()


def initialize():
    pyterrier_init()
    model = Model("foodresults.csv", "topics.csv", 'qrel.csv')
    return model


# Main function
if __name__ == '__main__':
    model = initialize()
    #Display
    st.title("SI 650 Final Project")

    query_form = st.form(key='user_query')
    query_text = query_form.text_input(label='Search on Food')
    query_submit = query_form.form_submit_button(label='Submit')

    if query_submit:
        results = model.get_query_results(query_text) # list of dict format

        # query_form = st.form(key='user_query_expand')
        
        results = pd.DataFrame(results)
        #df
        #expand query to get dropbox
        drop_box = model.expand_query(query_text, results)
        drop_box_select = st.selectbox(label = "We suggest", options=drop_box)
        if drop_box_select == drop_box[0]:
            results = model.get_query_results(drop_box[0])
            df = pd.DataFrame(results)
            #filter the result
            #df = model.filter_result(df)
            df
        #elif 


