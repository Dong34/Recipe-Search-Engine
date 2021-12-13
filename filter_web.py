from nltk.tokenize import word_tokenize
import streamlit as st
import altair as alt
import pandas as pd
import os
import re
import nltk
from nltk.stem import WordNetLemmatizer 
# from sklearn.model_selection import train_test_split
# from nltk import ngrams, FreqDist
# from nltk import word_tokenize
# from nltk.corpus import stopwords
# from nltk.tokenize import RegexpTokenizer


# ingredient type: First letter is capital and the remaining letters are lower case.
# Example: Beer, Apple pie spicy
def ingredient_sub(ingredient):
    substitution = pd.read_csv("substitutes.csv")
    
    lemmatizer = WordNetLemmatizer()
    for i in range(len(substitution)):
        sentence = substitution['Ingredient'][i]
        word_list = nltk.word_tokenize(sentence)
        # Lemmatize list of words and join
        substitution['Ingredient'][i] = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    
    substitution['Ingredient'] = substitution['Ingredient'].str.lower()
    

    candidate = substitution[substitution['Ingredient'].str.contains(ingredient)]

    if len(candidate) > 0:
        s = candidate['Substitutes']
        print(s)
        cut = ' ' * 4
        amount = re.split(cut + '|\n', str(candidate['Amount']))[1]
        s = re.split(cut + '|\n', s)[1]
        return amount, s

    return None

def filter_result(df, items, out=True):
    # Add filter here
    # items: the word we want to filter
    # option: tool or ingredient
    # Return: All the search results that do not contain the word ingredient in the "ingredients" description.

    if items == []:
        return df

    df['ingredients'] = df['ingredients'].str.lower()
    df['description'] = df['description'].str.lower()
    df['keywords'] = df['keywords'].str.lower()
    df['filtering'] = df['ingredients'] + ' ' + df['description'] + ' ' + df['keywords']
    

    # Init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    for i in range(len(df)):
        sentence = df['filtering'][i]
        word_list = nltk.word_tokenize(sentence)
        # Lemmatize list of words and join
        df['filtering'][i] = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    
    for i in items:
        w = lemmatizer.lemmatize(i)
        
        if out == True:
            df = df[(~(df['filtering'].str.contains(w)))]
        else:
            df = df[((df['filtering'].str.contains(w)))]

    
    return df


if __name__ == '__main__':

    # initialization
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    
    #Display
    st.title("Filter Test")

    # read search results
    search_result = pd.read_csv('searchResults.csv')
    search_result

    st.write("Filter any tools that you don't have:")
    oven = st.checkbox('oven')
    microwave = st.checkbox('microwave')
    filterword_out = st.text_input("Input anything that you want to filter out:")

    st.write("Filter any feature that you want your recipe to have:")
    healthy = st.checkbox('healthy')
    low_cholesterol = st.checkbox('low cholesterol')
    inexpensive = st.checkbox('inexpensive')
    filterword = st.text_input("Input anything that you want to filter:")
    # ingredient = st.text_input("Input any ingredient that you don't have:")

    wSet_out = []
    wSet = []
    if oven:
        wSet_out.append('oven')
    if microwave:
        wSet_out.append('microwave')
    if filterword_out:
        wSet_out.append(filterword_out)

    if healthy:
        wSet.append('healthy')
    if low_cholesterol:
        wSet.append('low cholesterol')
    if inexpensive:
        wSet.append('inexpensive')
    if filterword:
        wSet.append(filterword)


    st.write("----------------------------------------------")
    
    # filtered_result = filter_result(search_result.copy(), toolSet, 'tool')
    # filtered_result2 = filter_result(filtered_result.copy(), ingreSet, 'ingredient')
    filtered_result1 = filter_result(search_result.copy(deep=True), wSet_out, out=True)
    filtered_result2 = filter_result(filtered_result1.copy(deep=True), wSet, out=False)
    filtered_result2

    # ingredient substitution
    # num = st.number_input('Input a recipe number:')
    # if num:
    #     raw = search_result['ingredients'][int(num)]
    #     # print(raw)
    #     text = word_tokenize(raw)
    #     # print(text)
    #     processed = nltk.pos_tag(text)
    #     ingreSet = []    
    #     for ingre, typ in processed:
    #         if typ == 'NN':
    #             ingreSet.append(ingre)
        
    #     ingredient = st.selectbox("Choose the ingredient you want to substitute:", ingreSet)
    #     res = ingredient_sub(ingredient)
    #     if res != None:
    #         st.write(ingredient + ' corresponding amount: ' + res[0] + ' <---> ' + res[1])
    st.write('End')


