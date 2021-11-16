# -*- coding: utf-8 -*-
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#|r|e|d|a|n|d|g|r|e|e|n|.|c|o|.|u|k|
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

import os
import json
from pprint import pprint
import urllib.parse
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.loader import ItemLoader
from items import FoodcomItem
from scrapy import Request
from tqdm import tqdm


class foodcom(scrapy.Spider):

    name = 'food-com'
    custom_settings = {"FEEDS": {"foodresults_3.csv":{"format":"csv"}}}

    page = 1
    nxp = 1
    # https://www.food.com/recipe/all/healthy?pn=1
    list_url = 'https://www.food.com/recipe/all/healthy?pn='
    start_urls = [list_url + str(page)]

    headers = {
                'user-agent' : "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.93 Safari/537.36"
               }

    try:
        os.remove("foodresults.csv")
    except OSError:
        pass

    def parse(self,response):
        res = response.xpath('//script[@type="application/ld+json"]/text()').get()
        res = json.loads(res)
        #pprint(res)
        # iterate through each batch of recipes - 8 on very first 'page'
        for i in tqdm(range(8)):
            # print(res['itemListElement'][i]['url'])
            link = (res['itemListElement'][i]['url'])
            #yield {"link" : link }
            request = Request(url=link, headers=self.headers, callback=self.parse_details)
            yield request
        # add next_page code here
        next_page = (self.list_url+str(self.nxp+1))
        self.nxp +=1
        if response.xpath("//link/@rel='next\'").get() == "1":
            #print("GET SECOND PAGE")
            yield response.follow(url=next_page,callback=self.parse)

    def parse_details(self,response):
        print("OK")
        # get RECIPE
        res = response.xpath('//script[@type="application/ld+json"]/text()').get()
        recipe = json.loads(res)
        i = 0
        # get length of recipe
        #print(recipe)
        if 'aggregateRating' in recipe.keys():
            if 'ratingValue' in recipe['aggregateRating'].keys():
                rate = recipe['aggregateRating']['ratingValue']
            else:
                rate = 0
            if 'reviewCount' in recipe['aggregateRating'].keys():
                views = recipe['aggregateRating']['reviewCount']
            else:
                views = 0
        else:
            rate = 0
            views = 0
        if 'description' in recipe.keys():
            description = recipe['description']
        else:
            description = ''
        if 'keywords' in recipe.keys():
            keywords = recipe['keywords']
        else:
            keywords = ''
        if 'nutrition' in recipe.keys():
            if 'calories' in recipe['nutrition'].keys():
                calories = recipe['nutrition']['calories']
            else:
                calories = 0
            if 'fatContent' in recipe['nutrition'].keys():
                fatContent = recipe['nutrition']['fatContent']
            else:
                fatContent = 0
            if 'proteinContent' in recipe['nutrition'].keys():
                proteinContent = recipe['nutrition']['proteinContent']
            else:
                proteinContent = 0
        else:
            calories = 0
            fatContent = 0
            proteinContent = 0

        rcpl = (len(recipe['recipeInstructions']))
        ls =[]
        while i < rcpl:
            rcptx = (recipe['recipeInstructions'][i]['text'])
            ls.append(rcptx)
            i += 1
            #print(ls)


        # get INGREDIENTS
        ingredients = (recipe['recipeIngredient'])
        #print(ingredients)

        # get RECIPE NAME
        name = (recipe['name'])

        # yield to items / send to FEEDS
        yield{'name':name,'ingredients':ingredients,'recipe':ls,'rate':rate,'views':views,'description':description,
              'keywords':keywords,'calories':calories,'fatContent':fatContent, 'proteinContent':proteinContent}


# main driver #
if __name__ == "__main__":
    process = CrawlerProcess()
    process.crawl(foodcom)
    process.start()
