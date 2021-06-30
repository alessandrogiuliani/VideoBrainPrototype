# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:42:30 2020

@author: Alessandro Giuliani, Maria Madalina Stanciu

"""
import google_trends_wrapper as gt
import pandas as pd

class CategoryRead(object):

    def __init__(self, geo='US'):
        self.pytrend = gt.request.TrendReq(geo=geo)
        categories = self.pytrend.categories()
        self.__pytrendsCategories = categories
        self.geo = geo




    def __get_id_by_category_name(self, graph,searched_tag, tag='children'): 
        key=[] # List to keep track of visited nodes.
        visited=[]
        def inner_get_id_by_name(graph, searched_tag, key, visited, tag='children'):
            if (('name' in graph) and (searched_tag.lower() == graph['name'].lower())):
                key.append(graph["id"])
            elif (tag in graph):
                nodes = [node for node in graph[tag] if node not in visited]
                for node in nodes:
                    visited.append(node)
                    inner_get_id_by_name(node,searched_tag,key,visited)
            return key
        inner_get_id_by_name(graph, searched_tag,key,visited)
        return key[0]




    def get_category_id(self, category):
        self.category=category
        return self.__get_id_by_category_name(self.__pytrendsCategories, category)




    def set_related_searches(self, video_category, pytrend_category,  limit=None, rising=False):
        results = []
        self.pytrend.build_payload(kw_list=[], cat=pytrend_category,gprop='youtube', geo=self.geo,timeframe='today 1-m')
        related_queries_dict = self.pytrend.related_queries()
        if (video_category in related_queries_dict.keys()):
            results = self.__get_related_searches(related_queries_dict[video_category], rising=rising)
        else:
            results = self.__get_related_searches(related_queries_dict, rising=rising)
        related_queries_dict_topics =self.pytrend.related_topics()
        df = related_queries_dict_topics['top']
        for item in self.__get_related_topics_trends(df,pytrend_category):
            if item not in results:
                results.append(item)
        return results[:limit]




    def __get_related_searches(self, related_queries_dict, rising = False):
        result = []
        if (related_queries_dict['top'] is not None):
            top5 = related_queries_dict['top']['query'].values
            for item in top5:
                low = item.lower()
                if low not in result:
                    result.append(low)
        if rising:
            if (related_queries_dict['rising'] is not None):
                top5 = related_queries_dict['rising']['query'].values
                for low in top5:
                  result.append(low)
        return result




    def __get_related_topics_trends(self, df:pd.DataFrame,category):
        result = []
        df_temp = df.loc[df['topic_type'] == 'Topic']
        for item in df_temp.loc[df_temp['hasData'] == True]['topic_title'].values[0:3]:
            low = item.lower()
            result.append(low)
        for topic in df_temp.loc[df_temp['hasData']==True]['topic_mid'].values[0:3]:
            self.pytrend.build_payload(kw_list=[], gprop='youtube', geo=self.geo,timeframe='today 1-m', url=topic)
            related_queries_dict = self.pytrend.related_queries()
            related_topics_results=self.__get_related_searches(related_queries_dict[topic])
            for item in related_topics_results:
                low = item.lower()
                if low not in result:
                    result.append(low)
        return result




