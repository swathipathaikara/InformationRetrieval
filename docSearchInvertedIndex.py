#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 18:03:34 2022

@author: swathipathaikara
"""

import requests
#import json
import datetime as dt
from twisted.internet import reactor
from scrapy.crawler import CrawlerRunner
from scrapy.utils.project import get_project_settings

#from my_project.spiders.deals import DealsSpider
from bs4 import BeautifulSoup as bs

import urllib.request
import urllib.parse 



def schedule_next_crawl(null, hour, minute):
    nextweek = (
        dt.datetime.now() + dt.timedelta(seconds=7)
        ).replace(hour=hour, minute=minute, second=0, microsecond=0)
    sleep_time = (nextweek - dt.datetime.now()).total_seconds()
    reactor.callLater(sleep_time, crawl)

def crawl_job():
    settings = get_project_settings()
    runner = CrawlerRunner(settings)
    #return runner.crawl(DealsSpider)

def crawl():
    d = crawl_job()
    # crawl weekly at 1.30 am
    d.addCallback(schedule_next_crawl, hour=1, minute=30)
    
def get_page(url):
    response = urllib.request.urlopen(urllib.request.Request(url, 
                                                            headers={'User-Agent': 'Mozilla'} ))
    soup = bs(response, 
                         'html.parser', 
                         from_encoding=response.info().get_param('charset'))
    
    return soup
 
#import pandas as pd
def roboText(robots):
    disallow = []
    lines = str(robots).splitlines()
    for line in lines:

        if line.strip():
            if not line.startswith('#'):
                split = line.split(':', maxsplit=1)
                #data.append([split[0].strip(), split[1].strip()])
                if(split[0].strip() == 'Disallow'):
                    disallow.append(split[1].strip())

    return disallow


robots = get_page("https://www.coventry.ac.uk/robots.txt")
roboTextDisallowList = roboText(robots)

def mycrawler(seed, maxcount):    
    Details = {}
    DispDescription = []
    #DispList = []    
    Q = [seed]#this is the queue which initially contains the given seed URL
    count = 0
    Title = ''
    Authors = ''
    Abstract = ''
    KeyWords = ''
    Year = ''
    Month = ''
    DOI = ''
    response = requests.get(seed)
    content = bs(response.text,'html.parser')       
    Pagelinks = content.findAll('a', {'class': 'step'})
    for j in range(0, len(Pagelinks)):
        subPageLinks = Pagelinks[j]['href']
        subPageLinks = 'https://pureportal.coventry.ac.uk' + subPageLinks
        Q.append(subPageLinks)
    while(Q!=[] and count < maxcount):
        count +=1
        url = Q.pop(0)
        response = requests.get(url)
        content = bs(response.text,'html.parser')
        links = content.findAll('a', {'class': 'link'})
        for i in range(0, len(links)):
            #print(links[i],'\n')
            if(roboTextDisallowList.__contains__(links[i])):
                continue
            if not links[i]['href'].startswith('#') and not links[i]['href'].startswith('/') and '/persons/' in links[i]['href']:
                #print(links[i]['href'],'\n')
                subURL = links[i]['href']
                Q.append(subURL)
                pubURL = subURL + '/publications'
                #print(pubURL)
                pubResponse = requests.get(pubURL)
                pubContent = bs(pubResponse.text,'html.parser')
                pubLinks = pubContent.findAll('a')
                for j in range(0, len(pubLinks)):
                    if not pubLinks[j]['href'].startswith('#') and not pubLinks[j]['href'].startswith('/') and '/en/publications/' in pubLinks[j]['href']:
                        #print(pubLinks[j]['href'],'\n')
                        reqURL = pubLinks[j]['href']
                        if(reqURL != None and reqURL != '/'):
                            reqURL = reqURL.strip()
                            #print(reqURL)
                            reqResponse = requests.get(reqURL)
                            reqContent = bs(reqResponse.text,'html.parser')
                            description = reqContent.find('div',{'id' : 'cite-BIBTEX'}).get_text().replace('",','/').replace(',','//').replace('booktitle','book')
                            temp = description.replace('//',',').split('/')
                            #print(temp) 
                            for i in range(0, len(temp)):
                                if('title' in temp[i]):
                                    Title = temp[i].split(',')[1]
                                elif(temp[i].strip().startswith('author')):
                                    Authors = temp[i]    
                                elif(temp[i].strip().startswith('abstract')):
                                    Abstract = temp[i]
                                elif(temp[i].strip().startswith('keywords')):
                                    KeyWords = temp[i]
                                elif(temp[i].strip().startswith('year')):
                                    Year = temp[i]
                                elif(temp[i].strip().startswith('month')):
                                    Month = temp[i][0:16]
                                elif(temp[i].strip().startswith('doi')):
                                    DOI = temp[i]
                            Des = Title + Authors + Abstract + KeyWords + Year + Month + DOI
                            Details[pubLinks[j]['href']] = Des
                            #DispDetails = 'Author URL: ' + links[i]['href'] + '\n' + '   Publication URL: ' + pubLinks[j]['href'] + '  ' #+ description 
                            DispDetails = {
                                'authorUrl': subURL,
                                'publicationUrl': pubLinks[j]['href'],
                                'Description': Des
                            }
                            #print(DispDetails)
                            DispDescription.append(DispDetails)
    return(Details,DispDescription)
                        
urlDetails,DispDescription = mycrawler('https://pureportal.coventry.ac.uk/en/organisations/school-of-economics-finance-and-accounting/persons/',10)



import re
def docClensing(docs):
    CleanedDetails = {}
    for k, v in docs.items():
        temp = re.sub(r'[^\x00-\x7F]+','',str(v))
        temp = re.sub(r'@\w+','',temp)
        temp = re.sub(r'\n','',temp)
        temp = re.sub(r'//','',temp)
        temp = re.sub(r'/','',temp)
        temp = temp.replace('{','').replace('}','').replace('=','').replace('"','').replace('\'','')
        temp = temp.replace('author','').replace('title','').replace('month','').replace('year','').replace('day','')
        temp = temp.lower()
        temp = " ".join(temp.split())
        CleanedDetails[k] = temp
    return(CleanedDetails)

CleanedDetails = docClensing(urlDetails)

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

sw = stopwords.words('english')
nltk.download("punkt")
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

ps = PorterStemmer()
def stopWordStemRemoval(docs):
    StopWordStemRemoved = {}
    for k, v in docs.items():
        tokens = word_tokenize(v)
        tmp = ""
        for w in tokens:
            if w not in sw:
                tmp += ps.stem(w) + " "
        StopWordStemRemoved[k] = tmp

    return(StopWordStemRemoved)

StopWordStemRemoved = stopWordStemRemoval(CleanedDetails)


inverted_index = {}
def inverted_index_call(docs):
    new_terms = {}
    for k, doc in docs.items():
        new_tokens = doc.split()
        new_terms[k] = new_tokens
        for term in new_tokens:
            if term in inverted_index:
                inverted_index[term].add(k) #existing word
            else: inverted_index[term] = {k} #New word
    for term in inverted_index.keys():
        for k,doc in new_terms.items():
            if not doc.__contains__(term):
                val = list(inverted_index[term])
                if val.__contains__(k):
                    newVal = val.remove(k) #removing deleted word
                    inverted_index[term] = newVal
        
    
    return(inverted_index)
        
inverted_index = inverted_index_call(StopWordStemRemoved)


df = {}
for i in inverted_index:
    df[i] = len(inverted_index[i])

total_vocab = [x for x in df]
words_count = len(total_vocab)

import numpy as np

def tf_idf_cal(docs):
    tf_idf = {}
    N = len(docs.keys())
    for k, doc in docs.items():
        tokens = doc.split()
        #counter = i
        for token in np.unique(tokens):
            Df = df[token]
            tf = Df/words_count
            idf = np.log(N/(Df+1))
            tf_idf[k, token] = tf*idf
    return(tf_idf)

tf_idf = tf_idf_cal(StopWordStemRemoved)

words = np.array(list(inverted_index.keys()))



def vector_creation(docs):
    vector = {}
    i_tf_idf = []
    for k, doc in docs.items():
        i_tf_idf = []
        for i in range(0,len(words)):
            i_token =  words[i]
            i_vals = inverted_index[i_token]
            if(k in i_vals):
                i_tf_idf.append(tf_idf[k,i_token])
            else:
                i_tf_idf.append(0)
    
        vector[k] = np.array(i_tf_idf)
    return(vector)

vector = vector_creation(StopWordStemRemoved)


#####Query Processing
def or_postings(posting1, posting2):
    p1 = 0
    p2 = 0
    result = list()
    posting1.sort()
    posting2.sort()
    while p1 < len(posting1) and p2 < len(posting2):
        if(posting1[p1] == posting2[p2]):
            result.append(posting1[p1])
            p1 += 1
            p2 += 1
        elif posting1[p1] > posting2[p2]:
            result.append(posting2[p2])
            p2 += 1
        else:
            result.append(posting1[p1])
            p1 += 1
    while p1 < len(posting1):
        result.append(posting1[p1])
        p1 += 1
    while p2 < len(posting2):
        result.append(posting2[p2])
        p2 += 1
    return result    
        

def and_postings(posting1,posting2):
    p1 = 0
    p2 = 0
    result = list()
    posting1.sort()
    posting2.sort()
    while p1 < len(posting1) and p2 < len(posting2):
        if(posting1[p1] == posting2[p2]):
            result.append(posting1[p1])
            p1 += 1
            p2 += 1
        elif posting1[p1] > posting2[p2]:
            p2 += 1
        else:
            p1 += 1
    return result



def resultedPostinglist(query):
    tokens = word_tokenize(query)
    p1 = tokens[0]
    p2 = tokens[len(tokens)-1]
    if(' or' in query or 'or ' in query):
        resultedPostingList = or_postings(list(inverted_index[p1]),list(inverted_index[p2]))
    else:
        resultedPostingList = and_postings(list(inverted_index[p1]),list(inverted_index[p2]))
        
    return(resultedPostingList)     
    

def Query_TF_IDF(query):
    Qtf_idf = {}
    N = len(urlDetails)
    #counter = 1
    for token in word_tokenize(query):
        Df = df[token]
        tf = Df/words_count
        idf = np.log(N/(Df+1))
        Qtf_idf[1, token] = tf*idf
    return(Qtf_idf)


    
def QvectorCreation(query):
    Qvector = {}
    Qtf_idf = Query_TF_IDF(query.replace(' and','').replace('and ','').replace(' or','').replace('or ',''))
    Qi_tf_idf = []
    for i in range(0,len(words)):
        i_token =  words[i]
        #i_vals = inverted_index[i_token]
        if(i_token in word_tokenize(query)):
            Qi_tf_idf.append(Qtf_idf[1,i_token])
        else:
            Qi_tf_idf.append(0)
    
    Qvector[query] = np.array(Qi_tf_idf)
    return(Qvector)    
 
   
#import json

#import math
def calc_inner(v1, v2):
    ans = 0
    for i in range(len(v1)):
        ans += v1[i]*v2[i]
    return ans

def calc_length(v):
    tmp = 0
    for x in v:
        tmp += x**2
    return tmp 

def calc_cosine(v1,v2): 
    return calc_inner(v1,v2) / (calc_length(v1)*calc_length(v2))

def get_similarity(query):
    similarity = {}
    DispList = []
    query = query.lower()
    resultedPostingList = resultedPostinglist(query)
    Qvector = QvectorCreation(query)
    for i in range(0,len(DispDescription)):
        for j in range(0,len(resultedPostingList)):
            if(DispDescription[i].get('publicationUrl') == resultedPostingList[j]):
                similarity[i] = calc_cosine(Qvector[query],vector[resultedPostingList[j]])
                break
            else:
                similarity[i] = 0   
    similarity_sorted = sorted(similarity.items(),key = lambda X: X[1], reverse = True)
    for k, v in similarity_sorted:
        if(v != 0.0):
            DispList.append(DispDescription[k])
    return(DispList,len(DispList))
    
    
#q1 = 'eliana or 2016'
#sim = get_similarity(q1)
#for d in sim:
    #print(d,'\n')   
    