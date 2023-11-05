import requests
from bs4 import BeautifulSoup as bs
import re
import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import normalize
from _datetime import datetime


def textrank(x, df=0.85, max_iter=50): # df = Dumping Factor : 0.85라고 가정, max_iter = 최대 반복횟수, 제한 조건을 두고 적절한 값에 수렴할떄까지 반복해야 하나 임의적으로 50회라고 지정
    assert 0 < df < 1

    # initialize
    A = normalize(x, axis=0, norm='l1')
    R = np.ones(A.shape[0]).reshape(-1, 1)
    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1, 1)
    # iteration
    for _ in range(max_iter):
        R = df * (A @ R) + bias

    return R

def get_list(url): # 이 시각 많이 본 뉴스에서 리스트를 가져옴
    print(f"현재 시간은 {datetime.now()}입니다. 상위 10개의 기사를 조회합니다.")
    soup = get_soup(url)
    # news_list = soup.find('ol', attrs={"class" : "news_list"})
    news_list = soup.select('.news_list') # select를 써 본 기억이 없는것 같아 select로 구현
    news_link = {} # 딕셔너리로 선언
    for li in range(0, 10):
        print(f"{li+1}번 기사 : {news_list[0].findAll('a')[li].get_text()}")
        news_link[li] = news_list[0].findAll('a')[li].get('href')

    print("================================")

    checking_number = int(input("조회하고 싶은 기사를 입력하세요 : ")) # 조회를 원하는 기사 본문의 url 저장

    return "https://sports.news.naver.com/" + news_link[checking_number-1]


def get_soup(url): # soup 객체를 가져옴
    res = requests.get(url)
    if res.status_code == 200:
        return bs(res.text, 'html.parser')
    else:
        print(f"Super big fail! with {res.status_code}")

def get_content(soup):
    article = soup.find('div', attrs={"id": "newsEndContents"})

    content = article.text.replace("\n","")
    content = re.split("[\.?!]\s+", content) # 문장을 요소 단위로 분화, 문장 구분


    return content

def get_head(soup):
    head = soup.find('div', attrs={"class": "news_headline"})
    date_info = head.find('div', attrs={"class": "info"})

    print("기사 제목 : " + head.select_one('.title').get_text())
    print("기사 작성 시간 : " + date_info.select_one('span').get_text())
    # print("기자 이름 : " + head.find('em', attrs ={"class" : "media_end_head_journalist_name"}).get_text()[0:3]) , 적용이 안되는 기자가 많아 예외 처리

def summarize_text(content):
    data = []
    for text in content:
        if (text == "" or len(text) == 0):
            continue
        elif text == "All right reserved": # 기사가 끝났으면 반복문 종료
            break
        temp_dict = dict()
        temp_dict['sentence'] = text
        temp_dict['token_list'] = text.split()  # 기초적인 띄어쓰기 단위로 나누기
        # 한국어 전처리 분석기인 꼬꼬마 분석기 사용도 고려해봐야 할 것 같음
        data.append(temp_dict)

    data_frame = pd.DataFrame(data)  # DataFrame에 넣어 깔끔하게 보기
    print(data_frame)
    print("================================")

    # 여기서부터
    # reference : https://hoonzi-text.tistory.com/68, 문서 요약 하기 (with textrank)
    # reference2 : https://lovit.github.io/nlp/2019/04/30/textrank/, TextRank 를 이용한 키워드 추출과 핵심 문장 추출 (구현과 실험)

    similarity_matrix = []
    for i, row_i in data_frame.iterrows():
        i_row_vec = []
        for j, row_j in data_frame.iterrows():
            if i == j:
                i_row_vec.append(0.0)
            else:
                intersection = len(set(row_i['token_list']) & set(row_j['token_list'])) # 유사도 계산의 분자 부분
                log_i = math.log(len(set(row_i['token_list'])))
                log_j = math.log(len(set(row_j['token_list'])))
                similarity = intersection / (log_i + log_j)
                i_row_vec.append(similarity)

        similarity_matrix.append(i_row_vec)

        weightedGraph = np.array(similarity_matrix)

    R = textrank(weightedGraph)
    R = R.sum(axis=1)

    indexs = R.argsort()[-3:] # 랭크값 상위 세 문장의 인덱스를 가져옴
    print(indexs)
    for index in sorted(indexs): # 뉴스 구조의 순서를 유지하기 위해 정렬함
        print(data_frame['sentence'][index])
        print()

if __name__ == '__main__':
    url = get_list("https://sports.news.naver.com/wfootball/index")
    soup = get_soup(url)
    get_head(soup)
    print("================================")
    get_content(soup)
    summarize_text(get_content(soup))