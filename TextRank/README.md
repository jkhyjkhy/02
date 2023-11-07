## 네이버 해외축구 Top 10 기사 3줄 요약하기
### 목차
다음과 같은 목차로 구성됩니다.


0. [개요 ](#0-개요)
1. [Textrank 알고리즘의 기원 Pagerank란? ](#1-Textrank-알고리즘의-기원-Pagerank란-?)
2. [Textrank란? ](#2-Textrank란-?)
3. [TextRank 함수 설명 ](#3-TextRank-함수-설명)
4. [본문 요약 함수 설명 ](#4-본문-요약-함수-설명)
5. [개선해야 할 점 ](#5-개선해야-할-점)


---
### 0. 개요
•	제가 자연어처리 분야와 스포츠에 대한 큰 관심을 가지고 있던 상태에서, 두 관심사를 연결시키는 프로젝트를 진행하였습니다.


•	프로젝트의 목표는 네이버 해외축구 탭에서 현재 이슈가 되는 해외축구 기사 중 TOP 10을 요약문으로 사용자에게 제공하는 자연어처리 기반의 서비스를 개발하는 것이었습니다.


•	프로젝트에서는 Beautiful soup를 이용하여 사용자가 선택한 기사의 본문 내용을 추출하고, TextRank 알고리즘을 사용하여 본문 내에서 가장 가중치(중요도)가 높은 3개의 문장을 작성된 순서대로 추출하여 사용자에게 제공하도록 하였습니다.


TextRank 알고리즘을 구현해보는 프로젝트입니다.

---

### 1. Textrank 알고리즘의 기원 Pagerank란?
구글은 Pagerank라는 알고리즘을 통해 검색 서비스의 질을 크게 개선했다.


Pagerank의 핵심은 ＇더 중요하고 유용한 페이지는 다른 페이지로부터 더 많은 링크를 받는다.’ 라는 것이다.


웹 페이지 A, B, C, D가 있다고 가정해보자, 이때 각 페이지는 각각 같은 양의 초기 중요도(가중치)를 할당받는다.


A가 B, C, D에 링크를 걸었다면 이제부터 A의 가중치(이하 중요도)는 링크를 건 B, C, D에 영향을 받는다.


간단히 설명하자면 이제 B, C, D는 A/3(=A가 건 링크된 페이지의 개수) 만큼의 중요도(=Pagerank 값)를 가져온다.



이때 검색자(=Random Surfer)는 A 페이지를 방문 한 후 검색 결과에 만족하고 검색을 종료하거나 A 페이지에서 만족하지 못 한 채로 다른 페이지로 이동 할 것이다. 


이때 후자를 Damping Factor(=감쇠 인자, 이하 D)라고 하고 전자는 1-D로 표현 될 수 있다.


이제 페이지 A가 B,C,D로부터 링크를 받는다고 가정해보자


A의 중요도(P(A))를 계산하는 공식은 다음과 같다.


P(A) = (1-D)/N + D(P(B)/L(B) + P(C)/L(C) + P(D)/L(D)) (단, N = 전체 페이지의 개수, L(K) = K가 가지고 있는 링크의 수)


이때 모든 페이지의 중요도 합은 1이고, 논문에 따르면 일반적으로 D는 0.85로 계산된다

---
### 2. Textrank란?

PageRank의 이러한 알고리즘과 계산 공식을 적용한 것이 Text Rank 알고리즘이다.


문서에 나타난 토큰(문장, 단어 등)을 페이지에 대응하여 중요도를 계산하는 것이다.


이때 각 토큰들은 페이지들처럼 방향성을 가진 링크를 가지고 있지 않기 때문에 그래프로 표현된다.


따라서 토큰들을 노드라고 하고 ‘토큰 간 유사도’는 엣지에 가중치로 표현된다.


이 ‘토큰 간 유사도’는 TextRank를 제시한 논문에 따르면 다음과 같이 계산된다.


<img width="680" alt="image" src="https://github.com/jkhyjkhy/NLP_Project/assets/69373688/31bd351c-6701-4f72-98d6-857a3979cc7a">



위 식은 토큰 내부에 존재하는 요소(음절, 어절, 형태소 등)를 통해 계산하는데 풀어 설명하자면



**두 토큰 간의 유사도 = 두 토큰에 동시에 등장하는 요소 / (log(토큰 i의 요소 개수) + log(토큰 j의 요소 개수)**

이 된다.

---
### 3. TextRank 함수 설명
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


먼저 매트릭스 x를 매개변수로 받는다. 


df는 0.85를 디폴트로 설정하고 최대 반복횟수를 50회로 지정한다


사이킷 런의 normalize 함수를 이용해 정규화한 x의 값을 A에 저장한다.


A는 다른 토큰과의 유사도(링크) 정보를 가진 매트릭스이다.


이때 L1은 각 열(axis=0)의 합이 1이 되도록 정규화 하는 것이다.


초기 랭크값 R을 각 토큰에 할당한다.


토큰의 개수만큼 1열짜리의 넘파이 2차원 배열이 만들어진다.



다른 페이지로 이동하지 않고 머물 가능성인 편향치를 계산한다.


이때 편향치는 1-df와 토큰의 개수에 따라 결정된다.



사전에 정의해둔 최대 반복횟수 만큼 


df * ( 토큰간 유사도 * 현재 랭크값) + 편향치를 구한다.

---
### 4. 본문 요약 함수 설명
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


크롤링을 통해 가져온 뉴스 본문 데이터 content를 딕셔너리 형태로 저장한다.


각 문장은 key = ‘sentence’에 저장되고 


띄어쓰기 단위로 key =’token’에 저장한다.


(여기선 띄어쓰기 단위로 나눴지만 komoran과 같은 한국어 전처리 분석기 사용을 고려해야 할 것 같다.)



마지막으로 판다스 데이터 프레임에 넣어 확인해본다. 

---

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
    print(R.shape)
    R = R.sum(axis=1)
    print(R.shape)
    indexs = R.argsort()[-3:] # 랭크값 상위 세 문장의 인덱스를 가져옴
    print(indexs)
    for index in sorted(indexs): # 뉴스 구조의 순서를 유지하기 위해 정렬함
        print(data_frame['sentence'][index])
        print()
---

iterrows() 메소드를 사용해 매트릭스 전체를 순환하며 조사하는 반복문을 만들고(행 기준)


계산중인 문장이 같은 문장이라면 유사도 값을 0으로 만듦


두 문장내 토큰 교집합의 개수를 구함


로그값끼리 더한 값으로 나눠서 유사도를 만들어 그 값을 i_row_vec에 저장하고 


한 문장(행)에 대한 계산이 끝날때마다 similarity_matrix에 append한다.


넘파이 배열로 변환하여 가중치 그래프라고 명명하고


textrank함수에 가중치 그래프를 넣어 계산한 후


상위 3개의 중요도가 높은 문장을 추출해서 제공함, 이때 추출한 문장의 작성순서가 뒤바뀌지 않게 sorted된 indexs를 순회하는 반복문에서 출력

---
### 5. 개선해야 할 점
<img width="1026" alt="스크린샷 2023-11-07 오후 8 29 25" src="https://github.com/jkhyjkhy/NLP_Project/assets/69373688/d2108e7f-48f3-49b7-8c05-1d32f79522ee">
1. 추출적 접근법이 아니기 때문에 가끔 어색한 문장이 등장할때도 있음, 추후 요약문을 생성하는 추상적 접근법도 딥러닝의 관점에서 접근해볼 생각임
2. 문장과 문장의 구분이 .(마침표)?(물음표)!(느낌표)\s+(연속된 공백)으로 구분되어 있고 토큰의 구분 또한 단순한 띄어쓰기로 구분되어 있어 적합한 토큰 전처리 방법이 강구됨 --> 추후 프로젝트에서 꼬꼬마, 꼬모란, 한나눔, 메캅 등의 형태소 분석기를 이용한 전처리로 해결 시도 예정
3. 서비스의 목적으로 만들지 않아 웹이나 어떤 플랫폼에서 동작하는 것이 아닌 파이썬 코드로만 작성되어 사용자의 접근이 불편함
4. 크롤링 시 beautifulsoup를 이용했는데 사이트 별로 차이는 있지만 대부분 한 태그 내에 기사 본문 외에도 사진 해설, 기자 혹은 리포터 이름, 관련 기사 링크 등이 한 번에 들어 있어 필요한 본문 내용만 가져 올 수가 없었음 --> 사이트별로 본문 전처리 함수를 따로 만들거나 불용어 사전의 생성이 필요함
5. 키워드로서 의미가 없는 불용어 처리(조사 제거 등)가 되어 있지 않기 때문에 적합한 한국어 자연어처리 프로그램이라고 보기 어려움


#### reference : https://hoonzi-text.tistory.com/68, 문서 요약 하기 (with textrank)
#### reference2 : https://lovit.github.io/nlp/2019/04/30/textrank/, TextRank 를 이용한 키워드 추출과 핵심 문장 추출 (구현과 실험)
