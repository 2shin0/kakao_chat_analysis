# 데이터 분석 관련 코드
from matplotlib import font_manager, rcParams
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from utils import get_stopwords
from gensim import corpora, models
from konlpy.tag import Okt
import os
from io import BytesIO
import streamlit as st
import pandas as pd

classifier = pipeline("text-classification", model="matthewburke/korean_sentiment")
stopwords = get_stopwords()
font_path = './font/BMJUA_TTF.ttf'

# 기본통계
def generate_basic_statistics(df):
    # 메시지 수에 대한 기본 통계
    message_stats = df['text'].describe()
    message_stats_df = message_stats.to_frame().transpose()
    
    # 메시지 길이 분포 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # 메시지 길이 분포
    sns.histplot(df['msg_len'], bins=50, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('메시지 길이 분포')
    axes[0, 0].set_xlabel('메시지 길이')
    axes[0, 0].set_ylabel('빈도')

    # 사용자별 메시지 수
    user_message_count = df['user_name'].value_counts()
    top_users = user_message_count.head(10)
    top_users_df = top_users.reset_index()
    top_users_df.columns = ['사용자', '메시지 수']
    top_users.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('사용자별 메시지 수 (상위 10명)')
    axes[0, 1].set_xlabel('사용자')
    axes[0, 1].set_ylabel('메시지 수')
    axes[0, 1].tick_params(axis='x', rotation=90)

    # 월별 메시지 수
    df['month'] = df['date_time'].dt.month
    monthly_message_count = df.groupby('month').size()
    monthly_message_count_df = monthly_message_count.reset_index()
    monthly_message_count_df.columns = ['월', '메시지 수']  # 컬럼 이름 지정
    monthly_message_count_df.plot(kind='bar', x='월', y='메시지 수', ax=axes[1, 0])
    axes[1, 0].set_title('월별 메시지 수')
    axes[1, 0].set_xlabel('월')
    axes[1, 0].set_ylabel('메시지 수')
    
    # 메시지 길이에 대한 기본 통계 시각화 (Box plot 예시)
    sns.boxplot(x=df['msg_len'], ax=axes[1, 1])
    axes[1, 1].set_title('메시지 길이 통계')
    axes[1, 1].set_xlabel('메시지 길이')

    # Save the plots to a BytesIO object
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    plt.close()
    
    # Optionally save to disk (if needed)
    output_dir = './data'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'basic_statistics.png')
    img_stream.seek(0)
    with open(save_path, 'wb') as f:
        f.write(img_stream.read())

    # 데이터프레임을 세로로 나열하기 위해 col1, col2, col3 열 생성
    col1, col2, col3 = st.columns(3)

    # 첫 번째 열에 첫 번째 데이터프레임 배치
    with col1:
        st.write("#### 메시지 수에 대한 기본 통계")
        st.dataframe(message_stats_df)

    # 두 번째 열에 두 번째 데이터프레임 배치
    with col2:
        st.write("#### 사용자별 메시지 수 (상위 10명)")
        st.dataframe(top_users_df)

    # 세 번째 열에 세 번째 데이터프레임 배치
    with col3:
        st.write('#### 월별 메시지 수')
        st.dataframe(monthly_message_count_df)

    return img_stream



# 워드클라우드
def generate_wordcloud(df):
    # 명사만
    clean_text = ' '.join(df['clean_text'].dropna())

    okt = Okt()
    nouns_txt = okt.nouns(clean_text)

    # 한 글자는 확인 후 의미 없으므로 삭제
    filtered_nouns_txt = [word for word in nouns_txt if len(word) != 1]

    # 빈도 계산
    word_counts = Counter(filtered_nouns_txt)

    # 빈도가 특정 수 이상인 단어 중 stopwords를 제외한 단어들만 추출
    filtered_words = {word: count for word, count in word_counts.items() if count >= 10 and word not in stopwords}
    filtered_words = dict(sorted(filtered_words.items(), key=lambda item: item[1], reverse=True))

    wc = WordCloud(
        font_path=font_path,
        background_color="white",
        width=600,
        height=400,
        max_font_size=250, 
        min_font_size=10,
        contour_color='black',  # Optionally, add contour color,
        contour_width=1,
        colormap='viridis'
        )
    wc.generate_from_frequencies(filtered_words)

    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "result_review.png")
    wc.to_file(output_path)

    plt.figure(figsize=(10, 7))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')

    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)

    return img_stream




# 감정 분석
def extract_keywords(comments, stopwords, top_n=5):
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    X = vectorizer.fit_transform(comments)
    feature_names = vectorizer.get_feature_names_out()
    average_tfidf_scores = X.mean(axis=0).A1
    keywords = dict(zip(feature_names, average_tfidf_scores))
    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return sorted_keywords


def perform_sentiment_analysis(df):

    # 한글 글꼴 적용
    font_manager.fontManager.addfont(font_path)
    font_prop = font_manager.FontProperties(fname=font_path)
    rcParams['font.family'] = font_prop.get_name()

    # 파이차트 생성
    true_count = sum(df['is_positive'])
    false_count = len(df['is_positive']) - true_count

    sizes = [true_count, false_count]
    labels = ['긍정적인 채팅', '부정적인 채팅']
    colors = ['#66c2ff', '#ff9999']
    explode = [0.02, 0.02]
    autopct = make_autopct(sizes)

    plt.figure(figsize=(4, 4))
    plt.pie(sizes,
            labels=labels, 
            autopct=autopct, 
            startangle=260, 
            counterclock=False, 
            explode=explode, 
            shadow=True,
            colors=colors
            )
    plt.title('카톡방 채팅 감정 분석', fontsize=15, fontweight='bold')

    st.pyplot(plt)  # Streamlit에 파이차트 표시
    plt.close()  # matplotlib의 메모리 관리를 위해 추가

    # 데이터프레임을 저장할 변수 초기화
    df_keywords_positive = pd.DataFrame()
    df_keywords_negative = pd.DataFrame()

    # 긍정적인 댓글에서 키워드 추출
    comments_positive = df[df['is_positive']]['clean_text']
    sorted_keywords_positive = extract_keywords(comments_positive, stopwords, top_n=5)
    df_keywords_positive = pd.DataFrame(sorted_keywords_positive, columns=['Keyword', 'Score'])

    # 부정적인 댓글에서 키워드 추출
    comments_negative = df[~df['is_positive']]['clean_text']
    sorted_keywords_negative = extract_keywords(comments_negative, stopwords, top_n=5)
    df_keywords_negative = pd.DataFrame(sorted_keywords_negative, columns=['Keyword', 'Score'])

    # 두 데이터프레임을 나란히 배치
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('긍정적인 채팅 키워드')
        st.dataframe(df_keywords_positive)

    with col2:
        st.subheader('부정적인 채팅 키워드')
        st.dataframe(df_keywords_negative)

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return f'{val}개\n({pct:.2f}%)'
    return my_autopct



# 주제 분석
def preprocess_text(text, stopwords):
    """
    텍스트를 전처리하여 토큰 리스트를 반환합니다.

    :param text: 전처리할 텍스트
    :param stopwords: 불용어 리스트
    :return: 전처리된 토큰 리스트
    """
    okt = Okt()
    tokens = okt.morphs(text)
    tokens = [token for token in tokens if len(token) > 1 and token not in stopwords]
    return tokens

def perform_topic_analysis(df, num_topics=5):
    texts = []
    """
    주어진 데이터프레임의 'clean_text' 컬럼을 사용하여 주제 분석을 수행합니다.

    :param df: 분석할 데이터프레임
    :param num_topics: 추출할 주제의 수
    """
    stopwords = get_stopwords()  # stopwords를 가져옴

    # 'clean_text' 컬럼에서 텍스트 추출
    comments = df['clean_text'].dropna()  # NaN 값 제거

    # 전처리 수행
    preprocessed_comments = [preprocess_text(comment, stopwords) for comment in comments]

    # 사전 생성
    dictionary = corpora.Dictionary(preprocessed_comments)

    # 말뭉치 생성
    corpus = [dictionary.doc2bow(comment) for comment in preprocessed_comments]

    # LDA 모델 빌드
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # Print the topics and their associated keywords
    topics = []
    for idx, topic in lda_model.print_topics():
        topic_keywords = [keyword.split('*')[1].strip('"') for keyword in topic.split('+')]
        topics.append({'Topic': f'Topic {idx}', 'Keywords': ', '.join(topic_keywords)})

    topics_df = pd.DataFrame(topics)
    return topics_df

def making_topic_modeling(df):
    true_comments = df[df['is_positive']==1]
    false_comments = df[df['is_positive']==0]
    positive_topics_df = perform_topic_analysis(true_comments)
    negative_topics_df = perform_topic_analysis(false_comments)   
    return positive_topics_df, negative_topics_df