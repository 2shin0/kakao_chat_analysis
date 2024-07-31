import streamlit as st
from preprocessing import katalk_msg_parse
from analysis import generate_basic_statistics, generate_wordcloud, perform_sentiment_analysis, making_topic_modeling, making_cluster
from utils import load_uploaded_file
import pandas as pd
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="카톡 데이터 분석 웹 애플리케이션",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_process_data(uploaded_file):
    df = katalk_msg_parse(uploaded_file)
    return df


with st.sidebar:
    choice = option_menu("", ["데이터","데이터 분석"],
    icons=['house', 'bi bi-check2-all'],
    menu_icon="app-indicator", default_index=0,
    styles={
    "container": {"padding": "4!important", "background-color": "#fafafa"},
    "icon": {"color": "black", "font-size": "25px"},
    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#fafafa"},
    "nav-link-selected": {"background-color": "#08c7b4"},
    }
    )

if choice == '데이터':
    st.title('카톡 데이터 분석 웹 애플리케이션')
    st.write("아래 파일 업로드 버튼을 눌러 카톡 데이터(.txt 파일)을 업로드해 주세요!")

    uploaded_file = st.file_uploader("파일 업로드", type=["txt"])

    if uploaded_file:
        df = load_and_process_data(uploaded_file)
        st.sidebar.subheader('전처리')
        st.sidebar.write("필터링 없이 적용 버튼을 누르시면 전체 데이터를 확인하실 수 있습니다.")
        
        # 기본값 설정
        min_date = df['date_time'].min().date()
        max_date = df['date_time'].max().date()

        st.sidebar.subheader('필터링')
        user = st.sidebar.text_input("사용자명")
        text_filter = st.sidebar.text_input("내용 필터")
        start_date = st.sidebar.date_input("시작 날짜", min_value=min_date, max_value=max_date, value=min_date)
        end_date = st.sidebar.date_input("끝 날짜", min_value=min_date, max_value=max_date, value=max_date)

        df['date_time'] = pd.to_datetime(df['date_time'], format='%Y. %m. %d. %p %I:%M')

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        if st.sidebar.button('적용'):
            df_filtered = df[(df['date_time'] >= start_date) & (df['date_time'] <= end_date)]
            if user:
                df_filtered = df_filtered[df_filtered['user_name'].str.contains(user, case=False, na=False)]
            if text_filter:
                df_filtered = df_filtered[df_filtered['text'].str.contains(text_filter, case=False, na=False)]
            
            # Save to session state
            st.session_state.df_filtered = df_filtered
            
            st.info("데이터 분석 결과를 확인하고 싶다면 왼쪽 '데이터 분석' 메뉴를 확인해 주세요.")
            st.write(df_filtered)



elif choice == '데이터 분석':
    st.title('데이터 분석')
    
    if 'df_filtered' in st.session_state:
        df_filtered = st.session_state.df_filtered
        
        # Tabs for analysis
        tabs = st.tabs(['기본 통계', '워드클라우드', '감정 분석', '토픽 분석', '군집 분석'])
        
        with tabs[0]:
            statistics_image = generate_basic_statistics(df_filtered)
            st.image(statistics_image)
        
        with tabs[1]:
            wordcloud_img = generate_wordcloud(df_filtered)
            st.image(wordcloud_img)
        
        with tabs[2]:
            sentiment_results = perform_sentiment_analysis(df_filtered)
            st.write(sentiment_results)
        
        with tabs[3]:
            true, false = making_topic_modeling(df_filtered)
            
            st.write('#### 긍정 채팅 토픽 분석')
            st.write(true)
            
            st.write('#### 부정 채팅 토픽 분석')
            st.write(false)
        with tabs[4]:
            clustering_img = making_cluster(df_filtered)
            st.image(clustering_img)
    else:
        st.write("데이터가 필요합니다.")