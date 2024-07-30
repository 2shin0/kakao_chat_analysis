import re
import pandas as pd
import chardet   # 자동 인코딩 감지를 위한 라이브러리
import emoji
from urlextract import URLExtract
from transformers import pipeline


classifier = pipeline("text-classification", model="matthewburke/korean_sentiment")

def detect_encoding(file):
    raw_data = file.read(10000)
    result = chardet.detect(raw_data)
    return result['encoding']

def extract_emojis(text):
    emoji_list = [c for c in text if c in emoji.EMOJI_DATA]
    return emoji_list

# 각 비언어적 표현을 제거하는 함수
def remove_nonverbal(text, nonverbal_items):
    for item in nonverbal_items:
        text = text.replace(item, '')  # 각 아이템을 문자열에서 제거
    return text.strip()  # 양 끝 공백 제거


def assign_region(user_name, regions):
    for region in regions:
        if region in user_name:
            return region
    return None

# 사용자명에서 @ 부분 제거
def extract_user_names(df):
    user_names = df['user_name'].tolist()
    user_names = [re.sub(r'@\s*', '', name) for name in user_names]
    return user_names

def remove_user_mentions(clean_text, user_names):
    for user_name in user_names:
        pattern = rf'@\s*{re.escape(user_name)}'
        clean_text = re.sub(pattern, '', clean_text).strip()
    return clean_text


def classify_sentiment(text):
    result = classifier(text[:512])[0]
    label = result['label']
    score = result['score']
    return label, score



def katalk_msg_parse(file):
    encoding = detect_encoding(file)
    text_data = file.read().decode(encoding).splitlines()

    # file.seek(0)
    my_katalk_data = list()
    katalk_msg_pattern = "[0-9]{4}[년.] [0-9]{1,2}[월.] [0-9]{1,2}[일.] 오\S [0-9]{1,2}:[0-9]{1,2},.*:"
    date_info = "[0-9]{4}년 [0-9]{1,2}월 [0-9]{1,2}일 \S요일"
    in_out_info = "[0-9]{4}[년.] [0-9]{1,2}[월.] [0-9]{1,2}[일.] 오\S [0-9]{1,2}:[0-9]{1,2}:.*"

    for line in text_data:
        if re.match(date_info, line) or re.match(in_out_info, line):
            continue
        elif line == '\n':
            continue
        elif re.match(katalk_msg_pattern, line):
            try:
                line = line.split(",", 1)
                if len(line) < 2:
                    continue
                date_time = line[0].strip()
                user_text = line[1].split(" : ", maxsplit=1)
                if len(user_text) < 2:
                    continue
                user_name = user_text[0].strip()
                text = user_text[1].strip()
                my_katalk_data.append({'date_time': date_time,
                                        'user_name': user_name,
                                        'text': text
                                        })
            except Exception as e:
                print(f"Error processing line: {line}")
                print(f"Exception: {e}")
        else:
            if len(my_katalk_data) > 0:
                my_katalk_data[-1]['text'] += "\n" + line.strip()

    df = pd.DataFrame(my_katalk_data)

    # 전처리
    phrases_to_remove = [
        '/연수문의', '/8월컨퍼런스', '/톡방코드', '/기존선도교사이수기준', '/일반교사이수처리기준', 
        '/톡방사용법', '/전반적인사업안내trythis', '/공지만올라오게해주세요', '/닉네임규정'
    ]
    pattern = '|'.join(map(re.escape, phrases_to_remove))
    df = df[~df['text'].str.contains(pattern, regex=True)]

    # 오픈채팅봇 처리
    df = df[df['user_name'] != '오픈채팅봇']

    # 삭제된 메시지 처리
    df = df[df['text'] != '삭제된 메시지입니다.']

    # 지역 업데이트
    regions = [
        '서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', '경기', '강원',
        '충북', '충남', '전북', '전남', '경북', '경남', '제주'
    ]

    df['region'] = df['user_name'].apply(assign_region, args=(regions,))
    df['region'] = df['region'].where(df['region'].notna(), None)

    # 날짜 및 시간 처리
    df['date_time'] = df['date_time'].str.replace('오전', 'AM')
    df['date_time'] = df['date_time'].str.replace('오후', 'PM')
    df['date_time'] = pd.to_datetime(df['date_time'], format='%Y. %m. %d. %p %I:%M')
    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day
    df['weekday'] = df['date_time'].dt.day_name()
    df['hour'] = df['date_time'].dt.hour

    # 사진 및 동영상 처리
    df['msg_len'] = df['text'].str.len()
    df['msg_word_count'] = df['text'].str.split().str.len()
    audio_visual_text = '^동영상$|^사진$|^사진 [0-9]{1,2}장$'
    mask = df['text'].str.contains(audio_visual_text)
    df.loc[mask, 'audio_visual'] = 1
    df.loc[~mask, 'audio_visual'] = 0
    df.loc[mask, 'msg_len'] = 0
    df.loc[mask, 'msg_word_count'] = 0


    mimetic= "[ㄷㅋㅎㅠㅜ!?~]+"
    punctuations = "[,.]{2,}"
    emo_type1_facial1 = "[;:]{1}[\^\'-]?[)(DPpboOX]"
    emo_type1_facial2 = "[>ㅜㅠㅡ@\^][ㅁㅇ0oO\._\-]*[\^ㅜㅠㅡ@<];*"
    emo_type3 = "\(.+?\)"
    etc = '사진|동영상'

    nonverbal_list = [mimetic, punctuations, emo_type1_facial1, emo_type1_facial2, emo_type3, etc]

    df['nonverbal'] = df['text'].str.findall('|'.join(nonverbal_list)) + df['text'].map(extract_emojis)
    df['nonverbal_count'] = df['nonverbal'].apply(len)

    # URL 추출
    extractor = URLExtract()
    df['url'] = df['text'].apply(extractor.find_urls)
    df['url_count'] = df['url'].apply(len)


    df['clean_text'] = df.apply(lambda row: remove_nonverbal(row['text'], row['nonverbal']), axis=1)
    user_names_list = extract_user_names(df)
    df['clean_text'] = df['clean_text'].apply(lambda text: remove_user_mentions(text, user_names_list))
    df = df[~df['clean_text'].str.contains('@')]


    df['sentiment_label'] = df['clean_text'].apply(lambda text: classify_sentiment(text)[0])
    df['sentiment_score'] = df['clean_text'].apply(lambda text: classify_sentiment(text)[1])
    df['is_positive'] = df['sentiment_label'] == 'LABEL_1'  # 수정: 감성 분류 LABEL_1을 긍정으로 판단

    return df