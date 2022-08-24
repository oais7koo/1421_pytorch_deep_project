# -*- coding: utf-8 -*-
# author : oaiskoo
# date : 200211
# goal : 개인용 모듈
#

# IO
# oaislib.get_one_filename(dir_str, ext) # 파일패스에 있는 해당


## oaislib.fn_remove_all_files_in_folder(folder_name) # 폴더 내 파일 삭제 
## oaislib.fn_read_utftxt_to_list(filename)  #utf8의 텍스트를 읽어 들여서 리스트로 리턴 
## oaislib.fn_get_clean_text(text) #문장에서 기호제거
## oaislib.fn_get_text_ko(text)  #문장에서 한글 추출
## oaislib.fn_get_text_en(text) #문장에서 영문추출
## oaislib.fn_get_number(text) #문장에서 숫자 추출
##
## oaislib.fn_get_timetail() #파일명 생성을 위한 테일링 YYMMDD_HHMMSS 생성

## ## DB
## oaislib.fn_connect_dev_db() #개발부DB 연결
## oaislib.fn_connect_lab_db() #연구소DB 연결
## oaislib.fn_get_df_from_mstuv_dev_db(sql_str) #개발서버에서 sql결과를 df로 리턴
## oaislib.fn_get_df_from_lab_db(sql_str) #연구소 서버에서 sql결과를 df로 리턴
## oaislib.fn_get_df_from_test_mtrace_db(sql_str): test_mtrace db에서 쿼리 실행
## oaislib.fn_run_sql_to_mstuv_dev_db(sql_str) #mlab 개발DB에 sql문 실행
## oaislib.fn_run_sql_to_lab_db(sql_str) # mlab lab DB에 sql문 실행

## 날짜
### date_obj = oaislib.date_obj_gen_by_yyyymmdd(str) : 8자리 숫자를 날짜 객체로 리턴
### df = oaislib.date_sequence_gen(sdate, edate,date_type): 두 날짜객체 사이의 모든 날짜를 df로 구하기 
## oaislib.fn_get_date_str(): #YYYY-MM-DD의 날짜 문자열 리턴
## oaislib.get_8_date_str() # 8자리 날짜 스트링 리턴

## 기타
## oaislib.fn_change_element_in_list_to_int(lst) # 요소를 숫자로 변환
## oaislib.fn_disploop(prefix, i, interval, all) # loop display
## oaislib.fn_init_folder(folder_path) 폴더가 없으면 폴더 생성하고 폴더가 있으면 내부 파일 삭제
## oaislib.fn_rm_emoji(text) 텍스트에서 이모지 제거
## sim_df = oaislib.fn_sentence_cosine_sim(sen_df, num_sim_sentences)

# new_df, colnm_df  = oaislib.colnm_ko2en(df, var_prefix): # 한글 컬럼명을 영어로 변경 함수


# module for oaiskoo

'''
사용법
import shutil
if os.path.exists("../python/oaislib.py"):
    shutil.copy("../python/oaislib_org.py','oaislib.py")
import oaislib
'''
import os
import re
#import pymysql
import pandas as pd
from datetime import datetime
import glob
from os.path import basename
import os.path
import openpyxl
from openpyxl import load_workbook
import sys

#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity

## sen_df 인덱스와 문장의 컬럼으로 이루어진 입력문
## num_sim_sentence - 유사문장 도출 개수
def fn_sentence_cosine_sim(sen_df, num_sim_sentences):

    ## 문장에 null이 있으면 tfidf 적용시 에러가 발생하므로 제외
    sen_df = sen_df.dropna()

    ## 컬럼명 변경
    sen_df.columns = ['no','sentence']
    sen_cnt = len(sen_df)

    ## 전체 문장의 단어를 벡터화함
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sen_df['sentence'])

    ##ps 각 문장 별로 N개 이내 및 유사도가 0 초과인 유사문장을 받음
    ## 결과 벡터 생성
        
    sim_df = pd.DataFrame(index=range(sen_cnt * num_sim_sentences),
                          columns = ['no1', 'sen1', 'no2', 'sen2', 'rank', 'sim'])
    idx_total = -1
    
    ## 각 문장 별로 유사도가 높은 N개 이내의 유사문장 벡터를 받음
    for i in range(sen_cnt):
        idx1 = sen_df['no'].iloc[i]
        sentence1 = sen_df['sentence'].iloc[i]

        sub_sim_df = pd.DataFrame(index=range(sen_cnt),
                              columns = ['no1', 'sen1', 'no2', 'sen2', 'rank', 'sim'])
        idx_sub = -1
        ##하나의 문장에 대해서 다른 모든 문장의 유사도를 구함
        for j in range(sen_cnt):
            if i != j:
                idx2 = sen_df['no'].iloc[j]
                sen2 = sen_df['sentence'].iloc[j]
    
                rlt = cosine_similarity(tfidf_matrix[i:(i+1)], tfidf_matrix[j:(j+1)])
                
                idx_sub += 1
                sub_sim_df['no1'].iloc[idx_sub] = idx1
                sub_sim_df['sen1'].iloc[idx_sub] = sentence1
                sub_sim_df['no2'].iloc[idx_sub] = idx2
                sub_sim_df['sen2'].iloc[idx_sub] = sen2
                sub_sim_df['sim'].iloc[idx_sub] = rlt[0][0]
        sub_sim_df.dropna(subset = ['sim'], inplace=True)
        sub_sim_df.sort_values(by=['sim'], axis=0, ascending=False, inplace=True)
        sub_sim_df = sub_sim_df[sub_sim_df['sim'] > 0]
    
        ## insert rank
        num_sub = len(sub_sim_df)
        for j in range(num_sub):
            sub_sim_df['rank'].iloc[j] = j+1
        
        num_insert_row = min(num_sub, num_sim_sentences)
        
        for j in range(num_insert_row):
            idx_total += 1
            sim_df['no1'].iloc[idx_total] = sub_sim_df['no1'].iloc[j]
            sim_df['sen1'].iloc[idx_total] = sub_sim_df['sen1'].iloc[j]
            sim_df['no2'].iloc[idx_total] = sub_sim_df['no2'].iloc[j]
            sim_df['sen2'].iloc[idx_total] = sub_sim_df['sen2'].iloc[j]
            sim_df['rank'].iloc[idx_total] = sub_sim_df['rank'].iloc[j]
            sim_df['sim'].iloc[idx_total] = sub_sim_df['sim'].iloc[j]
    
    sim_df = sim_df[:idx_total+1]
    
    return sim_df



## remove all expt bmp
def fn_rm_all_expt_bmp(text):
    only_BMP_pattern = re.compile("["
    u"\U00010000-\U0010FFFF"  #BMP characters 이외
                           "]+", flags=re.UNICODE)
    rlt = only_BMP_pattern.sub(r'', text)
        
    return rlt

## remove emoji
def fn_rm_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    rlt_v = emoji_pattern.sub(r'', text)
    
    return rlt_v

## output dir gen
def fn_output_dir_gen(dir_str):
    if os.path.exists(dir_str):
        pass
    else:
        os.mkdir(dir_str)

# 폴더 내 파일 삭제 
def fn_remove_all_files_in_folder(folder_name):
    if os.path.exists(folder_name):
        for file in os.scandir(folder_name):
            os.remove(file.path)
        return 0
    else:
        return 1
    
# utf8의 텍스트를 읽어 들여서 리스트로 리턴 
def fn_read_utftxt_to_list(filename):
    output = list()
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip() #공백제거
            output.append(line)
    f.close()
    return output
# filename ='input/kor_sentences_utf8.txt'
#rlt = oais.read_utf_txt_to_list(filename)

# 문장에서 한글, 영어, 숫자만 추출
def fn_get_ko_en_num_from_text(text):
    text = text.replace('\n', ' ')
    text = re.sub('[^ㄱ-ㅎ|가-힣|a-z|A-Z|0-9|\*]',' ', text)
    text = " ".join(text.split()) # str을 리스트로 나눈 후 다시 join하면 공백이 하나로 합쳐짐(https://dayglo.tistory.com/54)
    return text

# 문장에서 기호 제거
def fn_get_clean_text(text):
    text = str(text)
    text = text.replace("\\","") #위의 표현으로 역슬래시가 제거가 안되어서 추가함
    text = re.sub('[-=+#/\?:^$.,@*\"※~&%ㆍ!ㅡ』\\;_‘|\(\)\[\]\<\>`\'…》]', '', text)
    return text

# 문장에서 한글만 추출
def fn_get_text_ko(text):
    text = str(text)
    text = re.compile('[가-힣]+').findall(text)
    text = ' '.join(text)
    return text

# 문장에서 영어만 추출
def fn_get_text_en(text):
    text = str(text)
    text = re.compile('[a-zA-Z]+').findall(text)
    text = ' '.join(text)
    return text

# 문장에서 숫자만 추출
def fn_get_number(text):
    text = str(text)
    text = re.compile('[0-9.]+').findall(text)
    text = ' '.join(text)
    return text
    
## 지금 시각을 YYMMDD_HHMMSS 형태로 리턴하기
def fn_get_timetail():
    now = datetime.now()
    yy =  str(now.year)[2:]
    mm = str(now.month)
    dd = str(now.day)
    hh = str(now.hour)
    mm = str(now.minute)
    ss = str(now.second)
    timetail = yy + mm + dd + '_'+ hh + mm + ss
    return timetail

## connect_db
def fn_connect_dev_db():
    db = pymysql.connect(host='bizdb.smartmlab.com',
                         port=33306, user='stuv', passwd='stuv!@#$',
                         db='stuv', charset='utf8')
    return db

def fn_connect_lab_db():
    db = pymysql.connect(host='192.168.0.190',
                         port=3306, user='oaiskoo', passwd='u4i3o2p1',
                         db='labmstuv', charset='utf8')
    return db

def fn_connect_test_mtrace_db():
    db = pymysql.connect(host='192.168.0.191',
                         port=3306, user='dev', passwd='mlab0201',
                         db='test_mtrace', charset='utf8')
    return db

## 위의 것을 사용(호환성때문에 둠)
def fn_lab_db_connect():
    db = pymysql.connect(host='192.168.0.190',
                         port=3306, user='oaiskoo', passwd='u4i3o2p1',
                         db='labmstuv', charset='utf8')
    return db

## mlab 개발 서버에서 쿼리 결과를 데이터 프레임으로 받음
def fn_get_df_from_mstuv_dev_db(sql_str):

    db = fn_connect_dev_db()
    
    try:
        cursor = db.cursor()
        cursor.execute(sql_str)
        table_col = cursor.description
        table_col = [items[0] for items in table_col]
        rlt = cursor.fetchall()
        rlt_df = pd.DataFrame(list(rlt), columns=table_col)
        rlt_df.columns = map(str.lower, rlt_df.columns)
        
    finally:
        db.close()

    return rlt_df

## 연구소 서버에서 쿼리 결과를 데이터 프레임으로 받음
def fn_get_df_from_lab_db(sql_str):
    db = fn_connect_lab_db()
    try:
        cursor = db.cursor()
        cursor.execute(sql_str)
        table_col = cursor.description
        table_col = [items[0] for items in table_col]
        rlt = cursor.fetchall()
        rlt_df = pd.DataFrame(list(rlt), columns=table_col)
        rlt_df.columns = map(str.lower, rlt_df.columns)

    finally:
        db.close()

    return rlt_df

## 연구소 서버 test_mtrace DB에서 쿼리 결과를 데이터 프레임으로 받음
def fn_get_df_from_test_mtrace_db(sql_str):
    db = fn_connect_test_mtrace_db()
    try:
        cursor = db.cursor()
        cursor.execute(sql_str)
        table_col = cursor.description
        table_col = [items[0] for items in table_col]
        rlt = cursor.fetchall()
        rlt_df = pd.DataFrame(list(rlt), columns=table_col)
        rlt_df.columns = map(str.lower, rlt_df.columns)

    finally:
        db.close()

    return rlt_df

## mlab 개발DB에 sql문 실행
def fn_run_sql_to_mstuv_dev_db(sql_str):
    db = fn_connect_dev_db()
    try:
        with db.cursor() as cursor:
            cursor.execute(sql_str)
            db.commit()
    finally:
        db.close()

## mlab lab DB에 sql문 실행
def fn_run_sql_to_lab_db(sql_str):
    db = fn_connect_lab_db()
    try:
        with db.cursor() as cursor:
            cursor.execute(sql_str)
            db.commit()
    finally:
        db.close()

## lab test_trace DB에 sql문 실행
def fn_run_sql_to_lab_db(sql_str):
    db = fn_connect_test_mtrace_db()    
    try:
        with db.cursor() as cursor:
            cursor.execute(sql_str)
            db.commit()
    finally:
        db.close()
        
        
## YYYY-MM-DD의 날짜 문자열 리턴
def fn_get_date_str():
    now_time = datetime.now()
    date_str = str(now_time.year) + "-" + str(now_time.month) + "-" + str(now_time.day)
    return date_str


################################################################################
## 8자리숫자를 날짜 객체로 변환
################################################################################
##
### date_obj = oaislib.date_obj_gen_by_yyyymmdd(str) : 8자리 숫자를 날짜 객체로 리턴
##
# 유효 날짜가 아니면 1900, 01, 01로 데이터 객체 생성
# 단 그러면, 유효성검증은 항상 해줘야 함
# null로 처리하는 것하고 어떤게 좋은지 고민이 필요

def date_obj_gen_by_yyyymmdd(val):
    
    date_str = str(val)
    if len(date_str) == 8: # 모든 글자가 숫자인지 확인 필요
        year = date_str[0:4]
        mon = date_str[4:6]
        day = date_str[6:8]
        date_obj = datetime.date(int(year), int(mon), int(day))
    else:
        date_obj = datetime.date(1900,1,1)
        
    return date_obj


################################################################################
## 두 날짜(YYYY-MM-DD) 사이의 모든 날짜를 df로 구하기 
################################################################################
## sdate, edate는 날짜 객체
## return 값은 스트링(yyyy-mm-dd)나 obj 형태가 있으며, 이는 output_type에서 결정
## output_type = 'str', 'date'

def date_sequence_gen(sdate, edate, output_type):

    delta = edate - sdate
    day_delta = delta.days + 1

    ## 날짜를 텍스트로 해서 df에 넣음
    df = pd.DataFrame(index=range(day_delta), columns = ['day_str'])

    for i in range(day_delta):

        if output_type == 'str':
            date_obj = sdate + datetime.timedelta(i)
            date_str = date_obj.strftime('%Y-%m-%d')
            df['day_str'].iloc[i] = date_str            
            
        elif output_type == 'date':
            df['day_str'].iloc[i] = sdate + datetime.timedelta(i)

    return df

################################################################################
## 요소를 숫자로 변환
################################################################################
def fn_change_element_in_list_to_int(lst):
    cnt = len(lst)
    for i in range(cnt):
        x = lst[i]
        lst[i] = int(x)
    return lst


## 폴더가 없으면 폴더 생성하고 폴더가 있으면 내부 파일 삭제
def fn_init_folder(folder_path):
    
    if os.path.exists(folder_path):
        fn_remove_all_files_in_folder(folder_path)
    else:
        os.makedirs(folder_path)

def fn_disploop(prefix ,i, interval, all):
    if i % interval == 0:
        print(prefix + '::' + str(i) + '/' + str(all))
            


# 한글 컬럼명을 영어로 변경 함수
# 영문 헤드 데이터 및 영문/한글 컬럼 테이블 리턴
# var_prefix = 'var'
def colnm_ko2en(df, var_prefix):
    col_cnt = len(df.columns)
    colnm_kr = df.columns.to_list()
    colnm_en = []

    for i in range(col_cnt):
        var_nm = var_prefix + "{0:02d}".format(i+1)
        colnm_en.append(var_nm)

    df.columns = colnm_en

    # colnm_df
    colnm_df = pd.DataFrame(index=range(col_cnt), columns=["idx","colnm_kr", "colnm_en"])
    colnm_df["idx"] = list(range(col_cnt))
    colnm_df["colnm_kr"] = colnm_kr
    colnm_df["colnm_en"] = colnm_en

    return df, colnm_df


# 8자리 날짜 스트링 리턴
def get_8_date_str():
    now_time = datetime.now()
    date_str = str(now_time.year)[0:3]+ str(now_time.month) + str(now_time.day)

    return date_str


def get_one_filename(dir_str, ext):
    search_path = dir_str + '/' + '*.' + ext
    rlt = glob.glob(search_path)

    if len(rlt) == 0:
        sys.exit( ext + 'file dont exist')
    elif len(rlt) > 0:
        sys.exit( 'more than one ' +  ext + 'file exist')
    else:
        return rlt[0]
        
# ################################################################################
# 액셀에 시트 넣기 
# ################################################################################
def insert_sheet_to_xlsx(filepath, sheet_name, df):
    book = load_workbook(filepath)
    writer = pd.ExcelWriter(filepath, engine = 'openpyxl')
    writer.book = book

    df.to_excel(writer, sheet_name = sheet_name)
    
    writer.save()
    writer.close()

if __name__ == '__main__':
    #folder_filepath = 'tmp'
    #fn_init_folder(folder_path)

    ## 날짜 시퀀스 구하기
    sdate_str = '2021-01-20'
    edate_str = '2021-02-10'
    df = date_sequence_gen(sdate_str, edate_str,'date')

    print(df)
    print('')

    
