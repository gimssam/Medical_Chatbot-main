import os
import csv
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from datetime import datetime
import openai
from dotenv import load_dotenv
from streamlit_geolocation import streamlit_geolocation

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 환경 변수를 로드합니다.
load_dotenv()

# API 키를 환경 변수에서 가져옵니다.
API_KEY = os.getenv("API_KEY")
os.environ["OPENAI_API_KEY"] = API_KEY 

# Langchain을 활용하기 위한 설정과 RAG 설정을 진행합니다.
llm = ChatOpenAI(model='gpt-4o-mini',
    temperature=0.1,
)

cache_dir = LocalFileStore("./.cache/practice/")

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
)

loader = UnstructuredFileLoader("./refer.txt")

docs = loader.load_and_split(text_splitter=splitter)

embeddings = OpenAIEmbeddings()

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

vectorstore = FAISS.from_documents(docs, cached_embeddings)

retriever = vectorstore.as_retriever()

# Streamlit 세션 상태를 초기화합니다. 이는 대화 내역을 저장하는 데 사용됩니다.
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 사용자 질문에 대한 응답을 처리하는 함수입니다.
def ask_gpt(user_question):
    # 이전 대화 내역을 기반으로 CHATGPT에게 요청할 쿼리를 생성합니다.
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 의사입니다. 응급 상황에 대한 조치 내용은 첨부된 문서를 참조하여 대답합니다. 
            의료와 관련된 질문에 친절히 대답하며, 심각하거나 큰 질병이 우려되는 상황에서는 병원 방문을 권유하세요:
            \n\n
            {context}",
            """
        ),
        ("human", "{question}"),
    ]
    )

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )
    result = chain.invoke(user_question)
        
    return result.content


# 지도 생성
def create_map(df_hospitals, my_locations):
    # 서울의 중심에 지도 생성
    latitude, longitude = my_locations
    m = folium.Map(location=[latitude, longitude], zoom_start=12)
    # 데이터 프레임의 각 행에 대하여
    for index, row in df_hospitals.iterrows():
        # 마커 추가
        folium.Marker(
            [row['위도'], row['경도']],
            tooltip=row['기관명'],  # 팝업에 표시될 내용row['기관명']  # 마우스 오버시 표시될 내용
        ).add_to(m)
        
    folium.Marker(
            [latitude, longitude],
            tooltip='내 위치',
            icon = folium.map.Icon('red')
        ).add_to(m)

    return m


# 메인 함수입니다.
if __name__ == "__main__":
    df_hospitals = pd.read_csv('의료기관통합.csv', index_col = 0)
    
    # Streamlit 페이지 설정
    st.set_page_config(layout="wide", page_title="LLM기반의 의료 상담 앱")
    # HTML 컴포넌트를 사용하여 위치 정보 표시
    # 페이지 제목
    st.title('LLM기반의 의료 상담 앱')

    # 질문 입력을 위한 텍스트 박스
    question = st.text_input("증상을 입력하세요", "")

    # 질문에 대한 답변을 생성하는 버튼
    if st.button('AI 분석 답변 생성 시작'):
        if question:
            answer = ask_gpt(question)  # GPT-3 모델을 호출하여 답변을 받습니다.
            st.session_state.chat_history.append(f"질문: {question}")
            st.session_state.chat_history.append(f"답변: {answer}")
            # 모든 대화 내역을 화면에 표시합니다.
            for message in st.session_state.chat_history:
                st.text(message)
        else:
            st.error("질문을 제공해주세요.")  # 필수 입력이 없을 경우 사용자에게 알림


    st.write("#### 현재 위치를 기준으로 주변 병원을 추천드리겠습니다.")
    location = streamlit_geolocation()
    
    locations = location['latitude'], location['longitude']
    
    th = 0.1
    if location['latitude']:
        print(location['latitude'])
        try: 
            df_hospitals = df_hospitals.loc[(abs(df_hospitals['위도'] - location['latitude'])<th) & (abs(df_hospitals['경도'] - location['longitude'])< th)]
            map = create_map(df_hospitals, locations)
            folium_static(map)
        except:
           st.write("위치 정보를 가져올 수 없습니다. 위치 서비스가 활성화되어 있는지 확인하세요.")
