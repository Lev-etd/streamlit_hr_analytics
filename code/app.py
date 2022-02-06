import os
import streamlit as st
import pathlib
from pathlib import Path

from utils import save_file, run_analysis, generate_pdf, run_emotion_analysis

root = os.getcwd()


def add_to_list(question):
    with open(f'{root}/question_list', 'a+') as f:
        f.seek(0)
        length_of_file = len(f.readlines())
        f.read()
        print(length_of_file)
        question = question.strip('\n')
        f.write(f'Вопрос номер {int(length_of_file / 2) + 1} \n')
        f.write(f'{question} \n')


def read_file_questions():
    with open(f'{root}/question_list', 'r') as f:
        for line in f.readlines():
            st.write(line.strip())


if __name__ == "__main__":
    # @st.cache
    if Path.exists(pathlib.Path(f'{root}/question_list')):
        pass
    else:
        with open(f'{root}/question_list', 'w'):
            pass

    st.set_page_config(page_title="STT for hr")

    starting_image_widget = st.empty()
    uploaded_video = starting_image_widget.file_uploader(
        "Загрузка видео",
        type=["mp4", "avi"],
        accept_multiple_files=True,
        help="Здесь можно загрузить одно или сразу несколько видео",
    )

    col1, col2, col3 = st.columns([3, 2, 2])
    with col2:
        im_display_slot = st.empty()

    if col2.button(label='Запустить анализ'):
        for ind, video in enumerate(uploaded_video):
            save_file(video)
            file_vid = pathlib.Path(f'{root}/uploaded_videos/{video.name}')
            file_audio = f'{root}/converted_audio/{str(pathlib.Path(video.name).with_suffix(".wav"))}'

            text = run_analysis(path_to_video=file_vid, path_to_audio=file_audio)
            predicted, predicted_logits = run_emotion_analysis(text)
            generate_pdf(text=text, emotion_info=predicted_logits, video_name=video.name, ind_of_video=ind)

    new_question = st.text_input(label='Напишите новый вопрос сюда, чтобы добавить в файл')
    col1, col2, col3 = st.columns([3, 2, 2])
    with col2:
        if st.button(label='Добавить вопрос'):
            add_to_list(new_question)

    if st.sidebar.button(label='Показать вопросы'):
        read_file_questions()
